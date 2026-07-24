"""
models.py
=========
The four networks, and the JEPA wrapper that ties them together.

    ContextEncoder   (B,K,ctx) -> c        "which plant am I in"  (implicit eta)
    StateEncoder     (B,obs)   -> z        "where am I right now" (from OBSERVATION)
    Predictor        (z,a,c)   -> z', dth  latent dynamics + heading change
    Decoder          (z)       -> core     latent -> clean physical state

Observation vs core
-------------------
The StateEncoder consumes the OBSERVATION (obs_dim), the Decoder emits the clean
CORE (3-D). In plain mode obs == core, so this collapses to the original model
and old checkpoints load unchanged. In enriched mode obs is the 10-channel noisy
sensor suite: the encoder must fuse and denoise it into a latent, while the
decoder recovers the clean state — which is exactly the denoising advantage a
JEPA is supposed to have over a direct predictor.

Why the context/state split, collapse, etc. — see the original notes below.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .data import N_CTX_FEAT, N_CORE, N_PRED, Normalizer


def dims_for(enriched: bool) -> Tuple[int, int]:
    """(obs_dim, ctx_dim) for a mode. Kept here so train and load agree."""
    if enriched:
        from .sensors import N_OBS
        return N_OBS, N_OBS + 2
    return N_CORE, N_CTX_FEAT


def _mlp(sizes, act=nn.SiLU, out_act=False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(act())
    return nn.Sequential(*layers)


# ── Encoders ─────────────────────────────────────────────────────────────────
class ContextEncoder(nn.Module):
    """K-step history -> a single dynamics embedding (order-aware GRU)."""

    def __init__(self, in_feat: int, d_ctx: int, hidden: int, n_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(in_feat, hidden, num_layers=n_layers, batch_first=True)
        self.head = _mlp([hidden, hidden, d_ctx])

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:      # (B,K,in) -> (B,d_ctx)
        _, h = self.gru(ctx)
        return self.head(h[-1])


class StateEncoder(nn.Module):
    """One observation -> latent. Also serves as the EMA target."""

    def __init__(self, in_dim: int, d_state: int, hidden: int):
        super().__init__()
        self.net = _mlp([in_dim, hidden, hidden, d_state])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:       # (B,in) -> (B,d)
        return self.net(obs)


# ── Predictor / Decoder ──────────────────────────────────────────────────────
class Predictor(nn.Module):
    """(z, action, c) -> (next latent, heading change), latent residual."""

    def __init__(self, d_state: int, d_ctx: int, hidden: int):
        super().__init__()
        self.net = _mlp([d_state + 2 + d_ctx, hidden, hidden, d_state + 1])
        self.d_state = d_state

    def forward(self, z, a, c) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([z, a, c], dim=-1))
        return z + out[..., :self.d_state], out[..., self.d_state:]


class Decoder(nn.Module):
    """Latent -> clean core state [v, delta, accel]."""

    def __init__(self, d_state: int, hidden: int):
        super().__init__()
        self.net = _mlp([d_state, hidden, hidden, N_CORE])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── The assembled model ──────────────────────────────────────────────────────
class JEPA(nn.Module):
    def __init__(self, obs_dim: int, ctx_dim: int, d_ctx: int, d_state: int,
                 hidden: int, n_ctx_layers: int, ema_tau: float,
                 norm: Normalizer, enriched: bool = False):
        super().__init__()
        self.enriched = enriched
        self.ctx_enc   = ContextEncoder(ctx_dim, d_ctx, hidden, n_ctx_layers)
        self.state_enc = StateEncoder(obs_dim, d_state, hidden)
        self.predictor = Predictor(d_state, d_ctx, hidden)
        self.decoder   = Decoder(d_state, hidden)

        self.target_enc = copy.deepcopy(self.state_enc)
        for p in self.target_enc.parameters():
            p.requires_grad_(False)
        self.ema_tau = ema_tau

        for name, arr in norm.to_dict().items():
            self.register_buffer(name, torch.tensor(arr, dtype=torch.float32))

    # ── normalisation helpers ────────────────────────────────────────────────
    def norm_ctx(self, x):    return (x - self.ctx_mean) / self.ctx_std
    def norm_core(self, x):   return (x - self.core_mean) / self.core_std
    def denorm_core(self, x): return x * self.core_std + self.core_mean
    def norm_pred(self, x):   return (x - self.pred_mean) / self.pred_std
    def denorm_pred(self, x): return x * self.pred_std + self.pred_mean

    def norm_obs(self, x):
        """OBS normalisation — separate buffers in enriched mode, core stats
        otherwise (in plain mode obs == core)."""
        if self.enriched:
            return (x - self.obs_mean) / self.obs_std
        return (x - self.core_mean) / self.core_std

    @torch.no_grad()
    def update_target(self) -> None:
        for tp, sp in zip(self.target_enc.parameters(), self.state_enc.parameters()):
            tp.mul_(self.ema_tau).add_(sp, alpha=1.0 - self.ema_tau)
        for tb, sb in zip(self.target_enc.buffers(), self.state_enc.buffers()):
            tb.copy_(sb)

    # ── training forward pass ────────────────────────────────────────────────
    def forward(self, ctx, obs0, acts, obs_tgt, core0, core_tgt
                ) -> Dict[str, torch.Tensor]:
        """
        ctx      (B,K,ctx_dim)   context features
        obs0     (B,obs_dim)     observation at t          (encoder input)
        acts     (B,H,2)         future actions
        obs_tgt  (B,H,obs_dim)   true future observations  (EMA target input)
        core0    (B,3)           clean core at t           (recon target)
        core_tgt (B,H,3)         clean future core         (decode target)
        """
        B, H, _ = acts.shape
        c = self.ctx_enc(self.norm_ctx(ctx))
        z = self.state_enc(self.norm_obs(obs0))

        z_preds, dth_preds, dec_preds = [], [], []
        for h in range(H):
            z, dth = self.predictor(z, acts[:, h], c)
            z_preds.append(z)
            dth_preds.append(dth)
            dec_preds.append(self.decoder(z))          # -> clean core (normalised)

        z_pred   = torch.stack(z_preds, dim=1)
        dth_pred = torch.cat(dth_preds, dim=1)
        dec_pred = torch.stack(dec_preds, dim=1)

        with torch.no_grad():
            z_tgt = self.target_enc(self.norm_obs(obs_tgt.reshape(-1, obs_tgt.shape[-1])))
            z_tgt = z_tgt.reshape(B, H, -1)

        recon0 = self.decoder(self.state_enc(self.norm_obs(obs0)))
        return dict(z_pred=z_pred, z_tgt=z_tgt, dth_pred=dth_pred,
                    dec_pred=dec_pred, recon0=recon0, c=c)

    # ── inference: multi-step rollout in physical units ──────────────────────
    @torch.no_grad()
    def rollout(self, ctx, obs0, acts) -> Tuple[torch.Tensor, torch.Tensor]:
        """Open-loop latent roll. Returns clean core (B,H,3) and dtheta (B,H),
        both RAW. Note the rollout uses LATENT recurrence — it consumes obs0 once
        and never re-observes, which is the whole point of the latent model."""
        B, H, _ = acts.shape
        c = self.ctx_enc(self.norm_ctx(ctx))
        z = self.state_enc(self.norm_obs(obs0))
        cores, dths = [], []
        for h in range(H):
            z, dth = self.predictor(z, acts[:, h], c)
            cores.append(self.denorm_core(self.decoder(z)))
            dths.append(dth[:, 0] * self.pred_std[3] + self.pred_mean[3])
        return torch.stack(cores, dim=1), torch.stack(dths, dim=1)

    @torch.no_grad()
    def context_embedding(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.ctx_enc(self.norm_ctx(ctx))

    # ── uniform one-step interface (shared with the predictive baseline) ─────
    @torch.no_grad()
    def encode_context(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.ctx_enc(self.norm_ctx(ctx))

    @torch.no_grad()
    def step_from_obs(self, obs_raw, a, c) -> Tuple[torch.Tensor, torch.Tensor]:
        """One step re-encoding a physical OBSERVATION (teacher forcing)."""
        z = self.state_enc(self.norm_obs(obs_raw))
        z2, dth = self.predictor(z, a, c)
        core_next = self.denorm_core(self.decoder(z2))
        dth_raw = dth[:, 0] * self.pred_std[3] + self.pred_mean[3]
        return core_next, dth_raw
