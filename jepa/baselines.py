"""
baselines.py
============
The autoregressive state-space predictor — the baseline the SkyJEPA paper
actually compares against (its "Predictive" model, Table III).

Same inputs as the JEPA (context + current observation + action); the only
differences are the two the paper isolates: it predicts in OBSERVATION space,
not latent space, and it feeds its own predictions back autoregressively.

In enriched mode this is exactly where the disadvantage should bite: the model
must predict the full 10-channel NOISY observation at every step and feed it
back, so noise compounds through the rollout — whereas the JEPA predicts a
denoised latent and only decodes at the end. A small readout maps the predicted
observation to the clean core for metrics. In plain mode observation == core,
the readout vanishes, and this is byte-identical to the original baseline.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .data import N_CORE, N_PRED, Normalizer
from .models import ContextEncoder, Decoder, _mlp, dims_for


class PredictiveModel(nn.Module):
    def __init__(self, obs_dim: int, ctx_dim: int, d_ctx: int, hidden: int,
                 n_ctx_layers: int, norm: Normalizer, enriched: bool = False):
        super().__init__()
        self.enriched = enriched
        self.enc_dim = obs_dim              # prediction is in observation space
        self.ctx_enc = ContextEncoder(ctx_dim, d_ctx, hidden, n_ctx_layers)
        # Predicts an observation INCREMENT (normalised) + heading change.
        self.predictor = _mlp([obs_dim + 2 + d_ctx, hidden, hidden, obs_dim + 1])
        # Readout observation -> clean core. Only needed when obs != core.
        if enriched:
            self.readout = _mlp([obs_dim, hidden, N_CORE])

        for name, arr in norm.to_dict().items():
            self.register_buffer(name, torch.tensor(arr, dtype=torch.float32))

    # normalisation
    def norm_ctx(self, x):    return (x - self.ctx_mean) / self.ctx_std
    def norm_core(self, x):   return (x - self.core_mean) / self.core_std
    def denorm_core(self, x): return x * self.core_std + self.core_mean

    def norm_enc(self, x):
        if self.enriched:
            return (x - self.obs_mean) / self.obs_std
        return (x - self.core_mean) / self.core_std

    def _enc_to_core_n(self, enc_n: torch.Tensor) -> torch.Tensor:
        """Normalised observation -> normalised clean core. Identity in plain
        mode (observation IS the core); a learned readout in enriched mode."""
        return self.readout(enc_n) if self.enriched else enc_n

    @torch.no_grad()
    def encode_context(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.ctx_enc(self.norm_ctx(ctx))

    # ── training forward: free multi-step autoregressive rollout ─────────────
    def forward(self, ctx, obs0, acts
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns normalised observation predictions (B,H,enc_dim), normalised
        heading changes (B,H), and normalised clean-core predictions (B,H,3)."""
        c = self.ctx_enc(self.norm_ctx(ctx))
        enc_n = self.norm_enc(obs0)
        enc_preds, dth_preds, core_preds = [], [], []
        H = acts.shape[1]
        for h in range(H):
            out = self.predictor(torch.cat([enc_n, acts[:, h], c], dim=-1))
            enc_n = enc_n + out[..., :self.enc_dim]
            enc_preds.append(enc_n)
            dth_preds.append(out[..., self.enc_dim:])
            core_preds.append(self._enc_to_core_n(enc_n))
        return (torch.stack(enc_preds, dim=1), torch.cat(dth_preds, dim=1),
                torch.stack(core_preds, dim=1))

    # ── uniform one-step interface (shared with the JEPA) ────────────────────
    @torch.no_grad()
    def step_from_obs(self, obs_raw, a, c) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_n = self.norm_enc(obs_raw)
        out = self.predictor(torch.cat([enc_n, a, c], dim=-1))
        core_next = self.denorm_core(self._enc_to_core_n(enc_n + out[..., :self.enc_dim]))
        dth_raw = out[..., self.enc_dim] * self.pred_std[3] + self.pred_mean[3]
        return core_next, dth_raw

    @torch.no_grad()
    def rollout(self, ctx, obs0, acts) -> Tuple[torch.Tensor, torch.Tensor]:
        """Open-loop roll in RAW units. Feeds its own predicted OBSERVATION back
        (autoregressive in observation space) — the compounding path the paper
        contrasts against the JEPA's latent rollout."""
        c = self.ctx_enc(self.norm_ctx(ctx))
        enc_n = self.norm_enc(obs0)
        cores, dths = [], []
        H = acts.shape[1]
        for h in range(H):
            out = self.predictor(torch.cat([enc_n, acts[:, h], c], dim=-1))
            enc_n = enc_n + out[..., :self.enc_dim]
            cores.append(self.denorm_core(self._enc_to_core_n(enc_n)))
            dths.append(out[..., self.enc_dim] * self.pred_std[3] + self.pred_mean[3])
        return torch.stack(cores, dim=1), torch.stack(dths, dim=1)


def load_predictive(ckpt_path: str, device: str = "cpu"):
    from .config import Config
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ck["config"])
    norm = Normalizer.from_dict(ck["norm"])
    obs_dim, ctx_dim = dims_for(cfg.data.enriched)
    model = PredictiveModel(obs_dim, ctx_dim, cfg.model.d_ctx, cfg.model.hidden,
                            cfg.model.n_ctx_layers, norm, cfg.data.enriched).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg
