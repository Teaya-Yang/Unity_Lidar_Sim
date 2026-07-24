"""
losses.py
=========
The composite training objective.

    total = w_latent * latent        JEPA: predicted latent vs EMA-target latent
          + w_decode * decode        decoder on PREDICTED latents (physical units)
          + w_recon  * recon         decoder on ENCODED latents (autoencoder anchor)
          + w_var    * variance      VICReg: keep every dimension alive
          + w_cov    * covariance    VICReg: keep dimensions decorrelated

On the two VICReg terms
-----------------------
They are not optional garnish. The latent term alone is minimised perfectly by a
constant embedding, and an EMA target only slows that collapse down rather than
preventing it. The variance hinge forces each latent dimension to keep a standard
deviation of at least `gamma` across the batch; the covariance term pushes the
off-diagonal of the embedding covariance to zero so dimensions carry distinct
information instead of duplicating one.

On w_decode
-----------
The decoder is trained on the PREDICTED latents, not only on encoded ones. That
matters: at evaluation time the decoder only ever sees latents that came out of
a multi-step roll, which drift away from the encoder's output distribution. A
decoder trained solely as an autoencoder is accurate exactly where it is never
used. `w_recon` keeps the anchor; `w_decode` makes it useful.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .config import LossConfig


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Hinge on per-dimension std. z: (..., D) -> scalar."""
    z = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z.var(dim=0) + 1e-8)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Sum of squared off-diagonal covariances, normalised by dimension."""
    z = z.reshape(-1, z.shape[-1])
    n, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return (off ** 2).sum() / d


def compute_losses(out: Dict[str, torch.Tensor],
                   core_tgt_norm: torch.Tensor,
                   dth_tgt_norm: torch.Tensor,
                   core0_norm: torch.Tensor,
                   cfg: LossConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    out           — the dict returned by JEPA.forward
    core_tgt_norm (B,H,3) true core states, normalised
    dth_tgt_norm  (B,H)   true heading changes, normalised
    core0_norm    (B,3)   true core state at t, normalised (recon target)

    Returns (total, per-term scalars for logging).
    """
    # 1. JEPA latent prediction (target already stop-gradded in forward()).
    l_latent = F.mse_loss(out["z_pred"], out["z_tgt"])

    # 2. Decoder on predicted latents — core state + heading change.
    l_core = F.mse_loss(out["dec_pred"], core_tgt_norm)
    l_dth  = F.mse_loss(out["dth_pred"], dth_tgt_norm)
    l_decode = l_core + l_dth

    # 3. Autoencoder anchor.
    l_recon = F.mse_loss(out["recon0"], core0_norm)

    # 4. Anti-collapse, applied to the predicted latents and the context embedding
    #    (a collapsed `c` would silently kill all adaptation, so guard it too).
    l_var = variance_loss(out["z_pred"], cfg.var_gamma) + \
            variance_loss(out["c"], cfg.var_gamma)
    l_cov = covariance_loss(out["z_pred"]) + covariance_loss(out["c"])

    total = (cfg.w_latent * l_latent + cfg.w_decode * l_decode +
             cfg.w_recon * l_recon + cfg.w_var * l_var + cfg.w_cov * l_cov)

    with torch.no_grad():
        emb_std = torch.sqrt(out["z_pred"].reshape(-1, out["z_pred"].shape[-1])
                             .var(dim=0) + 1e-8).mean()
        ctx_std = torch.sqrt(out["c"].var(dim=0) + 1e-8).mean()
        logs = dict(total=total.item(), latent=l_latent.item(),
                    decode=l_decode.item(), core=l_core.item(),
                    dtheta=l_dth.item(), recon=l_recon.item(),
                    var=l_var.item(), cov=l_cov.item(),
                    emb_std=emb_std.item(), ctx_std=ctx_std.item())
    return total, logs
