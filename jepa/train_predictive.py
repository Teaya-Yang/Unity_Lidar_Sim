"""
train_predictive.py
===================
Trains the autoregressive state-space baseline (baselines.PredictiveModel) on
condition A, so it can be compared against the JEPA under identical conditions.

    python -m jepa.train_predictive --steps 8000 --out runs/predictive

Deliberately uses the SAME data pipeline, context length K, horizon H, width,
optimiser and schedule as train.py. The only thing that differs from the JEPA
run is the model and its (plain MSE) objective — no latent target, no EMA, no
VICReg. That is exactly the ablation the SkyJEPA paper isolates.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import load_split, fit_normalizer, WindowSampler
from .baselines import PredictiveModel
from .models import dims_for
from .rollout import rollout_learned, per_horizon_errors, summarize


@torch.no_grad()
def validate(model, sampler, device, max_n=4000) -> Dict[str, float]:
    model.eval()
    b = sampler.all_windows(stride=7, max_n=max_n)
    pred = rollout_learned(model, b, device)
    errs = per_horizon_errors(pred, b["state_tgt"])
    model.train()
    return summarize(errs)


def train(cfg: Config) -> str:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = cfg.train.device
    os.makedirs(cfg.train.out_dir, exist_ok=True)

    tr = load_split(cfg.data.packed_root, "A", "train", cfg.data)
    va = load_split(cfg.data.packed_root, "A", "val", cfg.data)
    norm = fit_normalizer(tr, cfg.data.K, cfg.data.H, seed=cfg.train.seed)
    tr_s = WindowSampler(tr, cfg.data.K, cfg.data.H, cfg.train.seed, cfg.data.min_rollout_len)
    va_s = WindowSampler(va, cfg.data.K, cfg.data.H, cfg.train.seed + 1, cfg.data.min_rollout_len)
    print(f"[pred] windows: train={len(tr_s)} val={len(va_s)} (K={cfg.data.K}, H={cfg.data.H})"
          + (f"  [ENRICHED obs_dim={tr.obs_dim}]" if cfg.data.enriched else ""))

    obs_dim, ctx_dim = dims_for(cfg.data.enriched)
    model = PredictiveModel(obs_dim, ctx_dim, cfg.model.d_ctx, cfg.model.hidden,
                            cfg.model.n_ctx_layers, norm, cfg.data.enriched).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[pred] parameters: {n_par:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.steps)

    best, best_path = float("inf"), os.path.join(cfg.train.out_dir, "predictive_best.pt")
    hist, t0 = [], time.perf_counter()

    for step in range(1, cfg.train.steps + 1):
        b = tr_s.batch(cfg.train.batch_size)
        ctx   = torch.as_tensor(b["ctx"], dtype=torch.float32, device=device)
        obs0  = torch.as_tensor(b["obs0"], dtype=torch.float32, device=device)
        acts  = torch.as_tensor(b["acts"], dtype=torch.float32, device=device)
        obs_tgt  = torch.as_tensor(b["obs_tgt"], dtype=torch.float32, device=device)
        core_tgt = torch.as_tensor(b["core_tgt"], dtype=torch.float32, device=device)
        dth_tgt  = torch.as_tensor(b["pred_tgt"][:, :, 3], dtype=torch.float32, device=device)

        enc_pred_n, dth_pred_n, core_pred_n = model(ctx, obs0, acts)
        # Observation-space prediction target (== core in plain mode).
        enc_tgt_n  = model.norm_enc(obs_tgt)
        core_tgt_n = model.norm_core(core_tgt)
        dth_tgt_n  = (dth_tgt - model.pred_mean[3]) / model.pred_std[3]
        loss = F.mse_loss(enc_pred_n, enc_tgt_n) + F.mse_loss(dth_pred_n, dth_tgt_n)
        if model.enriched:      # readout supervision (obs -> clean core)
            loss = loss + F.mse_loss(core_pred_n, core_tgt_n)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        sched.step()

        if step % cfg.train.log_every == 0:
            el = time.perf_counter() - t0
            print(f"[{step:5d}/{cfg.train.steps}] loss={loss.item():.4f} "
                  f"({step/el:.1f} it/s)")
        if step % cfg.train.val_every == 0 or step == cfg.train.steps:
            v = validate(model, va_s, device)
            print("    val: " + "  ".join(f"{k}={x:.4f}" for k, x in v.items()))
            hist.append(dict(step=step, loss=loss.item(),
                             **{f"val/{k}": x for k, x in v.items()}))
            score = v.get("pos@15", v.get("pos@1", float("inf")))
            if score < best:
                best = score
                torch.save(dict(model=model.state_dict(), config=cfg.to_dict(),
                                norm=norm.to_dict(), kind="predictive",
                                step=step, val=v), best_path)
                print(f"    ↳ new best (pos@15={score:.4f}) -> {best_path}")

    with open(os.path.join(cfg.train.out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"\n[pred] done in {(time.perf_counter()-t0)/60:.1f} min. best pos@15 = {best:.4f}")
    return best_path


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train the autoregressive state-space baseline.")
    p.add_argument("--packed-root", default="packed")
    p.add_argument("--out", dest="out_dir", default="runs/predictive")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--H", type=int, default=15)
    p.add_argument("--d-ctx", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--enriched", action="store_true",
                   help="feed the noisy redundant sensor suite (must match the JEPA run)")
    p.add_argument("--sensor-iid", type=float, default=0.08)
    p.add_argument("--sensor-bias", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    cfg = Config()
    cfg.data.packed_root = a.packed_root
    cfg.data.K, cfg.data.H = a.K, a.H
    cfg.model.d_ctx, cfg.model.hidden = a.d_ctx, a.hidden
    cfg.train.steps, cfg.train.batch_size = a.steps, a.batch_size
    cfg.train.lr, cfg.train.seed = a.lr, a.seed
    cfg.train.val_every, cfg.train.log_every = a.val_every, a.log_every
    cfg.train.device, cfg.train.out_dir = a.device, a.out_dir
    cfg.data.enriched = a.enriched
    cfg.data.sensor_iid_scale, cfg.data.sensor_bias_scale = a.sensor_iid, a.sensor_bias
    return cfg


if __name__ == "__main__":
    train(_parse_args())
