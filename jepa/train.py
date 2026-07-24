"""
train.py
========
Trains the JEPA on condition A ONLY.

    python -m jepa.train --steps 8000 --out runs/jepa

Conditions B and C are never opened by this script. That separation is the whole
basis of the zero-shot claim, and it is enforced structurally: the only dataset
path this file constructs is `packed/A/...`.

What to watch in the log
------------------------
`emb_std` and `ctx_std`, not the loss. A collapsing joint-embedding model shows a
smoothly falling total loss while its embedding standard deviation slides toward
zero — the representation is going constant and the "improvement" is an illusion.
If emb_std drops below ~0.3, raise `--w-var` or lower the learning rate.

`val/pos@15` is the number that actually matters: open-loop position error after
15 steps on held-out condition-A rollouts. It is what the evaluator compares
against the analytic baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch

from .config import Config, DT
from .data import load_split, fit_normalizer, WindowSampler, N_CORE
from .models import JEPA, dims_for
from .losses import compute_losses
from .rollout import rollout_learned, per_horizon_errors, summarize


def _to_torch(batch: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in batch.items() if v.dtype != np.int64}


@torch.no_grad()
def validate(model: JEPA, sampler: WindowSampler, device: str,
             max_n: int = 4000) -> Dict[str, float]:
    """Open-loop rollout error on held-out condition-A windows, in physical units.

    Uses the same code path as the evaluator (rollout_learned) so the number
    printed during training is directly comparable to the final comparison table.
    """
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

    # ── Data: condition A only ───────────────────────────────────────────────
    tr = load_split(cfg.data.packed_root, "A", "train", cfg.data)
    va = load_split(cfg.data.packed_root, "A", "val", cfg.data)
    print(f"[train] A/train: {tr.n_rollouts} rollouts, {len(tr.x)} transitions"
          + (f"  [ENRICHED obs_dim={tr.obs_dim}]" if cfg.data.enriched else ""))

    norm = fit_normalizer(tr, cfg.data.K, cfg.data.H, seed=cfg.train.seed)
    tr_s = WindowSampler(tr, cfg.data.K, cfg.data.H, cfg.train.seed, cfg.data.min_rollout_len)
    va_s = WindowSampler(va, cfg.data.K, cfg.data.H, cfg.train.seed + 1, cfg.data.min_rollout_len)
    print(f"[train] windows: train={len(tr_s)}  val={len(va_s)}  (K={cfg.data.K}, H={cfg.data.H})")

    # ── Model ────────────────────────────────────────────────────────────────
    obs_dim, ctx_dim = dims_for(cfg.data.enriched)
    model = JEPA(obs_dim, ctx_dim, cfg.model.d_ctx, cfg.model.d_state, cfg.model.hidden,
                 cfg.model.n_ctx_layers, cfg.model.ema_tau, norm, cfg.data.enriched).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] parameters: {n_par:,}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.steps)

    best = float("inf")
    best_path = os.path.join(cfg.train.out_dir, "jepa_best.pt")
    hist = []
    t0 = time.perf_counter()

    for step in range(1, cfg.train.steps + 1):
        b = tr_s.batch(cfg.train.batch_size)
        t = _to_torch(b, device)

        out = model(t["ctx"], t["obs0"], t["acts"], t["obs_tgt"],
                    t["core0"], t["core_tgt"])

        core_tgt_n = model.norm_core(t["core_tgt"])
        # dtheta is channel 3 of the prediction target (see data.N_PRED layout)
        dth_tgt_n  = (t["pred_tgt"][:, :, 3] - model.pred_mean[3]) / model.pred_std[3]
        core0_n    = model.norm_core(t["core0"])

        loss, logs = compute_losses(out, core_tgt_n, dth_tgt_n, core0_n, cfg.loss)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        sched.step()
        model.update_target()

        if step % cfg.train.log_every == 0:
            el = time.perf_counter() - t0
            print(f"[{step:5d}/{cfg.train.steps}] loss={logs['total']:.4f} "
                  f"lat={logs['latent']:.4f} dec={logs['decode']:.4f} "
                  f"emb_std={logs['emb_std']:.3f} ctx_std={logs['ctx_std']:.3f} "
                  f"({step/el:.1f} it/s)")

        if step % cfg.train.val_every == 0 or step == cfg.train.steps:
            v = validate(model, va_s, device)
            msg = "  ".join(f"{k}={x:.4f}" for k, x in v.items())
            print(f"    val: {msg}")
            hist.append(dict(step=step, **logs, **{f"val/{k}": x for k, x in v.items()}))

            score = v.get("pos@15", v.get("pos@1", float("inf")))
            if score < best:
                best = score
                torch.save(dict(model=model.state_dict(),
                                config=cfg.to_dict(),
                                norm=norm.to_dict(),
                                step=step, val=v), best_path)
                print(f"    ↳ new best (pos@15={score:.4f}) -> {best_path}")

    with open(os.path.join(cfg.train.out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    el = time.perf_counter() - t0
    print(f"\n[train] done in {el/60:.1f} min. best val pos@15 = {best:.4f}")
    print(f"[train] checkpoint: {best_path}")
    return best_path


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train the JEPA dynamics model on condition A.")
    p.add_argument("--packed-root", default="packed")
    p.add_argument("--out", dest="out_dir", default="runs/jepa")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--K", type=int, default=16, help="context window length [steps]")
    p.add_argument("--H", type=int, default=15, help="training horizon [steps]")
    p.add_argument("--d-ctx", type=int, default=64)
    p.add_argument("--d-state", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--w-var", type=float, default=1.0,
                   help="VICReg variance weight — raise if emb_std collapses")
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--enriched", action="store_true",
                   help="feed the encoders the noisy redundant sensor suite (options 1+2)")
    p.add_argument("--sensor-iid", type=float, default=0.08, help="per-step sensor noise scale")
    p.add_argument("--sensor-bias", type=float, default=0.05, help="per-rollout sensor bias scale")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    cfg = Config()
    cfg.data.packed_root = a.packed_root
    cfg.data.K, cfg.data.H = a.K, a.H
    cfg.model.d_ctx, cfg.model.d_state, cfg.model.hidden = a.d_ctx, a.d_state, a.hidden
    cfg.loss.w_var = a.w_var
    cfg.train.steps, cfg.train.batch_size = a.steps, a.batch_size
    cfg.train.lr, cfg.train.seed = a.lr, a.seed
    cfg.train.val_every, cfg.train.log_every = a.val_every, a.log_every
    cfg.train.device, cfg.train.out_dir = a.device, a.out_dir
    cfg.data.enriched = a.enriched
    cfg.data.sensor_iid_scale, cfg.data.sensor_bias_scale = a.sensor_iid, a.sensor_bias
    return cfg


if __name__ == "__main__":
    train(_parse_args())
