"""
probe.py
========
Does the context embedding `c` actually encode eta?

    python -m jepa.probe --ckpt runs/jepa/jepa_best.pt

This is the cheapest decisive experiment in the whole project, and it should be
run BEFORE trusting any B/C result.

The reasoning: the entire generalisation argument rests on the claim that the
context encoder performs implicit system identification — that it reads the
plant's parameters out of the recent history. That claim is directly testable.
Freeze the encoder, take `c`, and fit a LINEAR map to the true eta (which the
dataset records per rollout in `eta_json`). If a linear probe recovers
DRAG_COEFF and ACCEL_TAU well above chance, the latent really is doing system ID.
If it cannot, then no amount of B/C evaluation will look good, and the fix is
upstream: a longer context window K, or richer excitation in the data.

Scoring is R² against a predict-the-mean baseline, per parameter. R² <= 0 means
the probe does no better than guessing the dataset mean — i.e. that parameter is
NOT identifiable from the context as encoded.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

from .data import load_split, WindowSampler
from .dynamics import NOMINAL
from .evaluate import load_model


# Parameters worth probing. The three that are strongly identifiable from a short
# window are listed first; the limits (A_MIN/A_MAX/DELTA_LIM) are only observable
# when the trajectory saturates them, so a low R² there is expected, not a bug.
PROBE_KEYS: List[str] = [
    "DRAG_COEFF", "ACCEL_TAU", "L",
    "MAX_STEER_RATE", "STEER_ROLLOFF_SPD", "STEER_ROLLOFF_MIN",
    "A_MIN", "A_MAX", "DELTA_LIM",
]


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form ridge regression with an intercept column."""
    X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    return np.linalg.solve(A, X1.T @ y)


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


@torch.no_grad()
def embed(model, data, K: int, H: int, stride: int, max_n: int, device: str):
    """Collect (context embedding, rollout index) over a deterministic sweep."""
    sampler = WindowSampler(data, K, H, seed=0)
    b = sampler.all_windows(stride=stride, max_n=max_n)
    c = model.context_embedding(
        torch.as_tensor(b["ctx"], dtype=torch.float32, device=device)).cpu().numpy()
    return c, b["rollout"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Linear probe: context embedding -> true eta.")
    ap.add_argument("--ckpt", default="runs/jepa/jepa_best.pt")
    ap.add_argument("--packed-root", default="packed")
    ap.add_argument("--fit-on", default="A", help="condition used to FIT the probe")
    ap.add_argument("--test-on", nargs="+", default=["A", "B"],
                    help="conditions to score the fitted probe on")
    ap.add_argument("--stride", type=int, default=11)
    ap.add_argument("--max-n", type=int, default=20000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, args.device)
    K, H = cfg.data.K, cfg.data.H

    # Fit on the training condition's val split (never the train split the encoder saw).
    fit_data = load_split(args.packed_root, args.fit_on,
                          "val" if args.fit_on == "A" else "test", cfg.data)
    Xf, rf = embed(model, fit_data, K, H, args.stride, args.max_n, args.device)
    print(f"[probe] fitting on {args.fit_on}: {len(Xf)} windows, d_ctx={Xf.shape[1]}")

    weights: Dict[str, np.ndarray] = {}
    for key in PROBE_KEYS:
        y = fit_data.eta_array(rf, key, NOMINAL.get(key, 0.0))
        weights[key] = ridge_fit(Xf, y)

    # Score on each requested condition.
    print("\n" + "=" * 66)
    print("LINEAR PROBE  c -> eta      (R², higher is better; <=0 = no signal)")
    print("=" * 66)
    hdr = f"  {'parameter':<20}" + "".join(f"{c:>12}" for c in args.test_on)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for key in PROBE_KEYS:
        cells = []
        for cond in args.test_on:
            d = load_split(args.packed_root, cond, "val" if cond == "A" else "test", cfg.data)
            X, r = embed(model, d, K, H, args.stride, args.max_n, args.device)
            y = d.eta_array(r, key, NOMINAL.get(key, 0.0))
            X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)
            cells.append(r2(X1 @ weights[key], y))
        print(f"  {key:<20}" + "".join(f"{v:>12.3f}" for v in cells))

    print("\n  DRAG_COEFF / ACCEL_TAU / L are the strongly identifiable three;")
    print("  high R² there confirms the context encoder is doing system ID.")
    print("  Low R² on A_MIN/A_MAX/DELTA_LIM is expected — a limit is only")
    print("  observable when the trajectory actually saturates it.")
    print("\n  A high R² on condition B is the stronger result: it means the")
    print("  encoder extrapolates its identification to parameter values")
    print("  outside anything it saw during training.")


if __name__ == "__main__":
    main()
