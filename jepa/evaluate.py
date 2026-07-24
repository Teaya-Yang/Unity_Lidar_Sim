"""
evaluate.py
===========
The head-to-head comparison the whole pipeline exists to produce.

    python -m jepa.evaluate --ckpt runs/jepa/jepa_best.pt

Four models, identical windows, identical actions, identical pose integrator:

  analytic_fixed   the deployed baseline — nominal parameters, no adaptation.
                   This is literally the model inside MPPI/MPC today.
  analytic_ls      STRONG baseline — the same equations, but with ACCEL_TAU,
                   DRAG_COEFF and L recovered by least squares from the same
                   context window the learned model sees. Adaptive analytic.
  analytic_oracle  CEILING — the same equations handed the TRUE eta. In sim the
                   analytic equations ARE the plant on A/B, so this is a bound no
                   learned model can beat there. On C it still has an irreducible
                   error floor, because no parameter value creates a v²·δ term.
  jepa             the learned model, trained on condition A only.

How to read the output
----------------------
The absolute error on C is not the headline. The headline is DEGRADATION:

    degradation = error(C) - error(A)

C's nine dynamics parameters are drawn from A's in-distribution band by
construction, so the ONLY difference between A and C is the unmodeled physics.
Any extra error on C is therefore attributable to structural mismatch and nothing
else. If the analytic models degrade more than the learned one, the structural
flexibility is real.

The `--slice-c` table is the decisive plot in tabular form: bin condition C by
slip_coeff and watch whether analytic error grows with it while the learned error
stays flat. Growth-with-slip is a signature that cannot be explained by tuning.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import Config
from .data import load_split, Normalizer, WindowSampler
from .dynamics import NOMINAL, estimate_eta_ls, eta_from_json
from .models import JEPA, dims_for
from .baselines import load_predictive
from .rollout import (rollout_analytic, rollout_learned,
                      per_horizon_errors, summarize, compounding_ratio)


# ── Model registry ───────────────────────────────────────────────────────────
def build_predictors(model: Optional[JEPA], device: str,
                     predictive=None) -> Dict[str, callable]:
    """Each entry maps (batch, data) -> predicted states (B,H,6)."""

    def analytic_fixed(b, data):
        return rollout_analytic(b["state0"].astype(np.float64), b["acts"])

    def analytic_ls(b, data):
        eta = estimate_eta_ls(b["ctx_state"].astype(np.float64),
                              b["ctx_act"].astype(np.float64),
                              b["ctx_next"].astype(np.float64))
        return rollout_analytic(b["state0"].astype(np.float64), b["acts"], eta)

    def analytic_oracle(b, data):
        etas = [data.eta[r] for r in b["rollout"]]
        eta = eta_from_json(etas)
        return rollout_analytic(b["state0"].astype(np.float64), b["acts"], eta)

    preds = {
        "analytic_fixed":  analytic_fixed,
        "analytic_ls":     analytic_ls,
        "analytic_oracle": analytic_oracle,
    }
    # The paper's actual baseline: a learned autoregressive state-space model.
    if predictive is not None:
        preds["predictive"] = lambda b, data: rollout_learned(predictive, b, device)
    if model is not None:
        preds["jepa"] = lambda b, data: rollout_learned(model, b, device)
    return preds


# ── Checkpoint loading ───────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ck["config"])
    norm = Normalizer.from_dict(ck["norm"])
    obs_dim, ctx_dim = dims_for(cfg.data.enriched)
    model = JEPA(obs_dim, ctx_dim, cfg.model.d_ctx, cfg.model.d_state, cfg.model.hidden,
                 cfg.model.n_ctx_layers, cfg.model.ema_tau, norm, cfg.data.enriched).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[eval] loaded {ckpt_path} (step {ck.get('step')}, "
          f"K={cfg.data.K}, H={cfg.data.H}"
          + (", ENRICHED" if cfg.data.enriched else "") + ")")
    return model, cfg


# ── Core evaluation ──────────────────────────────────────────────────────────
def evaluate_condition(preds: Dict[str, callable], data, K: int, H: int,
                       stride: int, max_n: int) -> Dict[str, Dict]:
    """Score every model on the SAME deterministic set of windows."""
    sampler = WindowSampler(data, K, H, seed=0)
    b = sampler.all_windows(stride=stride, max_n=max_n)
    truth = b["state_tgt"].astype(np.float64)
    n = len(b["state0"])

    results = {}
    for name, fn in preds.items():
        pred = fn(b, data)
        errs = per_horizon_errors(pred, truth)
        results[name] = dict(summary=summarize(errs, horizons=(1, 5, 10, H)),
                             curve={k: v.tolist() for k, v in errs.items()})
    return results, b, n


def slice_condition_c(preds: Dict[str, callable], data, K: int, H: int,
                      key: str, n_bins: int, stride: int, max_n: int) -> List[Dict]:
    """Bin condition-C windows by an unmodeled-effect magnitude and report each
    model's error per bin. This is the structural-mismatch signature."""
    sampler = WindowSampler(data, K, H, seed=0)
    b = sampler.all_windows(stride=stride, max_n=max_n)
    truth = b["state_tgt"].astype(np.float64)
    vals = data.eta_array(b["rollout"], key, 0.0)

    edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    rows = []
    for i in range(n_bins):
        m = (vals >= edges[i]) & (vals < edges[i + 1])
        if m.sum() < 20:
            continue
        sub = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(vals) else v)
               for k, v in b.items()}
        row = {key: float(vals[m].mean()), "n": int(m.sum())}
        for name, fn in preds.items():
            pred = fn(sub, data)
            errs = per_horizon_errors(pred, truth[m])
            row[name] = float(errs["pos"][-1])       # final-horizon position RMSE
        rows.append(row)
    return rows


# ── Reporting ────────────────────────────────────────────────────────────────
def _fmt_table(all_res: Dict[str, Dict], metric: str, models: List[str]) -> str:
    conds = list(all_res.keys())
    w = max(len(m) for m in models) + 2
    lines = ["  " + "model".ljust(w) + "".join(f"{c:>12}" for c in conds) +
             f"{'C - A':>12}"]
    lines.append("  " + "-" * (w + 12 * (len(conds) + 1)))
    for m in models:
        vals = [all_res[c][m]["summary"].get(metric, float("nan")) for c in conds]
        degr = (all_res["C"][m]["summary"].get(metric, float("nan")) -
                all_res["A"][m]["summary"].get(metric, float("nan"))) \
            if ("A" in all_res and "C" in all_res) else float("nan")
        lines.append("  " + m.ljust(w) + "".join(f"{v:>12.4f}" for v in vals) +
                     f"{degr:>12.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare the JEPA against analytic baselines on A/B/C.")
    ap.add_argument("--ckpt", default="runs/jepa/jepa_best.pt",
                    help="trained JEPA checkpoint; omit to score baselines only")
    ap.add_argument("--predictive-ckpt", default="runs/predictive/predictive_best.pt",
                    help="autoregressive state-space baseline (the paper's 'Predictive'); "
                         "loaded if present")
    ap.add_argument("--packed-root", default="packed")
    ap.add_argument("--conditions", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--stride", type=int, default=13,
                    help="window stride — windows overlap heavily, so subsample")
    ap.add_argument("--max-n", type=int, default=6000, help="max windows per condition")
    ap.add_argument("--slice-c", default="SLIP_COEFF",
                    help="condition-C eta key to bin by (SLIP_COEFF / BRAKE_ASYMMETRY / FRICTION_NOISE_AMP)")
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write full results to this JSON")
    args = ap.parse_args()

    model, cfg = (None, Config())
    if args.ckpt:
        try:
            model, cfg = load_model(args.ckpt, args.device)
        except FileNotFoundError:
            print(f"[eval] no JEPA checkpoint at {args.ckpt} — scoring baselines only")

    predictive = None
    if args.predictive_ckpt:
        try:
            predictive, pcfg = load_predictive(args.predictive_ckpt, args.device)
            print(f"[eval] loaded predictive baseline {args.predictive_ckpt}")
            if model is None:
                cfg = pcfg
        except FileNotFoundError:
            print(f"[eval] no predictive baseline at {args.predictive_ckpt} — skipping it")

    K, H = cfg.data.K, cfg.data.H
    preds = build_predictors(model, args.device, predictive)
    models = list(preds.keys())

    all_res, all_b = {}, {}
    for cond in args.conditions:
        split = "test"
        data = load_split(args.packed_root, cond, split, cfg.data)
        res, b, n = evaluate_condition(preds, data, K, H, args.stride, args.max_n)
        all_res[cond] = res
        all_b[cond] = data
        print(f"[eval] condition {cond}: {data.n_rollouts} rollouts, {n} windows scored")

    # ── Tables ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"OPEN-LOOP ROLLOUT ERROR  (H={H} steps = {H*0.1:.1f}s, RMSE over windows)")
    print("=" * 78)
    for metric, unit in ((f"pos@{H}", "m"), (f"theta@{H}", "rad"), (f"v@{H}", "m/s"),
                         ("pos@1", "m")):
        print(f"\n{metric}  [{unit}]")
        print(_fmt_table(all_res, metric, models))

    # ── Compounding-error analysis (the paper's headline: Fig. 6) ────────────
    # Latent (jepa) vs autoregressive state-space (predictive): how much extra
    # error does recursive rollout add over one-step-from-truth prediction?
    neural = [m for m in ("predictive", "jepa") if m in preds]
    if len(neural) >= 1:
        from .data import WindowSampler
        cond0 = args.conditions[0]
        sampler = WindowSampler(all_b[cond0], K, H, seed=0)
        b = sampler.all_windows(stride=args.stride, max_n=args.max_n)
        print("\n" + "=" * 78)
        print(f"COMPOUNDING-ERROR ANALYSIS on condition {cond0}   "
              f"(CR = rollout / teacher-forced position error)")
        print("=" * 78)
        hs = [h for h in (1, 5, 10, H) if h <= H]
        hdr = "  " + "model".ljust(14) + "".join(f"  CR@{h:<3}" for h in hs) + \
              "".join(f" ER@{h:<3}" for h in hs)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for m in neural:
            cm = compounding_ratio(preds and (model if m == "jepa" else predictive),
                                   b, args.device)
            row = "  " + m.ljust(14)
            row += "".join(f"  {cm['cr'][h-1]:5.2f}" for h in hs)
            row += "".join(f" {cm['er'][h-1]:5.3f}" for h in hs)
            print(row)
        print("\n  Lower CR (closer to 1) and lower ER = less recursive error")
        print("  accumulation. The paper's core claim is that the latent model")
        print("  (jepa) compounds less than the autoregressive baseline (predictive).")

    # ── The structural-mismatch signature ────────────────────────────────────
    if "C" in all_res and args.slice_c:
        rows = slice_condition_c(preds, all_b["C"], K, H, args.slice_c,
                                 args.n_bins, args.stride, args.max_n)
        if rows:
            print("\n" + "=" * 78)
            print(f"CONDITION C SLICED BY {args.slice_c}   "
                  f"(final-horizon position RMSE [m])")
            print("=" * 78)
            hdr = f"  {args.slice_c:>18} {'n':>7}" + "".join(f"{m:>17}" for m in models)
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for r in rows:
                print(f"  {r[args.slice_c]:>18.4f} {r['n']:>7}" +
                      "".join(f"{r[m]:>17.4f}" for m in models))
            print("\n  Reading: if an analytic row rises with the slice key while")
            print("  jepa stays flat, that is the structural gap — no parameter")
            print("  setting of the kinematic model can produce a v^2*delta term.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_res, f, indent=2)
        print(f"\n[eval] full curves written to {args.out}")


if __name__ == "__main__":
    main()
