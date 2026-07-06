"""
train_value.py
==============
Trains the belief-conditioned terminal value V(s, b) of value_net.py from a
dataset recorded by taxi_controller.py's RLDataRecorder.

Targets are the discounted COST-TO-GO under the recorded policy:

    G_t = Σ_{τ≥t} γ^(τ-t) · r_τ          (per episode, reset at terminals)
    y_t = −G_t                            (recorder stores REWARDS: higher =
                                           better; the planner minimises cost,
                                           so the value is trained negated)

Features come from value_net.features_from_obs — the exact encoder mppi()'s
terminal hook uses — with the per-slot beliefs saved by the recorder (falls
back to a uniform prior for datasets recorded before belief logging, which is
also how you train the V(s)-only ablation: --no-beliefs).

Monte-Carlo regression (no bootstrapping), pure-NumPy Adam — no torch.

Examples
--------
    # full mixed dataset, belief-conditioned
    python controller/train_value.py taxi_expert_data.npz -o value.npz

    # V(s)-only ablation (belief features forced to the uniform prior)
    python controller/train_value.py taxi_expert_data.npz -o value_nobelief.npz --no-beliefs

    # head-on task only, longer training
    python controller/train_value.py data.npz -o value_headon.npz --tasks 1 --epochs 300
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from value_net import FEAT_DIM, MLPParams, features_from_obs, forward


# ── Targets ────────────────────────────────────────────────────────────────────

def discounted_cost_to_go(rewards: np.ndarray, terminals: np.ndarray,
                          gamma: float) -> np.ndarray:
    """Per-step −(discounted reward-to-go), computed backwards, reset at episode ends."""
    G = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        if terminals[t]:
            running = 0.0                       # r_t is the terminal transition's reward
        running = rewards[t] + gamma * running
        G[t] = running
    return -G                                    # reward-to-go → cost-to-go


def build_dataset(path: str, gamma: float, tasks=None, use_beliefs=True):
    d = np.load(path)
    obs       = d["observations"].astype(np.float64)
    rewards   = d["rewards"].astype(np.float64)
    terminals = d["terminals"].astype(bool)

    if not terminals.any():
        raise ValueError(f"{path} has no terminal flags — cost-to-go targets need "
                         "episode boundaries. Re-record the dataset.")
    if not terminals[-1]:
        # A truncated recording (e.g. Unity stopped mid-episode) leaves a tail with no
        # terminal; its returns would leak across the cut, so drop it.
        last = int(np.where(terminals)[0][-1])
        obs, rewards, terminals = obs[:last + 1], rewards[:last + 1], terminals[:last + 1]
        task_ids = d["task_ids"][:last + 1]
    else:
        task_ids = d["task_ids"]

    y = discounted_cost_to_go(rewards, terminals, gamma)

    beliefs = None
    if use_beliefs and "beliefs" in d.files:
        beliefs = d["beliefs"].astype(np.float64)[:len(obs)]   # (N, K_OBS, N_ROUTE)
    elif use_beliefs:
        print("[train_value] WARNING: dataset has no 'beliefs' array — falling back to "
              "the uniform prior (this trains the V(s) ablation, not V(s,b)). "
              "Re-record with the current taxi_controller.py to get belief labels.")

    X = np.stack([
        features_from_obs(obs[i], beliefs[i] if beliefs is not None else None)
        for i in range(len(obs))
    ])

    if tasks is not None:
        mask = np.isin(task_ids, np.asarray(tasks, dtype=np.int32))
        X, y = X[mask], y[mask]
    if len(X) == 0:
        raise ValueError(f"No transitions in {path} for tasks={tasks}.")
    return X, y[:, None]                         # scalar targets as (N, 1)


# ── Training (NumPy Adam, MSE) ────────────────────────────────────────────────

def _standardise_stats(A: np.ndarray):
    """Column mean/std with a floor so constant columns don't divide by zero."""
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def train(X, Y, hidden_sizes, epochs, batch_size, lr, weight_decay,
          val_frac, seed, verbose=True):
    """Fit the value MLP by minibatch Adam on standardised features/targets."""
    rng = np.random.default_rng(seed)

    x_mean, x_std = _standardise_stats(X)
    y_mean, y_std = _standardise_stats(Y)
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    n = len(Xn)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr = Xn[tr_idx], Yn[tr_idx]
    Xval, Yval = Xn[val_idx], Yn[val_idx]

    params = MLPParams.init(X.shape[1], Y.shape[1], hidden_sizes, rng)
    params.x_mean, params.x_std = x_mean, x_std
    params.y_mean, params.y_std = y_mean, y_std

    tensors = params.weights + params.biases
    m = [np.zeros_like(t) for t in tensors]
    v = [np.zeros_like(t) for t in tensors]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    def mse(Xset, Yset):
        return float(np.mean((forward(params, Xset) - Yset) ** 2))

    best_val, best = np.inf, None
    n_layers = len(params.weights)
    n_tr = len(Xtr)

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_tr)
        for start in range(0, n_tr, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = Xtr[idx], Ytr[idx]

            pred, acts, pre = forward(params, xb, cache=True)
            b = len(xb)

            # Backprop (MSE). Output layer linear; hidden layers tanh.
            grads_w = [None] * n_layers
            grads_b = [None] * n_layers
            delta = (2.0 / b) * (pred - yb)
            for i in reversed(range(n_layers)):
                grads_w[i] = acts[i].T @ delta + weight_decay * params.weights[i]
                grads_b[i] = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ params.weights[i].T) * (1.0 - np.tanh(pre[i - 1]) ** 2)

            step += 1
            grads = grads_w + grads_b
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            for j, (t, g) in enumerate(zip(tensors, grads)):
                m[j] = beta1 * m[j] + (1 - beta1) * g
                v[j] = beta2 * v[j] + (1 - beta2) * (g * g)
                t -= lr * (m[j] / bc1) / (np.sqrt(v[j] / bc2) + eps)

        val = mse(Xval, Yval) if n_val > 0 else mse(Xtr, Ytr)
        if val < best_val:
            best_val = val
            best = ([w.copy() for w in params.weights],
                    [bb.copy() for bb in params.biases])
        if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}/{epochs}  train_mse={mse(Xtr, Ytr):.5f}  "
                  f"val_mse={val:.5f}")

    if best is not None:
        params.weights, params.biases = best
    return params, best_val


def run_training(dataset: str, out: str = "value_net.npz", gamma: float = 0.99,
                 tasks=None, use_beliefs: bool = True, hidden=(64, 64),
                 epochs: int = 150, batch_size: int = 256, lr: float = 1e-3,
                 weight_decay: float = 1e-5, val_frac: float = 0.1, seed: int = 0):
    """
    Function API for notebooks (Colab/Jupyter), where argparse would trip over the
    kernel's own arguments:

        from train_value import run_training
        run_training("expert.npz", out="value.npz")                    # V(s,b)
        run_training("expert.npz", out="value_nb.npz", use_beliefs=False)  # ablation

    Returns (params, best_val_mse); also writes the checkpoint to `out`.
    """
    X, Y = build_dataset(dataset, gamma, tasks, use_beliefs=use_beliefs)
    assert X.shape[1] == FEAT_DIM, f"feature dim {X.shape[1]} != {FEAT_DIM}"
    print(f"[train_value] {len(X)} samples  feat_dim={X.shape[1]}  "
          f"beliefs={'on' if use_beliefs else 'OFF (ablation)'}  "
          f"tasks={tasks or 'all'}  gamma={gamma}")
    print(f"[train_value] cost-to-go targets: mean={Y.mean():+.2f}  std={Y.std():.2f}  "
          f"min={Y.min():+.2f}  max={Y.max():+.2f}")
    print(f"[train_value] hidden={list(hidden)}  epochs={epochs}  "
          f"batch={batch_size}  lr={lr}")

    t0 = time.time()
    params, best_val = train(X, Y, list(hidden), epochs, batch_size, lr,
                             weight_decay, val_frac, seed)
    params.save(out)
    print(f"[train_value] done in {time.time()-t0:.1f}s  best_val_mse={best_val:.5f}  "
          f"→  {out}")
    return params, best_val


def main():
    p = argparse.ArgumentParser(
        description="Train the belief-conditioned terminal value V(s,b) from recorder data.")
    p.add_argument("dataset", help="Recorder .npz (observations, rewards, terminals[, beliefs]).")
    p.add_argument("-o", "--out", default="value_net.npz", help="Output checkpoint path.")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount for the cost-to-go targets. MUST match GAMMA_VALUE "
                        "in taxi_controller.py, which scales the terminal term by "
                        "gamma^H_MPPI at plan time.")
    p.add_argument("--tasks", type=int, nargs="+", default=None,
                   help="Restrict to these task_ids (e.g. 1 = head-on). Default: all.")
    p.add_argument("--no-beliefs", action="store_true",
                   help="Force the uniform prior in the belief features — trains the "
                        "V(s)-only ablation on the same data.")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, Y = build_dataset(args.dataset, args.gamma, args.tasks,
                         use_beliefs=not args.no_beliefs)
    assert X.shape[1] == FEAT_DIM, f"feature dim {X.shape[1]} != {FEAT_DIM}"
    print(f"[train_value] {len(X)} samples  feat_dim={X.shape[1]}  "
          f"beliefs={'on' if not args.no_beliefs else 'OFF (ablation)'}  "
          f"tasks={args.tasks or 'all'}  gamma={args.gamma}")
    print(f"[train_value] cost-to-go targets: mean={Y.mean():+.2f}  std={Y.std():.2f}  "
          f"min={Y.min():+.2f}  max={Y.max():+.2f}")
    print(f"[train_value] hidden={args.hidden}  epochs={args.epochs}  "
          f"batch={args.batch_size}  lr={args.lr}")

    t0 = time.time()
    params, best_val = train(
        X, Y, args.hidden, args.epochs, args.batch_size, args.lr,
        args.weight_decay, args.val_frac, args.seed)
    params.save(args.out)
    print(f"[train_value] done in {time.time()-t0:.1f}s  best_val_mse={best_val:.5f}  "
          f"→  {args.out}")


if __name__ == "__main__":
    main()
