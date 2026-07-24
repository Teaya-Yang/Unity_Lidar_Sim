"""
split_and_pack.py
==================
Turns a directory of per-rollout .npz shards + manifest.jsonl (as written by
collect_dataset.py) into:

  1. A train/val/test split done at the ROLLOUT level (never split a single
     rollout's transitions across sets — they're autocorrelated, so splitting
     by transition would leak near-duplicate GP-adjacent segments across the
     boundary).
  2. Packed, training-ready .npz archives (one per split) with all transitions
     concatenated, plus a per-transition rollout-id array for provenance.

Only condition A gets a train/val split. Conditions B and C are ALWAYS
evaluation-only — they must never be touched during training, since the whole
point of the zero-shot claim depends on that separation. This script enforces
that by construction: --condition B or C always emits a single "test" file with
100% of the rollouts, no matter what --train/--val fractions are passed.

Usage
-----
    python split_and_pack.py --in data/A --out packed/A --condition A
    python split_and_pack.py --in data/B --out packed/B --condition B
    python split_and_pack.py --in data/C --out packed/C --condition C
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

import numpy as np


def _load_manifest(in_dir: str) -> List[dict]:
    path = os.path.join(in_dir, "manifest.jsonl")
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _split_rollout_ids(entries: List[dict], condition: str,
                        train: float, val: float, seed: int) -> Dict[str, List[dict]]:
    """Shuffle rollout entries and cut into train/val/test BY ROLLOUT.

    Conditions B/C are forced to a single 'test' bucket regardless of the
    requested fractions — see module docstring.
    """
    rng = random.Random(seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)

    if condition != "A":
        return {"test": shuffled}

    n = len(shuffled)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def _pack_split(entries: List[dict], split_name: str, out_dir: str) -> None:
    """Concatenate every rollout's transitions into one archive for this split.

    Stores:
      x, a, x_next   : (N_total, 6) / (N_total, 2) / (N_total, 6)
      rollout_id      : (N_total,)  string id, so a training loop can still
                        group transitions by rollout (e.g. for the JEPA's
                        history-window construction, which needs CONTIGUOUS
                        same-rollout transitions, not an arbitrary shuffle).
      rollout_offsets : (n_rollouts+1,) cumulative transition-count boundaries,
                        so a loader can slice out rollout i as
                        x[rollout_offsets[i]:rollout_offsets[i+1]] without
                        scanning rollout_id.
      controller      : (n_rollouts,) which tracker produced this rollout.
      terminal_reason : (n_rollouts,) 'reached' / 'collision' / 'timeout' / ...
      eta_json        : (n_rollouts,) JSON string of the sampled eta, for
                        later slicing/analysis (e.g. "show me error vs
                        DRAG_COEFF value").
    """
    if not entries:
        print(f"[split] {split_name}: 0 rollouts, skipping")
        return

    xs, as_, xns, rids = [], [], [], []
    offsets = [0]
    controllers, terminal_reasons, etas, rollout_ids = [], [], [], []

    for e in entries:
        data = np.load(e["path"])
        n = len(data["x"])
        if n == 0:
            continue
        xs.append(data["x"])
        as_.append(data["a"])
        xns.append(data["x_next"])
        rids.append(np.full(n, e["rollout_id"], dtype=object))
        offsets.append(offsets[-1] + n)
        controllers.append(e["controller"])
        terminal_reasons.append(e["terminal_reason"])
        etas.append(json.dumps(e["eta"]))
        rollout_ids.append(e["rollout_id"])

    x = np.concatenate(xs, axis=0).astype(np.float32)
    a = np.concatenate(as_, axis=0).astype(np.float32)
    x_next = np.concatenate(xns, axis=0).astype(np.float32)
    rollout_id_per_step = np.concatenate(rids, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split_name}.npz")
    np.savez(
        out_path,
        x=x, a=a, x_next=x_next,
        rollout_id_per_step=rollout_id_per_step,
        rollout_offsets=np.asarray(offsets, dtype=np.int64),
        controller=np.asarray(controllers, dtype=object),
        terminal_reason=np.asarray(terminal_reasons, dtype=object),
        eta_json=np.asarray(etas, dtype=object),
        rollout_id=np.asarray(rollout_ids, dtype=object),
    )
    print(f"[split] {split_name}: {len(entries)} rollouts, "
          f"{len(x)} transitions -> {out_path}")


def split_and_pack(in_dir: str, out_dir: str, condition: str,
                    train: float = 0.8, val: float = 0.1, seed: int = 0) -> None:
    entries = _load_manifest(in_dir)
    print(f"[split] loaded {len(entries)} rollout entries from {in_dir} "
          f"(condition={condition})")
    buckets = _split_rollout_ids(entries, condition, train, val, seed)
    for split_name, split_entries in buckets.items():
        _pack_split(split_entries, split_name, out_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split + pack rollout shards for training.")
    p.add_argument("--in", dest="in_dir", required=True,
                    help="directory containing manifest.jsonl + rollout_*.npz shards")
    p.add_argument("--out", dest="out_dir", required=True,
                    help="output directory for packed split archives")
    p.add_argument("--condition", choices=("A", "B", "C"), required=True,
                    help="A gets a train/val/test split; B/C are always 100%% test")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.condition == "A":
        assert args.train + args.val < 1.0, "train+val must leave room for a test split"
    split_and_pack(args.in_dir, args.out_dir, args.condition,
                    train=args.train, val=args.val, seed=args.seed)