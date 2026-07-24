"""
collect_dataset.py
==================
Orchestrator for the taxi dynamics dataset. Runs N rollouts against a Unity
build (or Editor), sampling one eta and one GP reference per rollout, writing
each rollout to its own .npz shard plus one JSON-lines manifest describing
everything.

Sharding one file per rollout keeps IO simple, keeps failures local (a corrupt
shard loses one rollout, not the whole run), and lets the train/val/test split
work at the ROLLOUT level (see split_and_pack.py) without any special indexing.

Only ONE Unity environment is used per invocation — parallelism, if wanted,
is done at the process level: launch this script several times with disjoint
--seed-offset and --port ranges. That's simpler and more robust than trying to
share a single env across threads, and it matches what the paper does at scale.

Usage
-----
    python collect_dataset.py \
        --condition A \
        --n-rollouts 3000 \
        --exec ./build/Taxi \
        --port 5004 \
        --out data/A

Assumes taxi_controller_mppi.py and (if --controllers includes mpc)
taxi_controller_mpc.py are importable from the current directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from typing import Optional

import numpy as np

from domain_sampler import sample, RolloutEta
from reference_gen import ReferenceConfig, gp_control_sequence
from rollout_worker import run_rollout, RolloutResult


# ── Env factory ─────────────────────────────────────────────────────────────
def build_env(exec_path: Optional[str], port: int,
                seed: int, worker_id: int = 0):
    """Construct a UnityEnvironment with an EnvironmentParametersChannel bound.

    We register the channel BEFORE construction so it's guaranteed present in
    env._side_channel_manager._side_channels_dict when rollout_worker's
    push_eta_to_unity() looks for it. Some ML-Agents versions auto-register it,
    but this way we don't have to rely on that.
    """
    from mlagents_envs.environment import UnityEnvironment
    try:
        from mlagents_envs.side_channel.environment_parameters_channel import (
            EnvironmentParametersChannel)
    except ImportError as e:
        raise RuntimeError(
            "ml-agents-envs must be installed for data collection; "
            "pip install mlagents-envs") from e

    env_params = EnvironmentParametersChannel()
    env = UnityEnvironment(
        file_name=exec_path,
        base_port=port,
        seed=seed,
        no_graphics=exec_path is not None,
        worker_id=worker_id,
        side_channels=[env_params],
    )
    return env, env_params


def push_eta(env_params, payload: dict) -> None:
    """Direct channel push — simpler than reaching into env internals."""
    for k, v in payload.items():
        env_params.set_float_parameter(k, float(v))


# ── Manifest schema ─────────────────────────────────────────────────────────
def _rollout_entry(rollout_id: str, condition: str, controller: str,
                    shard_path: str, result: RolloutResult) -> dict:
    """One line of the JSONL manifest, describing one rollout."""
    return dict(
        rollout_id=rollout_id,
        condition=condition,
        controller=controller,
        path=shard_path,
        n_steps=int(result.n_steps),
        terminal_reason=result.terminal_reason,
        eta=result.eta,
        goal_xy=[float(result.goal_xy[0]), float(result.goal_xy[1])],
    )


# ── Main collection loop ────────────────────────────────────────────────────
def collect(condition: str, n_rollouts: int, out_dir: str,
             exec_path: Optional[str], port: int,
             controllers: tuple[str, ...],
             seed: int, max_steps: int, dt: float,
             sensor_noise_std: float, worker_id: int = 0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    ref_cfg = ReferenceConfig(n_steps=max_steps, dt=dt)

    env, env_params = build_env(exec_path, port, seed, worker_id=worker_id)
    print(f"[collect] env up on port {port}+{worker_id} (build={exec_path}); "
          f"condition={condition}; controllers={controllers}; "
          f"n_rollouts={n_rollouts}; seed={seed}")

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    # Append mode — a re-run with the same --out extends the dataset. Callers
    # who want a clean start delete the directory first.
    manifest_f = open(manifest_path, "a")

    behavior_name = None
    n_written = 0
    n_failed = 0
    t0 = time.perf_counter()
    try:
        for i in range(n_rollouts):
            eta_obj = sample(condition, rng)
            # Push eta BEFORE the reset, so TaxiAgent.OnEpisodeBegin() reads
            # the correct values when it applies them to ApplyBicycleDynamics.
            push_eta(env_params, eta_obj.env_channel_payload())
            env.reset()
            if behavior_name is None:
                behavior_name = list(env.behavior_specs.keys())[0]
                print(f"[collect] behavior={behavior_name}")

            controller = str(rng.choice(controllers))
            ref = gp_control_sequence(rng, ref_cfg)

            try:
                result = run_rollout(
                    env=env,
                    behavior_name=behavior_name,
                    eta=eta_obj.env_channel_payload(),
                    condition=condition,
                    controller=controller,
                    ref_controls=ref,
                    dt=dt,
                    max_steps=max_steps,
                    sensor_noise_std=sensor_noise_std,
                    push_eta=False,   # already pushed above
                )
            except Exception as e:
                # A single-rollout failure (e.g., MPC solver crashed on this
                # random eta) must not kill a 3k-rollout batch. Log and skip.
                print(f"[collect] rollout {i} FAILED: {type(e).__name__}: {e}")
                n_failed += 1
                continue

            if result.n_steps == 0:
                print(f"[collect] rollout {i}: 0 transitions recorded "
                      f"({result.terminal_reason}) — skipping")
                n_failed += 1
                continue

            rollout_id = f"{condition}_{seed:04d}_{i:06d}_{uuid.uuid4().hex[:6]}"
            shard_path = os.path.join(out_dir, f"rollout_{rollout_id}.npz")
            np.savez(
                shard_path,
                x=result.x, a=result.a, x_next=result.x_next,
                ref=result.ref, goal_xy=result.goal_xy,
            )
            manifest_f.write(json.dumps(
                _rollout_entry(rollout_id, condition, controller,
                                shard_path, result)) + "\n")
            manifest_f.flush()   # so a KeyboardInterrupt doesn't lose the last few
            n_written += 1

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta_s = (n_rollouts - (i + 1)) / max(rate, 1e-6)
                print(f"[collect] {i+1:5d}/{n_rollouts}  "
                      f"written={n_written} failed={n_failed}  "
                      f"({rate:.2f} rollouts/s, ETA {eta_s/60:.1f} min)")

    finally:
        manifest_f.close()
        try:
            env.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    print(f"\n[collect] done. {n_written} rollouts written to {out_dir}  "
          f"({n_failed} failed) in {elapsed/60:.1f} min")


# ── CLI ─────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Taxi dataset collector.")
    p.add_argument("--condition", choices=("A", "B", "C"), required=True,
                    help="A=training/in-dist, B=OOD params, C=unmodeled effect on")
    p.add_argument("--n-rollouts", type=int, required=True)
    p.add_argument("--out", default="data/A", help="output directory (dataset shard root)")
    p.add_argument("--exec", default=None,
                    help="path to Unity build; omit to attach to Editor Play mode")
    p.add_argument("--port", type=int, default=5004)
    p.add_argument("--worker-id", type=int, default=0,
                    help="ML-Agents worker offset; use to run several collectors in parallel")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=100,
                    help="max transitions per rollout (10s at dt=0.1)")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--controllers", nargs="+",
                    default=["mppi", "mpc", "open"],
                    choices=["mppi", "mpc", "open"],
                    help="pool of trackers to sample from per-rollout (uniformly)")
    p.add_argument("--sensor-noise-std", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collect(
        condition=args.condition,
        n_rollouts=args.n_rollouts,
        out_dir=args.out,
        exec_path=args.exec if args.exec != "None" else None,
        port=args.port,
        controllers=tuple(args.controllers),
        seed=args.seed,
        max_steps=args.max_steps,
        dt=args.dt,
        sensor_noise_std=args.sensor_noise_std,
        worker_id=args.worker_id,
    )