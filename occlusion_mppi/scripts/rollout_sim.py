#!/usr/bin/env python3
"""Offline MPPI rollout simulator -- NO ROS, NO MARSIM, NO Unity.

WHY THIS EXISTS
---------------
Debugging a planner inside the full stack means every run costs ~60 s of launch
and couples four processes, and a bad keep-out looks identical to a bad map, a
bad odometry frame, or a dropped topic. This harness feeds the planner a
SYNTHETIC boundary set whose geometry you already know, so any misbehaviour is
necessarily the planner's.

It is the same code path the ROS node uses: same BoundarySet, same
OcclusionMPPI, same cost.occlusion_stage_cost. Only the boundary source and the
plant are stubbed.

Run:
    python rollout_sim.py                 # text summary, exits non-zero on failure
    python rollout_sim.py --plot out.png  # also render the trajectory
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from occlusion_mppi.boundary import BoundarySet, OccupancySet  # noqa: E402
from occlusion_mppi.dynamics import DoubleIntegrator     # noqa: E402
from occlusion_mppi.mppi import OcclusionMPPI, MPPIConfig  # noqa: E402


def wall_shadow_boundary(wall_y=5.0, wall_x_half=3.0, res=0.1, depth=6.0):
    """Synthetic stand-in for what occlusion_boundary_node publishes for a wall.

    A wall spanning x in [-x_half, x_half] at y = wall_y, seen from the origin,
    casts a shadow behind it. The FRONTIER of that shadow -- the surface an unseen
    agent must cross -- is approximated here by the two lateral edges of the shadow
    wedge, which is where the nearest boundary to the ego actually lies.
    """
    pts = []
    for y in np.arange(wall_y, wall_y + depth, res):
        # shadow half-width grows linearly with range behind the wall (origin ego)
        hw = wall_x_half * (y / wall_y)
        pts.append([+hw, y, 1.0])
        pts.append([-hw, y, 1.0])
    # the silhouette edges themselves
    for z in np.arange(0.0, 3.0, res):
        pts.append([+wall_x_half, wall_y, z])
        pts.append([-wall_x_half, wall_y, z])
    return np.array(pts)


def wall_occupied_voxels(wall_y=5.0, wall_x_half=3.0, res=0.1, thickness=0.6):
    """The wall itself, as occupied voxels -- what ROG-Map's inf_occ would carry.

    Thickness matters: rollout poses are sampled every dt, so at v_max they are
    v_max*dt apart. A slab thinner than that step can be jumped over between two
    samples and the membership test never fires. 0.6 m clears 3 m/s * 0.1 s.
    """
    xs = np.arange(-wall_x_half, wall_x_half + res, res)
    ys = np.arange(wall_y - thickness / 2, wall_y + thickness / 2, res)
    zs = np.arange(0.0, 3.0, res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    wall = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # The ground plane, which the MARSIM scene also has. Included deliberately:
    # without it this harness cannot reproduce the failure where a wide z-band
    # folds the floor into every column and freezes the planner.
    # Sampled at the voxel resolution, not coarser: a floor sampled every 0.5 m
    # into 0.1 m voxels is a speckle with gaps the planner slips through, which
    # hides the very failure this is here to reproduce.
    fx = np.arange(-15.0, 15.0, res)
    fy = np.arange(-15.0, 15.0, res)
    gx, gy = np.meshgrid(fx, fy, indexing="ij")
    floor = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    return np.vstack([wall, floor])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", type=float, nargs=2, default=[0.0, 12.0])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--no-occlusion", action="store_true",
                    help="disable the keep-out, to see what it is actually changing")
    ap.add_argument("--no-collision", action="store_true",
                    help="also disable the occupancy term (rollouts may pass through the wall)")
    ap.add_argument("--occ-z-band", type=float, default=0.3,
                    help="vertical band of occupied voxels kept, around cruise height. "
                         "Raise past the cruise height (e.g. 1.5) to pull the floor in "
                         "and reproduce the all-rollouts-collide freeze.")
    args = ap.parse_args()

    cfg = MPPIConfig(samples=args.samples, horizon=30)
    cfg.use_occlusion = not args.no_occlusion

    plant = DoubleIntegrator(dim=2, dt=cfg.dt, v_max=3.0, a_max=4.0)
    planner = OcclusionMPPI(plant, cfg, rng=np.random.default_rng(0))

    raw = wall_shadow_boundary()
    bset = BoundarySet(raw, z_band=1.5, ego_z=1.0, planar=True)

    occ_raw = wall_occupied_voxels()
    # Same construction as the node, band included -- otherwise the harness cannot
    # reproduce what the node does.
    occ = None if args.no_collision else OccupancySet(
        occ_raw, resolution=0.1, planar=True, z_band=args.occ_z_band, ego_z=1.0)

    pos = np.array([0.0, 0.0])
    vel = np.array([0.0, 0.0])
    goal = np.array(args.goal)

    probe = OccupancySet(wall_occupied_voxels(), resolution=0.1, planar=True)

    path, min_clear, infeas, hits = [], np.inf, [], 0
    for i in range(args.steps):
        a, info = planner.plan(pos, vel, goal, bset, occ)
        st = plant.step(np.concatenate([pos, vel])[None, :], a[None, :])[0]
        pos, vel = st[:2], st[2:]
        path.append(pos.copy())
        min_clear = min(min_clear, float(bset.distance(pos[None, :])[0]))
        # Scored against the wall regardless of --no-collision, so the baseline run
        # reports its collisions instead of hiding them.
        hits += int(probe.inside(pos[None, :])[0])
        infeas.append(info["frac_infeasible"])
        if np.linalg.norm(pos - goal) < 0.5:
            break

    path = np.array(path)
    reached = np.linalg.norm(path[-1] - goal) < 0.5

    print(f"occlusion term                        : {'on' if cfg.use_occlusion else 'OFF'}")
    print(f"collision term                        : {'on' if occ is not None else 'OFF'}")
    print(f"boundary voxels (after z-band filter) : {len(bset)}")
    print(f"occupied columns                      : {len(probe)}")
    print(f"steps taken                           : {len(path)}")
    print(f"reached goal                          : {reached}")
    print(f"final position                        : {path[-1].round(2)}")
    print(f"min clearance to boundary             : {min_clear:.2f} m")
    print(f"keep-out at t=0                       : {cfg.d_safe:.2f} m")
    print(f"keep-out saturated                    : "
          f"{cfg.d_safe + cfg.v_target * cfg.t_grow_max:.2f} m")
    print(f"steps inside the wall                 : {hits}")
    print(f"mean fraction infeasible rollouts     : {np.mean(infeas):.2f}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.scatter(raw[:, 0], raw[:, 1], s=2, c="crimson", label="occlusion boundary")
            ax.plot(path[:, 0], path[:, 1], "-", lw=2, c="tab:blue", label="ego path")
            ax.plot(*goal, "*", ms=16, c="green", label="goal")
            ax.plot(0, 0, "o", ms=8, c="black", label="start")
            ax.hlines(5.0, -3, 3, colors="k", lw=4, label="wall")
            ax.set_aspect("equal"); ax.legend(); ax.grid(alpha=.3)
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            fig.savefig(args.plot, dpi=110, bbox_inches="tight")
            print(f"wrote {args.plot}")
        except ImportError:
            print("matplotlib unavailable, skipped plot")

    return 0 if reached else 1


if __name__ == "__main__":
    sys.exit(main())
