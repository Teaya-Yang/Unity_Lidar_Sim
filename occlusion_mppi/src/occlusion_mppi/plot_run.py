"""Post-run trajectory figure for the MARSIM wall scene.

Same idea as the 2D controller's `_save_trajectory`: a CSV of the executed run
plus ONE figure showing the executed path and the predicted rollout of a SINGLE
solve, with a handful of sampled horizon timestamps drawn together with the
expanding occlusion keep-out that applied at each of them.

WHAT IS DIFFERENT FROM THE 2D VERSION, AND WHY
----------------------------------------------
There, a boundary was a SEGMENT and its keep-out at t_k was a capsule, so the
figure could draw `capsule_polygon(seg, r_k)` per boundary and be done. ROG-Map
gives VOXELS instead (boundary.py's opening paragraph), so the enforced keep-out
is the union of |frontier| spheres of radius r_k -- routinely 10k+ of them.
Drawing one circle each is both unreadable and slow, and drawing a bounding
capsule would be a lie about what the cost charges for.

So the keep-out is drawn as a CONTOUR of the distance field instead: rasterise
`BoundarySet.distance` over the plot window and draw the level set at r_k. That
curve is exactly the set `{d == r_k}`, i.e. exactly the surface `cost.py` tests
`d < r_keep` against -- the same object the RViz `~keepout` SPHERE_LIST
approximates, minus the striding. A rollout crossing a drawn contour is a bug in
the planner or in cost.py, not in this figure.

The other substitutions:
  * Unity's static boxes -> the occupied cloud (`inf_occ`) the node was actually
    planning against, z-banded to flight altitude so the figure shows the wall
    rather than the floor, which is ~71% of wall_scene.pcd.
  * dynamic-obstacle bubbles -> nothing. The wall scene has no movers; the whole
    keep-out here comes from the occlusion frontier.
  * a z(t) panel is added under the XY axes when dim=3, because unlike the car
    the ego can climb OVER the occluder, and "went around" vs "went over" is
    invisible in a top-down plot.
"""

import os

import numpy as np

from .boundary import BoundarySet


def _r_keep(cfg, t_k):
    """Keep-out radius at horizon time t_k. Mirrors mppi.plan's `t_eff` clamp."""
    t_eff = min(t_k, cfg.t_grow_max) if cfg.t_grow_max is not None else t_k
    return cfg.d_safe + cfg.v_target * t_eff


def _frame_boundaries(fr):
    """Rebuild the frame's BoundarySet, so the figure measures distance with the
    SAME code the cost used rather than a plotting-local approximation."""
    return BoundarySet(fr["bnd"], planar=fr.get("planar", True),
                       query_z=fr.get("query_z"))


def save_run(out_dir, verdict, goal, traj, frames, cfg, dim=3,
             solve_t=None, max_frames=6, margin=6.0, grid=320,
             occ_band=0.6, **_ignored):
    """Write <out_dir>/traj.csv and <out_dir>/traj.png.

    traj   : (T, 1+2*dim) rows of [t, pos..., vel...] -- the EXECUTED run.
    frames : list of per-solve records, see MPPINode._record. Each needs
             t, ex (ego pos), plan (H,dim), bnd (N,3), occ (M,3), infeasible.
    cfg    : the MPPIConfig the run used; d_safe/v_target/t_grow_max are read
             from it so the drawn radii cannot drift from the enforced ones.
    solve_t: [s] pin WHICH solve is drawn -- the recorded one closest in episode
             time. Left None, the solve drawn is the one whose plan came closest
             to (or inside) its expanding keep-out; with no boundaries anywhere
             it falls back to the solve whose plan bent hardest.

    cfg.use_occlusion selects the two presentations. False (the ~use_occlusion
    baseline) still draws everything -- the node records the frontier either way
    -- but dotted, faded and labelled "not enforced", and the auto-selected solve
    is then the one that violated the ignored keep-out most deeply, which is the
    frame worth looking at on a baseline run.
    """
    os.makedirs(out_dir, exist_ok=True)
    # One fixed name, verdict-independent: a collision run must OVERWRITE the
    # previous figure, not add a second traj_collision.png next to a stale
    # traj_reached.png. The verdict is still in the title and in the CSV.
    stem = os.path.join(out_dir, "traj")

    traj = np.asarray(traj, dtype=float).reshape(-1, 1 + 2 * dim)
    cols = ["t"] + [f"{a}{b}" for a in ("", "v") for b in "xyz"[:dim]]
    np.savetxt(f"{stem}.csv", traj, delimiter=",", header=",".join(cols),
               comments="")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[traj] saved {stem}.csv (matplotlib missing -- no plot)")
        return stem

    frames = [f for f in (frames or []) if f.get("plan") is not None
              and len(f["plan"]) > 1]
    if not frames:
        print(f"[traj] saved {stem}.csv (no recorded solve with a plan -- no plot)")
        return stem

    x, y = traj[:, 1], traj[:, 2]

    # ---- pick the solve to draw ------------------------------------------
    def _curvature(fr):
        d = np.diff(np.asarray(fr["plan"])[:, :2], axis=0)
        th = np.arctan2(d[:, 1], d[:, 0])
        return float(np.abs(np.unwrap(th)[-1] - np.unwrap(th)[0]))

    def _tightest(fr):
        """max over the horizon of (r_keep(t_k) - distance to the frontier): how
        close this plan came to its own expanding keep-out. Positive => inside
        it, i.e. stages the hard term was charging w_collision for."""
        plan = np.asarray(fr["plan"])
        bs = _frame_boundaries(fr)
        if not len(bs):
            return -np.inf
        d = bs.distance(plan)
        t_k = (np.arange(len(plan)) + 1) * cfg.dt
        r = np.array([_r_keep(cfg, float(t)) for t in t_k])
        return float(np.max(r - d))

    with_bnd = [f for f in frames if f.get("bnd") is not None and len(f["bnd"])]
    if solve_t is not None:
        fr = min(frames, key=lambda f: abs(f["t"] - solve_t))
        if abs(fr["t"] - solve_t) > cfg.dt:
            print(f"[traj] no solve recorded at t={solve_t:.1f}s -- using the "
                  f"nearest, t={fr['t']:.1f}s (recorded "
                  f"{frames[0]['t']:.1f}..{frames[-1]['t']:.1f}s)")
    elif with_bnd:
        fr = max(with_bnd, key=_tightest)
    else:
        fr = max(frames, key=_curvature)

    plan = np.asarray(fr["plan"], dtype=float)
    tk = (np.arange(len(plan)) + 1) * cfg.dt
    bset = _frame_boundaries(fr)

    # BASELINE RUNS (~use_occlusion false in mppi.launch). The node subscribes to
    # ~boundary_topic and rebuilds the BoundarySet regardless of the flag -- only
    # mppi.plan's cost skips it -- so the frontier is still recorded and still
    # gets drawn. The KEEP-OUT does not: nothing enforced it, and a contour on
    # the figure is a claim that something did.
    on = bool(cfg.use_occlusion)
    if not len(bset):
        print(f"[traj] no occlusion keep-out on this figure: "
              f"{len(with_bnd)}/{len(frames)} solves had a frontier"
              + ("" if with_bnd else " -- none all run (check ~use_occlusion, "
                                     "~boundary_topic and ~boundary_z_min)"))

    # ---- window ----------------------------------------------------------
    # Zoom on the EGO's own extent (start -> end) plus the plan it is
    # executing. The wall and the outer keep-out contours are deliberately left
    # out of the bounds: a saturated keep-out ring is d_safe + v_target *
    # t_grow_max across and would zoom the ego down to a few pixels. Anything
    # outside the window simply gets clipped.
    allx = np.concatenate([x, plan[:, 0], [goal[0]]])
    ally = np.concatenate([y, plan[:, 1], [goal[1]]])
    cx, cy = 0.5 * (allx.min() + allx.max()), 0.5 * (ally.min() + ally.max())
    half_x = 0.5 * (allx.max() - allx.min()) + margin
    half_y = 0.5 * (ally.max() - ally.min()) + margin
    # Clamp the window's aspect both ways. A straight run at the wall is ~15 m
    # of y over ~1 m of x, and with equal aspect that window is a pencil: the
    # keep-out contours -- the point of the figure -- leave the frame within a
    # centimetre of the path. Widening the SHORT axis keeps metres square.
    half_x, half_y = max(half_x, 0.5 * half_y), max(half_y, 0.5 * half_x)

    # The altitude panel is gated on z actually MOVING, not on dim == 3. The
    # normal wall-scene config is dim=3 with lock_altitude=true and
    # boundary_planar=true, i.e. a_z is zeroed in mppi.plan and publish_cmd pins
    # the setpoint to cruise_z -- three nominal DOF, two real ones. Gating on
    # dim would put a flat line at cruise_z under every figure and imply the run
    # had vertical freedom it did not. Unlock the altitude and the panel returns
    # on its own, which is when the dive-under-the-occluder failure can happen.
    z_moves = dim == 3 and max(
        float(np.ptp(traj[:, 3])), float(np.ptp(plan[:, 2]))) > 0.05
    if z_moves:
        fig, (ax, axz) = plt.subplots(
            2, 1, figsize=(12, 13), gridspec_kw={"height_ratios": [4, 1]})
    else:
        fig, ax = plt.subplots(figsize=(12, 12))
        axz = None

    # ---- the map the node was planning against ---------------------------
    # Banded to the drawn ego's altitude: unbanded this is mostly floor, and a
    # top-down scatter of the floor hides everything else on the figure.
    occ = np.asarray(fr.get("occ", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    ez = float(fr["ex"][2]) if dim == 3 else float(fr.get("query_z", 1.0))
    if len(occ):
        occ = occ[np.abs(occ[:, 2] - ez) <= occ_band]
    if len(occ):
        ax.scatter(occ[:, 0], occ[:, 1], s=3, c="0.45", marker="s", zorder=0,
                   label=f"occupied voxels (inf_occ, |z-{ez:.1f}| < {occ_band} m)")

    ax.plot(x, y, "-", color="0.65", lw=1.4, zorder=1, label="executed trajectory")
    ax.plot(x[0], y[0], "o", color="tab:green", ms=9, zorder=3, label="start")
    ax.plot(x[-1], y[-1], "s", color="tab:red", ms=9, zorder=3, label="end")
    ax.plot(goal[0], goal[1], "*", color="gold", ms=18, mec="k", zorder=3,
            label="goal")

    ax.plot(plan[:, 0], plan[:, 1], "-", color="k", lw=2.0, zorder=4,
            label=("braking rollout -- all rollouts collided "
                   f"(solve at t = {fr['t']:.1f} s)" if fr.get("infeasible") else
                   f"predicted rollout (solve at t = {fr['t']:.1f} s)"))

    # ---- the expanding keep-out, per sampled horizon time -----------------
    # One distance field, contoured at each r_k: the field does not depend on
    # t_k, only the level does, so this is one KD-tree query for the whole set
    # of slices rather than one per slice.
    gx = np.linspace(cx - half_x, cx + half_x, grid)
    gy = np.linspace(cy - half_y, cy + half_y, grid)
    GX, GY = np.meshgrid(gx, gy)
    # Skipped outright on a baseline run: with no contours to draw there is
    # nothing to rasterise, and this is the expensive part of the figure.
    if len(bset) and on:
        q = np.column_stack([GX.ravel(), GY.ravel()])
        D = bset.distance(q).reshape(GX.shape)
    else:
        D = None

    stage_colors = ["#00e5ff", "#ff00a0", "#ffd400", "#00ff7f",
                    "#7c4dff", "#ff6d00", "#00b0ff", "#c6ff00"]

    # t_k = 0: the ego pose at this solve, with the un-expanded keep-out. Drawn
    # dashed dark rather than in the stage palette: it is the t=0 reference every
    # coloured contour grows out of, not another sampled stage.
    if D is not None:
        ax.contour(GX, GY, D, levels=[cfg.d_safe], colors="0.15",
                   linestyles="--", linewidths=1.4, zorder=1)
    ax.plot(fr["ex"][0], fr["ex"][1], "o", mfc="w", mec="k", ms=6, mew=0.9,
            zorder=5, label="$t_k$ = 0.0 s" + (f"   r = {cfg.d_safe:.1f} m"
                                               if on else ""))

    stages = np.linspace(0, len(plan) - 1, min(max_frames, len(plan)))
    stages = list(dict.fromkeys(stages.round().astype(int).tolist()))
    for si, k in enumerate(stages):
        col = stage_colors[si % len(stage_colors)]
        t_k = float(tk[k])
        # The keep-out AT THIS STAGE: a worst-case hidden agent leaving the
        # frontier at t=0 at v_target can be anywhere within
        # d_safe + v_target*t_k of it, so each sampled timestamp gets its own,
        # larger, level set -- capped at t_grow_max, exactly as mppi.plan caps it.
        r_k = _r_keep(cfg, t_k)
        if D is not None:
            ax.contour(GX, GY, D, levels=[r_k], colors=[col], linewidths=1.8,
                       zorder=1)
        # The radius is dropped from the label on a baseline run: r_keep is a
        # property of a term that was switched off, so quoting it next to a
        # stage the planner never scored against it is noise.
        ax.plot(plan[k, 0], plan[k, 1], "o", mfc=col, mec="k", ms=6, mew=0.9,
                zorder=5, label=f"$t_k$ = {t_k:.1f} s"
                                + (f"   r = {r_k:.1f} m" if on else ""))

    # The frontier voxels themselves -- what everything above is dilated FROM.
    if len(bset):
        b = bset.points
        ax.plot(b[:, 0], b[:, 1], ".", color="crimson", ms=1.5, zorder=6,
                label=f"occlusion frontier ({len(b)} voxels)")

    # The one distance the cost is reacting to at this solve (or WOULD be, on a
    # baseline run), drawn as the vector that produced it -- the figure's
    # equivalent of the node's ~nearest_boundary marker.
    if len(bset):
        d0, i0 = bset.nearest(np.asarray(fr["ex"], float)[None, :])
        if np.isfinite(d0[0]):
            nb = bset.points[int(i0[0])]
            ax.plot([fr["ex"][0], nb[0]], [fr["ex"][1], nb[1]], "-",
                    color="magenta", lw=1.6, zorder=7,
                    label=f"nearest frontier at the solve: {d0[0]:.2f} m")

    # ---- how much of the keep-out this run actually violated ---------------
    # Occlusion-aware runs only, for the same reason the contours are: these
    # count breaches of a keep-out, and on a baseline there was none to breach.
    # Two scopes, both measured against the frontier as it was AT THE TIME,
    # never against the final map:
    #   plan  -- stages of the drawn rollout inside their own r_keep(t_k), i.e.
    #            stages mppi.plan charged w_collision for
    #   run   -- recorded solves whose EXECUTED pose was inside the t=0 keep-out
    # Both should be ~0, so the line is a regression check on the term.
    breach_txt = ""
    if len(bset) and on:
        d_plan = bset.distance(plan)
        r_plan = np.array([_r_keep(cfg, float(t)) for t in tk])
        n_plan = int(np.sum(d_plan < r_plan))
        d_run = np.array([
            float(_frame_boundaries(f).distance(
                np.asarray(f["ex"], float)[None, :])[0])
            for f in with_bnd])
        n_run = int(np.sum(d_run < cfg.d_safe))
        breach_txt = (
            f"plan: {n_plan}/{len(plan)} stages inside r_keep   |   "
            f"run: {n_run}/{len(with_bnd)} solves with the ego inside d_safe, "
            f"closest approach {d_run.min():.2f} m")

    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy - half_y, cy + half_y)
    # Equal aspect on a square canvas would force the y window out to the x
    # window's size, undoing the zoom. Match the canvas to the data instead:
    # metres stay square while the window stays tight around the ego.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    if on:
        title = (f"{verdict} -- predicted rollout at t = {fr['t']:.1f} s with "
                 f"the expanding occlusion keep-out per sampled $t_k$\n"
                 f"d_safe {cfg.d_safe:.1f} m, v_target {cfg.v_target:.1f} m/s, "
                 f"t_grow_max {cfg.t_grow_max:.1f} s "
                 f"=> r_keep {cfg.d_safe:.1f} -> {_r_keep(cfg, 1e9):.1f} m"
                 + (f"\n{breach_txt}" if breach_txt else ""))
    else:
        # No keep-out geometry and no keep-out parameters: on a baseline the
        # figure is the executed run, the map, the rollout and the frontier
        # the planner was ignoring, and nothing about a term that was off.
        title = (f"{verdict} -- BASELINE: occlusion term OFF "
                 f"(~use_occlusion false), collision only\n"
                 f"predicted rollout at t = {fr['t']:.1f} s")
    ax.set_title(title, color="k" if on else "tab:red", fontsize=11)
    # Legend OUTSIDE the axes: the window is tight around the ego, so "best"
    # placement lands it on top of the rollout every time.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              borderaxespad=0.0)

    # ---- altitude panel (3D only) ----------------------------------------
    # "Went around the wall" and "went over it" are the same curve from above.
    if axz is not None:
        axz.plot(traj[:, 0], traj[:, 3], "-", color="0.35", lw=1.4,
                 label="executed z")
        t_plan = fr["t"] + tk
        axz.plot(t_plan, plan[:, 2], "-", color="k", lw=2.0,
                 label="predicted z")
        for si, k in enumerate(stages):
            axz.plot(t_plan[k], plan[k, 2], "o",
                     mfc=stage_colors[si % len(stage_colors)], mec="k", ms=6,
                     mew=0.9)
        for bnd, lbl in ((cfg.z_min, "z_min"), (cfg.z_max, "z_max")):
            if bnd is not None:
                axz.axhline(bnd, color="tab:red", ls="--", lw=1.0)
                axz.text(traj[0, 0], bnd, f" {lbl}", color="tab:red",
                         fontsize=7, va="bottom")
        axz.axvline(fr["t"], color="0.6", ls=":", lw=1.0)
        axz.set_xlabel("episode time [s]")
        axz.set_ylabel("z [m]")
        axz.grid(True, alpha=0.3)
        axz.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                   borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=120)
    plt.close(fig)
    print(f"[traj] saved {stem}.csv and {stem}.png "
          f"({'occlusion-aware' if on else 'BASELINE, occlusion term OFF'}; "
          f"rollout from the solve at t={fr['t']:.1f}s of {len(frames)} "
          f"recorded, {len(stages)} sampled stages, {len(bset)} frontier "
          f"voxels, {len(occ)} banded occupied voxels)")
    if breach_txt:
        print(f"[traj] {breach_txt}")
    return stem
