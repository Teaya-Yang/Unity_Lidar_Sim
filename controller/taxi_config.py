"""
Loader for config.yaml — the single source of truth for controller tuning.

Both controllers and the shared stage cost import CFG from here at module import
time and bind their module-level constants from it, so the ALL_CAPS names stay
exactly as they were at every call site; only their VALUES now come from the YAML.

Why the strict schema. A tuning file whose typos are silently ignored is worse
than no tuning file: you edit `d_safe_hard`, fat-finger it, see no error, and
conclude the parameter does not matter. So load() rejects BOTH unknown keys and
missing keys against SCHEMA below, naming the offender and its section. Adding a
parameter therefore means touching two places (config.yaml and SCHEMA), which is
the intended friction — it keeps the documented surface and the real one equal.

Loading an alternative tuning:

    python3 taxi_controller_mppi.py --config experiments/aggressive.yaml

The --config path is applied by reload() BEFORE the controllers read their
constants, which is why both controllers parse that flag early (see _early_config
in their argument handling).
"""

import os
from typing import Any, Dict

import numpy as np
import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

# section -> {key: kind}. "vec" is a fixed-length float list turned into np.ndarray;
# everything else is validated as the named Python type.
SCHEMA: Dict[str, Dict[str, str]] = {
    "vehicle": {
        "dt": "float", "wheelbase": "float", "v_des": "float",
        "drag_coeff": "float", "accel_tau": "float", "max_steer_rate": "float",
        "steer_rolloff_spd": "float", "steer_rolloff_min": "float",
    },
    "limits": {"a_min": "float", "a_max": "float", "delta_lim": "float"},
    "goal": {"slowdown_dist": "float", "min_speed": "float", "stop_dist": "float"},
    "cost": {
        "w_goal_run": "float", "w_goal_term": "float", "w_head": "float",
        "w_v": "float", "r_act": "vec", "r_dact": "vec",
    },
    "dynamic_obstacles": {
        "k_obs": "int", "d_safe": "float", "d_infl": "float", "w_obs": "float",
        "rho_slack": "float", "rho_slack2": "float", "d_infl_pass": "float",
        "unc_growth": "float", "unc_growth_max": "float",
        "headon_giveway": "float", "w_half": "float",
    },
    "keepout": {"d_safe_hard": "float", "w_hard": "float"},
    "occlusion": {
        "v_target": "float", "horizon": "float", "k_occ": "int", "query_r": "float",
        "fwd_half_angle": "float", "use_capsules": "bool", "single_circle": "bool",
        "w_sight": "float", "a_brake_sight": "float", "v_sight_floor": "float",
    },
    "occlusion_tracker": {
        "assoc_radius": "float", "alpha": "float", "ttl": "float", "min_hits": "int",
    },
    "dynamic_clusters": {
        "cell": "float", "min_points": "int", "max_radius": "float",
        "assoc_radius": "float", "alpha": "float", "vel_beta": "float",
        "ttl": "float", "min_hits": "int", "v_min": "float", "min_dyn_hits": "int",
        "dyn_window": "float", "extent_frac": "float", "resegment_ratio": "float",
        "require_motion": "bool", "grow_horizon": "float",
        "query_r": "float", "k_dyn": "int",
        "include_age": "bool",
    },
    "scan": {
        "fov_h": "float", "fov_v": "float", "res_h": "float", "res_v": "float",
        "max_range": "float",
    },
    "mppi": {
        "horizon": "int", "samples": "int", "lambda": "float",
        "sig_a": "float", "sig_d": "float",
        "w_lat": "float", "w_head": "float", "w_v": "float", "w_ctrl": "float",
        "w_off": "float", "w_prog": "float",
        "c_goal": "float", "c_goal_term": "float", "c_progress": "float",
        "n_scen": "int", "w_info": "float", "info_range": "float",
    },
    "mpc": {
        "horizon": "int", "k_static": "int", "static_query_r": "float",
        "sym_lat_thresh": "float", "sym_ahead_range": "float",
        "sym_bias": "float", "sym_bias_frac": "float",
    },
}


class ConfigError(ValueError):
    """Raised for a malformed, incomplete or unrecognised config file."""


def _coerce(section: str, key: str, kind: str, value: Any) -> Any:
    where = "{}.{}".format(section, key)
    if kind == "vec":
        if not isinstance(value, (list, tuple)) or not value:
            raise ConfigError("{}: expected a non-empty list, got {!r}".format(where, value))
        try:
            return np.array([float(x) for x in value], dtype=float)
        except (TypeError, ValueError):
            raise ConfigError("{}: list must be all numbers, got {!r}".format(where, value))
    if kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError("{}: expected true/false, got {!r}".format(where, value))
        return value
    # bool is a subclass of int in Python, so reject it explicitly for numbers.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("{}: expected a number, got {!r}".format(where, value))
    return int(value) if kind == "int" else float(value)


def load(path: str = None) -> Dict[str, Dict[str, Any]]:
    """Parse and validate a config file. Returns {section: {key: value}}."""
    path = DEFAULT_PATH if path is None else path
    try:
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError("config file not found: {}".format(path))
    except yaml.YAMLError as e:
        raise ConfigError("{} is not valid YAML: {}".format(path, e))

    if not isinstance(raw, dict):
        raise ConfigError("{}: top level must be a mapping of sections".format(path))

    unknown_sections = set(raw) - set(SCHEMA)
    if unknown_sections:
        raise ConfigError("{}: unknown section(s) {}".format(
            path, ", ".join(sorted(unknown_sections))))
    missing_sections = set(SCHEMA) - set(raw)
    if missing_sections:
        raise ConfigError("{}: missing section(s) {}".format(
            path, ", ".join(sorted(missing_sections))))

    out = {}
    for section, keys in SCHEMA.items():
        body = raw[section]
        if not isinstance(body, dict):
            raise ConfigError("{}: section '{}' must be a mapping".format(path, section))
        unknown = set(body) - set(keys)
        if unknown:
            raise ConfigError("{}: unknown key(s) in '{}': {}".format(
                path, section, ", ".join(sorted(unknown))))
        missing = set(keys) - set(body)
        if missing:
            raise ConfigError("{}: missing key(s) in '{}': {}".format(
                path, section, ", ".join(sorted(missing))))
        out[section] = {k: _coerce(section, k, kind, body[k]) for k, kind in keys.items()}

    _check_consistency(out, path)
    return out


def _check_consistency(cfg, path):
    """Cross-field invariants the schema alone cannot express.

    These are the relationships the inline comments in config.yaml describe; a
    config that violates one is not malformed, it is WRONG, and the failure mode
    is subtle behaviour rather than a crash — so catch it at startup.
    """
    def bad(msg):
        raise ConfigError("{}: {}".format(path, msg))

    if cfg["limits"]["a_min"] >= 0.0:
        bad("limits.a_min must be negative (it is the braking limit)")
    if cfg["limits"]["a_max"] <= 0.0:
        bad("limits.a_max must be positive")
    if cfg["vehicle"]["dt"] <= 0.0:
        bad("vehicle.dt must be positive")
    if cfg["dynamic_obstacles"]["d_infl"] < cfg["dynamic_obstacles"]["d_safe"]:
        bad("dynamic_obstacles.d_infl must be >= d_safe (the ring sits outside the keep-out)")
    if cfg["dynamic_obstacles"]["d_infl_pass"] <= cfg["dynamic_obstacles"]["d_safe"]:
        bad("dynamic_obstacles.d_infl_pass must exceed d_safe, or the ego only feels a "
            "frontal blocker at the last metre and clips the keep-out")
    if cfg["goal"]["min_speed"] <= 0.0:
        bad("goal.min_speed must be > 0: the steering authority rolls off toward zero "
            "speed, so a 0 floor stalls the ego mid-course")
    if cfg["occlusion"]["v_target"] <= 0.0:
        bad("occlusion.v_target must be positive")
    if not 0.0 < cfg["occlusion_tracker"]["alpha"] <= 1.0:
        bad("occlusion_tracker.alpha must be in (0, 1]")
    if not 0.0 < cfg["dynamic_clusters"]["alpha"] <= 1.0:
        bad("dynamic_clusters.alpha must be in (0, 1]")
    if not 0.0 < cfg["dynamic_clusters"]["vel_beta"] <= 1.0:
        bad("dynamic_clusters.vel_beta must be in (0, 1]")
    if cfg["dynamic_clusters"]["cell"] <= 0.0:
        bad("dynamic_clusters.cell must be positive")
    if cfg["dynamic_clusters"]["dyn_window"] <= 0.0:
        bad("dynamic_clusters.dyn_window must be positive (it divides the net displacement)")
    if cfg["dynamic_clusters"]["extent_frac"] < 0.0:
        bad("dynamic_clusters.extent_frac must be >= 0")
    if cfg["dynamic_clusters"]["resegment_ratio"] <= 1.0:
        bad("dynamic_clusters.resegment_ratio must exceed 1 (it is a ratio of extents, "
            "and <= 1 would declare EVERY window a re-segmentation, so nothing is ever "
            "classified as moving)")
    if cfg["dynamic_clusters"]["min_dyn_hits"] < 1:
        bad("dynamic_clusters.min_dyn_hits must be >= 1 (it is the number of still windows "
            "before demotion; 0 would demote a mover in the same window it was promoted)")
    if cfg["dynamic_clusters"]["assoc_radius"] <= cfg["occlusion"]["v_target"]:
        # The gate has to cover the distance an agent travels BETWEEN scans, else a moving
        # cluster is a brand-new track every scan and is never confirmed as dynamic. One
        # v_target of travel (~1 s of publishing) is the floor; the configured 8 m is ~1.6x.
        bad("dynamic_clusters.assoc_radius must exceed occlusion.v_target, or a moving "
            "cluster cannot be associated across scans and is never classified dynamic")
    if not 0.0 <= cfg["occlusion"]["fwd_half_angle"] <= 180.0:
        bad("occlusion.fwd_half_angle must be in [0, 180] degrees")
    if cfg["scan"]["res_h"] <= 0.0 or cfg["scan"]["res_v"] <= 0.0:
        bad("scan.res_h / scan.res_v must be positive")
    for planner in ("mppi", "mpc"):
        if cfg[planner]["horizon"] < 1:
            bad("{}.horizon must be >= 1".format(planner))
    if cfg["mppi"]["samples"] < 1:
        bad("mppi.samples must be >= 1")
    if cfg["mppi"]["lambda"] <= 0.0:
        bad("mppi.lambda must be positive (it divides the cost in the softmax)")


def _startup_path() -> str:
    """Resolve which config to load, BEFORE argparse has had a chance to run.

    This is the awkward part and it is unavoidable: the controllers bind their
    module-level ALL_CAPS constants at IMPORT time, which happens long before
    __main__ parses arguments. A --config handled in argparse would therefore
    arrive too late to affect a single constant.

    So the path is resolved here, at import, from (in order):
      1. --config PATH / --config=PATH on the command line
      2. $TAXI_CONFIG
      3. config.yaml next to this file
    Both controllers still DECLARE --config in their parsers, so it shows up in
    --help and is not rejected as unknown; the declaration is documentation, this
    function is the mechanism.
    """
    import sys
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--config":
            if i + 1 < len(argv):
                return argv[i + 1]
            raise ConfigError("--config requires a path argument")
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return os.environ.get("TAXI_CONFIG") or DEFAULT_PATH


CONFIG_PATH = _startup_path()
CFG = load(CONFIG_PATH)


def reload(path: str) -> Dict[str, Dict[str, Any]]:
    """Swap in an alternative config file, IN PLACE.

    Mutating CFG rather than rebinding it matters: the controllers may already
    hold a reference to the dict. Call this before they bind their constants —
    module-level names are snapshots, so a reload afterwards has no effect on
    them. Both controllers handle --config early for exactly this reason.
    """
    global CONFIG_PATH
    new = load(path)
    CFG.clear()
    CFG.update(new)
    CONFIG_PATH = path
    return CFG


def describe() -> str:
    """Flat 'section.key = value' dump — handy in a run's startup banner/log."""
    lines = []
    for section in SCHEMA:
        for key in SCHEMA[section]:
            lines.append("{}.{} = {}".format(section, key, CFG[section][key]))
    return "\n".join(lines)
