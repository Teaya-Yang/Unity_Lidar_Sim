"""
domain_sampler.py
=================
Per-rollout dynamics-parameter sampler for the taxi data-collection pipeline.

Three conditions, in one place so training and evaluation always agree on ranges:

  A  in-distribution   — sampled from the TRAINING randomization ranges. Used for
                          both dataset generation and the in-distribution eval slice.
  B  out-of-range      — sampled OUTSIDE the training band, in the outer shell only,
                          so no B-condition eta overlaps with an A-condition one.
                          Tests parameter-value extrapolation (like the paper's
                          payload / propeller-switching conditions).
  C  unmodeled effect  — A-condition dynamics PLUS an extra term Unity applies that
                          the Python analytic bicycle model structurally cannot
                          represent (e.g. a v^2 * delta slip term, or asymmetric
                          brake response). The JEPA is trained purely on A rollouts
                          where these switches are OFF, so C tests true structural
                          model mismatch — the strongest sim-to-real proxy.

The parameter names match the module constants of taxi_controller_mppi.py so a rollout
worker can push the sampled eta straight into Unity via the EnvironmentParametersChannel
and, mirror the same eta into the Python-side tracker if a controller happens to read
these constants at construction time. Nominal values are copied here (not imported)
so this file has zero dependency on ML-Agents / CasADi — it can be run standalone
for offline dataset planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict

import numpy as np


# ── Nominal dynamics parameters ─────────────────────────────────────────────
# These MUST match the module constants in taxi_controller_mppi.py. Copied here
# (rather than imported) so this file is import-safe from anywhere without
# pulling ML-Agents / CasADi. Keep in sync manually; the test in __main__
# below will flag a drift if the controller file is importable.
NOMINAL: Dict[str, float] = dict(
    L=6.0,                    # wheelbase [m]
    DRAG_COEFF=0.04,          # 1/s
    ACCEL_TAU=0.5,            # s
    MAX_STEER_RATE=0.6,       # rad/s
    STEER_ROLLOFF_SPD=15.0,   # m/s
    STEER_ROLLOFF_MIN=0.25,   # unitless fraction
    A_MIN=-4.0,               # m/s^2
    A_MAX=1.5,                # m/s^2
    DELTA_LIM=0.5,            # rad
)


# ── Training randomization half-widths (Condition A) ────────────────────────
# Each entry is the FRACTIONAL half-width around NOMINAL. Chosen conservatively:
# broad enough to give the JEPA meaningful coverage, narrow enough that the
# out-of-range Condition-B shell (RANGE_A * OUT_OF_RANGE_MULT) is still physically
# plausible. Sign-flipped parameters (A_MIN is negative) are handled correctly by
# multiplying the NOMINAL magnitude — see _bounds() below.
RANGE_A: Dict[str, float] = dict(
    L=0.20,
    DRAG_COEFF=0.40,
    ACCEL_TAU=0.30,
    MAX_STEER_RATE=0.20,
    STEER_ROLLOFF_SPD=0.20,
    STEER_ROLLOFF_MIN=0.20,
    A_MIN=0.20,
    A_MAX=0.20,
    DELTA_LIM=0.15,
)


# How far outside the training band Condition-B goes. 1.5x the half-width places
# B samples strictly outside A's support, in the outer shell — so B is never a
# subset of A even after RNG collisions.
OUT_OF_RANGE_MULT: float = 1.5


# Unmodeled-effect ranges for Condition C. These are read on the Unity side
# only (via the EnvironmentParametersChannel), so they never appear in the
# Python analytic dynamics — that's the point of the condition.
#
# NOTE on absorbability: slip / brake_asymmetry / friction are all LOCALLY
# absorbable — over a ~1.5 s horizon an online least-squares fit of effective
# (drag, lag, wheelbase) reproduces them almost exactly, so they do NOT actually
# probe structural mismatch (an adaptive analytic model neutralises them). The
# actuation delay below is the genuine structural effect: a pure time-shift of
# the command sequence that no static parameter set can represent. It is
# quantised (Unity rounds it to whole control steps) so 1-4 = 0.1-0.4 s of delay.
UNMODELED_C: Dict[str, tuple] = dict(
    slip_coeff=(0.02, 0.10),          # lateral force ~ slip_coeff * v^2 * delta
    brake_asymmetry=(1.15, 1.60),     # multiplier on |a_cmd| when a_cmd < 0
    friction_noise_amp=(0.0, 0.20),   # spatial 1/f friction multiplier amplitude
    actuation_delay_steps=(1.0, 4.0), # pure transport delay [control steps]; Unity rounds
)


@dataclass
class RolloutEta:
    """One rollout's sampled dynamics parameters, plus the condition label.

    `params` is the 9-entry dict Unity/Python both consume. `unmodeled` is
    empty for A/B; for C it carries the extra Unity-only terms whose keys
    match what TaxiAgent.cs reads from EnvironmentParameters. `condition`
    is stamped so downstream logging can slice by it without re-inferring
    the source at load time.
    """
    condition: str
    params: Dict[str, float]
    unmodeled: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def env_channel_payload(self) -> Dict[str, float]:
        """Flat {key: value} dict for the ML-Agents EnvironmentParametersChannel.

        Both the standard bicycle params and any unmodeled-C terms go on the
        same channel, since TaxiAgent.cs reads them via one API. Keys are
        lower-snake_case to match Unity's convention (the C# side does
        GetWithDefault("accel_tau", 0.5f) etc.).
        """
        out = {k.lower(): float(v) for k, v in self.params.items()}
        out.update({k.lower(): float(v) for k, v in self.unmodeled.items()})
        # Boolean gate so the Unity side knows to APPLY the unmodeled term at
        # all — set from the condition, not the presence of a key, so an
        # A/B rollout can never accidentally trigger the extra physics.
        out["unmodeled_enabled"] = 1.0 if self.condition == "C" else 0.0
        return out


def _bounds(nom: float, frac: float) -> tuple[float, float]:
    """[lo, hi] centred on nom with half-width |nom| * frac, sign-safe."""
    hw = abs(nom) * frac
    return nom - hw, nom + hw


def sample_eta_A(rng: np.random.Generator) -> RolloutEta:
    """Condition A: uniform inside the training band, per parameter."""
    params = {}
    for k, nom in NOMINAL.items():
        lo, hi = _bounds(nom, RANGE_A[k])
        params[k] = float(rng.uniform(lo, hi))
    return RolloutEta(condition="A", params=params)


def sample_eta_B(rng: np.random.Generator,
                  push: float = OUT_OF_RANGE_MULT) -> RolloutEta:
    """Condition B: uniform in the OUTER SHELL only — strictly outside A's support.

    For each parameter we choose left- or right-outer-shell with equal probability
    and sample uniformly within it. This guarantees no B sample can be a valid A
    sample by chance — the two supports are disjoint, which is what makes the
    zero-shot claim clean.
    """
    params = {}
    for k, nom in NOMINAL.items():
        w = RANGE_A[k]
        lo_out, hi_out = _bounds(nom, push * w)
        lo_in, hi_in = _bounds(nom, w)
        # Half the time in the LEFT shell [lo_out, lo_in), half in the RIGHT (hi_in, hi_out].
        if rng.random() < 0.5:
            params[k] = float(rng.uniform(lo_out, lo_in))
        else:
            params[k] = float(rng.uniform(hi_in, hi_out))
    return RolloutEta(condition="B", params=params)


def sample_eta_C(rng: np.random.Generator) -> RolloutEta:
    """Condition C: A-band dynamics + unmodeled effect(s) enabled on the Unity side.

    The bicycle-model constants stay INSIDE the training range so any residual
    tracking gap can be attributed to the unmodeled term, not to param drift.
    """
    base = sample_eta_A(rng)
    unmodeled = {k: float(rng.uniform(lo, hi))
                 for k, (lo, hi) in UNMODELED_C.items()}
    return RolloutEta(condition="C", params=base.params, unmodeled=unmodeled)


SAMPLERS = {"A": sample_eta_A, "B": sample_eta_B, "C": sample_eta_C}


def sample(condition: str, rng: np.random.Generator) -> RolloutEta:
    """Dispatch — string-keyed so the orchestrator can select via CLI flag."""
    if condition not in SAMPLERS:
        raise ValueError(f"unknown condition {condition!r}; expected one of A/B/C")
    return SAMPLERS[condition](rng)


if __name__ == "__main__":
    # Quick sanity: distributions look right, A and B never overlap on any key,
    # and if the real controller module is importable its NOMINALs still match.
    import statistics as _st
    rng = np.random.default_rng(0)
    for cond in ("A", "B", "C"):
        vals = [sample(cond, rng) for _ in range(500)]
        for k in NOMINAL:
            xs = [v.params[k] for v in vals]
            print(f"  {cond} {k:20s} mean={_st.mean(xs):+8.3f}  "
                  f"min={min(xs):+8.3f}  max={max(xs):+8.3f}")
        print()

    # Assert A/B disjoint on every parameter.
    a_vals = [sample("A", rng) for _ in range(2000)]
    b_vals = [sample("B", rng) for _ in range(2000)]
    for k in NOMINAL:
        a_hi = max(v.params[k] for v in a_vals)
        a_lo = min(v.params[k] for v in a_vals)
        b_in_a = sum(a_lo <= v.params[k] <= a_hi for v in b_vals)
        assert b_in_a == 0, f"A/B overlap on {k}: {b_in_a} B-samples inside A-range"
    print("A/B disjointness OK on all parameters.")

    # Optional drift check against the live controller module.
    try:
        import taxi_controller_mppi as tc
        for k, v in NOMINAL.items():
            live = getattr(tc, k)
            assert abs(live - v) < 1e-9, f"NOMINAL[{k}]={v} but controller has {live}"
        print("NOMINAL matches taxi_controller_mppi.py.")
    except ImportError:
        print("(taxi_controller_mppi not importable here — skip drift check)")