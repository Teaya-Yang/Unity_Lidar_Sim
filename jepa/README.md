# jepa — context-conditioned dynamics model + A/B/C evaluation

Learns the taxi's dynamics from condition-A rollouts and tests whether it
generalises to conditions B (out-of-range parameters) and C (unmodeled physics)
better than the analytic bicycle model the planners use today.

Everything runs on **CPU**. The model is ~268k parameters; a full training run is
~20 minutes on a laptop.

## Quick start

```bash
cd ~/Unity_Lidar_Sim
export PYTHONPATH="$PWD/controller"          # so the analytic baseline can import
                                              # the live controller constants
P=.venv/bin/python

$P -m jepa.train    --steps 8000 --out runs/jepa
$P -m jepa.probe    --ckpt runs/jepa/jepa_best.pt      # is `c` encoding eta?
$P -m jepa.evaluate --ckpt runs/jepa/jepa_best.pt      # the comparison table
```

Prerequisite: `packed/{A,B,C}/*.npz` from `dataset_collection/split_and_pack.py`.

## Layout

| file | role |
|---|---|
| `config.py` | every tunable, one dataclass tree; stored inside each checkpoint |
| `data.py` | packed `.npz` → context/horizon windows that never cross a rollout |
| `models.py` | ContextEncoder / StateEncoder / Predictor / Decoder + EMA target |
| `losses.py` | latent + decode + recon + VICReg anti-collapse |
| `dynamics.py` | analytic bicycle model, shared pose integrator, least-squares η fit |
| `rollout.py` | one open-loop rollout interface, used by *every* model |
| `train.py` | trains on condition A **only** |
| `evaluate.py` | A/B/C head-to-head vs three analytic baselines |
| `probe.py` | linear probe: does the context embedding encode η? |

## The architecture

```
context window (K past transitions)  ──► ContextEncoder ──► c   "which plant"
core state [v, δ, accel] at t        ──► StateEncoder   ──► z   "which state"

for h in range(H):
    z, dθ  = Predictor(z, action_h, c)          # latent step + heading change
    core_h = Decoder(z)                          # readout to physical units
```

Two design choices worth knowing:

- **`c` is computed once and held fixed across the horizon.** That encodes "the
  plant does not change within a rollout", which is exactly how the data was
  generated (one η per rollout).
- **The target is a single encoded state, not a window.** A sliding-window target
  would overlap the context by K−1 transitions and let the model score well by
  shifting and copying instead of learning dynamics.

## The state factorisation

The recorded state is `[x, y, θ, v, δ_actual, accel_actual]`, but the dynamics are
translation- and rotation-invariant. So the network only predicts the invariant
core `[v, δ, accel]` plus the per-step heading change `dθ`; position is then
integrated by `dynamics.integrate_pose`, which every analytic baseline also uses.
A position error therefore reflects the model, never a difference in integrator.

## The baselines

| baseline | η source | why it's there |
|---|---|---|
| `analytic_fixed` | nominal constants | the model inside MPPI/MPC **today** |
| `analytic_ls` | least squares on the context window | **strong** — adaptive analytic |
| `analytic_oracle` | true η from `eta_json` | **ceiling** — see below |
| `jepa` | inferred from context | the learned model |

`analytic_oracle` matters because *in simulation the analytic equations ARE the
plant*. Handed the true η it is exact on A and B — you should see `theta@15` and
`v@15` of literally `0.0000` there, and that is a correctness check on the whole
evaluation harness. On C it still has an irreducible error floor, because no
parameter value creates a `v²·δ` term. **That gap is the only place a learned
model can win structurally**, which is why C is the condition that matters.

## Reading the output

The headline is not absolute error on C — it is **degradation**, the `C - A`
column. Condition C's nine dynamics parameters are drawn from A's in-distribution
band by construction, so the only difference between A and C is the unmodeled
physics. Extra error on C is therefore attributable to structural mismatch and
nothing else.

The `SLIP_COEFF` slice table is the decisive result: if an analytic row rises with
slip while the learned row stays flat, that is a capability gap, not a tuning gap.

## Known caveat: the LS baseline partially absorbs slip

Empirically `analytic_ls` stays nearly flat across the slip bins. The reason is
that the slip term `slip·v²·δ` and the yaw term `(v/L)·tan δ` have similar shape
over the sampled range, so re-fitting an *effective* wheelbase absorbs much of the
slip effect. Condition C is therefore less structurally unrepresentable than
intended for this particular effect.

`BRAKE_ASYMMETRY` and `FRICTION_NOISE_AMP` are not absorbable this way — check
those slices too:

```bash
$P -m jepa.evaluate --ckpt runs/jepa/jepa_best.pt --slice-c BRAKE_ASYMMETRY
$P -m jepa.evaluate --ckpt runs/jepa/jepa_best.pt --slice-c FRICTION_NOISE_AMP
```

## Watch for collapse, not loss

A joint-embedding objective can always cheat by making every embedding equal; the
loss then falls beautifully while the representation carries nothing. Watch
`emb_std` and `ctx_std` in the training log — healthy is ≳0.4. If they slide
toward zero, raise `--w-var`.

## Enriched mode — noisy redundant sensors (`--enriched`)

The plain state is 3 clean numbers, which is too simple for latent modelling to
beat direct prediction. Enriched mode replaces the encoder input with a **noisy,
redundant 10-channel sensor suite** (`sensors.py`: four wheel speeds, a 2-axis
accelerometer, gyro, steering sensor, motor current, pitot speed — all encoding
the same 3-4 DOF, plus per-step noise and a per-rollout bias). The model still
**predicts the clean core**, so every metric and the A/B/C comparison are
unchanged; only the input becomes high-dimensional and noisy — the regime where
a JEPA's denoising/fusion is supposed to beat a direct predictor.

The sensors are synthesised in Python from the recorded clean state (they are
deterministic functions of it + noise), so **no Unity rebuild or re-collection is
needed** — the same packed datasets are reused.

```bash
# both models must be trained with the SAME --enriched settings
$P -m jepa.train            --enriched --steps 8000 --out runs/jepa_enr
$P -m jepa.train_predictive --enriched --steps 8000 --out runs/predictive_enr
$P -m jepa.evaluate --ckpt runs/jepa_enr/jepa_best.pt \
                    --predictive-ckpt runs/predictive_enr/predictive_best.pt \
                    --slice-c ACTUATION_DELAY_STEPS
```

The question this answers: does the JEPA's latent formulation finally beat the
autoregressive predictor once the observation is noisy and redundant? In plain
mode it did not (the state was too clean). `--sensor-iid` / `--sensor-bias`
control the noise levels. Plain-mode checkpoints and results are unaffected —
`enriched` defaults off and the plain architecture is byte-identical.

## Run the probe first

`probe.py` fits a linear map from the context embedding `c` to the true η. If
`DRAG_COEFF` / `ACCEL_TAU` / `L` come back with high R², the context encoder is
genuinely doing system identification and the B/C results mean something. If they
don't, fix that first — increase `--K`, or check that the data has enough
excitation. Low R² on `A_MIN`/`A_MAX`/`DELTA_LIM` is expected: a limit is only
observable when the trajectory actually saturates it.
