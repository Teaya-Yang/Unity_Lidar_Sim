"""
config.py
=========
Every tunable in one place. The training script, the evaluator and the probe all
read the SAME config object, so a run is reproducible from its checkpoint alone
(train.py stores the config dict inside the .pt file).

Two groups matter most:

  * K (context length) — how many past transitions the context encoder sees. This
    is the single most important hyperparameter in the whole design: it decides
    whether eta is *identifiable* at all. Too short and the encoder cannot infer
    drag/lag no matter how big the network is; too long and windows start to run
    off the end of short rollouts. 10-20 steps (1-2 s at dt=0.1) is the sane band.

  * H (horizon) — how many steps the predictor is rolled before scoring. Training
    on H=1 gives a model that looks great and then compounds badly inside MPPI's
    30-step rollout, so keep this comfortably multi-step.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple


# Physical constants that must match the simulator / controllers.
DT = 0.1              # Unity Fixed Timestep and controller DT [s]

# Indices into the recorded 6-D state [x, y, theta, v, delta_actual, accel_actual]
IDX_X, IDX_Y, IDX_TH, IDX_V, IDX_DELTA, IDX_ACCEL = range(6)

POSE_SLICE = slice(0, 3)    # [x, y, theta]  — integrated, not predicted directly
CORE_SLICE = slice(3, 6)    # [v, delta, accel] — the translation/rotation-invariant part


@dataclass
class DataConfig:
    """Where the packed datasets live and how windows are cut from them."""
    packed_root: str = "packed"
    K: int = 16               # context window length  [steps]
    H: int = 15               # prediction horizon     [steps]
    # A window needs K steps of history BEFORE t and H steps of future AFTER t,
    # so a rollout must be at least K + H + 1 transitions long to yield any sample.
    min_rollout_len: int = 40

    # Enriched-observation mode (options 1+2): feed the encoders a noisy,
    # redundant sensor suite instead of the clean 3-D core. The model still
    # PREDICTS the clean core (via the decoder), so metrics stay comparable —
    # only the input becomes high-dimensional and noisy. Off by default so the
    # existing results are untouched.
    enriched: bool = False
    sensor_iid_scale: float = 0.08
    sensor_bias_scale: float = 0.05
    sensor_seed: int = 0


@dataclass
class ModelConfig:
    """Network widths. Deliberately small — the core state is 3-D, so capacity is
    not the bottleneck; identifiability and training stability are."""
    d_ctx: int = 64           # context/dynamics embedding  (the implicit eta)
    d_state: int = 64         # state embedding
    hidden: int = 128
    n_ctx_layers: int = 2     # GRU layers in the context encoder
    ema_tau: float = 0.99     # target-encoder EMA rate


@dataclass
class LossConfig:
    """Loss weights.

    w_latent   — JEPA's own objective: predicted latent vs EMA-target latent.
    w_decode   — decoder trained on the PREDICTED latents (not just encoded ones).
                 This is what keeps the decoder honest for multi-step rollout and
                 is what the evaluator ultimately measures.
    w_recon    — decoder trained on encoded latents (an autoencoder anchor that
                 stops the latent drifting away from something decodable).
    w_var/w_cov— VICReg anti-collapse terms. Collapse is THE failure mode of a
                 joint-embedding objective; without these the loss falls
                 beautifully while the representation goes constant.
    """
    w_latent: float = 1.0
    w_decode: float = 1.0
    w_recon: float = 0.5
    w_var: float = 1.0
    w_cov: float = 0.04
    var_gamma: float = 1.0    # target per-dimension std for the variance hinge


@dataclass
class TrainConfig:
    batch_size: int = 256
    steps: int = 8000          # optimiser steps (not epochs — we sample windows i.i.d.)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    val_every: int = 250
    log_every: int = 100
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "runs/jepa"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict:
        return {k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                for k, v in asdict(self).items()}

    @staticmethod
    def from_dict(d: Dict) -> "Config":
        return Config(
            data=DataConfig(**d["data"]),
            model=ModelConfig(**d["model"]),
            loss=LossConfig(**d["loss"]),
            train=TrainConfig(**d["train"]),
        )
