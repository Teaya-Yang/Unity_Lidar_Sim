"""
jepa — a context-conditioned dynamics model for the taxi, plus the evaluation
that compares it against the analytic bicycle model on conditions A/B/C.

Layout
------
  config.py    every tunable, in one dataclass tree
  data.py      packed .npz -> context/horizon windows (never crossing a rollout)
  models.py    ContextEncoder / StateEncoder / Predictor / Decoder + EMA target
  losses.py    latent + decode + recon + VICReg anti-collapse
  dynamics.py  the analytic bicycle model, the pose integrator, LS parameter fit
  rollout.py   ONE open-loop rollout interface, shared by every model
  train.py     trains on condition A only
  evaluate.py  the A/B/C head-to-head against three analytic baselines
  probe.py     linear probe: does the context embedding encode eta?

Typical run
-----------
  python -m jepa.train    --steps 8000 --out runs/jepa
  python -m jepa.probe    --ckpt runs/jepa/jepa_best.pt
  python -m jepa.evaluate --ckpt runs/jepa/jepa_best.pt
"""

__all__ = ["config", "data", "models", "losses", "dynamics", "rollout",
           "train", "evaluate", "probe"]
