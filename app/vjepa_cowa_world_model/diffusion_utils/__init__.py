# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Diffusion utilities for trajectory prediction.

Adapted from XTR (xtr/diffusion_utils/).
Includes:
- VPSDE_linear: VP-SDE for the forward diffusion process
- DPM-Solver++: Fast ODE-based sampling
- dpm_sampler: Convenience wrapper for DPM-Solver++ sampling
"""

from .sampling import dpm_sampler
from .sde import SDE, VPSDE_linear

__all__ = [
    "SDE",
    "VPSDE_linear",
    "dpm_sampler",
]
