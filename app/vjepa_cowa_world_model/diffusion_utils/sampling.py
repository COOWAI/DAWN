"""
DPM-Solver++ sampling wrapper for diffusion models.

Adapted from XTR (xtr/diffusion_utils/sampling.py).
Uses DPM-Solver++ with 2 steps, 2nd order, logSNR schedule for fast sampling.
"""

from typing import Dict

import torch

from . import dpm_solver_pytorch as dpm


def dpm_sampler(
    model: torch.nn.Module,
    x_T,
    other_model_params: Dict = {},
    diffusion_steps=2,
    noise_schedule_params: Dict = {},
    model_wrapper_params: Dict = {},
    dpm_solver_params: Dict = {},
    sample_params: Dict = {},
):
    """
    Sample from a diffusion model using DPM-Solver++.

    Args:
        model: The denoising network. Must have a `model_type` property.
        x_T: Initial noise tensor [B, ...].
        other_model_params: Additional keyword args passed to the model.
        diffusion_steps: Number of DPM-Solver steps (default: 2).
        noise_schedule_params: Params for NoiseScheduleVP (default: linear schedule).
        model_wrapper_params: Params for model_wrapper (guidance type, etc.).
        dpm_solver_params: Params for DPM_Solver constructor.
        sample_params: Params for DPM_Solver.sample().

    Returns:
        (x_cls, x_0): Classification logits and denoised sample.
    """
    with torch.no_grad():
        noise_schedule = dpm.NoiseScheduleVP(schedule="linear", **noise_schedule_params)

        model_fn = dpm.model_wrapper(
            model, noise_schedule, model_type=model.model_type, model_kwargs=other_model_params, **model_wrapper_params
        )

        dpm_solver = dpm.DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++", **dpm_solver_params)

        sample_dpm = dpm_solver.sample(
            x_T,
            steps=diffusion_steps,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True,
            **sample_params
        )

    return sample_dpm
