"""Predictor JEPA loss helpers."""

from typing import Any, Callable, Optional, Tuple


def predictor_needs_z_ar_rollout(config: Any) -> bool:
    """Return whether the current config actually needs predictor z_ar outputs."""

    if config.train.predictor_use_z_ar_supervision:
        return True

    return bool(config.planner.use_planner and config.planner.planner_input_source != "z_tf")


def compute_predictor_jepa_losses_from_config(
    z_tf: Any,
    z_ar: Any,
    h_target: Any,
    config: Any,
    tokens_per_frame: int,
    loss_fn: Callable[..., Any],
    num_observed_steps: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """Compute predictor JEPA losses from config.

    Returns
    -------
    tuple:
        jepa_loss, jloss, sloss
    """

    observed_steps = int(num_observed_steps) if num_observed_steps is not None else config.train.num_observed_frames

    if bool(getattr(config.train, "use_parallel_predictor", False)):
        future_start_step = observed_steps if config.train.predictor_inference_consistent else 1
        future_offset = future_start_step * tokens_per_frame
        jloss = loss_fn(
            z_tf[:, future_offset:],
            h_target,
            offset=future_offset,
        )
        if config.train.predictor_use_z_ar_supervision:
            if z_ar is None:
                raise ValueError("z_ar must be provided when predictor_use_z_ar_supervision=True")
            sloss = loss_fn(z_ar, h_target, offset=future_offset)
        else:
            sloss = jloss * 0.0
        jepa_loss = jloss + sloss
        return jepa_loss, jloss, sloss

    if config.train.predictor_inference_consistent:
        ar_offset = observed_steps * tokens_per_frame
        observed_tokens = (observed_steps - 1) * tokens_per_frame
        jloss = loss_fn(
            z_tf[:, observed_tokens:],
            h_target,
            offset=ar_offset,
        )
    else:
        ar_offset = tokens_per_frame
        jloss = loss_fn(z_tf, h_target, offset=tokens_per_frame)

    if config.train.predictor_use_z_ar_supervision:
        if z_ar is None:
            raise ValueError("z_ar must be provided when predictor_use_z_ar_supervision=True")
        sloss = loss_fn(z_ar, h_target, offset=ar_offset)
    else:
        sloss = jloss * 0.0

    jepa_loss = jloss + sloss
    return jepa_loss, jloss, sloss


def compute_lewm_projected_jepa_losses_from_config(
    z_tf: Any,
    z_ar: Any,
    h_target: Any,
    config: Any,
    tokens_per_frame: int,
    project_fn: Callable[[Any], Any],
    loss_fn: Callable[..., Any],
    num_observed_steps: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """Project predictor outputs into le-wm space, then compute JEPA losses."""

    z_tf_projected = project_fn(z_tf)
    z_ar_projected = project_fn(z_ar) if config.train.predictor_use_z_ar_supervision else None

    return compute_predictor_jepa_losses_from_config(
        z_tf=z_tf_projected,
        z_ar=z_ar_projected,
        h_target=h_target,
        config=config,
        tokens_per_frame=tokens_per_frame,
        loss_fn=loss_fn,
        num_observed_steps=num_observed_steps,
    )
