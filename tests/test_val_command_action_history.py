from types import SimpleNamespace

import torch

from app.vjepa_cowa_world_model.val_command import validate_one_epoch


class _Sampler:
    def set_epoch(self, epoch):
        self.epoch = epoch


class _Encoder(torch.nn.Module):
    def __init__(self, tokens_per_frame=2, embed_dim=4):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        self.embed_dim = embed_dim

    def forward(self, inputs):
        batch_times_steps = inputs[0].shape[0]
        return [torch.zeros(batch_times_steps, self.tokens_per_frame, self.embed_dim)]


class _Predictor(torch.nn.Module):
    def forward(self, tokens, actions, states, extrinsics, action_mask=None):
        return torch.zeros_like(tokens)


class _Planner(torch.nn.Module):
    def __init__(self, num_poses=2):
        super().__init__()
        self.num_poses = num_poses
        self.received_action_history = None

    def forward(self, z_ar, status_feature, z_context=None, z_observed=None, action_history=None, anchor_state=None):
        self.received_action_history = action_history
        if action_history is None:
            raise AssertionError("expected validation to pass planner action_history")
        batch_size = z_ar.shape[0]
        return {
            "trajectories": torch.zeros(batch_size, 1, self.num_poses, 3),
            "confidences": torch.ones(batch_size, 1),
        }


def _config():
    return SimpleNamespace(
        meta=SimpleNamespace(seed=0),
        model=SimpleNamespace(backbone="vjepa2"),
        train=SimpleNamespace(use_parallel_predictor=False),
        data=SimpleNamespace(fps=2),
        planner=SimpleNamespace(use_action_history_for_planner=True, action_history_dim=3),
    )


def test_validate_one_epoch_passes_action_history_to_planner():
    batch_size = 1
    num_frames = 3
    context_frames = torch.zeros(batch_size, 3, num_frames, 4, 4)
    actions = torch.zeros(batch_size, num_frames - 1, 3)
    actions[:, 0, 0] = 1.0
    states = torch.zeros(batch_size, num_frames, 7)
    extrinsics = torch.zeros(batch_size, num_frames, 4, 4)
    sample = (context_frames, actions, states, extrinsics, None)

    planner = _Planner(num_poses=2)
    metrics = validate_one_epoch(
        encoder=_Encoder(tokens_per_frame=2, embed_dim=4),
        predictor=_Predictor(),
        planner=planner,
        val_loader=[sample],
        val_sampler=_Sampler(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        mixed_precision=False,
        tubelet_size=2,
        tokens_per_frame=2,
        num_poses=2,
        num_time_steps=num_frames,
        world_size=1,
        rank=0,
        epoch=0,
        normalize_reps=False,
        status_mode="first",
        z_ar_mode="full",
        use_z_context=False,
        use_tubelet_repeat=True,
        use_states_for_planner=False,
        action_dim=3,
        predictor_inference_consistent=False,
        num_observed_frames=2,
        use_states_for_predictor=True,
        predictor_no_aux_input=False,
        use_observed_tokens=False,
        state_dim=7,
        planner_status_dim=0,
        predictor_use_drive_command=True,
        planner_use_drive_command=True,
        planner_type="transformer",
        timestep_sec=0.5,
        compute_collision=False,
        config=_config(),
    )

    assert metrics["ade"] == 0.0
    assert planner.received_action_history is not None
    assert planner.received_action_history.shape == (batch_size, 2, 3)
