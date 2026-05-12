"""Inference-safe training helpers package.

Import concrete modules directly, for example::

    from app.vjepa_cowa_world_model.training.config import parse_training_config

Keeping this package initializer lightweight avoids importing optional training,
RL, and simulator dependencies during offline validation startup.
"""
