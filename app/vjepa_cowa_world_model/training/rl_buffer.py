from dataclasses import dataclass
from typing import Dict, List

import torch


def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    """Compute GAE(lambda) advantages and returns."""
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    next_value = last_value

    for step in reversed(range(steps)):
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * non_terminal - values[step]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[step] = gae
        next_value = values[step]

    returns = advantages + values
    return advantages, returns


@dataclass
class RolloutBatch:
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.95, normalize_advantage=True):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage
        self.reset()

    def reset(self):
        self.observations: List[Dict[str, torch.Tensor]] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []

    @staticmethod
    def _to_storage_tensor(tensor):
        return tensor.detach().cpu()

    def _to_storage_observation(self, observation):
        return {
            key: self._to_storage_tensor(value)
            for key, value in observation.items()
        }

    def add(self, observation, action, log_prob, value, reward, done):
        self.observations.append(self._to_storage_observation(observation))
        self.actions.append(self._to_storage_tensor(action))
        self.log_probs.append(self._to_storage_tensor(log_prob))
        self.values.append(self._to_storage_tensor(value))
        self.rewards.append(self._to_storage_tensor(reward))
        self.dones.append(self._to_storage_tensor(done))

    def __len__(self):
        return len(self.actions)

    def _stack_observations(self):
        if not self.observations:
            raise ValueError("RolloutBuffer is empty")
        keys = self.observations[0].keys()
        return {
            key: torch.stack([obs[key] for obs in self.observations], dim=0)
            for key in keys
        }

    def finalize(self, last_value):
        rewards = torch.stack(self.rewards, dim=0)
        values = torch.stack(self.values, dim=0)
        dones = torch.stack(self.dones, dim=0)
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value.detach().cpu(),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)

        batch = RolloutBatch(
            observations=self._stack_observations(),
            actions=torch.stack(self.actions, dim=0),
            old_log_prob=torch.stack(self.log_probs, dim=0),
            old_value=values,
            returns=returns,
            advantages=advantages,
        )
        return batch

    @staticmethod
    def iter_minibatches(batch: RolloutBatch, mini_batch_size, shuffle=True):
        batch_size = batch.actions.shape[0]
        indices = torch.arange(batch_size)
        if shuffle:
            indices = indices[torch.randperm(batch_size)]

        for start in range(0, batch_size, mini_batch_size):
            idx = indices[start:start + mini_batch_size]
            yield RolloutBatch(
                observations={k: v[idx] for k, v in batch.observations.items()},
                actions=batch.actions[idx],
                old_log_prob=batch.old_log_prob[idx],
                old_value=batch.old_value[idx],
                returns=batch.returns[idx],
                advantages=batch.advantages[idx],
            )
