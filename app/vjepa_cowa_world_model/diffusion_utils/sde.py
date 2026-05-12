"""
SDE (Stochastic Differential Equation) classes for diffusion models.

Adapted from XTR (xtr/diffusion_utils/sde.py).
Only includes VPSDE_linear which is used for trajectory prediction.
"""

import abc

import torch


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self):
        super().__init__()

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        """
        Returns the drift and diffusion coefficients of the SDE.

        Returns:
            (drift f(x,t), diffusion g(t))
        """
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """
        Parameters to determine the marginal distribution of the SDE, p_t(x).

        Returns:
            (mean, std)
        """
        pass

    @abc.abstractmethod
    def diffusion_coeff(self, t):
        """
        Returns the diffusion coefficient g(t) of the SDE.
        """
        pass

    @abc.abstractmethod
    def marginal_prob_std(self, t):
        """
        Returns the standard deviation of the marginal distribution p_t(x).
        """
        pass


class VPSDE_linear(SDE):
    """
    VP-SDE with linear beta schedule.

    SDE: dx = -beta(t)/2 * x dt + sqrt(beta(t)) dW_t

    where beta(t) = beta_min + (beta_max - beta_min) * t
    """

    def __init__(self, beta_max=20.0, beta_min=0.1):
        super().__init__()
        self._beta_max = beta_max
        self._beta_min = beta_min

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        shape = x.shape
        reshape = [-1] + [
            1,
        ] * (len(shape) - 1)
        t = t.reshape(reshape)

        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def marginal_prob(self, x, t):
        shape = x.shape
        reshape = [-1] + [
            1,
        ] * (len(shape) - 1)
        t = t.reshape(reshape)
        mean_log_coeff = -0.25 * t**2 * (self._beta_max - self._beta_min) - 0.5 * self._beta_min * t

        mean = torch.exp(mean_log_coeff) * x
        std = torch.sqrt(1 - torch.exp(2.0 * mean_log_coeff))
        return mean, std

    def diffusion_coeff(self, t):
        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        diffusion = torch.sqrt(beta_t)
        return diffusion

    def marginal_prob_std(self, t):
        discount = torch.exp(-0.5 * t**2 * (self._beta_max - self._beta_min) - self._beta_min * t)
        std = torch.sqrt(1 - discount)
        return std
