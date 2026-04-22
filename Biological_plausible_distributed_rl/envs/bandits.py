"""
Multi-Armed Bandit Environments for Distributional RL experiments.

Supports various reward distributions per arm, designed to test
risk-sensitive decision-making:
- Same mean, different variance (key test case)
- Bimodal distributions
- Heavy-tailed distributions
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ArmConfig:
    """Configuration for a single bandit arm."""
    name: str
    distribution: str  # "gaussian", "bimodal", "uniform", "lognormal"
    params: dict = field(default_factory=dict)
    
    def sample(self, n: int = 1) -> torch.Tensor:
        """Sample n rewards from this arm's distribution."""
        if self.distribution == "gaussian":
            mu = self.params.get("mean", 0.0)
            sigma = self.params.get("std", 1.0)
            return torch.normal(mu, sigma, size=(n,))
        
        elif self.distribution == "bimodal":
            # Mixture of two Gaussians
            mu1 = self.params.get("mean1", -1.0)
            mu2 = self.params.get("mean2", 1.0)
            sigma = self.params.get("std", 0.3)
            mix = self.params.get("mix", 0.5)
            mask = torch.bernoulli(torch.full((n,), mix)).bool()
            samples = torch.where(
                mask,
                torch.normal(mu1, sigma, size=(n,)),
                torch.normal(mu2, sigma, size=(n,)),
            )
            return samples
        
        elif self.distribution == "uniform":
            low = self.params.get("low", 0.0)
            high = self.params.get("high", 1.0)
            return torch.rand(n) * (high - low) + low
        
        elif self.distribution == "lognormal":
            mu = self.params.get("log_mean", 0.0)
            sigma = self.params.get("log_std", 1.0)
            shift = self.params.get("shift", 0.0)
            normal_samples = torch.normal(mu, sigma, size=(n,))
            return torch.exp(normal_samples) + shift
        
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
    
    @property
    def true_mean(self) -> float:
        """Analytical mean of this arm's distribution."""
        if self.distribution == "gaussian":
            return self.params.get("mean", 0.0)
        elif self.distribution == "bimodal":
            mix = self.params.get("mix", 0.5)
            return mix * self.params.get("mean1", -1.0) + (1 - mix) * self.params.get("mean2", 1.0)
        elif self.distribution == "uniform":
            return (self.params.get("low", 0.0) + self.params.get("high", 1.0)) / 2
        elif self.distribution == "lognormal":
            mu = self.params.get("log_mean", 0.0)
            sigma = self.params.get("log_std", 1.0)
            return np.exp(mu + sigma**2 / 2) + self.params.get("shift", 0.0)
        return 0.0

    @property
    def true_variance(self) -> float:
        """Analytical variance of this arm's distribution."""
        if self.distribution == "gaussian":
            return self.params.get("std", 1.0) ** 2
        elif self.distribution == "uniform":
            low = self.params.get("low", 0.0)
            high = self.params.get("high", 1.0)
            return (high - low) ** 2 / 12
        elif self.distribution == "bimodal":
            # Variance of mixture: E[X^2] - E[X]^2
            mix = self.params.get("mix", 0.5)
            mu1, mu2 = self.params.get("mean1", -1.0), self.params.get("mean2", 1.0)
            sigma = self.params.get("std", 0.3)
            ex2 = mix * (mu1**2 + sigma**2) + (1 - mix) * (mu2**2 + sigma**2)
            ex = self.true_mean
            return ex2 - ex**2
        return float("nan")


class MultiArmedBandit:
    """
    Multi-armed bandit environment.
    
    Usage:
        env = MultiArmedBandit.equal_mean_different_variance()
        for step in range(1000):
            action = agent.select_action()
            reward = env.step(action)
            agent.update(action, reward)
    """
    
    def __init__(self, arms: List[ArmConfig]):
        self.arms = arms
        self.n_arms = len(arms)
        self.step_count = 0
        self.action_counts = torch.zeros(self.n_arms)
        self.reward_history: List[Tuple[int, float]] = []
    
    def step(self, action: int) -> float:
        """Take an action, return a reward."""
        assert 0 <= action < self.n_arms, f"Invalid action {action}"
        reward = self.arms[action].sample(1).item()
        self.step_count += 1
        self.action_counts[action] += 1
        self.reward_history.append((action, reward))
        return reward
    
    def reset_stats(self):
        """Reset tracking statistics (not the environment itself)."""
        self.step_count = 0
        self.action_counts = torch.zeros(self.n_arms)
        self.reward_history = []
    
    def get_true_means(self) -> torch.Tensor:
        return torch.tensor([arm.true_mean for arm in self.arms])
    
    def get_true_variances(self) -> torch.Tensor:
        return torch.tensor([arm.true_variance for arm in self.arms])
    
    def __repr__(self) -> str:
        lines = [f"MultiArmedBandit({self.n_arms} arms):"]
        for i, arm in enumerate(self.arms):
            lines.append(
                f"  [{i}] {arm.name}: {arm.distribution} "
                f"(mean={arm.true_mean:.3f}, var={arm.true_variance:.3f})"
            )
        return "\n".join(lines)
    
    # --- Factory methods for standard experimental setups ---
    
    @classmethod
    def equal_mean_different_variance(
        cls, mean: float = 1.0, stds: List[float] = [0.5, 1.0, 2.0, 4.0]
    ) -> "MultiArmedBandit":
        """
        Key test case: all arms have the same expected value,
        but different variances. A scalar TD agent should be indifferent,
        while a distributional agent can learn risk-sensitive policies.
        """
        arms = [
            ArmConfig(
                name=f"σ={s:.1f}",
                distribution="gaussian",
                params={"mean": mean, "std": s},
            )
            for s in stds
        ]
        return cls(arms)
    
    @classmethod
    def safe_vs_risky(cls) -> "MultiArmedBandit":
        """
        Two arms with equal mean but very different risk profiles.
        Safe: deterministic reward of 1.0
        Risky: bimodal, either 0 or 2 with equal probability
        """
        arms = [
            ArmConfig(
                name="Safe",
                distribution="gaussian",
                params={"mean": 1.0, "std": 0.01},
            ),
            ArmConfig(
                name="Risky",
                distribution="bimodal",
                params={"mean1": 0.0, "mean2": 2.0, "std": 0.1, "mix": 0.5},
            ),
        ]
        return cls(arms)
    
    @classmethod
    def foraging_task(cls) -> "MultiArmedBandit":
        """
        Foraging-inspired environment:
        - Patch A: reliable small rewards (berries)
        - Patch B: variable medium rewards (variable fruit)
        - Patch C: rare large rewards (prey, high variance)
        All with similar expected values.
        """
        target_mean = 2.0
        arms = [
            ArmConfig(
                name="Berries (safe)",
                distribution="gaussian",
                params={"mean": target_mean, "std": 0.2},
            ),
            ArmConfig(
                name="Fruit (medium)",
                distribution="gaussian",
                params={"mean": target_mean, "std": 1.5},
            ),
            ArmConfig(
                name="Prey (risky)",
                distribution="bimodal",
                params={"mean1": 0.0, "mean2": 4.0, "std": 0.3, "mix": 0.5},
            ),
        ]
        return cls(arms)
