"""
Quantile Regression Agents for Multi-Armed Bandits.

Two versions:
1. TabularQR: Direct quantile estimation per arm (no neural net)
2. QRDQNAgent: Neural network version for state-dependent tasks (Phase 3)

Reference: Dabney et al. "Distributional Reinforcement Learning with
Quantile Regression" (AAAI 2018)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class QRDQNConfig:
    """Configuration for QR-DQN agent."""
    n_arms: int = 4
    n_quantiles: int = 32
    hidden_dim: int = 64
    lr: float = 1e-3
    epsilon: float = 0.1
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.01
    kappa: float = 1.0
    risk_measure: str = "neutral"
    risk_param: float = 0.0


class TabularQRAgent:
    """
    Tabular Quantile Regression agent for bandits.
    
    Each quantile θ_i(a) is updated directly via:
      δ = reward - θ_i(a)
      θ_i(a) ← θ_i(a) + α * [(1-τ_i)*𝟙(δ≥0) - τ_i*𝟙(δ<0)] * ρ'(δ)
    
    With Huber smoothing (κ > 0), ρ'(δ) = clip(δ, -κ, κ).
    """
    
    def __init__(self, config: QRDQNConfig):
        self.config = config
        self.n_arms = config.n_arms
        self.n_quantiles = config.n_quantiles
        self.lr = config.lr
        self.epsilon = config.epsilon
        self.step_count = 0
        
        # τ_i = (i + 0.5) / N
        self.taus = (torch.arange(config.n_quantiles).float() + 0.5) / config.n_quantiles
        
        # Quantile estimates: (n_arms, n_quantiles)
        self.theta = torch.zeros(config.n_arms, config.n_quantiles)
        
        self.theta_history: List[torch.Tensor] = []
        self.action_counts = torch.zeros(config.n_arms)
    
    def _risk_value(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Compute value from quantiles based on risk measure. Input: (..., n_quantiles)."""
        if self.config.risk_measure == "neutral":
            return quantiles.mean(dim=-1)
        sorted_q, _ = quantiles.sort(dim=-1)
        alpha = max(self.config.risk_param, 0.1)
        if self.config.risk_measure == "averse":
            n = max(1, int(self.n_quantiles * alpha))
            return sorted_q[..., :n].mean(dim=-1)
        elif self.config.risk_measure == "seeking":
            n = max(1, int(self.n_quantiles * alpha))
            return sorted_q[..., -n:].mean(dim=-1)
        return quantiles.mean(dim=-1)
    
    def get_action_values(self) -> torch.Tensor:
        return self._risk_value(self.theta)
    
    def select_action(self, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return self.get_action_values().argmax().item()
    
    def update(self, action: int, reward: float) -> float:
        delta = reward - self.theta[action]
        
        kappa = self.config.kappa
        if kappa > 0:
            grad = delta.clamp(-kappa, kappa)
        else:
            grad = delta.sign()
        
        weight = torch.where(delta >= 0, 1.0 - self.taus, self.taus)
        
        self.theta[action] += self.lr * weight * grad
        
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        self.step_count += 1
        self.action_counts[action] += 1
        
        if self.step_count % 50 == 0:
            self.theta_history.append(self.theta.clone())
        
        return delta.mean().item()
    
    def get_learned_distribution(self, action: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sorted_theta, _ = self.theta[action].sort()
        return self.taus.clone(), sorted_theta.clone()
    
    def get_all_distributions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sorted_theta, _ = self.theta.sort(dim=1)
        return self.taus.clone(), sorted_theta.clone()
    
    def get_asymmetry_factors(self) -> torch.Tensor:
        return (1.0 - self.taus) / self.taus
    
    def reset(self):
        self.theta = torch.zeros(self.n_arms, self.n_quantiles)
        self.epsilon = self.config.epsilon
        self.step_count = 0
        self.theta_history = []
        self.action_counts = torch.zeros(self.n_arms)
    
    def __repr__(self) -> str:
        values = self.get_action_values()
        v_str = ", ".join(f"{v:.3f}" for v in values)
        return f"TabularQR(V=[{v_str}], ε={self.epsilon:.3f}, N={self.n_quantiles}, steps={self.step_count})"
