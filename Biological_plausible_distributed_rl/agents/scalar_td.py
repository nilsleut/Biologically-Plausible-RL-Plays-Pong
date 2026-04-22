"""
Scalar TD Agent for Multi-Armed Bandits.

This is the baseline: a standard value-estimation agent that learns
E[R|a] for each action. By design, it cannot distinguish between
arms with equal mean but different variance — it's risk-neutral.

This limitation is exactly what motivates distributional RL.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class ScalarTDConfig:
    """Configuration for the scalar TD agent."""
    n_arms: int = 4
    lr: float = 0.1                 # Learning rate (step-size α)
    epsilon: float = 0.1            # ε-greedy exploration
    epsilon_decay: float = 0.999    # Decay per step
    epsilon_min: float = 0.01
    init_value: float = 0.0         # Initial Q-values (optimistic init if > 0)


class ScalarTDAgent:
    """
    Tabular scalar TD agent for bandits.
    
    Updates: Q(a) ← Q(a) + α * (r - Q(a))
    
    This is the simplest possible RL agent — it tracks a single
    scalar estimate of expected value per action.
    """
    
    def __init__(self, config: ScalarTDConfig):
        self.config = config
        self.q_values = torch.full((config.n_arms,), config.init_value)
        self.epsilon = config.epsilon
        self.step_count = 0
        
        # Logging
        self.q_history: List[torch.Tensor] = []
        self.td_errors: List[float] = []
    
    def select_action(self, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.config.n_arms)
        return self.q_values.argmax().item()
    
    def update(self, action: int, reward: float) -> float:
        """
        Standard TD update: Q(a) ← Q(a) + α * δ
        where δ = r - Q(a) is the TD error.
        
        Returns the TD error for logging.
        """
        td_error = reward - self.q_values[action].item()
        self.q_values[action] += self.config.lr * td_error
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )
        self.step_count += 1
        
        # Log
        self.q_history.append(self.q_values.clone())
        self.td_errors.append(td_error)
        
        return td_error
    
    def get_q_history_tensor(self) -> torch.Tensor:
        """Returns shape (T, n_arms) tensor of Q-value evolution."""
        if not self.q_history:
            return torch.empty(0, self.config.n_arms)
        return torch.stack(self.q_history)
    
    def reset(self):
        """Reset agent to initial state."""
        self.q_values = torch.full(
            (self.config.n_arms,), self.config.init_value
        )
        self.epsilon = self.config.epsilon
        self.step_count = 0
        self.q_history = []
        self.td_errors = []
    
    def __repr__(self) -> str:
        q_str = ", ".join(f"{q:.3f}" for q in self.q_values)
        return f"ScalarTD(Q=[{q_str}], ε={self.epsilon:.3f}, steps={self.step_count})"


class ScalarTDNetwork(nn.Module):
    """
    Neural network version of scalar TD for comparison.
    Maps state → Q-values. For bandits, the 'state' is constant,
    so this is equivalent to the tabular version but uses gradient descent.
    
    Included for architectural parity with the QR-DQN agent.
    """
    
    def __init__(self, n_arms: int, hidden_dim: int = 64):
        super().__init__()
        self.n_arms = n_arms
        # State input: just a constant 1 for bandits
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_arms),
        )
    
    def forward(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns Q-values for all arms. Shape: (batch, n_arms)."""
        if state is None:
            state = torch.ones(1, 1)
        return self.net(state)


class ScalarTDNetworkAgent:
    """
    Neural network scalar TD agent, for comparison with QR-DQN.
    Uses MSE loss on TD error.
    """
    
    def __init__(
        self,
        n_arms: int = 4,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
    ):
        self.n_arms = n_arms
        self.network = ScalarTDNetwork(n_arms, hidden_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.step_count = 0
        
        self.q_history: List[torch.Tensor] = []
        self.losses: List[float] = []
    
    def select_action(self, greedy: bool = False) -> int:
        with torch.no_grad():
            q = self.network().squeeze(0)
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return q.argmax().item()
    
    def update(self, action: int, reward: float) -> float:
        q_values = self.network().squeeze(0)
        target = reward
        loss = (q_values[action] - target) ** 2
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1
        
        with torch.no_grad():
            self.q_history.append(self.network().squeeze(0).clone())
        self.losses.append(loss.item())
        
        return (reward - q_values[action].item())
    
    def get_q_history_tensor(self) -> torch.Tensor:
        if not self.q_history:
            return torch.empty(0, self.n_arms)
        return torch.stack(self.q_history)
    
    def reset(self):
        for layer in self.network.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.epsilon = 0.1
        self.step_count = 0
        self.q_history = []
        self.losses = []
