"""
Stateful Agent Wrappers for Phase 3.

These wrappers adapt the bandit agents to stateful environments by:
1. Discretizing the state (energy level) into bins
2. Maintaining separate value estimates per state bin
3. Optionally adapting the risk measure based on state

The key experiment: HebbianStateful with adaptive risk readout
should learn state-dependent risk preferences (risk-averse when
energy is low, risk-neutral when high) — something ScalarStateful
fundamentally cannot do.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from agents.scalar_td import ScalarTDConfig
from agents.qr_dqn import QRDQNConfig
from agents.hebbian_distributional import HebbianConfig


class StatefulScalarTD:
    """
    Scalar TD agent with state-dependent Q-values.
    
    Discretizes energy into bins, maintains Q(state_bin, action).
    This is the best a scalar agent can do — it can learn different
    action preferences per state, but still only tracks means.
    """
    
    def __init__(self, n_actions: int, n_bins: int = 5, lr: float = 0.1,
                 epsilon: float = 0.15, epsilon_decay: float = 0.9998,
                 epsilon_min: float = 0.02):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: (n_bins, n_actions)
        self.q_table = torch.zeros(n_bins, n_actions)
        self.visit_counts = torch.zeros(n_bins, n_actions)
        self.step_count = 0
    
    def _discretize(self, state: torch.Tensor) -> int:
        """Map normalized energy [0,1] to bin index."""
        energy_frac = state[0].item()
        bin_idx = int(energy_frac * self.n_bins)
        return min(bin_idx, self.n_bins - 1)
    
    def select_action(self, state: torch.Tensor, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        bin_idx = self._discretize(state)
        return self.q_table[bin_idx].argmax().item()
    
    def update(self, state: torch.Tensor, action: int, reward: float):
        bin_idx = self._discretize(state)
        td_error = reward - self.q_table[bin_idx, action].item()
        self.q_table[bin_idx, action] += self.lr * td_error
        self.visit_counts[bin_idx, action] += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1
    
    def get_policy_map(self) -> torch.Tensor:
        """Return preferred action per state bin."""
        return self.q_table.argmax(dim=1)


class StatefulQR:
    """
    Tabular QR agent with state-dependent quantiles.
    Maintains full quantile estimates per (state_bin, action).
    """
    
    def __init__(self, n_actions: int, n_bins: int = 5, n_quantiles: int = 32,
                 lr: float = 0.05, epsilon: float = 0.15,
                 epsilon_decay: float = 0.9998, epsilon_min: float = 0.02,
                 risk_measure: str = "neutral", risk_param: float = 0.25):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.n_quantiles = n_quantiles
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.risk_measure = risk_measure
        self.risk_param = risk_param
        
        self.taus = (torch.arange(n_quantiles).float() + 0.5) / n_quantiles
        
        # Quantile table: (n_bins, n_actions, n_quantiles)
        self.theta = torch.zeros(n_bins, n_actions, n_quantiles)
        self.step_count = 0
    
    def _discretize(self, state: torch.Tensor) -> int:
        energy_frac = state[0].item()
        return min(int(energy_frac * self.n_bins), self.n_bins - 1)
    
    def _action_values(self, bin_idx: int) -> torch.Tensor:
        q = self.theta[bin_idx]  # (n_actions, n_quantiles)
        sorted_q, _ = q.sort(dim=1)
        
        if self.risk_measure == "neutral":
            return sorted_q.mean(dim=1)
        elif self.risk_measure == "averse":
            n = max(1, int(self.n_quantiles * self.risk_param))
            return sorted_q[:, :n].mean(dim=1)
        elif self.risk_measure == "seeking":
            n = max(1, int(self.n_quantiles * self.risk_param))
            return sorted_q[:, -n:].mean(dim=1)
        return sorted_q.mean(dim=1)
    
    def select_action(self, state: torch.Tensor, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        bin_idx = self._discretize(state)
        return self._action_values(bin_idx).argmax().item()
    
    def update(self, state: torch.Tensor, action: int, reward: float):
        bin_idx = self._discretize(state)
        theta_sa = self.theta[bin_idx, action]
        delta = reward - theta_sa
        kappa = 1.0
        grad = delta.clamp(-kappa, kappa)
        weight = torch.where(delta >= 0, 1.0 - self.taus, self.taus)
        self.theta[bin_idx, action] = theta_sa + self.lr * weight * grad
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1


class StatefulHebbian:
    """
    Hebbian distributional agent with state-dependent values
    AND adaptive risk readout.
    
    THIS IS THE KEY AGENT: it uses the energy level to modulate
    its risk readout from the neural population.
    
    Low energy → read out CVaR (bottom quantiles) → risk-averse
    High energy → read out mean (all quantiles) → risk-neutral
    
    This state-dependent readout is the biological hypothesis:
    the downstream circuit (e.g., striatum/cortex) adjusts HOW
    it reads the distributional code from dopamine neurons based
    on the animal's current need state.
    """
    
    def __init__(self, n_actions: int, n_bins: int = 5, n_neurons: int = 32,
                 base_lr: float = 0.05, epsilon: float = 0.15,
                 epsilon_decay: float = 0.9998, epsilon_min: float = 0.02,
                 adaptive_risk: bool = True,
                 cvar_alpha_low: float = 0.2,    # CVaR level at low energy
                 cvar_alpha_high: float = 1.0):   # 1.0 = full mean (neutral)
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.n_neurons = n_neurons
        self.base_lr = base_lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.adaptive_risk = adaptive_risk
        self.cvar_alpha_low = cvar_alpha_low
        self.cvar_alpha_high = cvar_alpha_high
        
        # Fixed uniform τ for asymmetric learning
        self.taus = (torch.arange(n_neurons).float() + 0.5) / n_neurons
        self.alpha_pos = base_lr * (1.0 - self.taus)
        self.alpha_neg = base_lr * self.taus
        
        # Values: (n_bins, n_neurons, n_actions)
        self.values = torch.zeros(n_bins, n_neurons, n_actions)
        self.step_count = 0
    
    def _discretize(self, state: torch.Tensor) -> int:
        energy_frac = state[0].item()
        return min(int(energy_frac * self.n_bins), self.n_bins - 1)
    
    def _get_cvar_alpha(self, state: torch.Tensor) -> float:
        """
        Compute the CVaR level based on energy.
        
        Low energy → small α (focus on worst-case quantiles)
        High energy → α=1.0 (use all quantiles = risk-neutral mean)
        
        Linear interpolation between cvar_alpha_low and cvar_alpha_high.
        """
        if not self.adaptive_risk:
            return 1.0  # Always risk-neutral
        
        energy_frac = state[0].item()
        # Interpolate: low energy → cvar_alpha_low, high → cvar_alpha_high
        alpha = self.cvar_alpha_low + energy_frac * (self.cvar_alpha_high - self.cvar_alpha_low)
        return max(0.1, min(1.0, alpha))
    
    def _action_values(self, bin_idx: int, cvar_alpha: float) -> torch.Tensor:
        """Read out action values with given CVaR level."""
        v = self.values[bin_idx]  # (n_neurons, n_actions)
        sorted_v, _ = v.sort(dim=0)
        
        if cvar_alpha >= 0.99:
            return sorted_v.mean(dim=0)
        
        n = max(1, int(self.n_neurons * cvar_alpha))
        return sorted_v[:n].mean(dim=0)  # Bottom quantiles = pessimistic
    
    def select_action(self, state: torch.Tensor, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        bin_idx = self._discretize(state)
        cvar_alpha = self._get_cvar_alpha(state)
        return self._action_values(bin_idx, cvar_alpha).argmax().item()
    
    def update(self, state: torch.Tensor, action: int, reward: float):
        bin_idx = self._discretize(state)
        v = self.values[bin_idx, :, action]  # (n_neurons,)
        
        rpe = reward - v
        lr = torch.where(rpe >= 0, self.alpha_pos, self.alpha_neg)
        self.values[bin_idx, :, action] = v + lr * rpe
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1
    
    def get_risk_profile(self, state: torch.Tensor) -> dict:
        """Diagnostic: return current risk readout for a state."""
        cvar_alpha = self._get_cvar_alpha(state)
        bin_idx = self._discretize(state)
        neutral_vals = self._action_values(bin_idx, 1.0)
        cvar_vals = self._action_values(bin_idx, cvar_alpha)
        return {
            "energy_frac": state[0].item(),
            "cvar_alpha": cvar_alpha,
            "neutral_values": neutral_vals.numpy(),
            "cvar_values": cvar_vals.numpy(),
            "preferred_action_neutral": neutral_vals.argmax().item(),
            "preferred_action_cvar": cvar_vals.argmax().item(),
        }
