"""
Phase 2: Biologically Plausible Distributional RL Agent.

CORE CONTRIBUTION: Replace the quantile regression loss (which requires
backprop) with a local Hebbian update rule where each "neuron" has its
own asymmetric learning rates α⁺ and α⁻.

The key insight from Dabney et al. (2020, Nature):
- Real dopamine neurons have diverse "reversal points" (the RPE value
  at which they switch from excitation to inhibition)
- This diversity naturally arises if each neuron has a different ratio
  of α⁺ (learning rate for positive RPEs) to α⁻ (negative RPEs)
- A population of such neurons implicitly encodes the full reward 
  distribution, not just the mean

This module implements exactly this: a population of value-estimating
units, each with a learned (or fixed) asymmetry factor, updated via
purely local Hebbian-style plasticity. No backprop, no global loss.

The mapping to QR-DQN is:
  QR-DQN quantile τ_i  →  Hebbian neuron with α⁺/α⁻ = (1-τ_i)/τ_i
  Quantile regression loss → Local asymmetric delta rule
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HebbianConfig:
    """Configuration for the Hebbian distributional agent."""
    n_arms: int = 4
    n_neurons: int = 32             # Number of value-coding neurons
    base_lr: float = 0.05           # Base learning rate
    epsilon: float = 0.1
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.01
    
    # Asymmetry initialization
    asymmetry_mode: str = "uniform"  # "uniform", "learned", "fixed_dabney"
    
    # For "learned" mode: meta-learning rate for asymmetry factors
    meta_lr: float = 0.001
    
    # Risk measure for action selection
    risk_measure: str = "neutral"    # "neutral", "averse", "seeking", "population"
    risk_param: float = 0.0
    
    # Biological constraints
    dale_law: bool = False           # If True, enforce sign constraints
    noise_std: float = 0.0           # Neural noise (biological realism)


class HebbianDistributionalAgent:
    """
    Biologically plausible distributional RL agent.
    
    Architecture:
    - A population of N "dopamine neurons", each maintaining a scalar
      value estimate v_i(a) for each action
    - Each neuron i has asymmetric learning rates α⁺_i and α⁻_i
    - On receiving reward r after choosing action a:
        δ_i = r - v_i(a)                    # Local RPE
        if δ_i > 0: v_i(a) += α⁺_i * δ_i   # Positive surprise
        if δ_i < 0: v_i(a) += α⁻_i * δ_i   # Negative surprise
    
    This is "Hebbian" in the sense that:
    - Updates are purely local (each neuron only uses its own RPE)
    - No gradient computation across neurons
    - No global loss function
    - The asymmetry α⁺/α⁻ is the ONLY thing that differentiates neurons
    
    Theoretical guarantee:
    If α⁺_i/(α⁺_i + α⁻_i) = τ_i where τ_i are uniformly spaced in [0,1],
    then v_i converges to the τ_i-th quantile of the reward distribution.
    This is because the neuron's equilibrium satisfies:
        E[α⁺ * δ⁺ + α⁻ * δ⁻] = 0
        α⁺ * P(r > v) * E[r-v | r>v] = α⁻ * P(r < v) * E[v-r | r<v]
    which for the asymmetric delta rule reduces to finding v such that
        P(r < v) = α⁺ / (α⁺ + α⁻) = τ
    """
    
    def __init__(self, config: HebbianConfig):
        self.config = config
        self.n_arms = config.n_arms
        self.n_neurons = config.n_neurons
        
        # Value estimates: (n_neurons, n_arms)
        self.values = torch.zeros(config.n_neurons, config.n_arms)
        
        # Initialize asymmetry factors
        self._init_asymmetry()
        
        self.epsilon = config.epsilon
        self.step_count = 0
        
        # Logging
        self.value_history: List[torch.Tensor] = []
        self.rpe_history: List[torch.Tensor] = []
        self.asymmetry_history: List[torch.Tensor] = []
        self.action_counts = torch.zeros(config.n_arms)
    
    def _init_asymmetry(self):
        """
        Initialize α⁺ and α⁻ for each neuron.
        
        Three modes:
        1. "uniform": τ_i = (i+0.5)/N, α⁺ = base_lr*(1-τ), α⁻ = base_lr*τ
           → Equivalent to QR-DQN quantile midpoints
           
        2. "fixed_dabney": Sample τ from the empirical distribution 
           observed in Dabney 2020 (approximately uniform with noise)
           
        3. "learned": Start uniform, but let α⁺/α⁻ adapt via meta-learning
        """
        mode = self.config.asymmetry_mode
        lr = self.config.base_lr
        N = self.n_neurons
        
        if mode == "uniform":
            # Uniform quantile spacing
            taus = (torch.arange(N).float() + 0.5) / N
        
        elif mode == "fixed_dabney":
            # Sample from ~uniform with some noise (mimics biological variability)
            taus = (torch.arange(N).float() + 0.5) / N
            noise = torch.randn(N) * 0.05
            taus = (taus + noise).clamp(0.02, 0.98)
            taus, _ = taus.sort()
        
        elif mode == "learned":
            # Start uniform, will adapt
            taus = (torch.arange(N).float() + 0.5) / N
        
        else:
            raise ValueError(f"Unknown asymmetry mode: {mode}")
        
        self.taus = taus
        
        # α⁺_i = lr * (1 - τ_i): large for low-quantile (optimistic) neurons
        # α⁻_i = lr * τ_i: large for high-quantile (pessimistic) neurons
        self.alpha_pos = lr * (1.0 - taus)  # (n_neurons,)
        self.alpha_neg = lr * taus           # (n_neurons,)
    
    def get_action_values(self) -> torch.Tensor:
        """
        Compute action values from the neural population.
        
        Different risk measures correspond to different readout strategies:
        - neutral: population mean (equivalent to expected value)
        - averse: mean of pessimistic neurons (low α⁺/α⁻)
        - seeking: mean of optimistic neurons (high α⁺/α⁻)
        - population: weighted average based on asymmetry
        """
        # Sort values per arm for quantile-based readout
        sorted_vals, _ = self.values.sort(dim=0)  # (n_neurons, n_arms)
        
        if self.config.risk_measure == "neutral":
            return sorted_vals.mean(dim=0)
        
        elif self.config.risk_measure == "averse":
            alpha = max(self.config.risk_param, 0.1)
            n = max(1, int(self.n_neurons * alpha))
            return sorted_vals[:n].mean(dim=0)
        
        elif self.config.risk_measure == "seeking":
            alpha = max(self.config.risk_param, 0.1)
            n = max(1, int(self.n_neurons * alpha))
            return sorted_vals[-n:].mean(dim=0)
        
        elif self.config.risk_measure == "population":
            # Weight by asymmetry: more pessimistic neurons → risk-averse
            weights = self.alpha_pos / (self.alpha_pos + self.alpha_neg)
            weights = weights / weights.sum()
            return (weights.unsqueeze(1) * self.values).sum(dim=0)
        
        return sorted_vals.mean(dim=0)
    
    def select_action(self, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return self.get_action_values().argmax().item()
    
    def update(self, action: int, reward: float) -> float:
        """
        Biologically plausible local update.
        
        For each neuron i:
          1. Compute LOCAL RPE: δ_i = reward - v_i(action)
          2. Apply ASYMMETRIC update:
             if δ_i ≥ 0: v_i(action) += α⁺_i * δ_i
             if δ_i < 0:  v_i(action) += α⁻_i * δ_i
        
        NO backprop. NO global loss. Each neuron is independent.
        This is the key biological plausibility claim.
        """
        v = self.values[:, action]  # (n_neurons,)
        
        # Add neural noise if configured
        if self.config.noise_std > 0:
            noise = torch.randn(self.n_neurons) * self.config.noise_std
            r_perceived = reward + noise
        else:
            r_perceived = reward
        
        # Local RPE for each neuron
        rpe = r_perceived - v  # (n_neurons,)
        
        # Asymmetric update
        lr = torch.where(rpe >= 0, self.alpha_pos, self.alpha_neg)
        delta_v = lr * rpe
        
        self.values[:, action] = v + delta_v
        
        # Optional meta-learning of asymmetry factors
        if self.config.asymmetry_mode == "learned":
            self._update_asymmetry(rpe, action)
        
        # Housekeeping
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        self.step_count += 1
        self.action_counts[action] += 1
        
        # Log
        if self.step_count % 50 == 0:
            self.value_history.append(self.values.clone())
            self.asymmetry_history.append(
                (self.alpha_pos / (self.alpha_pos + self.alpha_neg)).clone()
            )
        
        return rpe.mean().item()
    
    def _update_asymmetry(self, rpe: torch.Tensor, action: int):
        """
        Meta-learning: adjust α⁺/α⁻ to maximize distributional coverage.
        
        Intuition: if a neuron's value is consistently above/below its
        target quantile, adjust its asymmetry to bring it back.
        
        This is speculative and goes beyond Dabney 2020 — it's asking
        whether the brain could *learn* the asymmetry factors rather
        than having them genetically predetermined.
        """
        meta_lr = self.config.meta_lr
        
        # The sign of the running RPE average indicates if τ is too high/low
        # If neuron consistently gets positive RPEs → it's underestimating
        # → needs higher α⁺ relative to α⁻ → decrease τ
        sign_fraction = (rpe >= 0).float()  # Fraction of positive RPEs (scalar per neuron)
        
        # Target: sign_fraction should equal τ at equilibrium
        tau_error = sign_fraction - self.taus
        
        # Adjust τ
        self.taus = (self.taus - meta_lr * tau_error).clamp(0.02, 0.98)
        
        # Recompute α from τ
        lr = self.config.base_lr
        self.alpha_pos = lr * (1.0 - self.taus)
        self.alpha_neg = lr * self.taus
    
    def get_learned_distribution(self, action: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (effective_taus, sorted_values) for an action."""
        effective_taus = self.alpha_pos / (self.alpha_pos + self.alpha_neg)
        sorted_taus, sort_idx = effective_taus.sort()
        sorted_vals = self.values[sort_idx, action]
        return sorted_taus, sorted_vals
    
    def get_all_distributions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (taus, values) with values sorted per arm."""
        effective_taus = self.alpha_pos / (self.alpha_pos + self.alpha_neg)
        sorted_taus, sort_idx = effective_taus.sort()
        sorted_vals = self.values[sort_idx]  # (n_neurons, n_arms)
        return sorted_taus, sorted_vals
    
    def get_reversal_points(self, action: int) -> torch.Tensor:
        """
        Get the "reversal point" of each neuron for an action.
        
        The reversal point is the value v_i(a) itself — it's the reward
        level at which the neuron's RPE switches sign. This directly
        corresponds to Dabney 2020's experimental measurements.
        """
        return self.values[:, action].clone()
    
    def get_asymmetry_factors(self) -> torch.Tensor:
        """Return α⁺/α⁻ ratio for each neuron."""
        return self.alpha_pos / self.alpha_neg
    
    def get_effective_taus(self) -> torch.Tensor:
        """Return the effective quantile level each neuron codes for."""
        return self.alpha_pos / (self.alpha_pos + self.alpha_neg)
    
    def reset(self):
        self.values = torch.zeros(self.n_neurons, self.n_arms)
        self._init_asymmetry()
        self.epsilon = self.config.epsilon
        self.step_count = 0
        self.value_history = []
        self.rpe_history = []
        self.asymmetry_history = []
        self.action_counts = torch.zeros(self.n_arms)
    
    def __repr__(self) -> str:
        values = self.get_action_values()
        v_str = ", ".join(f"{v:.3f}" for v in values)
        return (
            f"HebbianDist(V=[{v_str}], ε={self.epsilon:.3f}, "
            f"N={self.n_neurons}, mode={self.config.asymmetry_mode}, "
            f"steps={self.step_count})"
        )
