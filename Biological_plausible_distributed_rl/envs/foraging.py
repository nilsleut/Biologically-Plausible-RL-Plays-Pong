"""
Phase 3: Stateful Foraging Environment with Energy Budget.

The agent forages across patches with different risk profiles.
Each timestep costs energy. If energy hits 0, the agent dies.

KEY DESIGN: When energy is LOW, risk-aversion is rational (avoid
starvation). When energy is HIGH, risk-neutrality or even risk-seeking
is rational (maximize long-term yield). A distributional agent with
CVaR readout can implement this state-dependent risk preference
automatically — scalar TD cannot.

This is the "so what" experiment: it shows that distributional
value coding isn't just a representational curiosity but enables
adaptive behavior that scalar methods fundamentally cannot produce.

Inspired by:
- Optimal foraging theory (Stephens & Krebs 1986)
- Risk-sensitive foraging in birds/rodents (Caraco et al. 1980)
- Dabney et al. 2020's hypothesis about distributional DA coding
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass 
class ForagingConfig:
    """Configuration for the foraging environment."""
    n_patches: int = 3              # Number of foraging patches (actions)
    max_energy: float = 20.0        # Energy capacity
    start_energy: float = 10.0      # Starting energy
    energy_cost: float = 0.5        # Energy cost per timestep
    max_steps: int = 200            # Episode length
    death_threshold: float = 0.0    # Die if energy <= this
    
    # Patch reward distributions (all same mean, different variance)
    patch_means: Tuple = (2.0, 2.0, 2.0)
    patch_stds: Tuple = (0.2, 1.0, 3.0)
    patch_names: Tuple = ("Safe berries", "Variable fruit", "Risky prey")


class ForagingEnv:
    """
    Stateful foraging environment.
    
    State: current energy level (normalized to [0, 1])
    Actions: choose a patch to forage
    Reward: food gained from patch (stochastic)
    Dynamics: energy += reward - cost, clipped to [0, max_energy]
    Terminal: energy <= 0 (death) or step >= max_steps (survived)
    
    The optimal policy depends on current energy:
    - Low energy → pick safe patch (avoid P(reward < cost) scenarios)
    - High energy → indifferent or risk-seeking (safe from starvation)
    
    This state-dependent risk preference is the key phenomenon.
    """
    
    def __init__(self, config: Optional[ForagingConfig] = None):
        self.config = config or ForagingConfig()
        self.energy = self.config.start_energy
        self.step_count = 0
        self.alive = True
        self.total_reward = 0.0
        
        # Episode logging
        self.energy_history: List[float] = [self.energy]
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
    
    def get_state(self) -> torch.Tensor:
        """Return normalized state vector: [energy/max_energy]."""
        return torch.tensor([self.energy / self.config.max_energy])
    
    def get_energy_level(self) -> str:
        """Categorize current energy as low/medium/high."""
        frac = self.energy / self.config.max_energy
        if frac < 0.25:
            return "low"
        elif frac < 0.6:
            return "medium"
        else:
            return "high"
    
    def step(self, action: int) -> Tuple[float, bool, dict]:
        """
        Take an action, return (reward, done, info).
        
        Reward is the food gained from the chosen patch.
        Energy is updated: energy += food - cost.
        """
        assert 0 <= action < self.config.n_patches
        assert self.alive, "Episode already terminated"
        
        # Sample food from chosen patch
        mean = self.config.patch_means[action]
        std = self.config.patch_stds[action]
        food = max(0.0, np.random.normal(mean, std))  # Food can't be negative
        
        # Update energy
        self.energy = min(
            self.config.max_energy,
            self.energy + food - self.config.energy_cost
        )
        
        self.step_count += 1
        self.total_reward += food
        
        # Check termination
        done = False
        info = {"cause": None, "energy": self.energy, "food": food}
        
        if self.energy <= self.config.death_threshold:
            self.alive = False
            done = True
            info["cause"] = "death"
        elif self.step_count >= self.config.max_steps:
            done = True
            info["cause"] = "survived"
        
        # Log
        self.energy_history.append(self.energy)
        self.action_history.append(action)
        self.reward_history.append(food)
        
        return food, done, info
    
    def reset(self) -> torch.Tensor:
        """Reset episode, return initial state."""
        self.energy = self.config.start_energy
        self.step_count = 0
        self.alive = True
        self.total_reward = 0.0
        self.energy_history = [self.energy]
        self.action_history = []
        self.reward_history = []
        return self.get_state()
    
    def __repr__(self) -> str:
        lines = [f"ForagingEnv(energy={self.energy:.1f}/{self.config.max_energy})"]
        for i in range(self.config.n_patches):
            lines.append(
                f"  [{i}] {self.config.patch_names[i]}: "
                f"μ={self.config.patch_means[i]:.1f}, σ={self.config.patch_stds[i]:.1f}"
            )
        return "\n".join(lines)

    # --- Factory methods ---
    
    @classmethod
    def easy(cls) -> "ForagingEnv":
        """High start energy, low cost — survival is easy."""
        return cls(ForagingConfig(
            start_energy=15.0, energy_cost=0.3, max_steps=200,
        ))
    
    @classmethod
    def hard(cls) -> "ForagingEnv":
        """Low start energy, high cost — survival pressure."""
        return cls(ForagingConfig(
            start_energy=5.0, energy_cost=0.8, max_steps=200,
        ))
    
    @classmethod
    def variable_pressure(cls) -> "ForagingEnv":
        """
        Medium start, moderate cost — the interesting regime where
        the agent regularly transitions between low/high energy states
        and should adapt its risk preference accordingly.
        """
        return cls(ForagingConfig(
            start_energy=8.0, energy_cost=0.6, max_steps=300,
            max_energy=20.0,
        ))


@dataclass
class GamblingConfig:
    """Configuration for the gambling paradigm."""
    n_trials: int = 200
    # Option A: certain reward
    certain_reward: float = 1.0
    # Option B: lottery with parametric risk
    lottery_high: float = 2.5
    lottery_low: float = 0.0
    lottery_p_high: float = 0.5  # P(high outcome)


class GamblingEnv:
    """
    Binary choice gambling paradigm (analogous to rodent experiments).
    
    Each trial: choose between
      A) Certain reward (e.g., always 1.0)
      B) Risky lottery (e.g., 2.5 with p=0.5, else 0.0)
    
    Both have the same expected value (1.0 vs 0.5*2.5 = 1.25... 
    actually we can tune this). The key measure is what fraction
    of the time the agent chooses the risky option.
    
    With equal EVs, a risk-neutral agent should be indifferent,
    a risk-averse agent should prefer certain, and a risk-seeking
    agent should prefer the lottery.
    
    Can also be run with parametric variation of lottery_p_high
    to generate psychometric curves.
    """
    
    def __init__(self, config: Optional[GamblingConfig] = None):
        self.config = config or GamblingConfig()
        self.trial = 0
    
    def step(self, action: int) -> Tuple[float, bool]:
        """action=0: certain, action=1: lottery. Returns (reward, done)."""
        assert action in (0, 1)
        
        if action == 0:
            reward = self.config.certain_reward
        else:
            if np.random.random() < self.config.lottery_p_high:
                reward = self.config.lottery_high
            else:
                reward = self.config.lottery_low
        
        self.trial += 1
        done = self.trial >= self.config.n_trials
        return reward, done
    
    def reset(self):
        self.trial = 0
    
    @classmethod
    def equal_ev(cls, certain: float = 1.0) -> "GamblingEnv":
        """Both options have expected value = certain."""
        return cls(GamblingConfig(
            certain_reward=certain,
            lottery_high=2 * certain,
            lottery_low=0.0,
            lottery_p_high=0.5,
        ))
    
    @classmethod
    def parametric_sweep(cls, certain: float = 1.0, p_values: Optional[List[float]] = None):
        """Create a list of envs with varying lottery probabilities."""
        if p_values is None:
            p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        envs = []
        for p in p_values:
            # Keep EV equal: certain = p * high + (1-p) * 0 → high = certain/p
            high = certain / p if p > 0 else 100.0
            envs.append(cls(GamblingConfig(
                certain_reward=certain,
                lottery_high=high,
                lottery_low=0.0,
                lottery_p_high=p,
            )))
        return envs, p_values
