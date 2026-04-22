"""
Fully Biologically Plausible Pong Agent.

PC Encoder (local learning) → features → Hebbian Distributional Value (local learning)

ZERO backpropagation in the entire pipeline.

This is the core contribution: a complete RL system that learns
to play Pong using only biologically plausible learning rules:
1. Predictive Coding for feature learning (predict next observation)
2. Asymmetric Hebbian plasticity for distributional value learning
3. Population readout for action selection

Connection to neuroscience:
- PC encoder ≈ visual cortex learning via prediction errors
- Hebbian value neurons ≈ dopamine neurons with diverse response profiles
- Action selection via population readout ≈ basal ganglia
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from agents.pc_encoder import PCEncoder, PCEncoderConfig


@dataclass
class BioAgentConfig:
    obs_dim: int = 6
    n_actions: int = 3
    
    # PC Encoder
    pc_hidden_dim: int = 64
    pc_feature_dim: int = 64         # Increased from 32 → more capacity
    pc_lr_encoder: float = 0.001
    pc_lr_predictor: float = 0.001
    pc_weight_decay: float = 0.0001
    pc_freeze_after: int = 0         # Freeze PC after N steps (0 = never freeze)
    
    # Also include engineered features alongside PC features
    use_engineered: bool = True    # Combine PC + engineered
    
    # Value population
    n_neurons: int = 32
    base_lr: float = 0.005
    gamma: float = 0.99
    
    # Action selection
    temperature: float = 0.5
    epsilon: float = 0.2
    epsilon_decay: float = 0.99998
    epsilon_min: float = 0.03
    
    # Eligibility traces
    trace_decay: float = 0.7
    use_traces: bool = True
    
    # Training
    total_timesteps: int = 500_000
    log_interval: int = 2000
    
    # PC warmup: train encoder for N steps before value learning starts
    pc_warmup_steps: int = 5000
    
    # Synaptic consolidation (bio-plausible EWC)
    # Protects important weights from being overwritten
    use_consolidation: bool = False
    consolidation_strength: float = 0.1   # How strongly to protect stable weights
    consolidation_ema: float = 0.999      # EMA decay for tracking weight stability


class BioAgent:
    """
    Fully biologically plausible Pong agent.
    
    Architecture:
      obs_t → PC Encoder → z_t (learned features)
           → Engineered features → f_t (optional)
           → concat [z_t, f_t] → combined features
           → Hebbian Value Population → v_i(a) per neuron per action
           → Population readout → action values
           → Softmax → action
    
    Learning:
      1. PC Encoder: trained by predicting obs_{t+1} from obs_t
         Update rule: local Hebbian (error * input)
      2. Value Population: trained by TD errors
         Update rule: asymmetric Hebbian (α⁺ for positive RPE, α⁻ for negative)
      
    Both updates are FULLY LOCAL. No backprop anywhere.
    """
    
    def __init__(self, config: BioAgentConfig):
        self.config = config
        c = config
        
        # --- PC Encoder ---
        pc_config = PCEncoderConfig(
            obs_dim=c.obs_dim,
            hidden_dim=c.pc_hidden_dim,
            feature_dim=c.pc_feature_dim,
            lr_encoder=c.pc_lr_encoder,
            lr_predictor=c.pc_lr_predictor,
            weight_decay=c.pc_weight_decay,
        )
        self.pc_encoder = PCEncoder(pc_config)
        
        # --- Engineered features (optional, for comparison) ---
        self.engineered_dim = 17 if c.use_engineered else 0
        
        # Total feature dimension
        self.feature_dim = c.pc_feature_dim + self.engineered_dim
        
        # --- Quantile levels for Hebbian value neurons ---
        self.taus = (torch.arange(c.n_neurons).float() + 0.5) / c.n_neurons
        self.alpha_pos = c.base_lr * (1.0 - self.taus)
        self.alpha_neg = c.base_lr * self.taus
        
        # --- Hebbian value weights ---
        # Shape: (n_actions, n_neurons, feature_dim)
        self.W_value = torch.zeros(c.n_actions, c.n_neurons, self.feature_dim)
        self.W_value += torch.randn_like(self.W_value) * 0.01
        
        # --- Eligibility traces ---
        if c.use_traces:
            self.traces = torch.zeros_like(self.W_value)
        
        # --- Synaptic consolidation ---
        # Tracks how "stable" each weight is via EMA of squared updates.
        # Stable weights (low variance) get protected from large changes.
        # Biological analogy: synaptic tagging & consolidation —
        # frequently-modified synapses remain plastic, stable ones
        # become resistant to change.
        if c.use_consolidation:
            # EMA of squared weight changes — high = volatile, low = stable
            self.weight_volatility = torch.zeros_like(self.W_value)
            # Anchor weights: snapshot of weights at their "best"
            self.weight_anchor = self.W_value.clone()
            self.best_eval_score = 0.0
        
        # --- State ---
        self.epsilon = c.epsilon
        self.step_count = 0
        self.prev_obs = None
        self.prev_features = None
        self.prev_action = None
        self.prev_values = None
        
        # --- Logging ---
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.log: List[dict] = []
    
    def _engineered_features(self, obs: np.ndarray) -> torch.Tensor:
        """Same engineered features as hebbian_pong.py."""
        bx, by, bvx, bvy, py, oy = obs
        bvx_raw = bvx / 20.0
        bvy_raw = bvy / 20.0
        
        features = []
        features.extend([bx, by, bvx, bvy, py, oy])
        
        ball_paddle_dy = by - py
        ball_paddle_dy_abs = abs(by - py)
        features.extend([ball_paddle_dy, ball_paddle_dy_abs])
        
        ball_approaching = 1.0 if bvx_raw > 0 else 0.0
        urgency = bx * bvx
        features.extend([ball_approaching, urgency])
        
        if bvx_raw > 0.001:
            time_to_reach = (0.9 - bx) / (bvx_raw + 1e-8)
            time_to_reach = min(time_to_reach, 10.0) / 10.0
        else:
            time_to_reach = 1.0
        features.append(time_to_reach)
        
        if bvx_raw > 0.001:
            steps_to_paddle = (0.9 - bx) / (bvx_raw + 1e-8)
            predicted_y = by + bvy_raw * steps_to_paddle
            predicted_y = np.clip(predicted_y, -1.0, 1.0)
            pred_error = predicted_y - py
        else:
            predicted_y = 0.0
            pred_error = -py
        features.extend([predicted_y, pred_error])
        
        features.append(ball_approaching * ball_paddle_dy)
        features.append(ball_approaching * pred_error)
        features.append((1.0 - ball_approaching) * py)
        features.append(1.0)  # bias
        
        return torch.FloatTensor(features)
    
    def _get_features(self, obs: np.ndarray) -> torch.Tensor:
        """
        Get combined feature vector: PC features + engineered features.
        """
        pc_features = self.pc_encoder.encode(obs)  # (pc_feature_dim,)
        
        if self.config.use_engineered:
            eng_features = self._engineered_features(obs)  # (17,)
            return torch.cat([pc_features, eng_features])
        
        return pc_features
    
    def _get_values(self, features: torch.Tensor) -> torch.Tensor:
        """Compute value for each (action, neuron). Returns (n_actions, n_neurons)."""
        return torch.einsum("aif,f->ai", self.W_value, features)
    
    def _action_values(self, neuron_values: torch.Tensor) -> torch.Tensor:
        """Population readout → action values."""
        sorted_v, _ = neuron_values.sort(dim=1)
        return sorted_v.mean(dim=1)
    
    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        features = self._get_features(obs)
        neuron_values = self._get_values(features)
        action_values = self._action_values(neuron_values)
        
        self.prev_features = features
        self.prev_values = neuron_values
        self.prev_obs = obs.copy()
        
        if not greedy and np.random.random() < self.epsilon:
            action = np.random.randint(self.config.n_actions)
        else:
            probs = F.softmax(action_values / self.config.temperature, dim=0)
            if greedy:
                action = probs.argmax().item()
            else:
                action = torch.multinomial(probs, 1).item()
        
        self.prev_action = action
        return action
    
    def update(self, obs: np.ndarray, reward: float, done: bool):
        """
        Update both PC encoder and Hebbian value weights.
        
        PC update: runs every step (predict next obs from prev obs)
        Value update: runs every step (TD error with asymmetric Hebbian)
        """
        c = self.config
        
        # === PC Encoder update ===
        # Learn to predict current obs from previous obs
        # Freeze after pc_freeze_after steps to stabilize features
        pc_frozen = (c.pc_freeze_after > 0 and self.step_count >= c.pc_freeze_after)
        if self.prev_obs is not None and not pc_frozen:
            self.pc_encoder.update(self.prev_obs, obs)
        
        # === Hebbian Value update ===
        if self.prev_features is not None and self.step_count >= c.pc_warmup_steps:
            features = self.prev_features
            action = self.prev_action
            old_values = self.prev_values[action]  # (n_neurons,)
            
            # Next state value
            if done:
                next_v = 0.0
            else:
                next_features = self._get_features(obs)
                next_neuron_values = self._get_values(next_features)
                next_v = next_neuron_values.mean(dim=0)
            
            # TD target
            if isinstance(next_v, float):
                target = reward
            else:
                target = reward + c.gamma * next_v
            
            # RPE per neuron
            rpe = target - old_values
            
            # Asymmetric learning rate
            lr = torch.where(rpe >= 0, self.alpha_pos, self.alpha_neg)
            
            # Hebbian update
            delta_w = torch.einsum("n,f->nf", lr * rpe, features)
            
            if c.use_traces:
                self.traces[action] = c.trace_decay * c.gamma * self.traces[action] + delta_w
                effective_delta = self.traces[action]
            else:
                effective_delta = delta_w
            
            # === Synaptic consolidation ===
            if c.use_consolidation:
                # Update volatility tracker: EMA of squared changes
                self.weight_volatility[action] = (
                    c.consolidation_ema * self.weight_volatility[action]
                    + (1 - c.consolidation_ema) * effective_delta.pow(2)
                )
                
                # Compute protection factor: stable weights (low volatility)
                # are protected more strongly.
                # importance = 1 / (volatility + eps) → high for stable weights
                importance = 1.0 / (self.weight_volatility[action].sqrt() + 1e-4)
                # Normalize to [0, 1] range
                importance = importance / (importance.max() + 1e-8)
                
                # Pull toward anchor: prevents drift from known-good weights
                anchor_pull = c.consolidation_strength * importance * (
                    self.weight_anchor[action] - self.W_value[action]
                )
                
                # Apply update with consolidation
                self.W_value[action] += effective_delta + anchor_pull
            else:
                self.W_value[action] += effective_delta
            
            if c.use_traces and done:
                self.traces.zero_()
            
            self.W_value.clamp_(-5.0, 5.0)
        
        # Epsilon decay
        self.epsilon = max(c.epsilon_min, self.epsilon * c.epsilon_decay)
        self.step_count += 1
        
        # Track rewards
        self.current_episode_reward += reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
    
    def train(self, env, n_timesteps: Optional[int] = None, callback=None) -> List[dict]:
        """Training loop."""
        n_timesteps = n_timesteps or self.config.total_timesteps
        c = self.config
        
        obs = env.reset()
        episode_count = 0
        wins, losses = 0, 0
        
        print(f"BioAgent Training: {n_timesteps:,} steps "
              f"(PC warmup: {c.pc_warmup_steps:,}, "
              f"PC freeze: {c.pc_freeze_after if c.pc_freeze_after > 0 else 'never':,}, "
              f"features: PC{'+eng' if c.use_engineered else ''} "
              f"dim={self.feature_dim})")
        
        for step in range(1, n_timesteps + 1):
            action = self.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            self.update(next_obs, reward, done)
            
            if done:
                episode_count += 1
                if info.get("event") == "win":
                    wins += 1
                elif info.get("event") == "lose":
                    losses += 1
                
                obs = env.reset()
                self.prev_obs = None
                self.prev_features = None
                self.prev_action = None
            else:
                obs = next_obs
            
            if step % c.log_interval == 0:
                pc_error = self.pc_encoder.get_prediction_error_history()
                log_entry = {
                    "step": step,
                    "episodes": episode_count,
                    "win_rate": wins / max(1, episode_count),
                    "epsilon": self.epsilon,
                    "pc_pred_error": pc_error,
                }
                self.log.append(log_entry)
                
                if callback:
                    callback(log_entry)
                
                if step % (c.log_interval * 10) == 0:
                    print(
                        f"  Step {step:,}/{n_timesteps:,} | "
                        f"Episodes: {episode_count} | "
                        f"Win rate: {log_entry['win_rate']:.1%} | "
                        f"PC error: {pc_error:.5f} | "
                        f"ε: {self.epsilon:.3f}"
                    )
        
        return self.log
    
    def get_learned_distributions(self, obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._get_features(obs)
        neuron_values = self._get_values(features)
        sorted_v, _ = neuron_values.sort(dim=1)
        return self.taus.clone(), sorted_v
    
    def get_stats(self) -> dict:
        return {
            "value_weight_norm": self.W_value.norm().item(),
            "pc_stats": self.pc_encoder.get_weight_stats(),
            "feature_dim": self.feature_dim,
            "step_count": self.step_count,
        }
    
    def update_anchor(self, eval_score: float):
        """
        Update the consolidation anchor if performance improved.
        
        Called externally after evaluation. If the current eval score
        is a new best, snapshot the weights as the new anchor.
        
        Biological interpretation: memory consolidation during "sleep"
        or rest periods — the brain stabilizes the synaptic configuration
        that produced good behavior.
        """
        if not self.config.use_consolidation:
            return
        
        if eval_score > self.best_eval_score:
            self.best_eval_score = eval_score
            self.weight_anchor = self.W_value.clone()