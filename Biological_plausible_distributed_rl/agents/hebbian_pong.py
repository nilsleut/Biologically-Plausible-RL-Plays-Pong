"""
Hebbian Distributional Agent for Pong.

This is the key integration: adapting the Hebbian distributional RL
framework from bandits/foraging to a real game (Pong).

Architecture:
  1. Feature layer: obs → features via fixed random projection 
     (no backprop; biological: random synaptic connectivity)
  2. Value population: N neurons per action, each with asymmetric
     Hebbian plasticity (α⁺/α⁻), learning distributional value
  3. Action selection: softmax over population readout (mean or CVaR)

Key design decisions:
- The feature layer is FIXED (random weights). This is both biologically
  motivated (sensory processing is largely innate/pretrained) and 
  practically necessary (Hebbian learning on raw features is too slow).
- Only the value weights are learned via Hebbian updates.
- Each value neuron computes: v_i(a) = w_i · φ(obs), where φ is the
  fixed feature transform and w_i are the Hebbian weights.
- The update rule is: Δw_i = α * δ_i * φ(obs), where δ_i = r + γV' - v_i
  with asymmetric α (α⁺ if δ>0, α⁻ if δ<0).

This is essentially a distributional TD(0) with linear function
approximation and local Hebbian weight updates.

Connection to DishBrain (Kagan et al. 2022):
- DishBrain's neurons receive game state as electrical stimulation
  → our fixed random projection
- DishBrain's learning signal is unpredictable stimulation for errors
  → our asymmetric RPE
- Both use populations of neurons to represent value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class HebbianPongConfig:
    obs_dim: int = 6
    n_actions: int = 3
    
    # Feature layer
    feature_dim: int = 128       # Dimension of random feature space
    feature_nonlinearity: str = "relu"  # "relu", "tanh", "none"
    feature_mode: str = "engineered"  # "random", "engineered", "both"
    
    # Value population
    n_neurons: int = 32          # Quantile neurons per action
    base_lr: float = 0.001      # Base Hebbian learning rate
    gamma: float = 0.99         # Discount factor
    
    # Action selection
    temperature: float = 1.0     # Softmax temperature
    epsilon: float = 0.1         # ε-greedy (on top of softmax)
    epsilon_decay: float = 0.99995
    epsilon_min: float = 0.02
    
    # Risk readout
    risk_measure: str = "neutral"  # "neutral", "averse", "seeking"
    risk_param: float = 0.25
    
    # Eligibility traces (helps with temporal credit assignment)
    trace_decay: float = 0.9     # λ for eligibility traces
    use_traces: bool = True
    
    # Training
    total_timesteps: int = 500_000
    log_interval: int = 1000


class HebbianPongAgent:
    """
    Biologically plausible agent for Pong.
    
    No backprop anywhere. All learning is local Hebbian plasticity.
    """
    
    def __init__(self, config: HebbianPongConfig):
        self.config = config
        c = config
        
        # --- Feature layer ---
        # Compute actual feature dimension based on mode
        self.engineered_dim = 17  # Fixed: number of handcrafted features
        if c.feature_mode == "random":
            self.actual_feature_dim = c.feature_dim
        elif c.feature_mode == "engineered":
            self.actual_feature_dim = self.engineered_dim
        elif c.feature_mode == "both":
            self.actual_feature_dim = c.feature_dim + self.engineered_dim
        else:
            raise ValueError(f"Unknown feature_mode: {c.feature_mode}")
        
        # Random projection weights (only used if mode includes "random")
        torch.manual_seed(0)
        self.W_feat = torch.randn(c.obs_dim, c.feature_dim) * 0.5
        self.b_feat = torch.randn(c.feature_dim) * 0.1
        torch.manual_seed(torch.seed())
        
        # --- Quantile levels ---
        self.taus = (torch.arange(c.n_neurons).float() + 0.5) / c.n_neurons
        self.alpha_pos = c.base_lr * (1.0 - self.taus)  # (n_neurons,)
        self.alpha_neg = c.base_lr * self.taus            # (n_neurons,)
        
        # --- Hebbian value weights ---
        # Shape: (n_actions, n_neurons, actual_feature_dim)
        self.W_value = torch.zeros(c.n_actions, c.n_neurons, self.actual_feature_dim)
        self.W_value += torch.randn_like(self.W_value) * 0.01
        
        # --- Eligibility traces ---
        if c.use_traces:
            self.traces = torch.zeros_like(self.W_value)
        
        # State
        self.epsilon = c.epsilon
        self.step_count = 0
        self.prev_features = None
        self.prev_action = None
        self.prev_values = None
        
        # Logging
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.log: List[dict] = []
    
    def _engineered_features(self, obs: np.ndarray) -> torch.Tensor:
        """
        Handcrafted features that make the Pong problem linearly separable.
        
        These encode the key information a player needs:
        - Where is the ball relative to my paddle?
        - Is the ball coming toward me?
        - How much time do I have to react?
        - Am I aligned with the ball's trajectory?
        
        Biologically motivated: sensory neurons in visual cortex extract
        oriented edges, motion direction, relative positions — these are
        the Pong equivalents.
        """
        # Unpack: [ball_x, ball_y, ball_vx*20, ball_vy*20, paddle_y, opponent_y]
        bx, by, bvx, bvy, py, oy = obs
        bvx_raw = bvx / 20.0  # Undo the scaling from env
        bvy_raw = bvy / 20.0
        
        features = []
        
        # 1. Raw state (normalized, 6 features)
        features.extend([bx, by, bvx, bvy, py, oy])
        
        # 2. Ball-paddle alignment (2 features) — KEY for learning to track
        ball_paddle_dy = by - py           # Vertical offset: + means ball above
        ball_paddle_dy_abs = abs(by - py)  # Distance regardless of direction
        features.extend([ball_paddle_dy, ball_paddle_dy_abs])
        
        # 3. Ball approaching? (2 features)
        ball_approaching = 1.0 if bvx_raw > 0 else 0.0  # Binary: ball coming to us
        urgency = bx * bvx  # High when ball is close AND moving toward us
        features.extend([ball_approaching, urgency])
        
        # 4. Time-to-impact estimate (1 feature)
        if bvx_raw > 0.001:  # Ball moving toward agent
            time_to_reach = (0.9 - bx) / (bvx_raw + 1e-8)
            time_to_reach = min(time_to_reach, 10.0) / 10.0  # Normalize
        else:
            time_to_reach = 1.0  # Far away / not approaching
        features.append(time_to_reach)
        
        # 5. Predicted ball y at paddle x (2 features) — simple linear prediction
        if bvx_raw > 0.001:
            steps_to_paddle = (0.9 - bx) / (bvx_raw + 1e-8)
            predicted_y = by + bvy_raw * steps_to_paddle
            # Rough wall bounce correction
            predicted_y = np.clip(predicted_y, -1.0, 1.0)
            pred_error = predicted_y - py  # How far off are we from intercept?
        else:
            predicted_y = 0.0
            pred_error = -py  # Move toward center when ball going away
        features.extend([predicted_y, pred_error])
        
        # 6. Interaction terms (3 features)
        features.append(ball_approaching * ball_paddle_dy)  # Offset only when ball coming
        features.append(ball_approaching * pred_error)       # Prediction error when relevant
        features.append((1.0 - ball_approaching) * py)       # Paddle position when ball away
        
        # 7. Bias term (1 feature)
        features.append(1.0)
        
        return torch.FloatTensor(features)
    
    def _random_features(self, obs: np.ndarray) -> torch.Tensor:
        """Fixed random projection features."""
        obs_t = torch.FloatTensor(obs)
        feat = obs_t @ self.W_feat + self.b_feat
        
        if self.config.feature_nonlinearity == "relu":
            feat = F.relu(feat)
        elif self.config.feature_nonlinearity == "tanh":
            feat = torch.tanh(feat)
        
        feat = feat / (feat.norm() + 1e-8) * np.sqrt(self.config.feature_dim)
        return feat
    
    def _features(self, obs: np.ndarray) -> torch.Tensor:
        """Compute features based on configured mode."""
        mode = self.config.feature_mode
        
        if mode == "random":
            return self._random_features(obs)
        elif mode == "engineered":
            return self._engineered_features(obs)
        elif mode == "both":
            rand_feat = self._random_features(obs)
            eng_feat = self._engineered_features(obs)
            return torch.cat([eng_feat, rand_feat])
    
    def _get_values(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute value for each (action, neuron) pair.
        
        Returns: (n_actions, n_neurons)
        """
        # v[a, i] = W_value[a, i, :] · features
        return torch.einsum("aif,f->ai", self.W_value, features)
    
    def _action_values(self, neuron_values: torch.Tensor) -> torch.Tensor:
        """
        Read out action values from the neural population.
        
        Args:
            neuron_values: (n_actions, n_neurons)
        
        Returns:
            (n_actions,) action values
        """
        sorted_v, _ = neuron_values.sort(dim=1)
        
        if self.config.risk_measure == "neutral":
            return sorted_v.mean(dim=1)
        elif self.config.risk_measure == "averse":
            n = max(1, int(self.config.n_neurons * self.config.risk_param))
            return sorted_v[:, :n].mean(dim=1)
        elif self.config.risk_measure == "seeking":
            n = max(1, int(self.config.n_neurons * self.config.risk_param))
            return sorted_v[:, -n:].mean(dim=1)
        return sorted_v.mean(dim=1)
    
    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Select action via softmax over population readout + ε-greedy."""
        features = self._features(obs)
        neuron_values = self._get_values(features)
        action_values = self._action_values(neuron_values)
        
        # Store for update
        self.prev_features = features
        self.prev_values = neuron_values
        
        if not greedy and np.random.random() < self.epsilon:
            action = np.random.randint(self.config.n_actions)
        else:
            # Softmax action selection
            probs = F.softmax(action_values / self.config.temperature, dim=0)
            if greedy:
                action = probs.argmax().item()
            else:
                action = torch.multinomial(probs, 1).item()
        
        self.prev_action = action
        return action
    
    def update(self, obs: np.ndarray, reward: float, done: bool):
        """
        Hebbian TD update.
        
        For each neuron i of the chosen action a:
          1. Compute next state features and values
          2. TD target: y = r + γ * mean(v_i(a', next_obs)) if not done, else r
             (we use the mean across actions for the next-state value,
              but each neuron has its OWN target based on its quantile)
          3. RPE: δ_i = y - v_i(a, obs)
          4. Hebbian update: ΔW[a,i,:] = α_i * δ_i * φ(obs)
             where α_i = α⁺ if δ_i ≥ 0, else α⁻
        """
        if self.prev_features is None:
            return
        
        c = self.config
        features = self.prev_features          # φ(s)
        action = self.prev_action
        old_values = self.prev_values[action]   # (n_neurons,) = v_i(a, s)
        
        # Next state value (for TD target)
        if done:
            next_v = 0.0
        else:
            next_features = self._features(obs)
            next_neuron_values = self._get_values(next_features)
            # Use mean across actions as bootstrap (like expected SARSA)
            next_v = next_neuron_values.mean(dim=0)  # (n_neurons,)
        
        # TD target per neuron
        if isinstance(next_v, float):
            target = reward
        else:
            target = reward + c.gamma * next_v  # (n_neurons,)
        
        # RPE per neuron
        rpe = target - old_values  # (n_neurons,)
        
        # Asymmetric learning rate
        lr = torch.where(rpe >= 0, self.alpha_pos, self.alpha_neg)
        
        # Hebbian update: ΔW = α * δ * φ
        # Shape: (n_neurons,) * (n_neurons,) → (n_neurons,) outer (feature_dim,)
        delta_w = torch.einsum("n,f->nf", lr * rpe, features)
        
        if c.use_traces:
            # Eligibility traces: accumulate gradient, decay over time
            self.traces[action] = c.trace_decay * c.gamma * self.traces[action] + delta_w
            self.W_value[action] += self.traces[action]
            
            if done:
                self.traces.zero_()
        else:
            self.W_value[action] += delta_w
        
        # Weight clipping to prevent explosion
        self.W_value.clamp_(-5.0, 5.0)
        
        # Epsilon decay
        self.epsilon = max(c.epsilon_min, self.epsilon * c.epsilon_decay)
        self.step_count += 1
        
        # Track episode stats
        self.current_episode_reward += reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
    
    def train(self, env, n_timesteps: Optional[int] = None, callback=None) -> List[dict]:
        """
        Training loop for single-env Hebbian agent.
        
        Unlike PPO, this is on-policy single-step — no rollout buffer,
        no minibatches. Each transition updates immediately.
        """
        n_timesteps = n_timesteps or self.config.total_timesteps
        
        obs = env.reset()
        episode_count = 0
        wins, losses = 0, 0
        
        print(f"Hebbian Training: {n_timesteps:,} timesteps")
        
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
                self.prev_features = None
                self.prev_action = None
            else:
                obs = next_obs
            
            # Logging
            if step % self.config.log_interval == 0:
                recent = self.episode_rewards[-20:] if self.episode_rewards else [0]
                log_entry = {
                    "step": step,
                    "episodes": episode_count,
                    "avg_reward": np.mean(recent),
                    "wins": wins,
                    "losses": losses,
                    "win_rate": wins / max(1, episode_count),
                    "epsilon": self.epsilon,
                }
                self.log.append(log_entry)
                
                if callback:
                    callback(log_entry)
                
                if step % (self.config.log_interval * 10) == 0:
                    print(
                        f"  Step {step:,}/{n_timesteps:,} | "
                        f"Episodes: {episode_count} | "
                        f"Win rate: {log_entry['win_rate']:.1%} | "
                        f"ε: {self.epsilon:.3f}"
                    )
        
        return self.log
    
    def get_learned_distributions(self, obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (taus, values) where values is (n_actions, n_neurons)
        for visualization.
        """
        features = self._features(obs)
        neuron_values = self._get_values(features)
        sorted_v, _ = neuron_values.sort(dim=1)
        return self.taus.clone(), sorted_v
    
    def get_weight_stats(self) -> dict:
        """Diagnostic: weight statistics."""
        return {
            "mean": self.W_value.mean().item(),
            "std": self.W_value.std().item(),
            "max": self.W_value.max().item(),
            "min": self.W_value.min().item(),
            "norm": self.W_value.norm().item(),
        }
