"""
Proximal Policy Optimization (PPO) from scratch.

Implements PPO-Clip with:
- Actor-Critic network (shared backbone optional)
- GAE (Generalized Advantage Estimation)
- Value function clipping
- Entropy bonus
- Mini-batch updates

Reference: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)

This is intentionally written from scratch — no stable-baselines,
no RLlib. The goal is to understand every component.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PPOConfig:
    # Architecture
    obs_dim: int = 6
    n_actions: int = 3
    hidden_dim: int = 64
    n_hidden: int = 2
    
    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    clip_eps: float = 0.2         # PPO clip range
    clip_value: bool = True       # Clip value function too
    vf_coef: float = 0.5          # Value loss coefficient
    ent_coef: float = 0.01        # Entropy bonus coefficient
    max_grad_norm: float = 0.5    # Gradient clipping
    
    # Training
    n_envs: int = 8               # Parallel environments
    n_steps: int = 128            # Steps per rollout per env
    n_epochs: int = 4             # PPO epochs per rollout
    n_minibatches: int = 4        # Minibatches per epoch
    total_timesteps: int = 500_000
    
    # Logging
    log_interval: int = 10        # Log every N updates


class ActorCritic(nn.Module):
    """
    Actor-Critic network with separate heads.
    
    Architecture:
        obs → [shared MLP] → actor head → action logits
                            → critic head → value
    """
    
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64,
                 n_hidden: int = 2):
        super().__init__()
        
        # Shared backbone
        layers = []
        in_dim = obs_dim
        for _ in range(n_hidden):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, n_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights (orthogonal init, standard for PPO)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Policy head: small init for exploration
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        # Value head: unit scale
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor):
        """Return (action_logits, value)."""
        features = self.backbone(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value
    
    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        """
        Sample action (or evaluate given action) and return everything
        needed for PPO: action, log_prob, entropy, value.
        """
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class RolloutBuffer:
    """
    Stores rollout data for PPO updates.
    
    After collecting n_steps * n_envs transitions, computes
    GAE advantages and returns, then provides minibatch iteration.
    """
    
    def __init__(self, n_steps: int, n_envs: int, obs_dim: int):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.batch_size = n_steps * n_envs
        
        # Storage
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=bool)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        
        # Computed after rollout
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        
        self.ptr = 0
    
    def add(self, obs, action, reward, done, log_prob, value):
        """Add one timestep of data."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1
    
    def compute_gae(self, last_value: np.ndarray, gamma: float, gae_lambda: float):
        """
        Compute Generalized Advantage Estimation.
        
        GAE(γ,λ): A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        
        This is the key advantage estimator for PPO — it balances
        bias (low λ → use value function more) vs variance (high λ 
        → use actual returns more).
        """
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
            
            # TD error
            delta = (self.rewards[t] + gamma * next_value * next_non_terminal 
                     - self.values[t])
            
            # GAE
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        self.returns = self.advantages + self.values
        self.ptr = 0
    
    def get_minibatches(self, n_minibatches: int):
        """
        Yield minibatches of flattened rollout data.
        Each yields dict of tensors with batch_size/n_minibatches samples.
        """
        batch_size = self.n_steps * self.n_envs
        mb_size = batch_size // n_minibatches
        
        # Flatten (steps, envs, ...) → (batch, ...)
        flat_obs = self.obs.reshape(batch_size, self.obs_dim)
        flat_actions = self.actions.reshape(batch_size)
        flat_log_probs = self.log_probs.reshape(batch_size)
        flat_values = self.values.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)
        
        # Random permutation
        indices = np.random.permutation(batch_size)
        
        for start in range(0, batch_size, mb_size):
            end = start + mb_size
            mb_idx = indices[start:end]
            
            yield {
                "obs": torch.FloatTensor(flat_obs[mb_idx]),
                "actions": torch.LongTensor(flat_actions[mb_idx]),
                "old_log_probs": torch.FloatTensor(flat_log_probs[mb_idx]),
                "old_values": torch.FloatTensor(flat_values[mb_idx]),
                "advantages": torch.FloatTensor(flat_advantages[mb_idx]),
                "returns": torch.FloatTensor(flat_returns[mb_idx]),
            }


class PPOAgent:
    """
    Complete PPO agent with training loop.
    
    Usage:
        agent = PPOAgent(PPOConfig())
        agent.train(vec_env)
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.network = ActorCritic(
            config.obs_dim, config.n_actions,
            config.hidden_dim, config.n_hidden,
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.lr, eps=1e-5
        )
        
        # Logging
        self.update_count = 0
        self.total_steps = 0
        self.log: List[dict] = []
    
    def collect_rollout(self, vec_env, buffer: RolloutBuffer) -> dict:
        """
        Collect n_steps of experience from n_envs parallel environments.
        Returns rollout statistics.
        """
        obs = vec_env.reset() if buffer.ptr == 0 else self._last_obs
        
        episode_rewards = []
        episode_wins = 0
        episode_losses = 0
        episode_count = 0
        
        for step in range(self.config.n_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
            
            actions_np = action.numpy()
            next_obs, rewards, dones, infos = vec_env.step(actions_np)
            
            buffer.add(
                obs, actions_np, rewards, dones,
                log_prob.numpy(), value.numpy()
            )
            
            # Track episodes
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    episode_count += 1
                    if info.get("event") == "win":
                        episode_wins += 1
                    elif info.get("event") == "lose":
                        episode_losses += 1
            
            obs = next_obs
            self.total_steps += self.config.n_envs
        
        self._last_obs = obs
        
        # Compute last value for GAE
        with torch.no_grad():
            _, last_value = self.network(torch.FloatTensor(obs))
        
        buffer.compute_gae(
            last_value.numpy(),
            self.config.gamma,
            self.config.gae_lambda,
        )
        
        win_rate = episode_wins / max(1, episode_count)
        
        return {
            "episode_count": episode_count,
            "win_rate": win_rate,
            "wins": episode_wins,
            "losses": episode_losses,
        }
    
    def update(self, buffer: RolloutBuffer) -> dict:
        """
        PPO update: multiple epochs of minibatch gradient descent
        on the clipped surrogate objective.
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clipfrac = 0
        n_updates = 0
        
        for epoch in range(self.config.n_epochs):
            for mb in buffer.get_minibatches(self.config.n_minibatches):
                # Get current policy's evaluation of old actions
                _, new_log_prob, entropy, new_value = \
                    self.network.get_action_and_value(mb["obs"], mb["actions"])
                
                # Normalize advantages (important for stability)
                advantages = mb["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # --- Policy loss (PPO-Clip) ---
                log_ratio = new_log_prob - mb["old_log_probs"]
                ratio = log_ratio.exp()
                
                # Clipped surrogate objective
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # --- Value loss ---
                if self.config.clip_value:
                    # Clipped value loss (PPO-style)
                    value_clipped = mb["old_values"] + torch.clamp(
                        new_value - mb["old_values"],
                        -self.config.clip_eps, self.config.clip_eps,
                    )
                    vf_loss1 = (new_value - mb["returns"]) ** 2
                    vf_loss2 = (value_clipped - mb["returns"]) ** 2
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * ((new_value - mb["returns"]) ** 2).mean()
                
                # --- Entropy bonus ---
                entropy_loss = -entropy.mean()
                
                # --- Total loss ---
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                )
                
                # --- Gradient step ---
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                
                # Logging
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                total_clipfrac += clipfrac.item()
                n_updates += 1
        
        self.update_count += 1
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "clipfrac": total_clipfrac / n_updates,
        }
    
    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Single-observation action selection for evaluation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            logits, _ = self.network(obs_tensor)
            
            if greedy:
                return logits.argmax(dim=1).item()
            
            dist = Categorical(logits=logits)
            return dist.sample().item()
    
    def train(self, vec_env, callback=None) -> List[dict]:
        """
        Full training loop.
        
        Args:
            vec_env: VectorPongEnv
            callback: optional function called each update with log dict
        
        Returns:
            Training log (list of dicts per update)
        """
        config = self.config
        buffer = RolloutBuffer(config.n_steps, config.n_envs, config.obs_dim)
        
        n_updates = config.total_timesteps // (config.n_steps * config.n_envs)
        self._last_obs = vec_env.reset()
        
        print(f"PPO Training: {config.total_timesteps} timesteps, "
              f"{n_updates} updates, {config.n_envs} envs")
        
        for update in range(1, n_updates + 1):
            # Collect rollout
            buffer.ptr = 0
            rollout_stats = self.collect_rollout(vec_env, buffer)
            
            # PPO update
            update_stats = self.update(buffer)
            
            # Log
            log_entry = {
                "update": update,
                "total_steps": self.total_steps,
                **rollout_stats,
                **update_stats,
            }
            self.log.append(log_entry)
            
            if callback:
                callback(log_entry)
            
            if update % config.log_interval == 0:
                print(
                    f"  Update {update}/{n_updates} | "
                    f"Steps: {self.total_steps:,} | "
                    f"Win rate: {rollout_stats['win_rate']:.1%} | "
                    f"Policy loss: {update_stats['policy_loss']:.4f} | "
                    f"Entropy: {update_stats['entropy']:.3f}"
                )
        
        return self.log
    
    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "update_count": self.update_count,
            "total_steps": self.total_steps,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_count = checkpoint["update_count"]
        self.total_steps = checkpoint["total_steps"]
