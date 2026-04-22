"""
Custom Pong Environment for RL experiments.

Lightweight 2D Pong with continuous physics, no external dependencies.
Designed to run 5000+ steps/second on CPU.

State: [ball_x, ball_y, ball_vx, ball_vy, paddle_y, opponent_y]
       All normalized to [-1, 1].
Action: discrete {0: stay, 1: up, 2: down} or continuous [-1, 1]

The opponent uses a simple tracking policy with configurable skill
(delay, noise, speed limit) to create a tunable difficulty.

Features:
- Vectorized: can run N environments in parallel (for PPO)
- Deterministic physics with configurable noise
- Gym-like API: reset() → obs, step(action) → (obs, reward, done, info)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PongConfig:
    # Court dimensions (normalized coordinates)
    court_width: float = 2.0    # x ∈ [-1, 1]
    court_height: float = 2.0   # y ∈ [-1, 1]
    
    # Paddle
    paddle_height: float = 0.3  # Half-height of paddle
    paddle_x: float = 0.9       # Distance from center (agent at +x, opponent at -x)
    paddle_speed: float = 0.04  # Max paddle movement per step
    
    # Ball
    ball_speed: float = 0.03    # Initial ball speed
    ball_speed_increase: float = 0.001  # Speed increase per rally hit
    ball_max_speed: float = 0.06
    ball_radius: float = 0.02
    
    # Opponent AI
    opponent_skill: float = 0.7    # 0=random, 1=perfect tracking
    opponent_speed: float = 0.03   # Max opponent paddle speed
    opponent_reaction_delay: int = 3  # Frames of delay
    
    # Game
    max_steps: int = 1000
    reward_win: float = 1.0
    reward_lose: float = -1.0
    reward_hit: float = 0.05      # Small reward for hitting the ball
    reward_step: float = 0.0      # Per-step reward (can set negative for urgency)
    
    # Action space
    discrete_actions: bool = True  # True: {0,1,2}, False: continuous [-1,1]


class PongEnv:
    """
    Single-instance Pong environment.
    
    Coordinate system:
    - x: left-right, agent paddle at x = +paddle_x, opponent at x = -paddle_x
    - y: up-down, court from -1 to +1
    - Ball starts at center, launched toward a random side
    """
    
    def __init__(self, config: Optional[PongConfig] = None):
        self.config = config or PongConfig()
        self.c = self.config  # Shorthand
        
        # State
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.paddle_y = 0.0      # Agent paddle center y
        self.opponent_y = 0.0    # Opponent paddle center y
        
        self.step_count = 0
        self.rally_length = 0
        self.agent_score = 0
        self.opponent_score = 0
        
        # Opponent reaction buffer
        self._opponent_target_buffer = []
        
        # State dimensions
        self.obs_dim = 6
        self.n_actions = 3 if self.c.discrete_actions else 1
    
    def reset(self) -> np.ndarray:
        """Reset and return initial observation."""
        self.ball_x = 0.0
        self.ball_y = np.random.uniform(-0.5, 0.5)
        
        # Launch ball in random direction
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        direction = np.random.choice([-1, 1])
        speed = self.c.ball_speed
        self.ball_vx = direction * speed * np.cos(angle)
        self.ball_vy = speed * np.sin(angle)
        
        self.paddle_y = 0.0
        self.opponent_y = 0.0
        self.step_count = 0
        self.rally_length = 0
        self._opponent_target_buffer = [0.0] * self.c.opponent_reaction_delay
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Return normalized observation vector."""
        return np.array([
            self.ball_x,
            self.ball_y,
            self.ball_vx * 20,  # Scale velocities to ~[-1,1]
            self.ball_vy * 20,
            self.paddle_y,
            self.opponent_y,
        ], dtype=np.float32)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take one step.
        
        Args:
            action: int (0=stay, 1=up, 2=down) if discrete,
                    float in [-1,1] if continuous
        
        Returns:
            obs, reward, done, info
        """
        # --- Parse action ---
        if self.c.discrete_actions:
            if action == 0:
                paddle_move = 0.0
            elif action == 1:
                paddle_move = self.c.paddle_speed
            else:
                paddle_move = -self.c.paddle_speed
        else:
            paddle_move = float(action) * self.c.paddle_speed
        
        # --- Move agent paddle ---
        self.paddle_y = np.clip(
            self.paddle_y + paddle_move,
            -1.0 + self.c.paddle_height,
            1.0 - self.c.paddle_height,
        )
        
        # --- Move opponent paddle (delayed tracking AI) ---
        self._move_opponent()
        
        # --- Move ball ---
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # --- Wall bounces (top/bottom) ---
        if self.ball_y >= 1.0 - self.c.ball_radius:
            self.ball_y = 2 * (1.0 - self.c.ball_radius) - self.ball_y
            self.ball_vy *= -1
        elif self.ball_y <= -1.0 + self.c.ball_radius:
            self.ball_y = 2 * (-1.0 + self.c.ball_radius) - self.ball_y
            self.ball_vy *= -1
        
        # --- Paddle collisions ---
        reward = self.c.reward_step
        done = False
        info = {"event": "none"}
        
        # Agent paddle (right side, x = +paddle_x)
        if (self.ball_x >= self.c.paddle_x - self.c.ball_radius and
            self.ball_vx > 0):
            if abs(self.ball_y - self.paddle_y) <= self.c.paddle_height:
                # Hit! Reflect ball
                self.ball_x = self.c.paddle_x - self.c.ball_radius
                self.ball_vx *= -1
                
                # Add angle based on where ball hit paddle
                hit_pos = (self.ball_y - self.paddle_y) / self.c.paddle_height
                self.ball_vy += hit_pos * 0.01
                
                # Speed up
                speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
                new_speed = min(speed + self.c.ball_speed_increase, self.c.ball_max_speed)
                scale = new_speed / (speed + 1e-8)
                self.ball_vx *= scale
                self.ball_vy *= scale
                
                self.rally_length += 1
                reward += self.c.reward_hit
                info["event"] = "hit"
            elif self.ball_x > 1.0:
                # Ball passed agent → opponent scores
                reward += self.c.reward_lose
                done = True
                self.opponent_score += 1
                info["event"] = "lose"
        
        # Opponent paddle (left side, x = -paddle_x)
        if (self.ball_x <= -self.c.paddle_x + self.c.ball_radius and
            self.ball_vx < 0):
            if abs(self.ball_y - self.opponent_y) <= self.c.paddle_height:
                # Opponent hit
                self.ball_x = -self.c.paddle_x + self.c.ball_radius
                self.ball_vx *= -1
                
                hit_pos = (self.ball_y - self.opponent_y) / self.c.paddle_height
                self.ball_vy += hit_pos * 0.01
                
                speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
                new_speed = min(speed + self.c.ball_speed_increase, self.c.ball_max_speed)
                scale = new_speed / (speed + 1e-8)
                self.ball_vx *= scale
                self.ball_vy *= scale
                
                self.rally_length += 1
                info["event"] = "opponent_hit"
            elif self.ball_x < -1.0:
                # Ball passed opponent → agent scores
                reward += self.c.reward_win
                done = True
                self.agent_score += 1
                info["event"] = "win"
        
        # Max steps
        self.step_count += 1
        if self.step_count >= self.c.max_steps:
            done = True
            info["event"] = "timeout"
        
        info["rally_length"] = self.rally_length
        info["step"] = self.step_count
        
        return self._get_obs(), reward, done, info
    
    def _move_opponent(self):
        """Simple tracking AI with delay and noise."""
        # Target: where ball will be (simple prediction)
        if self.ball_vx < 0:  # Ball moving toward opponent
            target = self.ball_y
        else:
            target = 0.0  # Return to center when ball going away
        
        # Add noise based on skill
        noise = (1.0 - self.c.opponent_skill) * np.random.randn() * 0.3
        target += noise
        
        # Reaction delay
        self._opponent_target_buffer.append(target)
        delayed_target = self._opponent_target_buffer.pop(0)
        
        # Move toward delayed target
        diff = delayed_target - self.opponent_y
        move = np.clip(diff, -self.c.opponent_speed, self.c.opponent_speed)
        
        # Apply skill-based accuracy
        move *= self.c.opponent_skill + (1 - self.c.opponent_skill) * np.random.random()
        
        self.opponent_y = np.clip(
            self.opponent_y + move,
            -1.0 + self.c.paddle_height,
            1.0 - self.c.paddle_height,
        )


class VectorPongEnv:
    """
    Vectorized Pong: run N environments in parallel.
    Essential for PPO which needs batched rollouts.
    """
    
    def __init__(self, n_envs: int = 8, config: Optional[PongConfig] = None):
        self.n_envs = n_envs
        self.envs = [PongEnv(config) for _ in range(n_envs)]
        self.obs_dim = self.envs[0].obs_dim
        self.n_actions = self.envs[0].n_actions
    
    def reset(self) -> np.ndarray:
        """Reset all envs, return (n_envs, obs_dim)."""
        return np.stack([env.reset() for env in self.envs])
    
    def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Step all envs. Auto-resets done envs.
        
        Args:
            actions: (n_envs,) array of actions
        
        Returns:
            obs (n_envs, obs_dim), rewards (n_envs,), 
            dones (n_envs,), infos (list of dicts)
        """
        obs_list, rew_list, done_list, info_list = [], [], [], []
        
        for i, env in enumerate(self.envs):
            obs, rew, done, info = env.step(actions[i])
            
            if done:
                # Auto-reset and return the reset observation
                obs = env.reset()
                info["terminal_obs"] = True
            
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            info_list.append(info)
        
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            info_list,
        )
    
    def get_scores(self) -> list:
        """Return (agent_score, opponent_score) for each env."""
        return [(e.agent_score, e.opponent_score) for e in self.envs]
