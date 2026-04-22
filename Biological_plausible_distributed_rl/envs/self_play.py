"""
Self-Play Pong Environment.

Instead of playing against a fixed AI opponent, the agent plays
against a frozen copy of itself. Periodically, the opponent is
updated to the current agent's weights.

This creates an automatic curriculum:
- Early training: opponent is random → easy wins
- Mid training: opponent is mediocre → balanced games  
- Late training: opponent is strong → must keep improving

Self-play is how AlphaGo and OpenAI Five learned. Doing it with
a fully biologically plausible agent (PC + Hebbian) is novel.

The key insight: self-play doesn't require backprop. The opponent
is just a frozen copy that selects actions — it doesn't learn.
Only the "student" agent learns, using the same local Hebbian rules.
"""

import numpy as np
import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Import will work for both flat and nested structures
try:
    from envs.pong import PongConfig
except ModuleNotFoundError:
    from pong import PongConfig


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""
    # How often to update the opponent (in episodes)
    opponent_update_interval: int = 50
    
    # Keep a pool of past opponents for diversity
    opponent_pool_size: int = 5
    
    # Probability of playing against a random past opponent vs latest
    past_opponent_prob: float = 0.2
    
    # Pong environment settings
    pong_config: PongConfig = None
    
    def __post_init__(self):
        if self.pong_config is None:
            self.pong_config = PongConfig(
                opponent_skill=1.0,  # Irrelevant — we override opponent movement
                max_steps=500,
                reward_hit=0.1,
            )


class SelfPlayPongEnv:
    """
    Pong environment where the opponent is controlled by a copy of the agent.
    
    The physics are the same as PongEnv, but instead of the built-in AI
    controlling the left paddle, a frozen agent copy does it.
    
    Key difference from regular Pong:
    - The opponent "sees" a mirrored observation (ball_x flipped, paddles swapped)
    - The opponent's actions control the left paddle
    - The opponent is periodically updated to the current agent's weights
    """
    
    def __init__(self, config: Optional[SelfPlayConfig] = None):
        self.config = config or SelfPlayConfig()
        self.c = self.config.pong_config
        
        # Game state
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.paddle_y = 0.0       # Agent (right)
        self.opponent_y = 0.0     # Opponent (left)
        
        self.step_count = 0
        self.agent_score = 0
        self.opponent_score = 0
        
        # Opponent agent (set externally)
        self.opponent_agent = None
        self.opponent_pool: List = []
        self.episodes_since_update = 0
        
        # Dimensions
        self.obs_dim = 6
        self.n_actions = 3
    
    def set_opponent(self, agent):
        """Set the opponent agent (a frozen copy)."""
        self.opponent_agent = agent
    
    def update_opponent_pool(self, agent):
        """Add current agent to opponent pool and select new opponent."""
        # Deep copy the agent's weights (not the whole object for memory)
        agent_snapshot = self._snapshot_agent(agent)
        self.opponent_pool.append(agent_snapshot)
        
        # Keep pool size bounded
        if len(self.opponent_pool) > self.config.opponent_pool_size:
            self.opponent_pool.pop(0)
        
        # Select opponent
        self._select_opponent()
        self.episodes_since_update = 0
    
    def _snapshot_agent(self, agent):
        """Create a lightweight snapshot of agent weights for the pool."""
        return copy.deepcopy(agent)
    
    def _select_opponent(self):
        """Select opponent from pool."""
        if not self.opponent_pool:
            return
        
        if (np.random.random() < self.config.past_opponent_prob 
            and len(self.opponent_pool) > 1):
            # Play against a random past version
            idx = np.random.randint(0, len(self.opponent_pool) - 1)
            self.opponent_agent = self.opponent_pool[idx]
        else:
            # Play against the latest version
            self.opponent_agent = self.opponent_pool[-1]
    
    def _get_agent_obs(self) -> np.ndarray:
        """Observation for the agent (right paddle)."""
        return np.array([
            self.ball_x,
            self.ball_y,
            self.ball_vx * 20,
            self.ball_vy * 20,
            self.paddle_y,
            self.opponent_y,
        ], dtype=np.float32)
    
    def _get_opponent_obs(self) -> np.ndarray:
        """
        Mirrored observation for the opponent (left paddle).
        
        The opponent sees the game from its own perspective:
        - ball_x is flipped (its paddle is at +x in its frame)
        - ball_vx is flipped
        - paddle positions are swapped
        """
        return np.array([
            -self.ball_x,           # Mirror x
            self.ball_y,
            -self.ball_vx * 20,     # Mirror vx
            self.ball_vy * 20,
            self.opponent_y,         # Its own paddle
            self.paddle_y,           # The "opponent" (which is the agent)
        ], dtype=np.float32)
    
    def reset(self) -> np.ndarray:
        self.ball_x = 0.0
        self.ball_y = np.random.uniform(-0.5, 0.5)
        
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        direction = np.random.choice([-1, 1])
        speed = self.c.ball_speed
        self.ball_vx = direction * speed * np.cos(angle)
        self.ball_vy = speed * np.sin(angle)
        
        self.paddle_y = 0.0
        self.opponent_y = 0.0
        self.step_count = 0
        
        # Check if opponent should be updated
        self.episodes_since_update += 1
        if (self.episodes_since_update >= self.config.opponent_update_interval
            and len(self.opponent_pool) > 0):
            self._select_opponent()
        
        return self._get_agent_obs()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step the environment.
        
        1. Agent moves right paddle (based on action)
        2. Opponent agent moves left paddle (based on mirrored observation)
        3. Ball physics
        4. Check scoring
        """
        c = self.c
        
        # --- Agent paddle movement ---
        if action == 0:
            agent_move = 0.0
        elif action == 1:
            agent_move = c.paddle_speed
        else:
            agent_move = -c.paddle_speed
        
        self.paddle_y = np.clip(
            self.paddle_y + agent_move,
            -1.0 + c.paddle_height, 1.0 - c.paddle_height,
        )
        
        # --- Opponent paddle movement ---
        if self.opponent_agent is not None:
            opp_obs = self._get_opponent_obs()
            opp_action = self.opponent_agent.select_action(opp_obs, greedy=True)
            
            if opp_action == 0:
                opp_move = 0.0
            elif opp_action == 1:
                opp_move = c.paddle_speed
            else:
                opp_move = -c.paddle_speed
            
            self.opponent_y = np.clip(
                self.opponent_y + opp_move,
                -1.0 + c.paddle_height, 1.0 - c.paddle_height,
            )
        
        # --- Ball physics (same as PongEnv) ---
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        # Wall bounces
        if self.ball_y >= 1.0 - c.ball_radius:
            self.ball_y = 2 * (1.0 - c.ball_radius) - self.ball_y
            self.ball_vy *= -1
        elif self.ball_y <= -1.0 + c.ball_radius:
            self.ball_y = 2 * (-1.0 + c.ball_radius) - self.ball_y
            self.ball_vy *= -1
        
        # Paddle collisions
        reward = 0.0
        done = False
        info = {"event": "none"}
        
        # Agent paddle (right)
        if self.ball_x >= c.paddle_x - c.ball_radius and self.ball_vx > 0:
            if abs(self.ball_y - self.paddle_y) <= c.paddle_height:
                self.ball_x = c.paddle_x - c.ball_radius
                self.ball_vx *= -1
                hit_pos = (self.ball_y - self.paddle_y) / c.paddle_height
                self.ball_vy += hit_pos * 0.01
                speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
                new_speed = min(speed + c.ball_speed_increase, c.ball_max_speed)
                scale = new_speed / (speed + 1e-8)
                self.ball_vx *= scale
                self.ball_vy *= scale
                reward += c.reward_hit
                info["event"] = "hit"
            elif self.ball_x > 1.0:
                reward += c.reward_lose
                done = True
                self.opponent_score += 1
                info["event"] = "lose"
        
        # Opponent paddle (left)
        if self.ball_x <= -c.paddle_x + c.ball_radius and self.ball_vx < 0:
            if abs(self.ball_y - self.opponent_y) <= c.paddle_height:
                self.ball_x = -c.paddle_x + c.ball_radius
                self.ball_vx *= -1
                hit_pos = (self.ball_y - self.opponent_y) / c.paddle_height
                self.ball_vy += hit_pos * 0.01
                speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
                new_speed = min(speed + c.ball_speed_increase, c.ball_max_speed)
                scale = new_speed / (speed + 1e-8)
                self.ball_vx *= scale
                self.ball_vy *= scale
                info["event"] = "opponent_hit"
            elif self.ball_x < -1.0:
                reward += c.reward_win
                done = True
                self.agent_score += 1
                info["event"] = "win"
        
        self.step_count += 1
        if self.step_count >= c.max_steps:
            done = True
            info["event"] = "timeout"
        
        info["rally_length"] = self.step_count
        
        return self._get_agent_obs(), reward, done, info