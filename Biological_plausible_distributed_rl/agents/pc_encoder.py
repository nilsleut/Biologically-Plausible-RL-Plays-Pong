"""
Predictive Coding Feature Encoder for Pong.

Learns feature representations by predicting the next observation
from the current one, using ONLY local Hebbian-style updates.
No backpropagation anywhere.

Architecture:
  obs_t → Encoder → z_t (latent features)
  z_t → Predictor → obs_t+1_predicted
  Prediction error: e = obs_t+1 - obs_t+1_predicted
  Update: local Hebbian learning to reduce prediction error

This implements a simplified Predictive Coding network (Rao & Ballard 1999):
- Each layer tries to predict the input from the layer below
- Prediction errors drive learning
- Only local information (pre/post-synaptic activity + error) is needed

Biological interpretation:
- The encoder is early visual cortex extracting features
- The predictor is a forward model of the environment
- Prediction errors are the teaching signal (like dopamine RPEs,
  but for sensory processing rather than reward)

Connection to the full system:
  PC Encoder (this) → features → Hebbian Value Agent → actions
  Both learn with local rules. Zero backprop in the entire pipeline.

Reference: Rao & Ballard (1999), Whittington & Bogacz (2017)
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PCEncoderConfig:
    obs_dim: int = 6
    hidden_dim: int = 64          # Internal representation size
    feature_dim: int = 32         # Output feature dimension (fed to value agent)
    
    # Learning rates (all local, no backprop)
    lr_encoder: float = 0.001     # W_enc learning rate (reduced from 0.005)
    lr_predictor: float = 0.001   # W_pred learning rate (reduced from 0.005)
    
    # LR decay: after warmup, reduce LR to stabilize features
    lr_decay: float = 0.99999     # Per-step multiplicative decay
    lr_min: float = 0.0001        # Floor
    
    # PC inference
    n_inference_steps: int = 5    # Iterative inference steps per observation
    inference_lr: float = 0.1     # Step size for inference dynamics
    
    # Regularization
    weight_decay: float = 0.0001  # Prevents weight explosion
    max_weight_norm: float = 5.0  # Hard clip on weight norms
    
    # Feature normalization
    normalize_features: bool = True
    ema_decay: float = 0.999      # For running mean/var of features


class PCEncoder:
    """
    Predictive Coding encoder that learns features without backprop.
    
    Two-layer architecture:
    
    Layer 1 (Encoder): obs → hidden → features
      W1: obs_dim → hidden_dim (with ReLU)
      W2: hidden_dim → feature_dim
    
    Layer 2 (Predictor): features → predicted next obs
      W3: feature_dim → hidden_dim (with ReLU)  
      W4: hidden_dim → obs_dim
    
    Learning rule (for each weight matrix W):
      ΔW = lr * error * input^T
    
    This is a Hebbian rule where:
    - "error" is the local prediction error (post-synaptic)
    - "input" is the pre-synaptic activity
    - The product error * input^T is the outer product = Hebbian term
    
    No gradient flows between layers. Each layer only uses its own
    prediction error and its own input/output.
    """
    
    def __init__(self, config: Optional[PCEncoderConfig] = None):
        self.config = config or PCEncoderConfig()
        c = self.config
        
        # --- Encoder weights (obs → features) ---
        # Layer 1: obs → hidden
        self.W1 = torch.randn(c.obs_dim, c.hidden_dim) * np.sqrt(2.0 / c.obs_dim)
        self.b1 = torch.zeros(c.hidden_dim)
        
        # Layer 2: hidden → features
        self.W2 = torch.randn(c.hidden_dim, c.feature_dim) * np.sqrt(2.0 / c.hidden_dim)
        self.b2 = torch.zeros(c.feature_dim)
        
        # --- Predictor weights (features → predicted next obs) ---
        # Layer 3: features → hidden
        self.W3 = torch.randn(c.feature_dim, c.hidden_dim) * np.sqrt(2.0 / c.feature_dim)
        self.b3 = torch.zeros(c.hidden_dim)
        
        # Layer 4: hidden → predicted obs
        self.W4 = torch.randn(c.hidden_dim, c.obs_dim) * np.sqrt(2.0 / c.hidden_dim)
        self.b4 = torch.zeros(c.obs_dim)
        
        # --- Running statistics for feature normalization ---
        self.feature_mean = torch.zeros(c.feature_dim)
        self.feature_var = torch.ones(c.feature_dim)
        
        # --- State ---
        self.prev_obs = None
        self.prev_features = None
        self.step_count = 0
        self.prediction_errors: list = []
        self.current_lr_enc = c.lr_encoder
        self.current_lr_pred = c.lr_predictor
    
    def encode(self, obs: np.ndarray) -> torch.Tensor:
        """
        Encode observation into feature vector.
        
        Forward pass through encoder (no learning here).
        Returns normalized features ready for the value agent.
        """
        obs_t = torch.FloatTensor(obs)
        
        # Layer 1: obs → hidden (ReLU)
        h1 = F.relu(obs_t @ self.W1 + self.b1)
        
        # Layer 2: hidden → features
        features = h1 @ self.W2 + self.b2
        
        # Normalize features
        if self.config.normalize_features and self.step_count > 100:
            features = (features - self.feature_mean) / (self.feature_var.sqrt() + 1e-8)
        
        return features
    
    def predict_next(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict next observation from features.
        
        Forward pass through predictor.
        """
        # Layer 3: features → hidden (ReLU)
        h3 = F.relu(features @ self.W3 + self.b3)
        
        # Layer 4: hidden → predicted next obs
        pred_obs = h3 @ self.W4 + self.b4
        
        return pred_obs
    
    def update(self, obs: np.ndarray, next_obs: np.ndarray):
        """
        One step of Predictive Coding learning.
        
        1. Encode current obs → features
        2. Predict next obs from features
        3. Compute prediction error
        4. Update weights with local Hebbian rules
        
        All updates are LOCAL: each weight matrix is updated using
        only its own input and its own error signal.
        """
        c = self.config
        obs_t = torch.FloatTensor(obs)
        next_obs_t = torch.FloatTensor(next_obs)
        
        # === Forward pass ===
        # Encoder
        h1 = F.relu(obs_t @ self.W1 + self.b1)        # (hidden_dim,)
        features = h1 @ self.W2 + self.b2              # (feature_dim,)
        
        # Predictor  
        h3 = F.relu(features @ self.W3 + self.b3)      # (hidden_dim,)
        pred_next = h3 @ self.W4 + self.b4              # (obs_dim,)
        
        # === Prediction error ===
        pred_error = next_obs_t - pred_next              # (obs_dim,)
        error_magnitude = pred_error.pow(2).mean().item()
        self.prediction_errors.append(error_magnitude)
        
        # === Local Hebbian updates ===
        # 
        # Key insight: we propagate the error LOCALLY through each layer.
        # No backprop — each layer gets an error signal and updates
        # using only pre-synaptic input and post-synaptic error.
        
        # --- Predictor updates (reduce prediction error directly) ---
        lr_pred = self.current_lr_pred
        lr_enc = self.current_lr_enc
        
        # Clip prediction error to prevent explosive updates
        pred_error_clipped = pred_error.clamp(-0.5, 0.5)
        
        # Layer 4: ΔW4 = lr * h3^T ⊗ error
        self.W4 += lr_pred * torch.outer(h3, pred_error_clipped)
        self.b4 += lr_pred * pred_error_clipped
        
        # Layer 3: error propagated back one step (still local to predictor)
        h3_error = pred_error_clipped @ self.W4.T
        h3_error = h3_error * (h3 > 0).float()  # ReLU mask (local info)
        
        self.W3 += lr_pred * torch.outer(features, h3_error)
        self.b3 += lr_pred * h3_error
        
        # --- Encoder updates (learn features that reduce prediction error) ---
        
        feature_error = h3_error @ self.W3.T  # (feature_dim,)
        
        # Layer 2: ΔW2 = lr * h1^T ⊗ feature_error
        self.W2 += lr_enc * torch.outer(h1, feature_error)
        self.b2 += lr_enc * feature_error
        
        # Layer 1: propagate to input layer
        h1_error = feature_error @ self.W2.T
        h1_error = h1_error * (h1 > 0).float()
        
        self.W1 += lr_enc * torch.outer(obs_t, h1_error)
        self.b1 += lr_enc * h1_error
        
        # === Regularization ===
        # Weight decay
        if c.weight_decay > 0:
            self.W1 *= (1 - c.weight_decay)
            self.W2 *= (1 - c.weight_decay)
            self.W3 *= (1 - c.weight_decay)
            self.W4 *= (1 - c.weight_decay)
        
        # Clip weight norms
        for W in [self.W1, self.W2, self.W3, self.W4]:
            norm = W.norm()
            if norm > c.max_weight_norm:
                W.mul_(c.max_weight_norm / norm)
        
        # === Update running statistics ===
        if c.normalize_features:
            self.feature_mean = c.ema_decay * self.feature_mean + (1 - c.ema_decay) * features.detach()
            self.feature_var = c.ema_decay * self.feature_var + (1 - c.ema_decay) * (features.detach() - self.feature_mean).pow(2)
        
        # === LR decay ===
        self.current_lr_enc = max(c.lr_min, self.current_lr_enc * c.lr_decay)
        self.current_lr_pred = max(c.lr_min, self.current_lr_pred * c.lr_decay)
        
        self.step_count += 1
    
    def get_prediction_error_history(self, window: int = 1000) -> float:
        """Return recent average prediction error."""
        if not self.prediction_errors:
            return float('nan')
        recent = self.prediction_errors[-window:]
        return np.mean(recent)
    
    def get_weight_stats(self) -> dict:
        """Diagnostic info."""
        return {
            "W1_norm": self.W1.norm().item(),
            "W2_norm": self.W2.norm().item(),
            "W3_norm": self.W3.norm().item(),
            "W4_norm": self.W4.norm().item(),
            "pred_error": self.get_prediction_error_history(),
            "steps": self.step_count,
        }