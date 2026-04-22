# Biologically Plausible RL Plays Pong

A technical exploration comparing standard deep RL (PPO) against biologically plausible learning rules on a custom Pong environment. Everything is implemented from scratch — no Stable-Baselines, no Gymnasium, no pre-built algorithms.

## What This Project Is

An engineering-focused learning project that asks: **how far can you get on a real game with zero backpropagation?**

The answer: surprisingly far. A Hebbian distributional agent with engineered features reaches 61% win rate against a rule-based opponent — matching PPO (59%) at 500k training steps. A fully biologically plausible variant using Predictive Coding for feature learning (zero backprop anywhere in the pipeline) reaches 57%.

Self-play training works in principle (the agent generates its own curriculum) but is unstable due to the plasticity-stability dilemma: the agent periodically discovers strong strategies (peak 83% win rate) but cannot reliably retain them without catastrophic forgetting.

## What This Project Is Not

This is not a research paper. The results are expected (backprop-free agents struggle with non-stationary learning), and the engineered features do most of the heavy lifting. The value lies in the implementation: PPO from scratch, a custom physics-based environment, Hebbian distributional value learning, Predictive Coding feature extraction, and self-play — all in ~1500 lines of readable PyTorch.

## Results

### PPO vs Hebbian vs BioAgent (easy opponent, 500k steps)

| Agent | Backprop | Win Rate | What It Shows |
|---|---|---|---|
| PPO | Yes (end-to-end) | 59% | Standard baseline |
| Hebbian (engineered features) | No | **61%** | Local Hebbian updates + good features ≥ PPO |
| BioAgent (PC + engineered) | No (zero backprop) | **57%** | Learned + handcrafted features ≈ PPO |
| BioAgent (PC only) | No (zero backprop) | 20% | PC features alone are insufficient |

### Self-Play Results (1M steps)

| Metric | Value |
|---|---|
| Self-play win rate | ~53% (balanced) |
| Peak vs Easy AI | 83% |
| Final vs Easy AI | 24–34% |
| Stability | Low — performance oscillates |

### Key Findings

**Engineered features matter more than the learning algorithm.** The Hebbian agent with handcrafted features (ball-paddle distance, time-to-impact, predicted intercept) outperforms PPO, which must learn its own features from raw observations. This suggests that in biological systems, innate sensory processing (evolved feature extraction) is critical for efficient learning.

**Predictive Coding features add value but don't replace engineering.** The PC encoder learns to predict the next observation using local Hebbian updates, and these learned representations improve performance when combined with engineered features (57% vs 44% without PC). However, PC features alone (20%) are far below the engineered baseline — the encoder learns features useful for prediction, not necessarily for control.

**Self-play creates automatic curriculum but exposes stability issues.** The self-play win rate converges to ~50% (balanced games), confirming that the opponent tracks the agent's skill. But absolute performance against fixed opponents is unstable — the agent discovers good strategies transiently but overwrites them as the opponent distribution shifts. This is the plasticity-stability dilemma, a fundamental challenge for biological and artificial learning systems alike.

## Architecture

```
Observation (ball_x, ball_y, ball_vx, ball_vy, paddle_y, opponent_y)
        │
        ├──→ PC Encoder (local Hebbian) ──→ learned features (64-dim)
        │
        ├──→ Engineered Features ──→ handcrafted features (17-dim)
        │
        └──→ [concatenated] ──→ Hebbian Value Population
                                   │
                                   ├── 32 neurons per action
                                   ├── each with asymmetric α⁺/α⁻
                                   ├── learns distributional value (quantiles)
                                   └── population readout → action selection
```

### Components

**Custom Pong Environment** (`envs/pong.py`). Lightweight 2D Pong with continuous physics, configurable opponent AI, and vectorized parallel execution for PPO. No external dependencies.

**PPO from Scratch** (`agents/ppo.py`). Proximal Policy Optimization with GAE, clipped surrogate objective, entropy bonus, value function clipping, orthogonal initialization, and minibatch updates. ~300 lines.

**Hebbian Distributional Agent** (`agents/hebbian_pong.py`). Population of value-estimating neurons with asymmetric learning rates (α⁺ ≠ α⁻), inspired by Dabney et al. (2020)'s finding that dopamine neurons encode distributional reward predictions. Supports random, engineered, or hybrid feature modes.

**Predictive Coding Encoder** (`agents/pc_encoder.py`). Two-layer encoder-predictor network trained to predict the next observation from the current one, using only local Hebbian updates. No backpropagation. Features are extracted from the encoder's hidden layer.

**BioAgent** (`agents/bio_agent.py`). Integrates PC encoder + Hebbian value learning into a single agent with zero backpropagation. Includes PC warmup, feature freezing, eligibility traces, and synaptic consolidation (weight protection for stable synapses).

**Self-Play Environment** (`envs/self_play.py`). The agent plays against a frozen copy of itself, periodically updated. Maintains a pool of past opponents for training diversity.

## Usage

```bash
pip install torch numpy matplotlib scipy

# PPO vs Hebbian comparison (easy/medium/hard opponent)
python experiments/compare_agents.py --difficulty easy --steps 500000

# Full comparison including BioAgent
python experiments/full_comparison.py --difficulty easy --steps 500000

# Self-play training
python experiments/train_self_play.py --steps 1000000
```

## Project Structure

```
├── envs/
│   ├── pong.py              # Custom Pong environment + vectorized version
│   └── self_play.py         # Self-play wrapper
├── agents/
│   ├── ppo.py               # PPO from scratch
│   ├── hebbian_pong.py      # Hebbian distributional agent
│   ├── pc_encoder.py        # Predictive Coding feature encoder
│   └── bio_agent.py         # Full bio-plausible agent (PC + Hebbian)
├── experiments/
│   ├── compare_agents.py    # PPO vs Hebbian comparison
│   ├── full_comparison.py   # All agents including BioAgent
│   ├── train_ppo.py         # PPO-only training
│   └── train_self_play.py   # Self-play training
└── results/                 # Generated plots and summaries
```

## Connections to Neuroscience

- **Distributional dopamine hypothesis** (Dabney et al., Nature 2020): Individual dopamine neurons encode different quantiles of the reward distribution via asymmetric learning rates. Our Hebbian value neurons implement this directly.
- **Predictive Coding** (Rao & Ballard 1999): Cortical circuits learn by predicting sensory input and updating based on prediction errors. Our PC encoder uses this principle for feature learning.
- **DishBrain** (Kagan et al. 2022): Biological neurons in vitro learned to play Pong via prediction-error-based stimulation. Our system uses the same principle (sensory stimulation → prediction error → local plasticity) but in silico.
- **Synaptic consolidation** (Kirkpatrick et al. 2017): Important synapses become resistant to change, preventing catastrophic forgetting. Our consolidation mechanism implements a biologically plausible version of this.

## Limitations & Future Work

- Engineered features are the primary driver of performance — the learned PC features add marginal value.
- Self-play is unstable without mechanisms like replay buffers or target networks, which lack clear biological analogues.
- The PC encoder's prediction error doesn't converge well on chaotic Pong dynamics — a contrastive or slowness-based objective might work better.
- Not benchmarked against standard Atari Pong (would require frame-based observations and convolutional architectures).

## Hardware

All experiments run on CPU (Ryzen 5 3600). No GPU required. Training times: ~15 min for 500k steps (single agent), ~60 min for full comparison, ~90 min for 1M self-play steps.

## Author

Nils Leutenegger — independent NeuroAI researcher, ETH Zürich Computer Science (starting autumn 2026).
