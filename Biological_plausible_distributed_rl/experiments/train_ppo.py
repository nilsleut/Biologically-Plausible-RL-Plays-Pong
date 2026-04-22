"""
Train PPO on Custom Pong and evaluate.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys
import json
from datetime import datetime

matplotlib.use("Agg")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.pong import PongEnv, VectorPongEnv, PongConfig
from agents.ppo import PPOAgent, PPOConfig


def evaluate(agent, config: PongConfig, n_episodes: int = 50) -> dict:
    """Evaluate agent with greedy policy."""
    env = PongEnv(config)
    wins, losses, timeouts = 0, 0, 0
    rally_lengths = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        while True:
            action = agent.select_action(obs, greedy=True)
            obs, reward, done, info = env.step(action)
            if done:
                if info["event"] == "win":
                    wins += 1
                elif info["event"] == "lose":
                    losses += 1
                else:
                    timeouts += 1
                rally_lengths.append(info["rally_length"])
                break
    
    return {
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "avg_rally": np.mean(rally_lengths) if rally_lengths else 0,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    save_dir = Path("results/ppo_pong")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PPO on Custom Pong")
    print("=" * 60)
    
    # Environment config
    pong_config = PongConfig(
        opponent_skill=0.5,
        opponent_speed=0.025,
        opponent_reaction_delay=5,
        max_steps=500,
        reward_hit=0.1,
    )
    
    # Test environment works
    print("\n--- Testing environment ---")
    env = PongEnv(pong_config)
    obs = env.reset()
    print(f"  Obs shape: {obs.shape}, sample: {obs.round(3)}")
    
    total_r = 0
    for _ in range(100):
        obs, r, done, info = env.step(np.random.randint(3))
        total_r += r
        if done:
            print(f"  Random agent episode: {info['event']}, "
                  f"rally={info['rally_length']}, steps={info['step']}")
            break
    
    # PPO config
    ppo_config = PPOConfig(
        obs_dim=6,
        n_actions=3,
        hidden_dim=64,
        n_hidden=2,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        n_envs=8,
        n_steps=128,
        n_epochs=4,
        n_minibatches=4,
        total_timesteps=100_000,  # Quick test; use 500k+ locally
        log_interval=10,
    )
    
    # Train
    print(f"\n--- Training PPO ({ppo_config.total_timesteps:,} steps) ---")
    vec_env = VectorPongEnv(n_envs=ppo_config.n_envs, config=pong_config)
    agent = PPOAgent(ppo_config)
    
    log = agent.train(vec_env)
    
    # Evaluate
    print("\n--- Evaluation ---")
    eval_results = evaluate(agent, pong_config, n_episodes=100)
    print(f"  Win rate:  {eval_results['win_rate']:.1%}")
    print(f"  Loss rate: {eval_results['loss_rate']:.1%}")
    print(f"  Timeouts:  {eval_results['timeout_rate']:.1%}")
    print(f"  Avg rally: {eval_results['avg_rally']:.1f}")
    
    # Evaluate against harder opponent
    hard_config = PongConfig(
        opponent_skill=0.8,
        opponent_speed=0.035,
        opponent_reaction_delay=2,
        max_steps=500,
        reward_hit=0.1,
    )
    hard_results = evaluate(agent, hard_config, n_episodes=100)
    print(f"\n  vs Hard opponent:")
    print(f"  Win rate:  {hard_results['win_rate']:.1%}")
    print(f"  Avg rally: {hard_results['avg_rally']:.1f}")
    
    # --- Plots ---
    print("\n--- Plotting ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    updates = [l["update"] for l in log]
    
    # Win rate over training
    ax = axes[0, 0]
    win_rates = [l["win_rate"] for l in log]
    window = 10
    if len(win_rates) > window:
        smoothed = np.convolve(win_rates, np.ones(window)/window, mode="valid")
        ax.plot(smoothed, linewidth=2, color="#228833")
        ax.plot(win_rates, alpha=0.2, color="#228833")
    else:
        ax.plot(win_rates, linewidth=2, color="#228833")
    ax.set_xlabel("Update")
    ax.set_ylabel("Win Rate")
    ax.set_title("PPO Win Rate vs Opponent (skill=0.5)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Policy and value loss
    ax = axes[0, 1]
    ax.plot([l["policy_loss"] for l in log], label="Policy loss", alpha=0.8)
    ax.plot([l["value_loss"] for l in log], label="Value loss", alpha=0.8)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Entropy
    ax = axes[1, 0]
    ax.plot([l["entropy"] for l in log], color="#EE6677", linewidth=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy (exploration)")
    ax.grid(True, alpha=0.3)
    
    # Clip fraction
    ax = axes[1, 1]
    ax.plot([l["clipfrac"] for l in log], color="#4477AA", linewidth=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Clip Fraction")
    ax.set_title("PPO Clip Fraction")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"PPO Training on Custom Pong ({ppo_config.total_timesteps:,} steps)", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / "01_ppo_training.png", bbox_inches="tight")
    plt.close()
    print("  Saved 01_ppo_training.png")
    
    # Save model
    agent.save(str(save_dir / "ppo_pong.pt"))
    print(f"  Saved model to {save_dir / 'ppo_pong.pt'}")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_timesteps": ppo_config.total_timesteps,
        "eval_win_rate": eval_results["win_rate"],
        "eval_vs_hard": hard_results["win_rate"],
        "eval_avg_rally": eval_results["avg_rally"],
        "final_entropy": log[-1]["entropy"],
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
