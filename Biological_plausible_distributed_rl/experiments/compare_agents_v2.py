"""
PPO vs Hebbian Distributional Agent on Pong.

The core comparison:
- PPO: backprop, actor-critic, GAE — standard RL
- Hebbian: local updates, asymmetric plasticity, no backprop — biologically plausible

We don't expect Hebbian to beat PPO. We expect it to LEARN, and then
we analyze HOW its representations differ.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys
import json
import argparse

matplotlib.use("Agg")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# Support both flat and nested directory structures
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from envs.pong import PongEnv, VectorPongEnv, PongConfig
    from agents.ppo import PPOAgent, PPOConfig
    from agents.hebbian_pong import HebbianPongAgent, HebbianPongConfig
except ModuleNotFoundError:
    from pong import PongEnv, VectorPongEnv, PongConfig
    from ppo import PPOAgent, PPOConfig
    from hebbian_pong import HebbianPongAgent, HebbianPongConfig


def evaluate_agent(agent, pong_config, n_episodes=50, agent_type="ppo"):
    """Evaluate any agent."""
    env = PongEnv(pong_config)
    wins, losses, timeouts = 0, 0, 0
    rallies = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        while True:
            if agent_type == "ppo":
                action = agent.select_action(obs, greedy=True)
            else:
                action = agent.select_action(obs, greedy=True)
            obs, _, done, info = env.step(action)
            if done:
                if info["event"] == "win": wins += 1
                elif info["event"] == "lose": losses += 1
                else: timeouts += 1
                rallies.append(info["rally_length"])
                break
    
    return {
        "win_rate": wins / n_episodes,
        "loss_rate": losses / n_episodes,
        "avg_rally": np.mean(rallies),
    }


def main():
    parser = argparse.ArgumentParser(description="PPO vs Hebbian on Pong")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"],
                        help="Opponent difficulty (default: easy)")
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Total timesteps (default: 500000)")
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    save_dir = Path(__file__).resolve().parent.parent / "results" / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Opponent difficulty presets
    difficulty_configs = {
        "easy": PongConfig(
            opponent_skill=0.4, opponent_speed=0.02,
            opponent_reaction_delay=6, max_steps=500, reward_hit=0.1,
        ),
        "medium": PongConfig(
            opponent_skill=0.6, opponent_speed=0.025,
            opponent_reaction_delay=4, max_steps=500, reward_hit=0.1,
        ),
        "hard": PongConfig(
            opponent_skill=0.8, opponent_speed=0.035,
            opponent_reaction_delay=2, max_steps=500, reward_hit=0.1,
        ),
    }
    
    pong_config = difficulty_configs[args.difficulty]
    N_TIMESTEPS = args.steps
    
    print("=" * 60)
    print(f"PPO vs Hebbian on Pong ({args.difficulty} opponent, {N_TIMESTEPS:,} steps)")
    print("=" * 60)
    
    # ================================================================
    # Train PPO
    # ================================================================
    print("\n--- Training PPO ---")
    ppo_config = PPOConfig(
        obs_dim=6, n_actions=3, hidden_dim=64, n_hidden=2,
        lr=3e-4, n_envs=8, n_steps=128, n_epochs=4,
        n_minibatches=4, total_timesteps=N_TIMESTEPS, log_interval=10,
    )
    vec_env = VectorPongEnv(n_envs=8, config=pong_config)
    ppo_agent = PPOAgent(ppo_config)
    ppo_log = ppo_agent.train(vec_env)
    
    ppo_eval = evaluate_agent(ppo_agent, pong_config, n_episodes=100)
    print(f"  PPO eval: win={ppo_eval['win_rate']:.0%}, rally={ppo_eval['avg_rally']:.1f}")
    
    # ================================================================
    # Train Hebbian (with sweep over feature modes and lr)
    # ================================================================
    print("\n--- Training Hebbian variants ---")
    
    hebb_configs = {
        "Hebbian\n(random feat)": HebbianPongConfig(
            obs_dim=6, n_actions=3, feature_dim=128, feature_mode="random",
            n_neurons=32, base_lr=0.002, gamma=0.99, temperature=0.5,
            epsilon=0.2, epsilon_decay=0.99998, epsilon_min=0.03,
            trace_decay=0.7, use_traces=True,
            total_timesteps=N_TIMESTEPS, log_interval=2000,
        ),
        "Hebbian\n(engineered)": HebbianPongConfig(
            obs_dim=6, n_actions=3, feature_dim=128, feature_mode="engineered",
            n_neurons=32, base_lr=0.005, gamma=0.99, temperature=0.5,
            epsilon=0.2, epsilon_decay=0.99998, epsilon_min=0.03,
            trace_decay=0.7, use_traces=True,
            total_timesteps=N_TIMESTEPS, log_interval=2000,
        ),
        "Hebbian\n(eng+random)": HebbianPongConfig(
            obs_dim=6, n_actions=3, feature_dim=64, feature_mode="both",
            n_neurons=32, base_lr=0.003, gamma=0.99, temperature=0.5,
            epsilon=0.2, epsilon_decay=0.99998, epsilon_min=0.03,
            trace_decay=0.7, use_traces=True,
            total_timesteps=N_TIMESTEPS, log_interval=2000,
        ),
    }
    
    hebb_results = {}
    for label, hc in hebb_configs.items():
        print(f"\n  {label}:")
        env = PongEnv(pong_config)
        agent = HebbianPongAgent(hc)
        agent.train(env, N_TIMESTEPS)
        
        ev = evaluate_agent(agent, pong_config, n_episodes=100, agent_type="hebbian")
        hebb_results[label] = {"log": agent.log, "eval": ev, "agent": agent}
        print(f"    Eval: win={ev['win_rate']:.0%}, rally={ev['avg_rally']:.1f}")
    
    # Pick best Hebbian for the main comparison
    best_hebb_label = max(hebb_results, key=lambda k: hebb_results[k]["eval"]["win_rate"])
    hebb_agent = hebb_results[best_hebb_label]["agent"]
    hebb_log = hebb_results[best_hebb_label]["log"]
    hebb_eval = hebb_results[best_hebb_label]["eval"]
    print(f"\n  Best Hebbian: {best_hebb_label} (win={hebb_eval['win_rate']:.0%})")
    
    # ================================================================
    # Comparison Plots
    # ================================================================
    print("\n--- Plotting ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Win rate comparison ---
    ax = axes[0, 0]
    
    # PPO win rate
    ppo_steps = [l["total_steps"] for l in ppo_log]
    ppo_wr = [l["win_rate"] for l in ppo_log]
    window = 10
    if len(ppo_wr) > window:
        ppo_smooth = np.convolve(ppo_wr, np.ones(window)/window, mode="valid")
        ppo_steps_smooth = ppo_steps[window-1:]
        ax.plot(ppo_steps_smooth, ppo_smooth, label="PPO", linewidth=2, color="#228833")
    
    # All Hebbian variants
    hebb_colors = ["#EE6677", "#4477AA", "#CCBB44"]
    for idx, (label, res) in enumerate(hebb_results.items()):
        if res["log"]:
            steps = [l["step"] for l in res["log"]]
            wr = [l["win_rate"] for l in res["log"]]
            ax.plot(steps, wr, label=label, linewidth=2, 
                    color=hebb_colors[idx % len(hebb_colors)], alpha=0.8)
    
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Win Rate")
    ax.set_title("Learning Curves")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # --- Final eval comparison ---
    ax = axes[0, 1]
    all_labels = ["PPO"] + list(hebb_results.keys())
    all_win_rates = [ppo_eval["win_rate"]] + [r["eval"]["win_rate"] for r in hebb_results.values()]
    all_rallies = [ppo_eval["avg_rally"]] + [r["eval"]["avg_rally"] for r in hebb_results.values()]
    bar_colors = ["#228833"] + hebb_colors[:len(hebb_results)]
    
    x = np.arange(len(all_labels))
    bars = ax.bar(x, all_win_rates, color=bar_colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=7, rotation=15)
    ax.set_ylabel("Win Rate")
    ax.set_title("Evaluation Win Rates")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars, all_win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.0%}", ha="center", fontsize=8)
    
    # --- Hebbian distributional representation ---
    ax = axes[1, 0]
    
    # Get learned distributions at a few different game states
    test_states = {
        "Ball center": np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0]),
        "Ball coming": np.array([0.5, 0.3, 0.5, 0.1, 0.3, -0.2]),
        "Ball away": np.array([-0.5, -0.2, -0.5, 0.0, 0.0, -0.1]),
    }
    
    colors_per_action = ["#228833", "#EE6677", "#4477AA"]
    action_names = ["Stay", "Up", "Down"]
    
    for state_label, state in test_states.items():
        taus, values = hebb_agent.get_learned_distributions(state)
        # Plot action 1 (up) distribution for each state
        ax.plot(taus.numpy(), values[1].numpy(),
                label=f"{state_label}", linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel("Quantile τ")
    ax.set_ylabel("Value")
    ax.set_title("Hebbian: Learned Value Distribution\n(Action=Up, different states)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Weight statistics over training ---
    ax = axes[1, 1]
    ws = hebb_agent.get_weight_stats()
    stats_text = "\n".join([f"{k}: {v:.4f}" for k, v in ws.items()])
    ax.text(0.1, 0.5, f"Best Hebbian: {best_hebb_label}\n\n"
            f"Weight Stats:\n{stats_text}\n\n"
            f"Feature dim: {hebb_agent.actual_feature_dim}\n"
            f"N neurons: {hebb_agent.config.n_neurons}\n"
            f"Base LR: {hebb_agent.config.base_lr}\n"
            f"Traces: {hebb_agent.config.use_traces}",
            fontsize=12, family="monospace", verticalalignment="center",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Hebbian Agent Configuration")
    ax.axis("off")
    
    plt.suptitle(f"PPO vs Hebbian on Custom Pong — {args.difficulty} opponent ({N_TIMESTEPS:,} steps)", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / f"01_ppo_vs_hebbian_{args.difficulty}.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved 01_ppo_vs_hebbian_{args.difficulty}.png")
    
    # Save summary
    summary = {
        "difficulty": args.difficulty,
        "timesteps": N_TIMESTEPS,
        "ppo_eval": ppo_eval,
        "hebbian_results": {k: v["eval"] for k, v in hebb_results.items()},
        "best_hebbian": best_hebb_label,
        "ppo_params": sum(p.numel() for p in ppo_agent.network.parameters()),
        "hebbian_params": hebb_agent.W_value.numel(),
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  PPO parameters:     {summary['ppo_params']:,}")
    print(f"  Hebbian parameters: {summary['hebbian_params']:,}")
    print("\nDone!")


if __name__ == "__main__":
    main()