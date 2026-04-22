"""
Full Comparison: PPO vs Hebbian (engineered) vs BioAgent (PC + Hebbian)

Three agents, one game:
1. PPO: backprop end-to-end (standard RL)
2. Hebbian (engineered): handcrafted features + local Hebbian value learning
3. BioAgent: PC-learned features + local Hebbian value learning (ZERO backprop)

Also tests BioAgent variants:
- PC only (no engineered features) — the purest biological model
- PC + engineered — hybrid for best performance
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from envs.pong import PongEnv, VectorPongEnv, PongConfig
    from agents.ppo import PPOAgent, PPOConfig
    from agents.hebbian_pong import HebbianPongAgent, HebbianPongConfig
    from agents.bio_agent import BioAgent, BioAgentConfig
except ModuleNotFoundError:
    from pong import PongEnv, VectorPongEnv, PongConfig
    from ppo import PPOAgent, PPOConfig
    from hebbian_pong import HebbianPongAgent, HebbianPongConfig
    from bio_agent import BioAgent, BioAgentConfig


def evaluate_agent(agent, pong_config, n_episodes=100):
    env = PongEnv(pong_config)
    wins = 0
    rallies = []
    for _ in range(n_episodes):
        obs = env.reset()
        while True:
            action = agent.select_action(obs, greedy=True)
            obs, _, done, info = env.step(action)
            if done:
                if info["event"] == "win":
                    wins += 1
                rallies.append(info["rally_length"])
                break
    return {"win_rate": wins / n_episodes, "avg_rally": np.mean(rallies)}


def main():
    parser = argparse.ArgumentParser(description="Full comparison: PPO vs Hebbian vs BioAgent")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--steps", type=int, default=500_000)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    save_dir = Path(__file__).resolve().parent.parent / "results" / "full_comparison"
    save_dir.mkdir(parents=True, exist_ok=True)
    
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
    N = args.steps
    
    print("=" * 60)
    print(f"Full Comparison — {args.difficulty} opponent, {N:,} steps")
    print("=" * 60)
    
    results = {}
    
    # ================================================================
    # 1. PPO
    # ================================================================
    print("\n--- PPO ---")
    ppo_config = PPOConfig(
        obs_dim=6, n_actions=3, hidden_dim=64, n_hidden=2,
        lr=3e-4, n_envs=8, n_steps=128, n_epochs=4,
        n_minibatches=4, total_timesteps=N, log_interval=10,
    )
    vec_env = VectorPongEnv(n_envs=8, config=pong_config)
    ppo = PPOAgent(ppo_config)
    ppo_log = ppo.train(vec_env)
    ppo_eval = evaluate_agent(ppo, pong_config)
    results["PPO\n(backprop)"] = {
        "log_steps": [l["total_steps"] for l in ppo_log],
        "log_wr": [l["win_rate"] for l in ppo_log],
        "eval": ppo_eval,
    }
    print(f"  Eval: {ppo_eval['win_rate']:.0%}")
    
    # ================================================================
    # 2. Hebbian (engineered features)
    # ================================================================
    print("\n--- Hebbian (engineered) ---")
    hebb = HebbianPongAgent(HebbianPongConfig(
        obs_dim=6, n_actions=3, feature_dim=128, feature_mode="engineered",
        n_neurons=32, base_lr=0.005, gamma=0.99, temperature=0.5,
        epsilon=0.2, epsilon_decay=0.99998, epsilon_min=0.03,
        trace_decay=0.7, use_traces=True,
        total_timesteps=N, log_interval=2000,
    ))
    env = PongEnv(pong_config)
    hebb.train(env, N)
    hebb_eval = evaluate_agent(hebb, pong_config)
    results["Hebbian\n(engineered)"] = {
        "log_steps": [l["step"] for l in hebb.log],
        "log_wr": [l["win_rate"] for l in hebb.log],
        "eval": hebb_eval,
    }
    print(f"  Eval: {hebb_eval['win_rate']:.0%}")
    
    # ================================================================
    # 3. BioAgent: PC + engineered + Hebbian
    # ================================================================
    print("\n--- BioAgent (PC + engineered) ---")
    bio_hybrid = BioAgent(BioAgentConfig(
        obs_dim=6, n_actions=3,
        pc_hidden_dim=64, pc_feature_dim=32,
        pc_lr_encoder=0.001, pc_lr_predictor=0.001,
        use_engineered=True,
        n_neurons=32, base_lr=0.005, gamma=0.99,
        temperature=0.5, epsilon=0.2, epsilon_decay=0.99998, epsilon_min=0.03,
        trace_decay=0.7, use_traces=True,
        pc_warmup_steps=20000,
        total_timesteps=N, log_interval=2000,
    ))
    env = PongEnv(pong_config)
    bio_hybrid.train(env, N)
    bio_hybrid_eval = evaluate_agent(bio_hybrid, pong_config)
    results["BioAgent\n(PC+eng)"] = {
        "log_steps": [l["step"] for l in bio_hybrid.log],
        "log_wr": [l["win_rate"] for l in bio_hybrid.log],
        "eval": bio_hybrid_eval,
        "pc_errors": [l["pc_pred_error"] for l in bio_hybrid.log],
    }
    print(f"  Eval: {bio_hybrid_eval['win_rate']:.0%}")
    
    # ================================================================
    # 4. BioAgent: PC only (PURE biological — no engineered features)
    # ================================================================
    print("\n--- BioAgent (PC only — pure bio) ---")
    bio_pure = BioAgent(BioAgentConfig(
        obs_dim=6, n_actions=3,
        pc_hidden_dim=64, pc_feature_dim=32,
        pc_lr_encoder=0.001, pc_lr_predictor=0.001,
        use_engineered=False,  # <-- NO engineered features
        n_neurons=32, base_lr=0.005, gamma=0.99,
        temperature=0.5, epsilon=0.2, epsilon_decay=0.99998, epsilon_min=0.03,
        trace_decay=0.7, use_traces=True,
        pc_warmup_steps=20000,
        total_timesteps=N, log_interval=2000,
    ))
    env = PongEnv(pong_config)
    bio_pure.train(env, N)
    bio_pure_eval = evaluate_agent(bio_pure, pong_config)
    results["BioAgent\n(PC only)"] = {
        "log_steps": [l["step"] for l in bio_pure.log],
        "log_wr": [l["win_rate"] for l in bio_pure.log],
        "eval": bio_pure_eval,
        "pc_errors": [l["pc_pred_error"] for l in bio_pure.log],
    }
    print(f"  Eval: {bio_pure_eval['win_rate']:.0%}")
    
    # ================================================================
    # Plots
    # ================================================================
    print("\n--- Plotting ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {"PPO\n(backprop)": "#228833", "Hebbian\n(engineered)": "#4477AA",
              "BioAgent\n(PC+eng)": "#EE6677", "BioAgent\n(PC only)": "#CCBB44"}
    
    # --- Learning curves ---
    ax = axes[0, 0]
    for label, res in results.items():
        steps = res["log_steps"]
        wr = res["log_wr"]
        if len(wr) > 10:
            w = 10
            smooth = np.convolve(wr, np.ones(w)/w, mode="valid")
            ax.plot(steps[w-1:], smooth, label=label, linewidth=2,
                    color=colors.get(label, "gray"), alpha=0.8)
        else:
            ax.plot(steps, wr, label=label, linewidth=2,
                    color=colors.get(label, "gray"), alpha=0.8)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Win Rate")
    ax.set_title("Learning Curves")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # --- Eval bar chart ---
    ax = axes[0, 1]
    labels = list(results.keys())
    win_rates = [results[l]["eval"]["win_rate"] for l in labels]
    bar_colors = [colors.get(l, "gray") for l in labels]
    bars = ax.bar(range(len(labels)), win_rates, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Win Rate")
    ax.set_title("Evaluation Win Rates")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.0%}", ha="center", fontsize=9)
    
    # --- PC prediction error ---
    ax = axes[1, 0]
    for label in ["BioAgent\n(PC+eng)", "BioAgent\n(PC only)"]:
        if label in results and "pc_errors" in results[label]:
            steps = results[label]["log_steps"]
            errors = results[label]["pc_errors"]
            ax.plot(steps, errors, label=label, linewidth=2,
                    color=colors.get(label, "gray"))
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Prediction Error (MSE)")
    ax.set_title("PC Encoder: Prediction Error Over Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Summary table ---
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for label in labels:
        ev = results[label]["eval"]
        table_data.append([
            label.replace("\n", " "),
            f"{ev['win_rate']:.0%}",
            f"{ev['avg_rally']:.1f}",
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Agent", "Win Rate", "Avg Rally"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    ax.set_title(f"Results Summary — {args.difficulty} opponent", pad=20)
    
    plt.suptitle(
        f"PPO vs Hebbian vs BioAgent (PC) — {args.difficulty} opponent ({N:,} steps)\n"
        f"BioAgent uses ZERO backpropagation",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(save_dir / f"full_comparison_{args.difficulty}.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved full_comparison_{args.difficulty}.png")
    
    # Save JSON
    summary = {
        "difficulty": args.difficulty,
        "steps": N,
        "results": {k.replace("\n", " "): v["eval"] for k, v in results.items()},
    }
    with open(save_dir / f"summary_{args.difficulty}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()