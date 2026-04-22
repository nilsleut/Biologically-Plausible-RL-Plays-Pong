"""
Self-Play Training: BioAgent learns Pong by playing against itself.

The complete biologically plausible pipeline:
1. Predictive Coding for feature learning (local Hebbian)
2. Distributional Hebbian for value learning (local asymmetric)
3. Self-Play for automatic curriculum (no fixed opponent)

ZERO backpropagation. The opponent is a frozen copy, periodically
updated to the current agent.

Usage:
    python experiments/train_self_play.py --steps 1000000
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
from pathlib import Path
import sys
import json
import argparse

matplotlib.use("Agg")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from envs.self_play import SelfPlayPongEnv, SelfPlayConfig
    from envs.pong import PongEnv, PongConfig
    from agents.bio_agent import BioAgent, BioAgentConfig
    from agents.hebbian_pong import HebbianPongAgent, HebbianPongConfig
except ModuleNotFoundError:
    from self_play import SelfPlayPongEnv, SelfPlayConfig
    from pong import PongEnv, PongConfig
    from bio_agent import BioAgent, BioAgentConfig
    from hebbian_pong import HebbianPongAgent, HebbianPongConfig


def evaluate_vs_fixed(agent, difficulty="easy", n_episodes=100):
    """Evaluate agent against fixed AI opponents."""
    configs = {
        "easy": PongConfig(opponent_skill=0.4, opponent_speed=0.02, opponent_reaction_delay=6, max_steps=500, reward_hit=0.1),
        "medium": PongConfig(opponent_skill=0.6, opponent_speed=0.025, opponent_reaction_delay=4, max_steps=500, reward_hit=0.1),
        "hard": PongConfig(opponent_skill=0.8, opponent_speed=0.035, opponent_reaction_delay=2, max_steps=500, reward_hit=0.1),
    }
    env = PongEnv(configs[difficulty])
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
    parser = argparse.ArgumentParser(description="Self-Play BioAgent on Pong")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--update_interval", type=int, default=50,
                        help="Episodes between opponent updates")
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    save_dir = Path(__file__).resolve().parent.parent / "results" / "self_play"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    N = args.steps
    
    print("=" * 60)
    print(f"Self-Play BioAgent Training ({N:,} steps)")
    print("Zero backpropagation — PC + Hebbian + Self-Play")
    print("=" * 60)
    
    # --- Create BioAgent ---
    agent = BioAgent(BioAgentConfig(
        obs_dim=6, n_actions=3,
        pc_hidden_dim=64, pc_feature_dim=64,
        pc_lr_encoder=0.001, pc_lr_predictor=0.001,
        pc_freeze_after=50000,
        use_engineered=True,
        n_neurons=32, base_lr=0.005, gamma=0.99,
        temperature=0.5, epsilon=0.3,
        epsilon_decay=0.99999, epsilon_min=0.05,
        trace_decay=0.7, use_traces=True,
        pc_warmup_steps=20000,
        use_consolidation=True,              # NEW: synaptic consolidation
        consolidation_strength=0.1,
        consolidation_ema=0.999,
        total_timesteps=N, log_interval=2000,
    ))
    
    # --- Create Self-Play environment ---
    sp_config = SelfPlayConfig(
        opponent_update_interval=200,               # Fix 2: slower opponent updates (50→200)
        opponent_pool_size=5,
        past_opponent_prob=0.2,
    )
    env = SelfPlayPongEnv(sp_config)
    
    # --- Initialize opponent with random agent ---
    initial_opponent = copy.deepcopy(agent)
    env.update_opponent_pool(initial_opponent)
    
    # --- Training loop ---
    print(f"\n--- Training ---")
    
    obs = env.reset()
    episode_count = 0
    wins, losses = 0, 0
    log = []
    eval_log = []
    
    # Track per-episode for opponent updates
    episode_in_batch = 0
    
    for step in range(1, N + 1):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.update(next_obs, reward, done)
        
        if done:
            episode_count += 1
            episode_in_batch += 1
            if info.get("event") == "win":
                wins += 1
            elif info.get("event") == "lose":
                losses += 1
            
            # Update opponent periodically
            if episode_in_batch >= sp_config.opponent_update_interval:
                env.update_opponent_pool(copy.deepcopy(agent))
                episode_in_batch = 0
            
            obs = env.reset()
            agent.prev_obs = None
            agent.prev_features = None
            agent.prev_action = None
        else:
            obs = next_obs
        
        # Logging
        if step % 2000 == 0:
            sp_wr = wins / max(1, episode_count)
            pc_error = agent.pc_encoder.get_prediction_error_history()
            
            log_entry = {
                "step": step,
                "episodes": episode_count,
                "self_play_win_rate": sp_wr,
                "pc_pred_error": pc_error,
                "epsilon": agent.epsilon,
                "opponent_pool_size": len(env.opponent_pool),
            }
            log.append(log_entry)
        
        # Periodic evaluation against fixed opponents
        if step % 20000 == 0:
            eval_easy = evaluate_vs_fixed(agent, "easy")
            eval_medium = evaluate_vs_fixed(agent, "medium")
            eval_hard = evaluate_vs_fixed(agent, "hard")
            
            eval_entry = {
                "step": step,
                "easy": eval_easy["win_rate"],
                "medium": eval_medium["win_rate"],
                "hard": eval_hard["win_rate"],
            }
            eval_log.append(eval_entry)
            
            # Update consolidation anchor if this is the best performance
            combined_score = eval_easy["win_rate"] + 0.5 * eval_medium["win_rate"]
            agent.update_anchor(combined_score)
            
            print(
                f"  Step {step:,}/{N:,} | "
                f"Self-play WR: {sp_wr:.1%} | "
                f"vs Easy: {eval_easy['win_rate']:.0%} | "
                f"vs Med: {eval_medium['win_rate']:.0%} | "
                f"vs Hard: {eval_hard['win_rate']:.0%} | "
                f"PC err: {pc_error:.4f} | "
                f"ε: {agent.epsilon:.3f}"
            )
    
    # --- Final evaluation ---
    print("\n--- Final Evaluation ---")
    final_easy = evaluate_vs_fixed(agent, "easy")
    final_medium = evaluate_vs_fixed(agent, "medium")
    final_hard = evaluate_vs_fixed(agent, "hard")
    
    print(f"  vs Easy:   {final_easy['win_rate']:.0%} (rally: {final_easy['avg_rally']:.1f})")
    print(f"  vs Medium: {final_medium['win_rate']:.0%} (rally: {final_medium['avg_rally']:.1f})")
    print(f"  vs Hard:   {final_hard['win_rate']:.0%} (rally: {final_hard['avg_rally']:.1f})")
    
    # --- Plots ---
    print("\n--- Plotting ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Self-play win rate
    ax = axes[0, 0]
    if log:
        steps = [l["step"] for l in log]
        sp_wr = [l["self_play_win_rate"] for l in log]
        ax.plot(steps, sp_wr, linewidth=2, color="#228833", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, label="50% (balanced)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Win Rate")
    ax.set_title("Self-Play Win Rate\n(should approach 50% as opponent improves)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Evaluation vs fixed opponents over training
    ax = axes[0, 1]
    if eval_log:
        eval_steps = [e["step"] for e in eval_log]
        ax.plot(eval_steps, [e["easy"] for e in eval_log], "o-", label="vs Easy", linewidth=2, color="#228833")
        ax.plot(eval_steps, [e["medium"] for e in eval_log], "s-", label="vs Medium", linewidth=2, color="#CCBB44")
        ax.plot(eval_steps, [e["hard"] for e in eval_log], "^-", label="vs Hard", linewidth=2, color="#EE6677")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Win Rate vs Fixed AI")
    ax.set_title("Evaluation Against Fixed Opponents\n(measures absolute skill improvement)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # PC prediction error
    ax = axes[1, 0]
    if log:
        ax.plot([l["step"] for l in log], [l["pc_pred_error"] for l in log],
                linewidth=1.5, color="#EE6677", alpha=0.8)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Prediction Error (MSE)")
    ax.set_title("PC Encoder Learning")
    ax.grid(True, alpha=0.3)
    
    # Final results summary
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        ["vs Easy", f"{final_easy['win_rate']:.0%}", f"{final_easy['avg_rally']:.1f}"],
        ["vs Medium", f"{final_medium['win_rate']:.0%}", f"{final_medium['avg_rally']:.1f}"],
        ["vs Hard", f"{final_hard['win_rate']:.0%}", f"{final_hard['avg_rally']:.1f}"],
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Opponent", "Win Rate", "Avg Rally"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    ax.set_title("Final Evaluation — Self-Play Trained BioAgent\n(PC + Hebbian, ZERO backprop)", pad=20)
    
    plt.suptitle(f"Self-Play BioAgent — {N:,} steps, zero backpropagation", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / "self_play_results.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved self_play_results.png")
    
    # Save JSON
    summary = {
        "steps": N,
        "final_eval": {
            "easy": final_easy,
            "medium": final_medium,
            "hard": final_hard,
        },
        "opponent_updates": len(env.opponent_pool),
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()