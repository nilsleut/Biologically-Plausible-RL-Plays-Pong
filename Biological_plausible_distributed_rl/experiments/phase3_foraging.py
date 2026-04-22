"""
Phase 3: Risk-Sensitive Foraging and Gambling Tasks.

Key experiments:
1. Foraging survival: does adaptive risk readout improve survival rate?
2. State-dependent policy: does the agent switch from safe→risky as energy rises?
3. Gambling psychometrics: do risk preferences match experimental animal data?
4. Ablation: adaptive risk vs fixed risk vs scalar TD
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys
from collections import defaultdict

matplotlib.use("Agg")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.foraging import ForagingEnv, ForagingConfig, GamblingEnv
from agents.stateful_agents import StatefulScalarTD, StatefulQR, StatefulHebbian


def run_foraging_episode(env: ForagingEnv, agent, train: bool = True):
    """Run one episode, return metrics."""
    state = env.reset()
    total_reward = 0.0
    
    while True:
        action = agent.select_action(state, greedy=not train)
        reward, done, info = env.step(action)
        if train:
            agent.update(state, action, reward)
        total_reward += reward
        state = env.get_state()
        if done:
            break
    
    return {
        "survived": info["cause"] == "survived",
        "total_reward": total_reward,
        "steps": env.step_count,
        "final_energy": env.energy,
        "energy_history": list(env.energy_history),
        "action_history": list(env.action_history),
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    save_dir = Path("results/phase3")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    N_EPISODES = 500
    N_EVAL = 100
    
    print("=" * 60)
    print("Phase 3: Risk-Sensitive Foraging & Gambling")
    print("=" * 60)
    
    # ================================================================
    # Exp 1: Foraging Survival Rates
    # ================================================================
    print("\n--- Exp 1: Foraging Survival ---")
    
    env_config = ForagingConfig(
        n_patches=3, max_energy=15.0, start_energy=5.0,
        energy_cost=1.8, max_steps=200,
        patch_means=(2.0, 2.0, 2.0),
        patch_stds=(0.1, 1.5, 4.0),
        patch_names=("Safe", "Medium", "Risky"),
    )
    
    agents = {
        "Scalar TD": lambda: StatefulScalarTD(n_actions=3, n_bins=5, lr=0.1),
        "QR neutral": lambda: StatefulQR(n_actions=3, n_bins=5, risk_measure="neutral"),
        "QR risk-averse\n(fixed CVaR 0.25)": lambda: StatefulQR(
            n_actions=3, n_bins=5, risk_measure="averse", risk_param=0.25
        ),
        "Hebbian\n(adaptive risk)": lambda: StatefulHebbian(
            n_actions=3, n_bins=5, adaptive_risk=True,
            cvar_alpha_low=0.2, cvar_alpha_high=1.0,
        ),
        "Hebbian\n(always neutral)": lambda: StatefulHebbian(
            n_actions=3, n_bins=5, adaptive_risk=False,
        ),
    }
    
    all_metrics = {}
    
    for label, make_agent in agents.items():
        print(f"  Training {label}...")
        agent = make_agent()
        env = ForagingEnv(env_config)
        
        # Training
        train_survival = []
        for ep in range(N_EPISODES):
            result = run_foraging_episode(env, agent, train=True)
            train_survival.append(result["survived"])
        
        # Evaluation (greedy)
        eval_results = []
        for ep in range(N_EVAL):
            result = run_foraging_episode(env, agent, train=False)
            eval_results.append(result)
        
        survival_rate = np.mean([r["survived"] for r in eval_results])
        avg_reward = np.mean([r["total_reward"] for r in eval_results])
        avg_steps = np.mean([r["steps"] for r in eval_results])
        
        all_metrics[label] = {
            "survival_rate": survival_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "train_survival": train_survival,
            "eval_results": eval_results,
            "agent": agent,
        }
        
        print(f"    Survival: {survival_rate:.1%}, Reward: {avg_reward:.1f}, Steps: {avg_steps:.0f}")
    
    # --- Fig 1: Survival rate comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    labels = list(all_metrics.keys())
    rates = [all_metrics[l]["survival_rate"] for l in labels]
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]
    bars = ax.bar(range(len(labels)), rates, color=colors[:len(labels)], alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Survival Rate (eval)")
    ax.set_title("Foraging Survival Rate by Agent Type")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.0%}", ha="center", fontsize=9)
    
    # Training curve (smoothed survival)
    ax = axes[1]
    window = 50
    for idx, (label, metrics) in enumerate(all_metrics.items()):
        surv = np.array(metrics["train_survival"], dtype=float)
        smoothed = np.convolve(surv, np.ones(window)/window, mode="valid")
        ax.plot(smoothed, label=label, color=colors[idx], alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Survival Rate (rolling {window})")
    ax.set_title("Training Survival Over Time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_dir / "01_survival_rates.png", bbox_inches="tight")
    plt.close()
    print("  Saved 01_survival_rates.png")
    
    # ================================================================
    # Exp 2: State-dependent policy analysis
    # ================================================================
    print("\n--- Exp 2: State-Dependent Policy ---")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    energy_levels = np.linspace(0.05, 0.95, 50)
    patch_names = env_config.patch_names
    
    for idx, (label, metrics) in enumerate(all_metrics.items()):
        if idx >= 3:
            break  # Only plot 3 agents in top row
        
        agent = metrics["agent"]
        ax = axes[0, idx]
        
        # Compute greedy action at each energy level
        action_prefs = np.zeros((len(energy_levels), env_config.n_patches))
        
        for i, e in enumerate(energy_levels):
            state = torch.tensor([e])
            # Run multiple times to account for any stochasticity
            for _ in range(100):
                a = agent.select_action(state, greedy=True)
                action_prefs[i, a] += 1
            action_prefs[i] /= 100
        
        for a in range(env_config.n_patches):
            ax.plot(energy_levels, action_prefs[:, a],
                    label=patch_names[a], linewidth=2, alpha=0.8)
        
        ax.set_xlabel("Energy Level (normalized)")
        ax.set_ylabel("Action Probability")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axvline(0.25, color="red", linestyle=":", alpha=0.3, label="Low threshold")
    
    # Bottom row: example trajectories
    for idx, (label, metrics) in enumerate(list(all_metrics.items())[:3]):
        ax = axes[1, idx]
        # Pick a few eval episodes
        for ep_idx in range(min(5, len(metrics["eval_results"]))):
            ep = metrics["eval_results"][ep_idx]
            ax.plot(ep["energy_history"], alpha=0.4, linewidth=1)
        
        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(env_config.max_energy * 0.25, color="orange", linestyle=":", alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy")
        ax.set_title(f"{label}: Example Trajectories")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("State-Dependent Risk Preferences", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / "02_state_dependent_policy.png", bbox_inches="tight")
    plt.close()
    print("  Saved 02_state_dependent_policy.png")
    
    # ================================================================
    # Exp 3: Action distribution by energy level
    # ================================================================
    print("\n--- Exp 3: Action Distribution by Energy Level ---")
    
    fig, axes = plt.subplots(1, len(all_metrics), figsize=(4 * len(all_metrics), 4))
    
    energy_bins = ["Low\n(<25%)", "Med\n(25-60%)", "High\n(>60%)"]
    
    for idx, (label, metrics) in enumerate(all_metrics.items()):
        ax = axes[idx]
        
        # Aggregate actions by energy level from eval episodes
        action_by_level = {level: [] for level in ["low", "medium", "high"]}
        
        for ep in metrics["eval_results"]:
            for t, (energy, action) in enumerate(
                zip(ep["energy_history"][:-1], ep["action_history"])
            ):
                frac = energy / env_config.max_energy
                if frac < 0.25:
                    level = "low"
                elif frac < 0.6:
                    level = "medium"
                else:
                    level = "high"
                action_by_level[level].append(action)
        
        # Compute frequencies
        freqs = np.zeros((3, env_config.n_patches))
        for i, level in enumerate(["low", "medium", "high"]):
            if action_by_level[level]:
                actions = np.array(action_by_level[level])
                for a in range(env_config.n_patches):
                    freqs[i, a] = (actions == a).mean()
        
        x = np.arange(3)
        width = 0.25
        for a in range(env_config.n_patches):
            ax.bar(x + a * width - width, freqs[:, a], width,
                   label=patch_names[a], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(energy_bins, fontsize=8)
        ax.set_ylabel("Action Frequency")
        ax.set_title(f"{label}", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.suptitle("Action Choice by Energy Level", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_dir / "03_action_by_energy.png", bbox_inches="tight")
    plt.close()
    print("  Saved 03_action_by_energy.png")
    
    # ================================================================
    # Exp 4: Gambling psychometric curves
    # ================================================================
    print("\n--- Exp 4: Gambling Psychometrics ---")
    
    p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    N_GAMBLING_TRIALS = 500
    N_GAMBLING_RUNS = 10
    
    gambling_agents = {
        "Scalar TD": lambda: StatefulScalarTD(n_actions=2, n_bins=1, lr=0.1, epsilon=0.15, epsilon_decay=0.999, epsilon_min=0.02),
        "Hebbian neutral": lambda: StatefulHebbian(n_actions=2, n_bins=1, adaptive_risk=False),
        "Hebbian averse": lambda: StatefulHebbian(n_actions=2, n_bins=1, adaptive_risk=False, cvar_alpha_low=0.25, cvar_alpha_high=0.25),
    }
    
    # Override _get_cvar_alpha for the "averse" agent
    # (since adaptive_risk=False uses 1.0, we need a fixed-averse version)
    class FixedAverseHebbian(StatefulHebbian):
        def _get_cvar_alpha(self, state):
            return 0.25
    
    gambling_agents["Hebbian averse\n(CVaR 0.25)"] = lambda: FixedAverseHebbian(
        n_actions=2, n_bins=1, adaptive_risk=False,
    )
    # Remove the broken one
    del gambling_agents["Hebbian averse"]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for agent_label, make_agent in gambling_agents.items():
        risky_fracs = []
        
        for p in p_values:
            run_fracs = []
            for run in range(N_GAMBLING_RUNS):
                agent = make_agent()
                env = GamblingEnv.equal_ev(certain=1.0)
                # Override lottery prob
                env.config.lottery_p_high = p
                env.config.lottery_high = 1.0 / p if p > 0 else 100.0
                env.reset()
                
                state = torch.tensor([0.5])  # Dummy state
                risky_count = 0
                
                for trial in range(N_GAMBLING_TRIALS):
                    action = agent.select_action(state)
                    reward, done = env.step(action)
                    agent.update(state, action, reward)
                    if action == 1:
                        risky_count += 1
                    if done:
                        env.reset()
                
                # Use last 200 trials
                run_fracs.append(risky_count / N_GAMBLING_TRIALS)
            
            risky_fracs.append(np.mean(run_fracs))
        
        ax.plot(p_values, risky_fracs, "o-", label=agent_label, linewidth=2, markersize=5)
    
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, label="Indifference")
    ax.set_xlabel("P(high outcome)")
    ax.set_ylabel("Fraction choosing risky option")
    ax.set_title("Gambling Psychometrics: Risky Choice vs Lottery Probability\n"
                 "(Equal expected value throughout)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    fig.savefig(save_dir / "04_gambling_psychometrics.png", bbox_inches="tight")
    plt.close()
    print("  Saved 04_gambling_psychometrics.png")
    
    # ================================================================
    # Exp 5: Hebbian adaptive risk readout visualization
    # ================================================================
    print("\n--- Exp 5: Adaptive Risk Readout ---")
    
    hebb_adaptive = all_metrics["Hebbian\n(adaptive risk)"]["agent"]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: CVaR alpha as function of energy
    ax = axes[0]
    energies = np.linspace(0.01, 0.99, 100)
    alphas = [hebb_adaptive._get_cvar_alpha(torch.tensor([e])) for e in energies]
    ax.plot(energies, alphas, linewidth=2.5, color="#228833")
    ax.fill_between(energies, 0, alphas, alpha=0.15, color="#228833")
    ax.set_xlabel("Energy Level (normalized)")
    ax.set_ylabel("CVaR α (readout breadth)")
    ax.set_title("Adaptive Risk Readout\n"
                 "Low energy → narrow CVaR (pessimistic)\n"
                 "High energy → full mean (neutral)")
    ax.annotate("Risk-averse zone", xy=(0.1, 0.25), fontsize=10, color="red")
    ax.annotate("Risk-neutral zone", xy=(0.7, 0.85), fontsize=10, color="green")
    ax.grid(True, alpha=0.3)
    
    # Right: value readout at different energy levels
    ax = axes[1]
    test_energies = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for e in test_energies:
        state = torch.tensor([e])
        profile = hebb_adaptive.get_risk_profile(state)
        ax.bar(
            np.arange(env_config.n_patches) + (test_energies.index(e) - 2) * 0.15,
            profile["cvar_values"],
            width=0.15,
            label=f"E={e:.1f} (α={profile['cvar_alpha']:.2f})",
            alpha=0.7,
        )
    
    ax.set_xticks(range(env_config.n_patches))
    ax.set_xticklabels(list(patch_names), fontsize=9)
    ax.set_ylabel("Action Value (CVaR readout)")
    ax.set_title("Action Values Under Different Energy Levels")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    fig.savefig(save_dir / "05_adaptive_readout.png", bbox_inches="tight")
    plt.close()
    print("  Saved 05_adaptive_readout.png")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    
    for label, metrics in all_metrics.items():
        print(f"\n{label}:")
        print(f"  Survival: {metrics['survival_rate']:.1%}")
        print(f"  Avg reward: {metrics['avg_reward']:.1f}")
        print(f"  Avg steps: {metrics['avg_steps']:.0f}")
    
    print(f"\nAll figures saved to {save_dir}/")
    print("Phase 3 complete!")


if __name__ == "__main__":
    main()
