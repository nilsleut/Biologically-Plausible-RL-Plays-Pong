"""
Phase 1: Scalar TD vs. Tabular QR-DQN on Multi-Armed Bandits.

Key demonstrations:
1. Both agents converge to correct mean values
2. QR-DQN additionally learns the full reward distribution
3. QR-DQN with risk measures → different arm preferences
4. Asymmetry factors match Dabney 2020 dopamine neuron findings
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

from envs.bandits import MultiArmedBandit, ArmConfig
from agents.scalar_td import ScalarTDAgent, ScalarTDConfig
from agents.qr_dqn import TabularQRAgent, QRDQNConfig


def run_experiment(env, agent, n_steps):
    """Run agent on bandit, return (rewards, actions)."""
    rewards, actions = [], []
    for _ in range(n_steps):
        a = agent.select_action()
        r = env.step(a)
        agent.update(a, r)
        rewards.append(r)
        actions.append(a)
    return np.array(rewards), np.array(actions)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    save_dir = Path("results/phase1")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    N_STEPS = 10_000
    
    print("=" * 60)
    print("Phase 1: Scalar TD vs Distributional RL (Tabular QR)")
    print("=" * 60)
    
    # ================================================================
    # Experiment 1: Equal mean, different variance
    # ================================================================
    print("\n--- Exp 1: Equal Mean, Different Variance ---")
    env = MultiArmedBandit.equal_mean_different_variance(mean=1.0, stds=[0.5, 1.0, 2.0, 4.0])
    print(env)
    
    # Scalar TD
    scalar = ScalarTDAgent(ScalarTDConfig(n_arms=4, lr=0.1, epsilon=0.15, epsilon_decay=0.9995, epsilon_min=0.02))
    s_rewards, s_actions = run_experiment(env, scalar, N_STEPS)
    env.reset_stats()
    
    # Tabular QR (risk neutral)
    qr = TabularQRAgent(QRDQNConfig(n_arms=4, n_quantiles=32, lr=0.05, epsilon=0.15, epsilon_decay=0.9995, epsilon_min=0.02))
    q_rewards, q_actions = run_experiment(env, qr, N_STEPS)
    
    # --- Fig 1: Convergence ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    q_hist = scalar.get_q_history_tensor().numpy()
    # Smooth with rolling average
    window = 100
    for arm_idx in range(4):
        vals = q_hist[:, arm_idx]
        smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=env.arms[arm_idx].name, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, label="True mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("Q(a)")
    ax.set_title("Scalar TD: Q-value Convergence (smoothed)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    taus, all_q = qr.get_all_distributions()
    for arm_idx in range(4):
        ax.plot(taus.numpy(), all_q[arm_idx].numpy(), label=env.arms[arm_idx].name, linewidth=2, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, label="True mean")
    ax.set_xlabel("Quantile τ")
    ax.set_ylabel("Quantile Value θ(τ)")
    ax.set_title("QR Agent: Learned Quantile Functions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_dir / "01_convergence.png", bbox_inches="tight")
    plt.close()
    print("  Saved 01_convergence.png")
    
    # --- Fig 2: Learned distributions vs ground truth ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for arm_idx in range(4):
        ax = axes[arm_idx]
        # True distribution
        samples = env.arms[arm_idx].sample(10000).numpy()
        ax.hist(samples, bins=60, density=True, alpha=0.4, color=f"C{arm_idx}", label="True")
        
        # Learned CDF → PDF via finite differences
        taus_a, q_a = qr.get_learned_distribution(arm_idx)
        q_np = q_a.numpy()
        taus_np = taus_a.numpy()
        
        # Plot quantile positions as rug
        ax.plot(q_np, np.zeros_like(q_np), '|', color='black', markersize=10, alpha=0.5, label="Learned quantiles")
        
        ax.set_title(f"{env.arms[arm_idx].name}\nμ={env.arms[arm_idx].true_mean:.1f}, σ²={env.arms[arm_idx].true_variance:.1f}")
        ax.legend(fontsize=7)
        ax.set_xlabel("Reward")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("QR Agent: Learned Quantile Positions vs True Distribution", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(save_dir / "02_distributions.png", bbox_inches="tight")
    plt.close()
    print("  Saved 02_distributions.png")
    
    # --- Fig 3: Action selection comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    arm_names = [a.name for a in env.arms]
    window = 500
    
    for ax, actions, title in [(axes[0], s_actions, "Scalar TD"), (axes[1], q_actions, "QR Agent (neutral)")]:
        freqs = np.zeros((len(actions) - window, 4))
        for t in range(len(actions) - window):
            for a in range(4):
                freqs[t, a] = (actions[t:t+window] == a).mean()
        for a in range(4):
            ax.plot(freqs[:, a], label=arm_names[a], alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Action Freq (rolling {window})")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 0.7)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Action Selection: Both agents ~uniform (equal means)", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_dir / "03_action_selection.png", bbox_inches="tight")
    plt.close()
    print("  Saved 03_action_selection.png")
    
    # ================================================================
    # Experiment 2: Risk Sensitivity (Safe vs Risky)
    # ================================================================
    print("\n--- Exp 2: Risk Sensitivity ---")
    env2 = MultiArmedBandit.safe_vs_risky()
    print(env2)
    
    N_RUNS = 5
    N_STEPS_RISK = 3000
    
    configs = {
        "Scalar TD": lambda: ScalarTDAgent(ScalarTDConfig(n_arms=2, lr=0.1, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.02)),
        "QR neutral": lambda: TabularQRAgent(QRDQNConfig(n_arms=2, n_quantiles=32, lr=0.05, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.02, risk_measure="neutral")),
        "QR risk-averse\n(CVaR 0.25)": lambda: TabularQRAgent(QRDQNConfig(n_arms=2, n_quantiles=32, lr=0.05, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.02, risk_measure="averse", risk_param=0.25)),
        "QR risk-seeking\n(top 25%)": lambda: TabularQRAgent(QRDQNConfig(n_arms=2, n_quantiles=32, lr=0.05, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.02, risk_measure="seeking", risk_param=0.25)),
    }
    
    all_freqs = {}
    for label, make_agent in configs.items():
        run_freqs = []
        for run in range(N_RUNS):
            env2.reset_stats()
            agent = make_agent()
            _, actions = run_experiment(env2, agent, N_STEPS_RISK)
            final = actions[-1000:]
            run_freqs.append([(final == a).mean() for a in range(2)])
        all_freqs[label] = np.array(run_freqs)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(2)
    width = 0.18
    offsets = np.linspace(-1.5*width, 1.5*width, len(all_freqs))
    
    for idx, (label, freqs) in enumerate(all_freqs.items()):
        m, s = freqs.mean(axis=0), freqs.std(axis=0)
        ax.bar(x + offsets[idx], m, width, yerr=s, label=label, alpha=0.8, capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(["Safe (σ≈0)", "Risky (bimodal)"])
    ax.set_ylabel("Selection Frequency (last 1000 steps)")
    ax.set_title("Risk Sensitivity: Same Mean, Different Variance\n"
                 "Risk-averse → Safe, Risk-seeking → Risky")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    fig.savefig(save_dir / "04_risk_sensitivity.png", bbox_inches="tight")
    plt.close()
    print("  Saved 04_risk_sensitivity.png")
    
    # ================================================================
    # Experiment 3: Asymmetry factors (connection to Dabney 2020)
    # ================================================================
    print("\n--- Exp 3: Asymmetry Factors ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: the (1-τ)/τ curve
    ax = axes[0]
    taus_plot = qr.taus.numpy()
    ratios = qr.get_asymmetry_factors().numpy()
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(taus_plot)))
    ax.bar(range(len(taus_plot)), ratios, color=colors, alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="Symmetric (α⁺=α⁻)")
    ax.set_xlabel("Quantile Index")
    ax.set_ylabel("Asymmetry α⁺/α⁻ = (1−τ)/τ")
    ax.set_title("QR-DQN: Built-in Asymmetry\n(cf. Dabney 2020, dopamine neurons)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ticks = list(range(0, len(taus_plot), max(1, len(taus_plot)//8)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"τ={taus_plot[i]:.2f}" for i in ticks], fontsize=8)
    
    # Right: reversal point distribution (what Dabney found in real neurons)
    ax = axes[1]
    # In QR-DQN, each quantile neuron has a "reversal point" = its current value
    # The reversal point is where the TD error sign flips
    # Dabney found these are uniformly distributed → codes the full distribution
    for arm_idx in range(4):
        _, q_a = qr.get_learned_distribution(arm_idx)
        ax.scatter(
            q_a.numpy(), np.full(len(q_a), arm_idx),
            c=plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(q_a))),
            s=30, alpha=0.7, zorder=3,
        )
    ax.set_xlabel("Reversal Point (θ_i value)")
    ax.set_ylabel("Arm Index")
    ax.set_yticks(range(4))
    ax.set_yticklabels([a.name for a in env.arms])
    ax.set_title("Learned Reversal Points per Arm\n(wider spread = higher variance arm)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_dir / "05_asymmetry_and_reversals.png", bbox_inches="tight")
    plt.close()
    print("  Saved 05_asymmetry_and_reversals.png")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nScalar TD final Q-values: {scalar.q_values.numpy().round(3)}")
    print(f"True means:              {env.get_true_means().numpy()}")
    
    taus, quants = qr.get_all_distributions()
    print(f"\nQR mean values:          {quants.mean(dim=1).numpy().round(3)}")
    print(f"QR std estimates:        {quants.std(dim=1).numpy().round(3)}")
    print(f"True std:                {env.get_true_variances().sqrt().numpy().round(3)}")
    
    print(f"\nAction counts (Scalar):  {np.bincount(s_actions, minlength=4)}")
    print(f"Action counts (QR):      {np.bincount(q_actions, minlength=4)}")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_steps": N_STEPS,
        "scalar_q": scalar.q_values.numpy().tolist(),
        "qr_means": quants.mean(dim=1).numpy().tolist(),
        "qr_stds": quants.std(dim=1).numpy().tolist(),
        "true_means": env.get_true_means().numpy().tolist(),
        "true_stds": env.get_true_variances().sqrt().numpy().tolist(),
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll figures saved to {save_dir}/")
    print("Phase 1 complete!")


if __name__ == "__main__":
    main()
