"""
Phase 2 Experiment: Biologically Plausible Distributional RL.

Key questions:
1. Does the Hebbian agent learn the same quantile functions as QR-DQN?
2. Can it do risk-sensitive action selection?
3. How do the learned reversal points compare to Dabney 2020?
4. Does the "learned asymmetry" mode discover good τ distributions?
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

from envs.bandits import MultiArmedBandit
from agents.scalar_td import ScalarTDAgent, ScalarTDConfig
from agents.qr_dqn import TabularQRAgent, QRDQNConfig
from agents.hebbian_distributional import HebbianDistributionalAgent, HebbianConfig


def run_experiment(env, agent, n_steps):
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
    
    save_dir = Path("results/phase2")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    N_STEPS = 10_000
    N = 32  # neurons/quantiles
    
    print("=" * 60)
    print("Phase 2: Hebbian Distributional RL")
    print("=" * 60)
    
    # ================================================================
    # Exp 1: Distribution learning — Hebbian vs QR-DQN
    # ================================================================
    print("\n--- Exp 1: Distribution Learning Comparison ---")
    env = MultiArmedBandit.equal_mean_different_variance(mean=1.0, stds=[0.5, 1.0, 2.0, 4.0])
    print(env)
    
    # QR-DQN baseline
    qr = TabularQRAgent(QRDQNConfig(n_arms=4, n_quantiles=N, lr=0.05, epsilon=0.15, epsilon_decay=0.9995, epsilon_min=0.02))
    
    # Hebbian (uniform τ — should match QR-DQN exactly)
    hebb_uniform = HebbianDistributionalAgent(HebbianConfig(
        n_arms=4, n_neurons=N, base_lr=0.05, epsilon=0.15,
        epsilon_decay=0.9995, epsilon_min=0.02, asymmetry_mode="uniform",
    ))
    
    # Hebbian (Dabney-style noisy τ)
    hebb_dabney = HebbianDistributionalAgent(HebbianConfig(
        n_arms=4, n_neurons=N, base_lr=0.05, epsilon=0.15,
        epsilon_decay=0.9995, epsilon_min=0.02, asymmetry_mode="fixed_dabney",
    ))
    
    # Hebbian (learned τ)
    hebb_learned = HebbianDistributionalAgent(HebbianConfig(
        n_arms=4, n_neurons=N, base_lr=0.05, epsilon=0.15,
        epsilon_decay=0.9995, epsilon_min=0.02, asymmetry_mode="learned",
        meta_lr=0.001,
    ))
    
    agents = {
        "QR-DQN": qr,
        "Hebbian (uniform)": hebb_uniform,
        "Hebbian (Dabney)": hebb_dabney,
        "Hebbian (learned)": hebb_learned,
    }
    
    results = {}
    for label, agent in agents.items():
        env.reset_stats()
        r, a = run_experiment(env, agent, N_STEPS)
        results[label] = {"rewards": r, "actions": a}
        print(f"  {label}: mean reward = {r.mean():.3f}")
    
    # --- Fig 1: Learned quantile functions comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    arm_idx_to_show = [0, 3]  # σ=0.5 and σ=4.0
    
    for row, arm_idx in enumerate(arm_idx_to_show):
        # Left: all agents' quantile functions for this arm
        ax = axes[row, 0]
        for label, agent in agents.items():
            taus, vals = agent.get_learned_distribution(arm_idx)
            ax.plot(taus.numpy(), vals.numpy(), label=label, linewidth=2, alpha=0.7)
        
        # True quantile function
        arm = env.arms[arm_idx]
        true_taus = np.linspace(0.01, 0.99, 200)
        from scipy.stats import norm
        true_quantiles = norm.ppf(true_taus, loc=arm.params["mean"], scale=arm.params["std"])
        ax.plot(true_taus, true_quantiles, "k--", linewidth=1.5, alpha=0.5, label="True")
        
        ax.set_xlabel("Quantile τ")
        ax.set_ylabel("θ(τ)")
        ax.set_title(f"Arm: {arm.name} — Quantile Functions")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Right: reversal point distributions
        ax = axes[row, 1]
        for idx, (label, agent) in enumerate(agents.items()):
            if hasattr(agent, 'get_reversal_points'):
                rps = agent.get_reversal_points(arm_idx).numpy()
            else:
                _, vals = agent.get_learned_distribution(arm_idx)
                rps = vals.numpy()
            ax.scatter(
                rps, np.full(len(rps), idx) + np.random.randn(len(rps)) * 0.05,
                s=20, alpha=0.6, label=label
            )
        
        ax.set_xlabel("Reversal Point Value")
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(list(agents.keys()), fontsize=8)
        ax.set_title(f"Arm: {arm.name} — Reversal Points")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Hebbian vs QR-DQN: Distribution Learning", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / "01_distribution_comparison.png", bbox_inches="tight")
    plt.close()
    print("  Saved 01_distribution_comparison.png")
    
    # ================================================================
    # Exp 2: Risk sensitivity comparison
    # ================================================================
    print("\n--- Exp 2: Risk Sensitivity ---")
    env2 = MultiArmedBandit.safe_vs_risky()
    
    N_RUNS = 5
    N_STEPS_R = 3000
    
    risk_configs = {
        "Scalar TD": lambda: ScalarTDAgent(ScalarTDConfig(n_arms=2, lr=0.1, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.02)),
        "QR neutral": lambda: TabularQRAgent(QRDQNConfig(n_arms=2, n_quantiles=N, lr=0.05, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.02)),
        "QR averse": lambda: TabularQRAgent(QRDQNConfig(n_arms=2, n_quantiles=N, lr=0.05, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.02, risk_measure="averse", risk_param=0.25)),
        "Hebbian neutral": lambda: HebbianDistributionalAgent(HebbianConfig(n_arms=2, n_neurons=N, base_lr=0.05, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.02, risk_measure="neutral")),
        "Hebbian averse": lambda: HebbianDistributionalAgent(HebbianConfig(n_arms=2, n_neurons=N, base_lr=0.05, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.02, risk_measure="averse", risk_param=0.25)),
        "Hebbian seeking": lambda: HebbianDistributionalAgent(HebbianConfig(n_arms=2, n_neurons=N, base_lr=0.05, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.02, risk_measure="seeking", risk_param=0.25)),
    }
    
    all_freqs = {}
    for label, make_agent in risk_configs.items():
        run_freqs = []
        for run in range(N_RUNS):
            env2.reset_stats()
            agent = make_agent()
            _, actions = run_experiment(env2, agent, N_STEPS_R)
            final = actions[-1000:]
            run_freqs.append([(final == a).mean() for a in range(2)])
        all_freqs[label] = np.array(run_freqs)
    
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(2)
    width = 0.12
    offsets = np.linspace(-(len(all_freqs)-1)/2*width, (len(all_freqs)-1)/2*width, len(all_freqs))
    
    for idx, (label, freqs) in enumerate(all_freqs.items()):
        m, s = freqs.mean(axis=0), freqs.std(axis=0)
        ax.bar(x + offsets[idx], m, width, yerr=s, label=label, alpha=0.8, capsize=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(["Safe (σ≈0)", "Risky (bimodal)"])
    ax.set_ylabel("Selection Freq (last 1000)")
    ax.set_title("Risk Sensitivity: Hebbian vs QR-DQN\n(Same mean=1.0, different variance)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    fig.savefig(save_dir / "02_risk_sensitivity.png", bbox_inches="tight")
    plt.close()
    print("  Saved 02_risk_sensitivity.png")
    
    # ================================================================
    # Exp 3: Asymmetry factor analysis (Dabney 2020 comparison)
    # ================================================================
    print("\n--- Exp 3: Asymmetry Analysis ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Asymmetry ratios for uniform Hebbian
    ax = axes[0, 0]
    ratios = hebb_uniform.get_asymmetry_factors().numpy()
    taus_eff = hebb_uniform.get_effective_taus().numpy()
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(ratios)))
    ax.bar(range(len(ratios)), ratios, color=colors, alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Symmetric")
    ax.set_xlabel("Neuron Index (sorted by τ)")
    ax.set_ylabel("α⁺/α⁻")
    ax.set_title("Hebbian (uniform): Asymmetry Ratios")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top right: Effective τ distribution for all modes
    ax = axes[0, 1]
    for label, agent in [("Uniform", hebb_uniform), ("Dabney", hebb_dabney), ("Learned", hebb_learned)]:
        taus = agent.get_effective_taus().numpy()
        taus_sorted = np.sort(taus)
        ax.plot(range(len(taus_sorted)), taus_sorted, "o-", label=label, markersize=4, alpha=0.7)
    
    ax.plot(range(N), np.linspace(0, 1, N), "k--", alpha=0.3, label="Ideal uniform")
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Effective τ")
    ax.set_title("τ Distributions Across Modes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Bottom left: Reversal point distribution vs reward distribution
    ax = axes[1, 0]
    arm_idx = 0  # σ=0.5
    samples = env.arms[arm_idx].sample(5000).numpy()
    ax.hist(samples, bins=50, density=True, alpha=0.3, color="gray", label=f"True rewards ({env.arms[arm_idx].name})")
    
    for label, agent in [("QR-DQN", qr), ("Hebbian", hebb_uniform)]:
        if hasattr(agent, 'get_reversal_points'):
            rps = agent.get_reversal_points(arm_idx).numpy()
        else:
            _, vals = agent.get_learned_distribution(arm_idx)
            rps = vals.numpy()
        ax.plot(np.sort(rps), np.linspace(0, 1, len(rps)) * 2, "o-", label=f"{label} reversal pts", markersize=4)
    
    ax.set_xlabel("Value")
    ax.set_ylabel("Density / Scaled CDF")
    ax.set_title(f"Reversal Points vs True Distribution ({env.arms[arm_idx].name})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Bottom right: Quantile estimation error
    ax = axes[1, 1]
    from scipy.stats import norm
    
    errors_per_agent = {}
    for label, agent in agents.items():
        errs = []
        for arm_idx in range(4):
            arm = env.arms[arm_idx]
            taus, vals = agent.get_learned_distribution(arm_idx)
            true_q = torch.tensor(norm.ppf(taus.numpy(), loc=arm.params["mean"], scale=arm.params["std"]))
            err = (vals - true_q).abs().mean().item()
            errs.append(err)
        errors_per_agent[label] = errs
    
    x = np.arange(4)
    width = 0.18
    for idx, (label, errs) in enumerate(errors_per_agent.items()):
        ax.bar(x + idx*width - 1.5*width, errs, width, label=label, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([a.name for a in env.arms])
    ax.set_ylabel("Mean |θ_learned - θ_true|")
    ax.set_title("Quantile Estimation Error per Arm")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.suptitle("Phase 2: Asymmetry & Reversal Point Analysis", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / "03_asymmetry_analysis.png", bbox_inches="tight")
    plt.close()
    print("  Saved 03_asymmetry_analysis.png")
    
    # ================================================================
    # Exp 4: Learned asymmetry evolution
    # ================================================================
    print("\n--- Exp 4: Asymmetry Learning Dynamics ---")
    
    if hebb_learned.asymmetry_history:
        fig, ax = plt.subplots(figsize=(10, 5))
        hist = torch.stack(hebb_learned.asymmetry_history).numpy()
        # Plot every 4th neuron
        for i in range(0, N, max(1, N//8)):
            ax.plot(hist[:, i], alpha=0.6, label=f"Neuron {i}")
        ax.set_xlabel("Logging Step (×50)")
        ax.set_ylabel("Effective τ = α⁺/(α⁺+α⁻)")
        ax.set_title("Learned Asymmetry: τ Evolution During Training")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_dir / "04_learned_asymmetry_evolution.png", bbox_inches="tight")
        plt.close()
        print("  Saved 04_learned_asymmetry_evolution.png")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    
    for label, agent in agents.items():
        taus, vals = agent.get_all_distributions()
        means = vals.mean(dim=0).numpy()
        stds = vals.std(dim=0).numpy()
        print(f"\n{label}:")
        print(f"  Means: {means.round(3)}")
        print(f"  Stds:  {stds.round(3)}")
    
    print(f"\nTrue means: {env.get_true_means().numpy()}")
    print(f"True stds:  {env.get_true_variances().sqrt().numpy().round(3)}")
    
    print(f"\nAll figures saved to {save_dir}/")
    print("Phase 2 complete!")


if __name__ == "__main__":
    main()
