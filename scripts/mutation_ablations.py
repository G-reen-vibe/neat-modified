"""
Run additional ablations specifically on mutation type combinations.

Tests:
    - weight only (no conn/neuron/prune mutations)
    - weight + conn
    - weight + neuron
    - weight + conn + neuron
    - weight + conn + neuron + prune (baseline)
    - conn only (no weight)
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from neat import Config
from neat.config import MutationPolicyKind
from neat.envs import make_env
from neat.analysis import visualize_genome, capture_training_trajectory
from scripts.train import build_default_config
from scripts.evaluate import SOLVED_THRESHOLDS
from scripts.ablations import run_ablation, get_base_config


def mutation_ablations(env_name: str) -> list:
    """Build ablation configs that vary mutation type combinations."""
    base = get_base_config(env_name)
    ablations = []
    # Note: weight is always kept on (otherwise no learning)
    configs = [
        ("weight_only", {"conn_prob": 0.0, "neuron_prob": 0.0, "prune_prob": 0.0},
         "Only weight mutation (no topology changes)"),
        ("weight_conn", {"neuron_prob": 0.0, "prune_prob": 0.0},
         "Weight + connection mutation (grow only)"),
        ("weight_neuron", {"conn_prob": 0.0, "prune_prob": 0.0},
         "Weight + neuron mutation (split only)"),
        ("weight_conn_neuron", {"prune_prob": 0.0},
         "Weight + conn + neuron (no pruning)"),
        ("conn_only", {"weight_prob": 0.0, "neuron_prob": 0.0, "prune_prob": 0.0},
         "Only connection mutation (topology-only, no weight learning)"),
    ]
    for name, overrides, desc in configs:
        cfg = base.copy()
        for k, v in overrides.items():
            setattr(cfg.policy, k, v)
        ablations.append({
            "name": f"{env_name}_mut_{name}",
            "cfg": cfg,
            "description": desc,
        })
    return ablations


def main():
    env_name = "CartPole-v1"
    print(f"=== Mutation-Type Ablations on {env_name} ===")
    ablations = mutation_ablations(env_name)
    results = []
    max_steps = 500
    for ab in ablations:
        ab["cfg"].generation.pop_size = 40
        ab["cfg"].seed = hash(ab["name"]) % 10000
        result = run_ablation(
            ab["name"], env_name, ab["cfg"],
            n_gens=15, eval_episodes=10, max_steps=max_steps,
            capture_genome_gif=False, verbose=True,
        )
        results.append({
            "name": result["name"],
            "description": ab["description"],
            "best_fitness_train": result["best_fitness_train"],
            "eval_mean": result["eval_mean"],
            "eval_std": result["eval_std"],
            "solved": result["solved"],
            "elapsed_s": result["elapsed_s"],
        })

    print("\n=== Mutation-Type Ablation Summary ===")
    threshold = SOLVED_THRESHOLDS.get(env_name, (float("inf"), 100))[0]
    print(f"[{env_name}] (threshold={threshold})")
    for r in sorted(results, key=lambda x: -x["eval_mean"]):
        marker = "✓" if r["solved"] else "✗"
        print(f"  {marker} {r['name']:40s}  eval={r['eval_mean']:7.2f} "
              f"± {r['eval_std']:5.2f}  t={r['elapsed_s']:.1f}s")
    # Save
    os.makedirs("results/ablations", exist_ok=True)
    with open("results/ablations/mutation_type_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
