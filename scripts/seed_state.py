"""Seed the AutoTuner state with known-good configs for each env.

Run this once to initialize results/autotune_v2.json with the best configs
discovered through manual experimentation.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat import Config
from scripts.train import build_default_config
from scripts.autotune import EnvBestConfig, TuningResult


# Best known configs per env (from manual experimentation)
BEST_CONFIGS = {
    "CartPole-v1": {
        "weight_std": 0.3, "conn_std": 0.5, "weight_prob": 0.81,
        "conn_prob": 0.471, "neuron_prob": 0.137, "prune_prob": 0.0,
        "elitism": 2, "cull_pct": 0.5, "target_species": 8,
        "n_conn_multiplier": 1.0, "weight_init_multiplier": 1.0,
        "best_eval_mean": 500.0, "best_eval_std": 0.0, "best_round": 1,
    },
    "Acrobot-v1": {
        "weight_std": 0.3, "conn_std": 0.5, "weight_prob": 0.9,
        "conn_prob": 0.3, "neuron_prob": 0.1, "prune_prob": 0.04,
        "elitism": 2, "cull_pct": 0.5, "target_species": 10,
        "n_conn_multiplier": 1.13, "weight_init_multiplier": 2.0,
        "best_eval_mean": -92.14, "best_eval_std": 19.63, "best_round": 0,
    },
    "MountainCar-v0": {
        "weight_std": 0.5, "conn_std": 0.7, "weight_prob": 1.0,
        "conn_prob": 0.5, "neuron_prob": 0.1, "prune_prob": 0.03,
        "elitism": 2, "cull_pct": 0.5, "target_species": 12,
        "n_conn_multiplier": 2.5, "weight_init_multiplier": 1.5,
        "best_eval_mean": -115.70, "best_eval_std": 22.73, "best_round": 0,
    },
    "LunarLander-v3": {
        "weight_std": 0.215, "conn_std": 0.4, "weight_prob": 1.0,
        "conn_prob": 0.3, "neuron_prob": 0.13, "prune_prob": 0.03,
        "elitism": 1, "cull_pct": 0.57, "target_species": 10,
        "n_conn_multiplier": 0.99, "weight_init_multiplier": 1.08,
        "best_eval_mean": -76.56, "best_eval_std": 59.55, "best_round": 0,
    },
    "Blackjack-v1": {
        "weight_std": 0.3, "conn_std": 0.665, "weight_prob": 1.0,
        "conn_prob": 0.3, "neuron_prob": 0.069, "prune_prob": 0.036,
        "elitism": 2, "cull_pct": 0.33, "target_species": 8,
        "n_conn_multiplier": 1.5, "weight_init_multiplier": 2.0,
        "best_eval_mean": 0.01, "best_eval_std": 0.95, "best_round": 0,
    },
}


def apply_overrides(cfg: Config, o: dict) -> Config:
    cfg.mutation.weight_std = float(o["weight_std"])
    cfg.mutation.conn_std = float(o["conn_std"])
    cfg.policy.weight_prob = float(o["weight_prob"])
    cfg.policy.conn_prob = float(o["conn_prob"])
    cfg.policy.neuron_prob = float(o["neuron_prob"])
    cfg.policy.prune_prob = float(o["prune_prob"])
    cfg.generation.elitism = int(o["elitism"])
    cfg.generation.cull_pct = float(o["cull_pct"])
    cfg.speciation.target_species = int(o["target_species"])
    cfg.init.n_conn_multiplier = float(o["n_conn_multiplier"])
    cfg.init.weight_init_multiplier = float(o["weight_init_multiplier"])
    return cfg


def main():
    state = {
        "algorithm_version": 2,
        "algo_changes": [
            {"round": 0, "version": 2, "description":
             "Aggressive reward shaping for LunarLander; multi-seed training "
             "for Acrobot/MountainCar; stagnation penalty (15 gens); "
             "genome repair for output disconnected."},
        ],
        "env_bests": {},
    }
    for env_name, best in BEST_CONFIGS.items():
        cfg = build_default_config(env_name, seed=0, pop_size=50)
        cfg = apply_overrides(cfg, best)
        eval_mean = best["best_eval_mean"]
        eval_std = best["best_eval_std"]
        threshold_map = {
            "CartPole-v1": 475.0, "Acrobot-v1": -100.0,
            "MountainCar-v0": -110.0, "LunarLander-v3": 200.0,
            "Blackjack-v1": -0.2,
        }
        solved = eval_mean >= threshold_map[env_name]
        state["env_bests"][env_name] = {
            "best_eval_mean": eval_mean,
            "best_eval_std": eval_std,
            "best_train_best": eval_mean,  # approx
            "best_round": best["best_round"],
            "best_config": cfg.to_dict(),
            "history": [
                {
                    "round": 0, "env": env_name, "changes": best,
                    "train_best": eval_mean, "eval_mean": eval_mean,
                    "eval_std": eval_std, "eval_min": eval_mean - eval_std,
                    "eval_max": eval_mean + eval_std, "solved": solved,
                    "elapsed_s": 0.0, "is_best": True,
                    "notes": "manual_seed",
                }
            ],
        }
    os.makedirs("results", exist_ok=True)
    with open("results/autotune_v2.json", "w") as f:
        json.dump(state, f, indent=2)
    print("Seeded results/autotune_v2.json with manual best configs:")
    for name, best in BEST_CONFIGS.items():
        threshold_map = {
            "CartPole-v1": 475.0, "Acrobot-v1": -100.0,
            "MountainCar-v0": -110.0, "LunarLander-v3": 200.0,
            "Blackjack-v1": -0.2,
        }
        solved = best["best_eval_mean"] >= threshold_map[name]
        marker = "✓ SOLVED" if solved else "✗"
        print(f"  {name:20s} best_eval={best['best_eval_mean']:7.2f}  {marker}")


if __name__ == "__main__":
    main()
