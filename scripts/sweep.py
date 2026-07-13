"""
Hyperparameter sweep for CartPole-v1.

Tries different combinations of:
    - weight_std
    - conn_prob
    - neuron_prob
    - elitism
    - target_species
    - optimizer on/off

For each combo, trains for N generations and evaluates the best genome over
K episodes.  Reports the best combo.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
from typing import Dict, List, Any
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat import Config, Population
from neat.envs import make_env
from scripts.train import build_default_config, train
from scripts.evaluate import evaluate_genome


def sweep_cartpole(gens: int = 20, eval_eps: int = 20, pop: int = 50):
    """Run a small hyperparameter sweep on CartPole-v1."""
    base_cfg = build_default_config("CartPole-v1", seed=0, pop_size=pop)

    # Define sweep space (small for speed)
    sweeps = [
        {"name": "baseline", "overrides": {}},
        {"name": "low_mut", "overrides": {
            "weight_prob": 0.5, "conn_prob": 0.15, "neuron_prob": 0.03}},
        {"name": "high_elite", "overrides": {"elitism": 4}},
        {"name": "more_species", "overrides": {"target_species": 12}},
        {"name": "optimizer", "overrides": {"optimizer_enabled": True}},
        {"name": "optimizer_low_mut", "overrides": {
            "optimizer_enabled": True, "weight_prob": 0.5}},
        {"name": "aggressive", "overrides": {
            "weight_prob": 1.0, "conn_prob": 0.4, "neuron_prob": 0.1}},
        {"name": "small_weight_std", "overrides": {"weight_std": 0.15}},
        {"name": "big_weight_std", "overrides": {"weight_std": 0.5}},
    ]

    results = []
    for sw in sweeps:
        cfg = build_default_config("CartPole-v1", seed=0, pop_size=pop)
        # Apply overrides
        o = sw["overrides"]
        if "weight_prob" in o: cfg.policy.weight_prob = o["weight_prob"]
        if "weight_std" in o:  cfg.mutation.weight_std = o["weight_std"]
        if "conn_prob" in o:   cfg.policy.conn_prob = o["conn_prob"]
        if "neuron_prob" in o: cfg.policy.neuron_prob = o["neuron_prob"]
        if "elitism" in o:     cfg.generation.elitism = o["elitism"]
        if "target_species" in o: cfg.speciation.target_species = o["target_species"]
        if "optimizer_enabled" in o: cfg.optimizer.enabled = o["optimizer_enabled"]

        print(f"\n--- {sw['name']} ---")
        t0 = time.time()
        result = train("CartPole-v1", cfg, gens, max_steps=500,
                       n_eval_episodes=1, verbose=False)
        best = result["best_genome"]
        eval_res = evaluate_genome(best, "CartPole-v1", n_episodes=eval_eps,
                                    max_steps=500, seed=99999)
        elapsed = time.time() - t0
        r = {
            "name": sw["name"],
            "overrides": sw["overrides"],
            "best_fitness_train": result["best_fitness"],
            "eval_mean": eval_res["mean_reward"],
            "eval_std": eval_res["std_reward"],
            "eval_min": eval_res["min_reward"],
            "eval_max": eval_res["max_reward"],
            "solved": eval_res["solved"],
            "elapsed_s": elapsed,
        }
        results.append(r)
        print(f"  train_best={r['best_fitness_train']:.1f} "
              f"eval_mean={r['eval_mean']:.1f}±{r['eval_std']:.1f} "
              f"solved={r['solved']} t={r['elapsed_s']:.1f}s")

    # Sort by eval_mean descending
    results.sort(key=lambda r: r["eval_mean"], reverse=True)
    print("\n=== SWEEP SUMMARY (sorted by eval_mean) ===")
    for r in results:
        print(f"  {r['name']:25s}  eval={r['eval_mean']:6.1f}±{r['eval_std']:5.1f}  "
              f"solved={r['solved']}  t={r['elapsed_s']:.1f}s")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=20)
    p.add_argument("--eval-eps", type=int, default=20)
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--output", default="results/sweep_cartpole.json")
    args = p.parse_args()
    results = sweep_cartpole(gens=args.gens, eval_eps=args.eval_eps, pop=args.pop)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
