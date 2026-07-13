"""
Run a battery of benchmarks across multiple envs and configurations.

Each benchmark:
    - Trains for N generations
    - Reports: best fitness, mean fitness, generations to solve (if applicable),
      wall time, final #species, final #conns/nodes
    - Saves a JSON results file

Usage:
    python scripts/benchmark.py --output results/benchmarks.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat import Config, Population
from neat.envs import make_env
from scripts.train import build_default_config, train


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------
BENCHMARKS = [
    {
        "name": "CartPole-v1_default",
        "env": "CartPole-v1",
        "gens": 50,
        "max_steps": 500,
        "config_overrides": {},
        "solved_at": 475,  # CartPole-v1 is "solved" at >=475 over 100 consecutive episodes
    },
    {
        "name": "CartPole-v1_optimizer",
        "env": "CartPole-v1",
        "gens": 50,
        "max_steps": 500,
        "config_overrides": {"optimizer_enabled": True},
        "solved_at": 475,
    },
    {
        "name": "CartPole-v1_aggressive",
        "env": "CartPole-v1",
        "gens": 50,
        "max_steps": 500,
        "config_overrides": {
            "weight_std": 0.5, "conn_std": 0.8,
            "conn_prob": 0.4, "neuron_prob": 0.08,
        },
        "solved_at": 475,
    },
    {
        "name": "Acrobot-v1_default",
        "env": "Acrobot-v1",
        "gens": 50,
        "max_steps": 500,
        "config_overrides": {},
        "solved_at": -100,  # Acrobot: solved when avg reward >= -100 over 100 eps
    },
    {
        "name": "MountainCar-v0_default",
        "env": "MountainCar-v0",
        "gens": 50,
        "max_steps": 200,
        "config_overrides": {},
        "solved_at": -110,
    },
]


def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Apply simple top-level overrides to a config."""
    if "optimizer_enabled" in overrides:
        cfg.optimizer.enabled = bool(overrides["optimizer_enabled"])
    if "weight_std" in overrides:
        cfg.mutation.weight_std = float(overrides["weight_std"])
    if "conn_std" in overrides:
        cfg.mutation.conn_std = float(overrides["conn_std"])
    if "conn_prob" in overrides:
        cfg.policy.conn_prob = float(overrides["conn_prob"])
    if "neuron_prob" in overrides:
        cfg.policy.neuron_prob = float(overrides["neuron_prob"])
    if "prune_prob" in overrides:
        cfg.policy.prune_prob = float(overrides["prune_prob"])
    if "pop_size" in overrides:
        cfg.generation.pop_size = int(overrides["pop_size"])
    if "target_species" in overrides:
        cfg.speciation.target_species = int(overrides["target_species"])
    return cfg


def run_benchmark(bench: Dict, seed: int = 0, pop_size: int = 100,
                   eval_episodes: int = 1) -> Dict:
    """Run one benchmark and return results."""
    print(f"\n=== {bench['name']} ===")
    cfg = build_default_config(bench["env"], seed=seed, pop_size=pop_size)
    cfg = apply_overrides(cfg, bench["config_overrides"])
    t0 = time.time()
    result = train(bench["env"], cfg, bench["gens"],
                   max_steps=bench["max_steps"],
                   n_eval_episodes=eval_episodes,
                   verbose=True)
    elapsed = time.time() - t0
    # Determine generations-to-solve
    history = result["history"]
    solved_gen = None
    for i, h in enumerate(history):
        if h["fitness_max"] >= bench["solved_at"]:
            solved_gen = i + 1
            break
    return {
        "name": bench["name"],
        "env": bench["env"],
        "gens_run": len(history),
        "best_fitness": result["best_fitness"],
        "solved_at": bench["solved_at"],
        "solved_gen": solved_gen,
        "elapsed_s": elapsed,
        "final_stats": history[-1] if history else {},
        "history_max": [h["fitness_max"] for h in history],
        "history_mean": [h["fitness_mean"] for h in history],
        "history_species": [h["n_species"] for h in history],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/benchmarks.json")
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--gens", type=int, default=50, help="Override #generations for all benchmarks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--filter", default="", help="Only run benchmarks whose name contains this")
    args = p.parse_args()

    if args.gens:
        for b in BENCHMARKS:
            b["gens"] = args.gens

    results = []
    for bench in BENCHMARKS:
        if args.filter and args.filter not in bench["name"]:
            continue
        try:
            r = run_benchmark(bench, seed=args.seed, pop_size=args.pop,
                              eval_episodes=args.eval_episodes)
            results.append(r)
        except Exception as e:
            print(f"FAILED: {bench['name']}: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": bench["name"], "error": str(e)})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {args.output}")
    # Summary
    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"  {r['name']}: ERROR ({r['error']})")
        else:
            print(f"  {r['name']}: best={r['best_fitness']:.2f} "
                  f"solved_gen={r['solved_gen']} time={r['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
