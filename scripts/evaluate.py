"""
Evaluate a trained genome over multiple episodes to verify "solved" status.

CartPole-v1 is "solved" when avg reward >= 475 over 100 consecutive episodes.
LunarLander-v3 is "solved" when avg reward >= 200 over 100 consecutive episodes.

Usage:
    python scripts/evaluate.py --env CartPole-v1 --genome checkpoints/best.pkl --episodes 100
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat import Config, Genome, GlobalIndex
from neat.envs import make_env
from scripts.train import build_default_config, train


SOLVED_THRESHOLDS = {
    "CartPole-v1": (475.0, 100),
    "CartPole-v1-max": (500.0, 100),  # max reward
    "LunarLander-v3": (200.0, 100),
    "Acrobot-v1": (-100.0, 100),
    "MountainCar-v0": (-110.0, 100),
}


def evaluate_genome(g: Genome, env_name: str, n_episodes: int = 100,
                     max_steps: int = 1000, seed: int = 0) -> dict:
    """Evaluate `g` over `n_episodes` random episodes and return stats."""
    env = make_env(env_name, max_steps=max_steps, n_eval_episodes=1, seed=seed)
    rewards = []
    steps_list = []
    for ep in range(n_episodes):
        r, steps, _ = env.evaluate(g, episode_seed=seed * 10000 + ep)
        rewards.append(r)
        steps_list.append(steps)
    rewards = np.array(rewards)
    threshold, n_req = SOLVED_THRESHOLDS.get(env_name, (float("inf"), 100))
    return {
        "env": env_name,
        "n_episodes": n_episodes,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "median_reward": float(np.median(rewards)),
        "solved_threshold": threshold,
        "solved": bool(rewards.mean() >= threshold),
        "rewards": rewards.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--optimizer", action="store_true")
    p.add_argument("--output", default=None, help="Save JSON results here")
    args = p.parse_args()

    print(f"Training {args.env} for {args.gens} generations...")
    cfg = build_default_config(args.env, seed=args.seed, pop_size=args.pop,
                                optimizer=args.optimizer)
    result = train(args.env, cfg, args.gens, max_steps=args.max_steps,
                   n_eval_episodes=1, verbose=True)
    best = result["best_genome"]
    print(f"\nBest fitness during training: {result['best_fitness']:.2f}")
    print(f"\nEvaluating best genome over {args.episodes} episodes...")
    eval_result = evaluate_genome(best, args.env, n_episodes=args.episodes,
                                   max_steps=args.max_steps, seed=args.seed + 99999)
    print(f"\n=== Evaluation Results ===")
    print(f"  Mean reward:   {eval_result['mean_reward']:.2f}")
    print(f"  Std reward:    {eval_result['std_reward']:.2f}")
    print(f"  Min reward:    {eval_result['min_reward']:.2f}")
    print(f"  Max reward:    {eval_result['max_reward']:.2f}")
    print(f"  Median reward: {eval_result['median_reward']:.2f}")
    print(f"  Solved threshold: {eval_result['solved_threshold']:.2f}")
    print(f"  SOLVED: {eval_result['solved']}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "training_history": result["history"],
                "evaluation": eval_result,
                "config": cfg.to_dict(),
            }, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
