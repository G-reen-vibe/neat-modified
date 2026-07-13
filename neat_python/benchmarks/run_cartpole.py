"""
CartPole-v1 benchmark for the modified NEAT algorithm.

Run:
    python3 benchmarks/run_cartpole.py [--gens N] [--pop N] [--seed S]

Solves CartPole-v1 = avg reward >= 475 over 100 consecutive episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np

# allow running from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym

from neat import (
    Population, MutationPolicy, GRPOOptimizer,
    MP_PER_TYPE_PROB, MP_SINGLE_PICK, MP_NESTED,
)
from neat import mutations as M
from neat.network import act_discrete, forward


# ----------------------------------------------------------------- fitness -
def evaluate_genome_cartpole(genome, env_seed: int = 0, max_steps: int = 500,
                              render: bool = False) -> float:
    """Run one CartPole episode. Returns total reward."""
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    obs, _ = env.reset(seed=env_seed)
    total = 0.0
    for _ in range(max_steps):
        a = act_discrete(genome, np.asarray(obs, dtype=np.float64))
        obs, r, terminated, truncated, _ = env.step(a)
        total += r
        if terminated or truncated:
            break
    env.close()
    return total


def make_fitness_fn(env_seed: int = 0, max_steps: int = 500, n_avg: int = 1):
    """Return a fitness function that evaluates each genome on n_avg episodes
    with different seeds."""
    def fit(genome) -> float:
        rs = []
        for k in range(n_avg):
            rs.append(evaluate_genome_cartpole(genome, env_seed=env_seed + k, max_steps=max_steps))
        return float(np.mean(rs))
    return fit


# ----------------------------------------------------------------- presets -
def make_population(args) -> Population:
    """Build the NEAT Population with reasonable hyperparameters."""
    optimizer = GRPOOptimizer(
        enabled=not args.no_optimizer,
        lr=args.opt_lr,
        weight_std=args.weight_std,
        l2=args.opt_l2,
        l1=0.0,
        method=args.opt_method,
        beta1=0.9, beta2=0.999, eps=1e-8,
        similarity_method="percentage",
    )
    mut_policy = MutationPolicy(
        method=MP_PER_TYPE_PROB,
        weight_prob=args.weight_prob,
        connection_prob=args.connection_prob,
        neuron_prob=args.neuron_prob,
        pruning_prob=args.pruning_prob,
        weight_cfg={
            "selection": M.W_SELECT_PROB, "pct": args.weight_pct,
            "mod": M.M_GAUSSIAN, "mod_param": args.weight_std,
        },
        connection_cfg={
            "selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
            "mod": M.M_GAUSSIAN, "mod_param": args.conn_std,
        },
        neuron_cfg={
            "selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
            "mod": M.N_SPLIT_INCOMING_ONE,
        },
        pruning_cfg={
            "selection": M.P_SELECT_PCT_SHUFFLED, "pct": 0.0,
        },
    )
    pop = Population(
        n_inputs=4, n_outputs=2, size=args.pop,
        init_conns_multiplier=args.init_mult,
        init_neuron_range=(0, args.init_neurons),
        asexual_pct=0.75, crossover_pct=0.25,
        n_interspecies=1, n_elitism=1, cull_pct=0.5,
        optimizer=optimizer,
        mutation_policy=mut_policy,
        speciation_policy="purge_then_standard",
        target_species=args.target_species,
        threshold=args.threshold,
        min_threshold=0.05, max_threshold=0.5,
        threshold_adjust=0.025,
        similarity_method="percentage",
        seed=args.seed,
    )
    return pop


# ----------------------------------------------------------------- main ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--pop", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=42)
    p.add_argument("--n-avg", type=int, default=1, help="episodes per genome per gen")
    p.add_argument("--max-steps", type=int, default=500)
    # hyperparams
    p.add_argument("--init-mult", type=float, default=2.0)
    p.add_argument("--init-neurons", type=int, default=2)
    p.add_argument("--weight-prob", type=float, default=1.0)
    p.add_argument("--weight-pct", type=float, default=0.5)
    p.add_argument("--weight-std", type=float, default=0.1)
    p.add_argument("--connection-prob", type=float, default=0.3)
    p.add_argument("--conn-std", type=float, default=0.5)
    p.add_argument("--neuron-prob", type=float, default=0.1)
    p.add_argument("--pruning-prob", type=float, default=0.05)
    p.add_argument("--target-species", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--opt-lr", type=float, default=0.05)
    p.add_argument("--opt-l2", type=float, default=0.0)
    p.add_argument("--opt-method", type=str, default="adam", choices=["sgd", "momentum", "rmsprop", "adam"])
    p.add_argument("--no-optimizer", action="store_true")
    p.add_argument("--out", type=str, default=None, help="output JSON file")
    p.add_argument("--snapshot-every", type=int, default=1)
    args = p.parse_args()

    print(f"CartPole-v1 NEAT benchmark")
    print(f"  population: {args.pop}, generations: {args.gens}, seed: {args.seed}")
    print(f"  optimizer: {args.opt_method} (lr={args.opt_lr}, l2={args.opt_l2})")
    print(f"  mutations: weight={args.weight_prob}, conn={args.connection_prob}, "
          f"neuron={args.neuron_prob}, prune={args.pruning_prob}")
    print()

    pop = make_population(args)
    fit_fn = make_fitness_fn(env_seed=args.env_seed, max_steps=args.max_steps, n_avg=args.n_avg)

    snapshots = []
    t0 = time.time()
    for gen in range(args.gens):
        t_gen = time.time()
        stats = pop.step(fit_fn)
        dt = time.time() - t_gen
        print(f"Gen {stats['generation']:>3d}: best={stats['best_fitness']:>6.1f} "
              f"mean={stats['mean_fitness']:>6.1f} "
              f"species={stats['n_species']:>2d} "
              f"avg_conns={stats['avg_conns']:>4.1f} "
              f"avg_nodes={stats['avg_nodes']:>4.1f} "
              f"thresh={stats['species_threshold']:.3f} "
              f"({dt:.1f}s)")
        if gen % args.snapshot_every == 0 or gen == args.gens - 1:
            snapshots.append(pop.snapshot())
        if stats["best_fitness"] >= 475.0:
            print(f"  -> solved at gen {stats['generation']}!")
    total_dt = time.time() - t0
    print(f"\nTotal time: {total_dt:.1f}s ({total_dt / args.gens:.2f}s/gen)")
    print(f"Best fitness: {pop.best_fitness:.1f}")

    if args.out:
        out = {
            "args": vars(args),
            "history": pop.history,
            "best_fitness": pop.best_fitness,
            "total_time": total_dt,
            "snapshots": snapshots,
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
