"""
Run a full training offline and save all snapshots to a JSON file
for the visualizer to play back.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gymnasium as gym

from neat import Population, MutationPolicy, GRPOOptimizer, MP_PER_TYPE_PROB
from neat import mutations as M
from neat.network import act_discrete
from neat.genome import Genome


def evaluate_with_trace(genome, env, max_steps=500):
    obs, _ = env.reset(seed=42)
    trace = {"obs": [], "actions": [], "reward": 0.0, "steps": 0}
    for _ in range(max_steps):
        a = act_discrete(genome, np.asarray(obs, dtype=np.float64))
        trace["obs"].append([float(x) for x in obs])
        trace["actions"].append(int(a))
        obs, r, terminated, truncated, _ = env.step(a)
        trace["reward"] += r
        trace["steps"] += 1
        if terminated or truncated:
            break
    return trace


def main():
    config = {
        "pop_size": 40,
        "generations": 60,
        "n_avg": 1,
        "init_neurons": 1,
        "init_mult": 2.0,
        "weight_prob": 0.8,
        "weight_pct": 0.3,
        "weight_std": 0.05,
        "connection_prob": 0.1,
        "neuron_prob": 0.05,
        "pruning_prob": 0.02,
        "target_species": 4,
        "threshold": 0.25,
        "elitism": 2,
        "cull_pct": 0.5,
        "optimizer_enabled": False,
        "seed": 0,
    }

    opt = GRPOOptimizer(
        enabled=config["optimizer_enabled"],
        lr=0.02, weight_std=config["weight_std"],
        method="adam", similarity_method="percentage",
    )
    mut = MutationPolicy(
        method=MP_PER_TYPE_PROB,
        weight_prob=config["weight_prob"],
        connection_prob=config["connection_prob"],
        neuron_prob=config["neuron_prob"],
        pruning_prob=config["pruning_prob"],
        weight_cfg={"selection": M.W_SELECT_PROB, "pct": config["weight_pct"],
                     "mod": M.M_GAUSSIAN, "mod_param": config["weight_std"]},
        connection_cfg={"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
                         "mod": M.M_GAUSSIAN, "mod_param": 0.3},
        neuron_cfg={"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
                     "mod": M.N_SPLIT_INCOMING_ONE},
        pruning_cfg={"selection": M.P_SELECT_PCT_SHUFFLED, "pct": 0.0},
    )
    pop = Population(
        n_inputs=4, n_outputs=2, size=config["pop_size"],
        init_conns_multiplier=config["init_mult"],
        init_neuron_range=(0, config["init_neurons"]),
        asexual_pct=0.8, crossover_pct=0.2,
        n_interspecies=1, n_elitism=config["elitism"],
        cull_pct=config["cull_pct"],
        optimizer=opt, mutation_policy=mut,
        speciation_policy="purge_then_standard",
        target_species=config["target_species"],
        threshold=config["threshold"],
        min_threshold=0.05, max_threshold=0.4, threshold_adjust=0.02,
        similarity_method="percentage",
        seed=config["seed"],
    )

    env = gym.make("CartPole-v1")
    episode_buffer = []

    def fit(genome):
        trace = evaluate_with_trace(genome, env, max_steps=500)
        if trace["reward"] >= 100:
            episode_buffer.append({
                "trace": trace,
                "generation": pop.generation,
                "fitness": float(trace["reward"]),
            })
            if len(episode_buffer) > 5:
                episode_buffer.pop(0)
        return float(trace["reward"])

    snapshots = []
    t0 = time.time()
    for gen in range(config["generations"]):
        stats = pop.step(fit)
        snap = pop.snapshot()
        if "history" in snap and len(snap["history"]) > 100:
            snap["history"] = snap["history"][-100:]
        if "genomes" in snap and len(snap["genomes"]) > 10:
            snap["genomes"] = snap["genomes"][:10]
        snap["episode_buffer"] = list(episode_buffer[-3:])
        snap["running"] = True
        snapshots.append(snap)
        print(f"Gen {stats['generation']:>3d}: best={stats['best_fitness']:>6.1f} "
              f"mean={stats['mean_fitness']:>6.1f} species={stats['n_species']} "
              f"({time.time()-t0:.1f}s)", flush=True)

    env.close()

    # mark last as not running
    if snapshots:
        snapshots[-1]["running"] = False

    out_path = "/home/z/my-project/public/training-data.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": config, "snapshots": snapshots}, f, default=str)
    print(f"\nSaved {len(snapshots)} snapshots to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
