"""
Standalone training process that writes snapshots to a file.
The FastAPI server reads this file to serve to the visualizer.

Run:
    python3 training_worker.py [--config config.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import signal
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym

from neat import Population, MutationPolicy, GRPOOptimizer, MP_PER_TYPE_PROB
from neat import mutations as M
from neat.network import act_discrete
from neat.genome import Genome


STATE_FILE = "/home/z/my-project/.zscripts/training-state.json"
CONFIG_FILE = "/home/z/my-project/.zscripts/training-config.json"
CONTROL_FILE = "/home/z/my-project/.zscripts/training-control.json"  # {"stop": true}

STOP_FLAG = False


def handle_signal(signum, frame):
    global STOP_FLAG
    STOP_FLAG = True


def build_population(config: Dict) -> Population:
    opt = GRPOOptimizer(
        enabled=config.get("optimizer_enabled", False),
        lr=config.get("opt_lr", 0.02),
        weight_std=config.get("weight_std", 0.05),
        method=config.get("opt_method", "adam"),
        similarity_method="percentage",
    )
    mut = MutationPolicy(
        method=MP_PER_TYPE_PROB,
        weight_prob=config.get("weight_prob", 0.8),
        connection_prob=config.get("connection_prob", 0.1),
        neuron_prob=config.get("neuron_prob", 0.05),
        pruning_prob=config.get("pruning_prob", 0.02),
        weight_cfg={
            "selection": M.W_SELECT_PROB,
            "pct": config.get("weight_pct", 0.3),
            "mod": M.M_GAUSSIAN,
            "mod_param": config.get("weight_std", 0.05),
        },
        connection_cfg={"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
                         "mod": M.M_GAUSSIAN, "mod_param": 0.3},
        neuron_cfg={"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
                     "mod": M.N_SPLIT_INCOMING_ONE},
        pruning_cfg={"selection": M.P_SELECT_PCT_SHUFFLED, "pct": 0.0},
    )
    pop = Population(
        n_inputs=4, n_outputs=2, size=config.get("pop_size", 30),
        init_conns_multiplier=config.get("init_mult", 2.0),
        init_neuron_range=(0, config.get("init_neurons", 0)),
        asexual_pct=0.8, crossover_pct=0.2,
        n_interspecies=1, n_elitism=config.get("elitism", 2),
        cull_pct=config.get("cull_pct", 0.5),
        optimizer=opt, mutation_policy=mut,
        speciation_policy="purge_then_standard",
        target_species=config.get("target_species", 4),
        threshold=config.get("threshold", 0.25),
        min_threshold=0.05, max_threshold=0.4, threshold_adjust=0.02,
        similarity_method="percentage",
        seed=config.get("seed", 0),
    )
    return pop


def evaluate_with_trace(genome: Genome, env: gym.Env, max_steps: int = 500) -> Dict:
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


def make_fitness_fn(env: gym.Env, max_steps: int, episode_buffer: list, generation_ref: list):
    def fit(genome: Genome) -> float:
        trace = evaluate_with_trace(genome, env, max_steps)
        if trace["reward"] >= 100:
            episode_buffer.append({
                "trace": trace,
                "generation": generation_ref[0],
                "fitness": float(trace["reward"]),
            })
            if len(episode_buffer) > 10:
                episode_buffer.pop(0)
        return float(trace["reward"])
    return fit


def write_state(state: Dict) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, default=str)
    os.rename(tmp, STATE_FILE)


def read_control() -> Dict:
    try:
        with open(CONTROL_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def clear_control() -> None:
    try:
        os.remove(CONTROL_FILE)
    except FileNotFoundError:
        pass


def main():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        # check config file
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}

    print(f"Training worker started with config: {config}", flush=True)

    pop = build_population(config)
    env = gym.make("CartPole-v1")
    episode_buffer = []
    generation_ref = [0]

    n_gens = config.get("generations", 100)
    max_steps = config.get("max_steps", 500)
    delay = config.get("delay", 0.0)
    fit_fn = make_fitness_fn(env, max_steps, episode_buffer, generation_ref)

    for gen in range(n_gens):
        if STOP_FLAG:
            break
        # check for stop control
        ctrl = read_control()
        if ctrl.get("stop"):
            clear_control()
            break
        # check for delay update
        if "delay" in ctrl:
            delay = ctrl["delay"]
            clear_control()

        stats = pop.step(fit_fn)
        generation_ref[0] = stats["generation"]
        snap = pop.snapshot()
        # cap data
        if "history" in snap and len(snap["history"]) > 100:
            snap["history"] = snap["history"][-100:]
        if "genomes" in snap and len(snap["genomes"]) > 10:
            snap["genomes"] = snap["genomes"][:10]
        snap["episode_buffer"] = episode_buffer[-3:]
        snap["running"] = True
        write_state(snap)
        print(f"Gen {stats['generation']}: best={stats['best_fitness']:.1f} mean={stats['mean_fitness']:.1f}", flush=True)
        if delay > 0:
            time.sleep(delay)

    env.close()
    # mark as stopped
    try:
        with open(STATE_FILE, "r") as f:
            final = json.load(f)
        final["running"] = False
        write_state(final)
    except Exception:
        pass
    print("Training worker done", flush=True)


if __name__ == "__main__":
    main()
