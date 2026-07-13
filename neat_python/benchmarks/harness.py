"""
Environment wrappers + benchmark harness for NEAT on 5 difficult envs.

Supports:
  * Discrete action envs (MountainCar, Acrobot, LunarLander)
  * Continuous action envs (Pendulum, BipedalWalker) - tanh-squashed output
  * Observation normalization (running mean/std)
  * Episode evaluation with optional trace recording
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import random
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat import (
    Population, MutationPolicy, GRPOOptimizer,
    MP_PER_TYPE_PROB, MP_SINGLE_PICK,
)
from neat import mutations as M
from neat.network import forward, act_discrete
from neat.genome import Genome


# --------------------------------------------------------------- obs norm ----
class ObsNormalizer:
    """Running mean/std normalization for observations."""
    def __init__(self, n_obs: int, clip: float = 5.0) -> None:
        self.n = 0
        self.mean = np.zeros(n_obs, dtype=np.float64)
        self.M2 = np.zeros(n_obs, dtype=np.float64)
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        var = self.M2 / (self.n - 1)
        return np.sqrt(np.maximum(var, 1e-8))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        out = (x - self.mean) / self.std
        return np.clip(out, -self.clip, self.clip)


# ------------------------------------------------------------- env config ---
@dataclass
class EnvConfig:
    name: str
    n_inputs: int
    n_outputs: int
    discrete: bool
    action_low: Optional[Tuple[float, ...]] = None  # for continuous
    action_high: Optional[Tuple[float, ...]] = None
    solved_reward: float = 0.0       # reward threshold to consider "solved"
    max_steps: int = 500
    # fitness shaping: how to transform raw episode reward into NEAT fitness
    # 'raw'       = sum of rewards (good when rewards are positive)
    # 'shift'     = sum - min_possible (shifts to be >= 0)
    # 'normalize' = sum / max_steps (in [-1, 0] for penalty envs)
    # 'shape_pos' = sum + abs_shift (shifts to be >= 0, useful for negative rewards)
    fitness_mode: str = "raw"
    # absolute shift applied in 'shape_pos' mode
    abs_shift: float = 0.0
    # for envs like MountainCar where sparse reward is bad, add a shaping bonus
    shape_bonus: float = 0.0  # bonus per step closer to goal


ENVS: Dict[str, EnvConfig] = {
    "MountainCar-v0": EnvConfig(
        name="MountainCar-v0", n_inputs=2, n_outputs=3, discrete=True,
        solved_reward=-110.0, max_steps=200,
        fitness_mode="shape_pos", abs_shift=200.0,
        # raw reward is -1 per step; max -200. Shift by +200 to make [0, +200].
        # higher = fewer steps to goal = better.
    ),
    "Acrobot-v1": EnvConfig(
        name="Acrobot-v1", n_inputs=6, n_outputs=3, discrete=True,
        solved_reward=-100.0, max_steps=500,
        fitness_mode="shape_pos", abs_shift=500.0,
        # raw is -1 per step; max -500. Shift +500 to make [0, +500].
    ),
    "Pendulum-v1": EnvConfig(
        name="Pendulum-v1", n_inputs=3, n_outputs=1, discrete=False,
        action_low=(-2.0,), action_high=(2.0,),
        solved_reward=-200.0, max_steps=200,
        fitness_mode="shape_pos", abs_shift=1700.0,
        # raw is roughly [-1700, 0]. Shift +1700 to make [0, +1700].
    ),
    "LunarLander-v3": EnvConfig(
        name="LunarLander-v3", n_inputs=8, n_outputs=4, discrete=True,
        solved_reward=200.0, max_steps=1000,
        fitness_mode="shape_pos", abs_shift=300.0,
        # rewards can go slightly negative on crash; shift to ensure >= 0.
    ),
    "BipedalWalker-v3": EnvConfig(
        name="BipedalWalker-v3", n_inputs=24, n_outputs=4, discrete=False,
        action_low=(-1.0, -1.0, -1.0, -1.0), action_high=(1.0, 1.0, 1.0, 1.0),
        solved_reward=300.0, max_steps=1600,
        fitness_mode="shape_pos", abs_shift=100.0,
        # rewards can go very negative on falls; shift to ensure >= 0.
    ),
}


def shape_fitness(shaped_reward: float, cfg: EnvConfig, steps_alive: int = 0,
                   raw_reward: float = 0.0) -> float:
    """Transform episode reward into a NEAT-friendly fitness.

    For envs where agents fail early (BipedalWalker, LunarLander), we add
    a small per-step survival bonus so that "staying alive longer" gives
    a measurable fitness gradient even when the raw reward is identical.
    """
    base = shaped_reward
    if cfg.fitness_mode == "shape_pos":
        base = shaped_reward + cfg.abs_shift
    elif cfg.fitness_mode == "normalize":
        base = shaped_reward / cfg.max_steps
    # survival bonus: +0.5 per step alive (only for envs where early failure
    # is common and we need a gradient)
    if cfg.name in ("BipedalWalker-v3", "LunarLander-v3"):
        base += float(steps_alive) * 0.5
    return base


# ------------------------------------------------------------ evaluate ----
def evaluate_episode(
    genome: Genome,
    env: gym.Env,
    cfg: EnvConfig,
    obs_norm: ObsNormalizer,
    seed: int = 0,
    record_trace: bool = False,
) -> Dict:
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float64)
    obs_norm.update(obs)
    norm_obs = obs_norm.normalize(obs)
    total_reward = 0.0  # TRUE raw reward (for solved check)
    shaped_reward = 0.0  # shaped reward (for fitness)
    steps = 0
    trace = {"obs": [], "actions": [], "rewards": []} if record_trace else None
    # for potential-based shaping (Pendulum)
    prev_potential = 0.0
    if cfg.name == "Pendulum-v1":
        prev_potential = float(obs[0])  # cos(theta)
    for _ in range(cfg.max_steps):
        if cfg.discrete:
            a = act_discrete(genome, norm_obs)
        else:
            raw = forward(genome, norm_obs)
            squashed = np.tanh(raw)
            low = np.array(cfg.action_low)
            high = np.array(cfg.action_high)
            a = low + (squashed + 1.0) * 0.5 * (high - low)
            a = np.clip(a, low, high)
        if record_trace:
            trace["obs"].append(obs.tolist())
            trace["actions"].append(a.tolist() if not cfg.discrete else int(a))
            trace["rewards"].append(0.0)
        obs, r, terminated, truncated, _ = env.step(a)
        obs = np.asarray(obs, dtype=np.float64)
        obs_norm.update(obs)
        norm_obs = obs_norm.normalize(obs)
        total_reward += float(r)
        # potential-based shaping for Pendulum
        if cfg.name == "Pendulum-v1":
            cur_potential = float(obs[0])
            shaped_r = r + 0.9 * cur_potential - prev_potential
            shaped_reward += shaped_r
            prev_potential = cur_potential
        else:
            shaped_reward = total_reward  # no extra shaping
        if record_trace:
            trace["rewards"][-1] = float(r)
        steps += 1
        if terminated or truncated:
            break
    return {
        "reward": total_reward,        # true raw reward
        "shaped_reward": shaped_reward, # shaped (for fitness)
        "steps": steps,
        "trace": trace,
    }


# --------------------------------------------------------- pop builder ----
def build_population(cfg: EnvConfig, hp: Dict) -> Population:
    opt = GRPOOptimizer(
        enabled=hp.get("optimizer_enabled", False),
        lr=hp.get("opt_lr", 0.02),
        weight_std=hp.get("weight_std", 0.1),
        l2=hp.get("opt_l2", 0.0),
        method=hp.get("opt_method", "adam"),
        similarity_method="percentage",
    )
    mut = MutationPolicy(
        method=MP_PER_TYPE_PROB,
        weight_prob=hp.get("weight_prob", 0.9),
        connection_prob=hp.get("connection_prob", 0.15),
        neuron_prob=hp.get("neuron_prob", 0.08),
        pruning_prob=hp.get("pruning_prob", 0.03),
        weight_cfg={
            "selection": M.W_SELECT_PROB,
            "pct": hp.get("weight_pct", 0.5),
            "mod": M.M_GAUSSIAN,
            "mod_param": hp.get("weight_std", 0.1),
        },
        connection_cfg={
            "selection": hp.get("conn_select", M.C_SELECT_PCT_SHUFFLED),
            "pct": 0.0,
            "mod": M.M_GAUSSIAN,
            "mod_param": hp.get("conn_std", 0.5),
        },
        neuron_cfg={
            "selection": M.C_SELECT_PCT_SHUFFLED,
            "pct": 0.0,
            "mod": M.N_SPLIT_INCOMING_ONE,
        },
        pruning_cfg={
            "selection": M.P_SELECT_PCT_SHUFFLED,
            "pct": 0.0,
        },
    )
    pop = Population(
        n_inputs=cfg.n_inputs, n_outputs=cfg.n_outputs, size=hp.get("pop_size", 60),
        init_conns_multiplier=hp.get("init_mult", 2.0),
        init_neuron_range=(0, hp.get("init_neurons", 1)),
        asexual_pct=hp.get("asexual_pct", 0.8),
        crossover_pct=hp.get("crossover_pct", 0.2),
        n_interspecies=hp.get("n_interspecies", 1),
        n_elitism=hp.get("elitism", 2),
        cull_pct=hp.get("cull_pct", 0.5),
        optimizer=opt, mutation_policy=mut,
        speciation_policy="purge_then_standard",
        target_species=hp.get("target_species", 5),
        threshold=hp.get("threshold", 0.25),
        min_threshold=0.05, max_threshold=0.5, threshold_adjust=0.025,
        similarity_method="percentage",
        # For continuous control: use identity output (we tanh-squash in eval).
        # For discrete: use tanh (default) so argmax is well-defined.
        output_activation="identity" if not cfg.discrete else "tanh",
        seed=hp.get("seed", 0),
    )
    return pop


# ----------------------------------------------------------- run round ----
@dataclass
class RoundResult:
    env: str
    round_id: int
    hyperparams: Dict[str, Any]
    best_fitness: float = -float("inf")
    mean_fitness: float = 0.0
    final_best: float = -float("inf")
    final_mean: float = 0.0
    n_species: int = 0
    n_gens: int = 0
    solved_gen: Optional[int] = None
    avg_conns: float = 0.0
    avg_nodes: float = 0.0
    time_s: float = 0.0
    history: List[Dict] = field(default_factory=list)
    notes: str = ""


def run_round(
    env_name: str,
    round_id: int,
    hp: Dict,
    n_gens: int = 30,
    n_avg: int = 1,
    env_seed: int = 42,
    notes: str = "",
    verbose: bool = True,
) -> RoundResult:
    """Run one hyperparameter round on the given env."""
    cfg = ENVS[env_name]
    env = gym.make(cfg.name)
    pop = build_population(cfg, hp)
    obs_norm = ObsNormalizer(cfg.n_inputs)

    # Fitness function
    raw_best_tracker = {"best": -float("inf")}  # mutable closure
    def fit(genome: Genome) -> float:
        rs = []
        ss = []
        steps_total = 0
        for k in range(n_avg):
            res = evaluate_episode(genome, env, cfg, obs_norm,
                                    seed=env_seed + k * 1000)
            rs.append(res["reward"])
            ss.append(res["shaped_reward"])
            steps_total += res["steps"]
        raw = float(np.mean(rs))
        shaped = float(np.mean(ss))
        if raw > raw_best_tracker["best"]:
            raw_best_tracker["best"] = raw
        avg_steps = steps_total / max(1, n_avg)
        return shape_fitness(shaped, cfg, steps_alive=int(avg_steps), raw_reward=raw)

    result = RoundResult(env=env_name, round_id=round_id, hyperparams=dict(hp), notes=notes)
    t0 = time.time()
    for gen in range(n_gens):
        stats = pop.step(fit)
        raw_best_seen = raw_best_tracker["best"]
        # mean: subtract abs_shift from shaped mean (approximate, ignores survival bonus)
        raw_mean = stats["mean_fitness"] - cfg.abs_shift
        if cfg.name in ("BipedalWalker-v3", "LunarLander-v3"):
            # approximate: subtract average survival bonus (assume ~max_steps/2 for survivors)
            raw_mean -= cfg.max_steps * 0.25  # rough correction
        result.history.append({
            "gen": stats["generation"],
            "best": raw_best_seen,
            "mean": raw_mean,
            "best_shaped": stats["best_fitness"],
            "mean_shaped": stats["mean_fitness"],
            "species": stats["n_species"],
            "conns": stats["avg_conns"],
            "nodes": stats["avg_nodes"],
            "threshold": stats["species_threshold"],
        })
        if result.solved_gen is None:
            if cfg.solved_reward >= 0 and raw_best_seen >= cfg.solved_reward:
                result.solved_gen = stats["generation"]
            elif cfg.solved_reward < 0 and raw_best_seen >= cfg.solved_reward:
                result.solved_gen = stats["generation"]
        if verbose:
            print(f"  [{env_name} R{round_id}] gen {stats['generation']:>3d}: "
                  f"best={raw_best_seen:>8.2f} mean={raw_mean:>8.2f} "
                  f"species={stats['n_species']:>2d} "
                  f"conns={stats['avg_conns']:>4.1f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    result.time_s = time.time() - t0
    if result.history:
        result.final_best = result.history[-1]["best"]
        result.final_mean = result.history[-1]["mean"]
        result.best_fitness = max(h["best"] for h in result.history)
        result.mean_fitness = sum(h["mean"] for h in result.history) / len(result.history)
        result.n_species = result.history[-1]["species"]
        result.n_gens = len(result.history)
        result.avg_conns = result.history[-1]["conns"]
        result.avg_nodes = result.history[-1]["nodes"]
    env.close()
    return result


# -------------------------------------------------------------- main -------
def main():
    """Quick smoke test of the harness."""
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="MountainCar-v0", choices=list(ENVS.keys()))
    p.add_argument("--gens", type=int, default=10)
    p.add_argument("--pop", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    hp = {
        "pop_size": args.pop,
        "weight_prob": 0.9,
        "connection_prob": 0.15,
        "neuron_prob": 0.08,
        "pruning_prob": 0.03,
        "weight_pct": 0.5,
        "weight_std": 0.1,
        "conn_std": 0.5,
        "init_mult": 2.0,
        "init_neurons": 1,
        "target_species": 5,
        "threshold": 0.25,
        "elitism": 2,
        "cull_pct": 0.5,
        "optimizer_enabled": False,
        "seed": args.seed,
    }
    res = run_round(args.env, 0, hp, n_gens=args.gens)
    print(f"\n=== Result ===")
    print(f"Best: {res.best_fitness:.2f}, Final best: {res.final_best:.2f}")
    print(f"Mean: {res.mean_fitness:.2f}, Final mean: {res.final_mean:.2f}")
    print(f"Solved at gen: {res.solved_gen}")
    print(f"Time: {res.time_s:.1f}s")


if __name__ == "__main__":
    main()
