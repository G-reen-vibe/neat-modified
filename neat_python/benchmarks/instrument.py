"""
Instrumentation hooks for NEAT training.

Provides:
  * VideoRecorder: wraps a gym env to record MP4 episodes
  * GenomeExporter: saves genome topology + weights to JSON
  * TrainingStatsCollector: tracks per-generation stats (diversity, weight dist,
    species composition, activation usage, etc.)
  * run_instrumented_round: run a round with full instrumentation
"""
from __future__ import annotations

import os
import sys
import json
import time
import math
import random
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat import (
    Population, MutationPolicy, GRPOOptimizer,
    MP_PER_TYPE_PROB, MP_SINGLE_PICK,
)
from neat import mutations as M
from neat.network import forward, act_discrete
from neat.genome import Genome
from neat.similarity import similarity

from benchmarks.harness import (
    ENVS, EnvConfig, ObsNormalizer, build_population,
    evaluate_episode, shape_fitness,
)


# ---------------------------------------------------------- video ----------
def make_video_env(env_name: str, video_dir: str, episode_trigger=None,
                    name_prefix: str = "agent") -> gym.Env:
    """Create a gym env that records MP4 videos."""
    os.makedirs(video_dir, exist_ok=True)
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env, video_dir,
        episode_trigger=episode_trigger or (lambda ep_id: True),
        name_prefix=name_prefix,
    )
    return env


# ------------------------------------------------------- genome export ----
def genome_to_json(genome: Genome, extra: Optional[Dict] = None) -> Dict:
    """Export a genome to a JSON-serializable dict with full topology."""
    nodes = []
    for nid, n in sorted(genome.nodes.items()):
        nodes.append({
            "id": nid,
            "kind": n.kind,
            "activation": n.activation,
        })
    conns = []
    for innov, c in sorted(genome.conns.items()):
        conns.append({
            "innov": innov,
            "in": c.in_node,
            "out": c.out_node,
            "weight": c.weight,
            "enabled": c.enabled,
        })
    return {
        "n_inputs": genome.n_inputs,
        "n_outputs": genome.n_outputs,
        "nodes": nodes,
        "connections": conns,
        "fitness": genome.fitness,
        "species_id": genome.species_id,
        "generation": genome.generation,
        "extra": extra or {},
    }


def save_genome_json(genome: Genome, path: str, extra: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(genome_to_json(genome, extra), f, indent=2)


# ----------------------------------------------- training stats collector --
@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    n_species: int
    species_sizes: List[int]
    species_best: List[float]
    species_avg: List[float]
    species_staleness: List[int]
    threshold: float
    avg_conns: float
    avg_nodes: float
    max_conns: int
    min_conns: int
    weight_mean: float
    weight_std: float
    weight_min: float
    weight_max: float
    activation_counts: Dict[str, int]
    n_evals: int
    time_s: float


class TrainingStatsCollector:
    """Collects detailed per-generation statistics from a Population."""

    def __init__(self) -> None:
        self.history: List[GenerationStats] = []
        self.best_genome_snapshots: List[Dict] = []  # JSON snapshots of best genome per gen

    def collect(self, pop: Population, stats: Dict, gen_time: float) -> GenerationStats:
        genomes = pop.genomes
        fitnesses = [g.fitness for g in genomes]
        conn_counts = [len(g.conns) for g in genomes]
        node_counts = [len(g.nodes) for g in genomes]
        all_weights = [c.weight for g in genomes for c in g.conns.values()]
        act_counts: Dict[str, int] = defaultdict(int)
        for g in genomes:
            for n in g.nodes.values():
                act_counts[n.activation] += 1

        species = pop.speciator.species
        species_sizes = [len(sp.members) for sp in species.values()]
        species_best = [max((m.fitness for m in sp.members), default=0.0) for sp in species.values()]
        species_avg = [sum(m.fitness for m in sp.members) / max(1, len(sp.members)) for sp in species.values()]
        species_staleness = [sp.staleness for sp in species.values()]

        gs = GenerationStats(
            generation=stats["generation"],
            best_fitness=stats["best_fitness"],
            mean_fitness=stats["mean_fitness"],
            std_fitness=float(np.std(fitnesses)),
            n_species=stats["n_species"],
            species_sizes=species_sizes,
            species_best=species_best,
            species_avg=species_avg,
            species_staleness=species_staleness,
            threshold=stats["species_threshold"],
            avg_conns=stats["avg_conns"],
            avg_nodes=stats["avg_nodes"],
            max_conns=max(conn_counts) if conn_counts else 0,
            min_conns=min(conn_counts) if conn_counts else 0,
            weight_mean=float(np.mean(all_weights)) if all_weights else 0.0,
            weight_std=float(np.std(all_weights)) if all_weights else 0.0,
            weight_min=min(all_weights) if all_weights else 0.0,
            weight_max=max(all_weights) if all_weights else 0.0,
            activation_counts=dict(act_counts),
            n_evals=len(genomes),
            time_s=gen_time,
        )
        self.history.append(gs)

        # snapshot best genome every 5 gens
        if pop.best_genome is not None and (len(self.history) % 5 == 0 or len(self.history) <= 3):
            self.best_genome_snapshots.append(genome_to_json(pop.best_genome, {
                "generation": stats["generation"],
                "fitness": pop.best_fitness,
            }))

        return gs

    def to_dict(self) -> Dict:
        return {
            "history": [asdict(h) for h in self.history],
            "best_genome_snapshots": self.best_genome_snapshots,
        }


# ------------------------------------------- instrumented round runner ----
def run_instrumented_round(
    env_name: str,
    round_id: int,
    hp: Dict,
    n_gens: int = 30,
    n_avg: int = 1,
    env_seed: int = 42,
    notes: str = "",
    record_video: bool = False,
    video_dir: Optional[str] = None,
    save_genomes_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run a round with full instrumentation."""
    cfg = ENVS[env_name]
    env = gym.make(cfg.name)
    pop = build_population(cfg, hp)
    obs_norm = ObsNormalizer(cfg.n_inputs)
    collector = TrainingStatsCollector()

    raw_best_tracker = {"best": -float("inf")}

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

    history = []
    t0 = time.time()
    for gen in range(n_gens):
        t_gen = time.time()
        stats = pop.step(fit)
        gen_time = time.time() - t_gen
        gs = collector.collect(pop, stats, gen_time)
        raw_best_seen = raw_best_tracker["best"]
        raw_mean = stats["mean_fitness"] - cfg.abs_shift
        if cfg.name in ("BipedalWalker-v3", "LunarLander-v3"):
            raw_mean -= cfg.max_steps * 0.25
        history.append({
            "gen": stats["generation"],
            "best": raw_best_seen,
            "mean": raw_mean,
            "species": stats["n_species"],
            "conns": stats["avg_conns"],
            "nodes": stats["avg_nodes"],
            "threshold": stats["species_threshold"],
        })
        if verbose:
            print(f"  [{env_name} R{round_id}] gen {stats['generation']:>3d}: "
                  f"best={raw_best_seen:>8.2f} mean={raw_mean:>8.2f} "
                  f"species={stats['n_species']:>2d} "
                  f"conns={stats['avg_conns']:>4.1f} "
                  f"w_std={gs.weight_std:.3f} "
                  f"({gen_time:.1f}s)", flush=True)

    env.close()

    # record video of best genome
    video_path = None
    if record_video and pop.best_genome is not None and video_dir:
        video_path = os.path.join(video_dir, f"{env_name}-R{round_id}")
        try:
            venv = make_video_env(cfg.name, video_path,
                                   name_prefix=f"{env_name}-R{round_id}")
            vobs_norm = ObsNormalizer(cfg.n_inputs)
            for s in range(5):
                obs, _ = venv.reset(seed=env_seed + s)
                vobs_norm.update(np.asarray(obs, dtype=np.float64))
            evaluate_episode(pop.best_genome, venv, cfg, vobs_norm, seed=env_seed)
            venv.close()
        except Exception as e:
            print(f"  video recording failed: {e}")

    # save best genome JSON
    if save_genomes_dir and pop.best_genome is not None:
        save_genome_json(pop.best_genome,
                          os.path.join(save_genomes_dir, f"{env_name}-R{round_id}-best.json"),
                          {"round_id": round_id, "env": env_name})

    total_time = time.time() - t0
    return {
        "env": env_name,
        "round_id": round_id,
        "hyperparams": dict(hp),
        "notes": notes,
        "history": history,
        "stats": collector.to_dict(),
        "best_fitness": max(h["best"] for h in history) if history else -float("inf"),
        "final_best": history[-1]["best"] if history else -float("inf"),
        "final_mean": history[-1]["mean"] if history else 0.0,
        "solved_gen": next((h["gen"] for h in history if h["best"] >= cfg.solved_reward), None),
        "time_s": total_time,
        "video_path": video_path,
    }


if __name__ == "__main__":
    # quick smoke test
    res = run_instrumented_round(
        "MountainCar-v0", 999,
        {"pop_size": 20, "seed": 0},
        n_gens=5, verbose=True,
        save_genomes_dir="/home/z/my-project/download/genomes",
    )
    print(f"\nBest: {res['best_fitness']:.2f}")
    print(f"Stats collected: {len(res['stats']['history'])} gens")
    print(f"Genome snapshots: {len(res['stats']['best_genome_snapshots'])}")
