"""
Run ablation experiments on the modified NEAT algorithm.

Ablations:
    1. Speciation mode (Single, Standard, Purge)
    2. Optimizer on/off
    3. Mutation policy (per_type, single)
    4. Crossover topology method (FITTER, MORE_CONNS, COMBINE)
    5. Crossover weight method (INDEPENDENT, AVERAGE, BY_NEURON)
    6. Similarity method (STANDARD, PERCENTAGE)
    7. Mutation types on/off (weight only, +conn, +neuron, +prune)
    8. Population size (small vs large)
    9. Elitism count
    10. Reward shaping on/off

For each ablation, we train for a small number of generations and record:
    - Best fitness over time
    - Genome size over time
    - Species count over time
    - Final eval mean reward

Results are saved to results/ablations/<ablation_name>.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import copy
import numpy as np
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat import Config, Population
from neat.config import (
    SpeciationKind, SimilarityKind, MutationPolicyKind,
    CrossoverTopology, CrossoverWeights,
)
from neat.envs import make_env
from neat.analysis import TrainingStats, visualize_genome, capture_training_trajectory
from scripts.train import build_default_config, train
from scripts.evaluate import evaluate_genome, SOLVED_THRESHOLDS


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------
def run_ablation(name: str, env_name: str, cfg: Config, n_gens: int,
                  eval_episodes: int = 20, max_steps: int = 500,
                  capture_genome_gif: bool = False,
                  verbose: bool = True) -> Dict[str, Any]:
    """Run one ablation experiment.

    Returns:
        {
            "name": ...,
            "env": ...,
            "config": cfg.to_dict(),
            "history": [...],  # per-gen stats
            "best_fitness_train": ...,
            "eval_mean": ...,
            "eval_std": ...,
            "best_genome": {...},  # final best genome dict
            "elapsed_s": ...,
        }
    """
    if verbose:
        print(f"\n--- Ablation: {name} on {env_name} ---")
    t0 = time.time()

    # Build population and training stats collector
    pop = Population(cfg)
    stats = TrainingStats(snapshot_every=max(1, n_gens // 5))
    env_wrapper = make_env(env_name, max_steps=max_steps, n_eval_episodes=1, seed=cfg.seed)

    best_ever_fitness = -float("inf")
    best_ever_genome = None

    for gen in range(n_gens):
        # Multi-seed eval (more stable)
        n_train_seeds = 1
        if env_name == "Blackjack-v1":
            n_train_seeds = 10
        elif env_name == "MountainCar-v0":
            n_train_seeds = 5
        elif env_name in ("Acrobot-v1", "LunarLander-v3"):
            n_train_seeds = 3

        base_seed = cfg.seed * 10000 + gen * 137
        def eval_fn(g, _base=base_seed, _n=n_train_seeds):
            if _n <= 1:
                r, _, _ = env_wrapper.evaluate(g, episode_seed=_base)
                return r
            rewards = []
            for k in range(_n):
                r, _, _ = env_wrapper.evaluate(g, episode_seed=_base + k * 7919)
                rewards.append(r)
            return float(np.mean(rewards))

        pop.evaluate(eval_fn)
        gen_best = pop.best()
        if gen_best and gen_best.fitness > best_ever_fitness:
            best_ever_fitness = gen_best.fitness
            best_ever_genome = gen_best.clone()

        stats.record_generation(pop, gen, eval_fn)
        if verbose and (gen % max(1, n_gens // 10) == 0 or gen == n_gens - 1):
            elapsed = time.time() - t0
            h = stats.history[-1]
            print(f"  gen {gen+1:3d}/{n_gens} | max={h['fitness']['max']:7.2f} "
                  f"mean={h['fitness']['mean']:7.2f} | sp={h['species']['count']:2d} "
                  f"conns={h['genome_size']['avg_conns']:5.1f} t={elapsed:.1f}s")
        pop.step()

    # Evaluate best genome with raw rewards
    eval_rewards = []
    eval_env = make_env(env_name, max_steps=max_steps, n_eval_episodes=1,
                         seed=cfg.seed + 99999)
    for ep in range(eval_episodes):
        r, _, _ = eval_env.evaluate_raw(best_ever_genome,
                                          episode_seed=cfg.seed * 1000 + ep)
        eval_rewards.append(r)
    eval_mean = float(np.mean(eval_rewards))
    eval_std = float(np.std(eval_rewards))
    threshold = SOLVED_THRESHOLDS.get(env_name, (float("inf"), 100))[0]

    result = {
        "name": name,
        "env": env_name,
        "config": cfg.to_dict(),
        "history": stats.history,
        "best_fitness_train": float(best_ever_fitness),
        "eval_mean": eval_mean,
        "eval_std": eval_std,
        "solved": bool(eval_mean >= threshold),
        "threshold": threshold,
        "elapsed_s": time.time() - t0,
        "best_genome": stats._genome_to_dict(best_ever_genome) if best_ever_genome else None,
    }

    # Save stats object too (for plotting)
    output_dir = os.path.join("results", "ablations")
    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, f"{name}_stats.json")
    stats.save(stats_file)

    # Visualize the best genome
    if best_ever_genome:
        viz_path = os.path.join(output_dir, f"{name}_genome.png")
        try:
            visualize_genome(best_ever_genome, viz_path,
                              title=f"{name} - Best Genome (fit={best_ever_fitness:.1f})")
            result["genome_viz_path"] = viz_path
        except Exception as e:
            print(f"  WARNING: visualize_genome failed: {e}")

        # Capture a behavior GIF
        if capture_genome_gif:
            try:
                gif_path = os.path.join(output_dir, f"{name}_behavior.gif")
                capture_training_trajectory(best_ever_genome, env_name, gif_path,
                                              max_steps=max_steps, seed=cfg.seed + 42)
                result["behavior_gif_path"] = gif_path
            except Exception as e:
                print(f"  WARNING: capture_training_trajectory failed: {e}")

    # Save result
    result_file = os.path.join(output_dir, f"{name}.json")
    with open(result_file, "w") as f:
        # Don't save the full history again (it's in stats_file)
        slim_result = {k: v for k, v in result.items()
                        if k not in ("history", "config")}
        slim_result["config"] = cfg.to_dict()
        # Don't write best_genome here either (too verbose)
        slim_result["best_genome_summary"] = {
            "n_nodes": result["best_genome"]["n_nodes"] if result["best_genome"] else 0,
            "n_conns": result["best_genome"]["n_conns"] if result["best_genome"] else 0,
            "fitness": result["best_genome"]["fitness"] if result["best_genome"] else 0,
        }
        json.dump(slim_result, f, indent=2)

    if verbose:
        print(f"  => eval_mean={eval_mean:.2f}±{eval_std:.2f}  "
              f"solved={result['solved']}  t={result['elapsed_s']:.1f}s")
    return result


# ---------------------------------------------------------------------------
# Build ablation configs
# ---------------------------------------------------------------------------
def get_base_config(env_name: str, pop_size: int = 50) -> Config:
    """Get the best known config for an env (used as ablation baseline)."""
    cfg = build_default_config(env_name, seed=0, pop_size=pop_size)
    # Apply best known hyperparameters
    if env_name == "CartPole-v1":
        cfg.mutation.weight_std = 0.3
        cfg.policy.weight_prob = 0.81
        cfg.policy.conn_prob = 0.471
        cfg.policy.neuron_prob = 0.137
        cfg.generation.elitism = 2
    elif env_name == "Acrobot-v1":
        cfg.mutation.weight_std = 0.3
        cfg.mutation.conn_std = 0.5
        cfg.policy.weight_prob = 0.9
        cfg.policy.conn_prob = 0.3
        cfg.policy.neuron_prob = 0.1
        cfg.policy.prune_prob = 0.04
        cfg.generation.elitism = 2
        cfg.init.n_conn_multiplier = 1.13
        cfg.init.weight_init_multiplier = 2.0
    elif env_name == "MountainCar-v0":
        cfg.mutation.weight_std = 0.6
        cfg.mutation.conn_std = 0.8
        cfg.policy.weight_prob = 1.0
        cfg.policy.conn_prob = 0.5
        cfg.policy.neuron_prob = 0.12
        cfg.policy.prune_prob = 0.02
        cfg.generation.elitism = 3
        cfg.generation.cull_pct = 0.6
        cfg.speciation.target_species = 12
        cfg.init.n_conn_multiplier = 3.0
        cfg.init.weight_init_multiplier = 2.0
    elif env_name == "Blackjack-v1":
        cfg.mutation.weight_std = 0.3
        cfg.mutation.conn_std = 0.665
        cfg.policy.weight_prob = 1.0
        cfg.policy.conn_prob = 0.3
        cfg.policy.neuron_prob = 0.069
        cfg.generation.elitism = 2
        cfg.generation.cull_pct = 0.33
        cfg.init.n_conn_multiplier = 1.5
        cfg.init.weight_init_multiplier = 2.0
    elif env_name == "LunarLander-v3":
        cfg.mutation.weight_std = 0.215
        cfg.mutation.conn_std = 0.4
        cfg.policy.weight_prob = 1.0
        cfg.policy.conn_prob = 0.3
        cfg.policy.neuron_prob = 0.13
        cfg.policy.prune_prob = 0.03
        cfg.generation.elitism = 1
        cfg.generation.cull_pct = 0.57
        cfg.speciation.target_species = 10
        cfg.init.n_conn_multiplier = 0.99
        cfg.init.weight_init_multiplier = 1.08
    return cfg


def ablation_configs(env_name: str) -> List[Dict[str, Any]]:
    """Return a list of (name, cfg, description) for each ablation."""
    ablations = []
    base = get_base_config(env_name)

    # 1. Baseline (best known)
    ablations.append({
        "name": f"{env_name}_baseline",
        "cfg": base.copy(),
        "description": "Best known config (baseline)",
    })

    # 2. Speciation: Single (no speciation)
    cfg = base.copy()
    cfg.speciation.initial_kind = SpeciationKind.SINGLE
    cfg.speciation.subsequent_kind = SpeciationKind.SINGLE
    ablations.append({
        "name": f"{env_name}_spec_single",
        "cfg": cfg,
        "description": "No speciation (single species)",
    })

    # 3. Speciation: Standard only (no purge)
    cfg = base.copy()
    cfg.speciation.initial_kind = SpeciationKind.STANDARD
    cfg.speciation.subsequent_kind = SpeciationKind.STANDARD
    ablations.append({
        "name": f"{env_name}_spec_standard",
        "cfg": cfg,
        "description": "Standard speciation (no purge)",
    })

    # 4. Optimizer on
    cfg = base.copy()
    cfg.optimizer.enabled = True
    cfg.optimizer.lr = 0.1
    cfg.optimizer.method = "adam"
    ablations.append({
        "name": f"{env_name}_optimizer_on",
        "cfg": cfg,
        "description": "GRPO optimizer enabled (Adam)",
    })

    # 5. Single mutation policy
    cfg = base.copy()
    cfg.policy.kind = MutationPolicyKind.SINGLE
    ablations.append({
        "name": f"{env_name}_policy_single",
        "cfg": cfg,
        "description": "Single-pick mutation policy",
    })

    # 6. Crossover: COMBINE topology
    cfg = base.copy()
    cfg.crossover.topology = CrossoverTopology.COMBINE
    ablations.append({
        "name": f"{env_name}_xover_combine",
        "cfg": cfg,
        "description": "Combine-topology crossover",
    })

    # 7. Crossover: INDEPENDENT weights
    cfg = base.copy()
    cfg.crossover.weights = CrossoverWeights.INDEPENDENT
    ablations.append({
        "name": f"{env_name}_xover_indep",
        "cfg": cfg,
        "description": "Independent-weight crossover",
    })

    # 8. Similarity: STANDARD (NEAT paper)
    cfg = base.copy()
    cfg.speciation.similarity = SimilarityKind.STANDARD
    ablations.append({
        "name": f"{env_name}_sim_standard",
        "cfg": cfg,
        "description": "Standard NEAT similarity (vs Percentage)",
    })

    # 9. No neuron mutation
    cfg = base.copy()
    cfg.policy.neuron_prob = 0.0
    ablations.append({
        "name": f"{env_name}_no_neuron",
        "cfg": cfg,
        "description": "Disable neuron-split mutation",
    })

    # 10. No pruning
    cfg = base.copy()
    cfg.policy.prune_prob = 0.0
    ablations.append({
        "name": f"{env_name}_no_prune",
        "cfg": cfg,
        "description": "Disable pruning mutation",
    })

    # 11. No elitism
    cfg = base.copy()
    cfg.generation.elitism = 0
    ablations.append({
        "name": f"{env_name}_no_elite",
        "cfg": cfg,
        "description": "Disable elitism",
    })

    # 12. High elitism
    cfg = base.copy()
    cfg.generation.elitism = 5
    ablations.append({
        "name": f"{env_name}_high_elite",
        "cfg": cfg,
        "description": "Elitism = 5 (vs 1-3 baseline)",
    })

    return ablations


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", default="CartPole-v1,MountainCar-v0,Blackjack-v1",
                   help="Comma-separated env names")
    p.add_argument("--gens", type=int, default=20, help="Generations per ablation")
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--filter", default="", help="Only run ablations matching this substring")
    p.add_argument("--capture-gif", action="store_true",
                   help="Capture agent behavior GIF (slow)")
    args = p.parse_args()

    env_names = [s.strip() for s in args.envs.split(",")]
    all_results = {}

    for env_name in env_names:
        print(f"\n{'='*60}")
        print(f"ABLATIONS ON {env_name}")
        print(f"{'='*60}")
        # Per-env max steps
        max_steps_map = {"CartPole-v1": 500, "Acrobot-v1": 500,
                         "MountainCar-v0": 200, "LunarLander-v3": 1000,
                         "Blackjack-v1": 100}
        max_steps = max_steps_map.get(env_name, args.max_steps)

        ablations = ablation_configs(env_name)
        env_results = []
        for ab in ablations:
            if args.filter and args.filter not in ab["name"]:
                continue
            ab["cfg"].generation.pop_size = args.pop
            # Vary seed by ablation name hash for diversity
            ab["cfg"].seed = hash(ab["name"]) % 10000
            result = run_ablation(
                ab["name"], env_name, ab["cfg"],
                n_gens=args.gens,
                eval_episodes=args.eval_episodes,
                max_steps=max_steps,
                capture_genome_gif=args.capture_gif,
                verbose=True,
            )
            env_results.append({
                "name": result["name"],
                "description": ab["description"],
                "best_fitness_train": result["best_fitness_train"],
                "eval_mean": result["eval_mean"],
                "eval_std": result["eval_std"],
                "solved": result["solved"],
                "elapsed_s": result["elapsed_s"],
                "best_genome_summary": result.get("best_genome_summary"),
            })
        all_results[env_name] = env_results

    # Print summary
    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)
    for env_name, results in all_results.items():
        threshold = SOLVED_THRESHOLDS.get(env_name, (float("inf"), 100))[0]
        print(f"\n  [{env_name}] (threshold={threshold})")
        for r in sorted(results, key=lambda x: -x["eval_mean"]):
            marker = "✓" if r["solved"] else "✗"
            print(f"    {marker} {r['name']:40s}  eval={r['eval_mean']:7.2f} "
                  f"± {r['eval_std']:5.2f}  t={r['elapsed_s']:.1f}s")

    # Save summary
    os.makedirs("results/ablations", exist_ok=True)
    with open("results/ablations/summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/ablations/summary.json")


if __name__ == "__main__":
    main()
