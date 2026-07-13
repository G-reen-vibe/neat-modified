"""
Run one well-instrumented 'showcase' run per env with the best config found.
Generates: dashboard PNG, genome graph snapshots, training video.
"""
from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.harness import ENVS
from benchmarks.sweep import DEFAULT_HP, merge_hp
from benchmarks.instrument import run_instrumented_round, save_genome_json
from benchmarks.viz import (
    plot_dashboard, plot_training_curves, plot_species_composition,
    plot_weight_distribution, plot_topology_growth, plot_activation_usage,
    plot_genome_graph,
)


SHOWCASE_DIR = "/home/z/my-project/download/showcase"
PLOTS_DIR = os.path.join(SHOWCASE_DIR, "plots")
VIDEOS_DIR = os.path.join(SHOWCASE_DIR, "videos")
GENOMES_DIR = os.path.join(SHOWCASE_DIR, "genomes")


# Best configs per env (from prior sweep)
BEST_CONFIGS = {
    "MountainCar-v0": {
        "pop_size": 50, "init_neurons": 0, "init_mult": 2.0,
        "neuron_prob": 0.15, "connection_prob": 0.2, "conn_std": 0.8,
        "weight_std": 0.1, "target_species": 6, "threshold": 0.2,
        "elitism": 2, "optimizer_enabled": True, "opt_lr": 0.01,
        "n_avg": 1,
    },
    "Acrobot-v1": {
        "pop_size": 60, "init_neurons": 1, "init_mult": 2.0,
        "neuron_prob": 0.1, "connection_prob": 0.15, "conn_std": 0.5,
        "weight_std": 0.1, "target_species": 5, "threshold": 0.2,
        "elitism": 2,
    },
    "Pendulum-v1": {
        "pop_size": 80, "init_neurons": 3, "init_mult": 3.0,
        "neuron_prob": 0.05, "connection_prob": 0.3, "conn_std": 1.5,
        "weight_std": 0.2, "weight_pct": 0.7,
        "target_species": 8, "threshold": 0.15, "elitism": 3, "n_avg": 2,
    },
    "LunarLander-v3": {
        "pop_size": 50, "init_neurons": 0, "neuron_prob": 0.1,
        "connection_prob": 0.15, "target_species": 6, "threshold": 0.2,
        "elitism": 3, "optimizer_enabled": True, "opt_lr": 0.01,
    },
    "BipedalWalker-v3": {
        "pop_size": 40, "init_neurons": 0, "init_mult": 1.0,
        "neuron_prob": 0.05, "connection_prob": 0.1, "weight_std": 0.05,
        "target_species": 6, "threshold": 3.0,
        "min_threshold": 0.5, "max_threshold": 10.0, "threshold_adjust": 0.5,
        "similarity_method": "standard", "elitism": 3,
    },
}


def run_showcase(env_name: str, n_gens: int = 40) -> Dict:
    cfg = BEST_CONFIGS[env_name]
    hp = merge_hp(cfg)
    print(f"\n{'='*60}\n  Showcase: {env_name} ({n_gens} gens)\n{'='*60}")
    env_dir = env_name.replace("-", "_")
    video_dir = os.path.join(VIDEOS_DIR, env_dir)
    genomes_dir = os.path.join(GENOMES_DIR, env_dir)
    plots_dir = os.path.join(PLOTS_DIR, env_dir)
    for d in [video_dir, genomes_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    res = run_instrumented_round(
        env_name, 0, hp, n_gens=n_gens, n_avg=hp.get("n_avg", 1),
        notes=f"showcase: best config for {env_name}",
        record_video=True, video_dir=video_dir,
        save_genomes_dir=genomes_dir,
        verbose=True,
    )
    # generate all plots
    stats_history = res["stats"]["history"]
    history = res["history"]
    plot_dashboard(stats_history, history,
                    os.path.join(plots_dir, "dashboard.png"),
                    title=f"{env_name} - Training Dashboard")
    plot_training_curves([history], os.path.join(plots_dir, "curves.png"),
                          title=f"{env_name} - Training Curves")
    plot_species_composition(stats_history, os.path.join(plots_dir, "species.png"),
                              title=f"{env_name} - Species Composition")
    plot_weight_distribution(stats_history, os.path.join(plots_dir, "weights.png"),
                              title=f"{env_name} - Weight Distribution")
    plot_topology_growth(stats_history, os.path.join(plots_dir, "topology.png"),
                          title=f"{env_name} - Topology Growth")
    plot_activation_usage(stats_history, os.path.join(plots_dir, "activations.png"),
                           title=f"{env_name} - Activation Usage")
    # plot genome snapshots
    for i, snap in enumerate(res["stats"]["best_genome_snapshots"]):
        gen = snap["extra"]["generation"]
        plot_genome_graph(snap, os.path.join(plots_dir, f"genome_gen{gen}.png"),
                           title=f"{env_name} - Best Genome @ gen {gen} (fit={snap['extra']['fitness']:.1f})")
    # save summary
    summary = {
        "env": env_name,
        "best_fitness": res["best_fitness"],
        "final_best": res["final_best"],
        "solved_gen": res["solved_gen"],
        "time_s": res["time_s"],
        "n_gens": n_gens,
        "config": cfg,
        "stats_history": stats_history,
    }
    with open(os.path.join(plots_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Best: {res['best_fitness']:.2f}, solved_gen: {res['solved_gen']}")
    print(f"  Plots in: {plots_dir}")
    return summary


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="all")
    p.add_argument("--gens", type=int, default=40)
    args = p.parse_args()

    envs = list(ENVS.keys()) if args.env == "all" else [args.env]
    summaries = {}
    for env in envs:
        try:
            summaries[env] = run_showcase(env, n_gens=args.gens)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
    with open(os.path.join(SHOWCASE_DIR, "summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nAll showcases done. Summaries in {SHOWCASE_DIR}/summaries.json")


if __name__ == "__main__":
    main()
