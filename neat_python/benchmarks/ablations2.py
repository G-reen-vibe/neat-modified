"""
Additional ablations: crossover methods, fitness shaping, init config.
"""
from __future__ import annotations

import os
import sys
import json
import time
import copy
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.harness import ENVS
from benchmarks.sweep import DEFAULT_HP, merge_hp
from benchmarks.instrument import run_instrumented_round
from benchmarks.viz import plot_training_curves, plot_ablation_bars
from benchmarks.ablations import save_result, PLOTS_DIR, GENOMES_DIR

from neat.crossover import T_FITTER, T_MORE_CONNS, T_COMBINE, W_INDEPENDENT, W_AVERAGE, W_BY_NEURON


def ablation_crossover(env: str, n_gens: int = 25, pop: int = 40) -> None:
    """Compare crossover methods."""
    ablation = "crossover"
    print(f"\n{'='*60}\n  Ablation: crossover methods [{env}]\n{'='*60}")
    base_hp = {"pop_size": pop, "init_neurons": 1 if ENVS[env].n_inputs < 10 else 0,
               "target_species": 5, "threshold": 0.2,
               "crossover_pct": 0.5, "asexual_pct": 0.5}
    if ENVS[env].n_inputs >= 10:
        base_hp.update({"similarity_method": "standard", "threshold": 3.0,
                         "min_threshold": 0.5, "max_threshold": 10.0, "threshold_adjust": 0.5})
    variants = {
        "fitter_avg": {"topology_method": T_FITTER, "weight_method": W_AVERAGE},
        "fitter_indep": {"topology_method": T_FITTER, "weight_method": W_INDEPENDENT},
        "moreconns_avg": {"topology_method": T_MORE_CONNS, "weight_method": W_AVERAGE},
        "combine_avg": {"topology_method": T_COMBINE, "weight_method": W_AVERAGE},
        "byneuron": {"topology_method": T_FITTER, "weight_method": W_BY_NEURON},
    }
    results = []
    for vname, vhp in variants.items():
        hp = merge_hp({**base_hp, **vhp})
        label = f"xover {vname}"
        print(f"\n>>> {label}")
        res = run_instrumented_round(
            env, 0, hp, n_gens=n_gens, n_avg=1,
            notes=f"{ablation}: {label}",
            record_video=False, save_genomes_dir=GENOMES_DIR, verbose=True,
        )
        res["label"] = label
        res["variant"] = vname
        save_result(ablation, env, vname, res)
        results.append(res)
    histories = [r["history"] for r in results]
    labels = [r["label"] for r in results]
    plot_training_curves(histories, os.path.join(PLOTS_DIR, f"{ablation}_{env}_curves.png"),
                          title=f"Crossover Ablation - {env}", labels=labels)
    plot_ablation_bars(results, os.path.join(PLOTS_DIR, f"{ablation}_{env}_bars.png"),
                        title=f"Crossover Best Reward - {env}")


def ablation_init(env: str, n_gens: int = 25, pop: int = 40) -> None:
    """Compare initialization configs."""
    ablation = "init"
    print(f"\n{'='*60}\n  Ablation: initialization [{env}]\n{'='*60}")
    base_hp = {"pop_size": pop, "target_species": 5, "threshold": 0.2}
    if ENVS[env].n_inputs >= 10:
        base_hp.update({"similarity_method": "standard", "threshold": 3.0,
                         "min_threshold": 0.5, "max_threshold": 10.0, "threshold_adjust": 0.5})
    variants = {
        "minimal": {"init_neurons": 0, "init_mult": 1.0},
        "default": {"init_neurons": 1, "init_mult": 2.0},
        "bigger": {"init_neurons": 2, "init_mult": 2.5},
        "maximal": {"init_neurons": 4, "init_mult": 3.0},
    }
    results = []
    for vname, vhp in variants.items():
        hp = merge_hp({**base_hp, **vhp})
        label = f"init {vname}"
        print(f"\n>>> {label}")
        res = run_instrumented_round(
            env, 0, hp, n_gens=n_gens, n_avg=1,
            notes=f"{ablation}: {label}",
            record_video=False, save_genomes_dir=GENOMES_DIR, verbose=True,
        )
        res["label"] = label
        res["variant"] = vname
        save_result(ablation, env, vname, res)
        results.append(res)
    histories = [r["history"] for r in results]
    labels = [r["label"] for r in results]
    plot_training_curves(histories, os.path.join(PLOTS_DIR, f"{ablation}_{env}_curves.png"),
                          title=f"Initialization Ablation - {env}", labels=labels)
    plot_ablation_bars(results, os.path.join(PLOTS_DIR, f"{ablation}_{env}_bars.png"),
                        title=f"Init Best Reward - {env}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=["crossover", "init", "both"], default="both")
    p.add_argument("--env", default="all")
    p.add_argument("--gens", type=int, default=25)
    p.add_argument("--pop", type=int, default=40)
    args = p.parse_args()

    envs = list(ENVS.keys()) if args.env == "all" else [args.env]
    if args.ablation in ("crossover", "both"):
        for env in envs:
            ablation_crossover(env, n_gens=args.gens, pop=args.pop)
    if args.ablation in ("init", "both"):
        for env in envs:
            ablation_init(env, n_gens=args.gens, pop=args.pop)


if __name__ == "__main__":
    main()
