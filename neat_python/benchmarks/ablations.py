"""
Ablation study runner.

Runs controlled experiments comparing algorithm variants, with full
instrumentation (stats, videos, genome snapshots). Results saved as
JSON + PNG plots for the report.

Each ablation run is identified by:
  ablation_name / env / variant
"""
from __future__ import annotations

import os
import sys
import json
import time
import copy
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.harness import ENVS
from benchmarks.sweep import DEFAULT_HP, merge_hp
from benchmarks.instrument import run_instrumented_round
from benchmarks.viz import (
    plot_dashboard, plot_training_curves, plot_species_composition,
    plot_weight_distribution, plot_topology_growth, plot_activation_usage,
    plot_ablation_bars, plot_genome_graph,
)


OUT_DIR = "/home/z/my-project/download/ablation"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
VIDEOS_DIR = os.path.join(OUT_DIR, "videos")
GENOMES_DIR = os.path.join(OUT_DIR, "genomes")
RESULTS_FILE = os.path.join(OUT_DIR, "results.json")


def ensure_dirs():
    for d in [PLOTS_DIR, VIDEOS_DIR, GENOMES_DIR]:
        os.makedirs(d, exist_ok=True)


def save_result(ablation: str, env: str, variant: str, result: Dict) -> None:
    """Append result to results JSON."""
    all_results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            all_results = json.load(f)
    key = f"{ablation}/{env}/{variant}"
    all_results[key] = {
        "ablation": ablation,
        "env": env,
        "variant": variant,
        "result": result,
        "timestamp": time.time(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


# --------------------------------------------------------- ablations -------
def ablation_grpo(env: str, n_gens: int = 30, pop: int = 40) -> None:
    """Compare GRPO optimizer on vs off."""
    ablation = "grpo"
    print(f"\n{'='*60}\n  Ablation: GRPO optimizer on vs off [{env}]\n{'='*60}")
    base_hp = {"pop_size": pop, "init_neurons": 1 if ENVS[env].n_inputs < 10 else 0,
               "target_species": 5, "threshold": 0.2}
    if ENVS[env].n_inputs >= 10:
        base_hp.update({"similarity_method": "standard", "threshold": 3.0,
                         "min_threshold": 0.5, "max_threshold": 10.0, "threshold_adjust": 0.5})
    variants = {
        "off": {"optimizer_enabled": False},
        "on_lr0.01": {"optimizer_enabled": True, "opt_lr": 0.01, "opt_method": "adam"},
        "on_lr0.05": {"optimizer_enabled": True, "opt_lr": 0.05, "opt_method": "adam"},
    }
    results = []
    for vname, vhp in variants.items():
        hp = merge_hp({**base_hp, **vhp})
        label = f"GRPO {vname}"
        print(f"\n>>> {label}")
        res = run_instrumented_round(
            env, 0, hp, n_gens=n_gens, n_avg=1,
            notes=f"{ablation}: {label}",
            record_video=False,
            save_genomes_dir=GENOMES_DIR,
            verbose=True,
        )
        res["label"] = label
        res["variant"] = vname
        save_result(ablation, env, vname, res)
        results.append(res)
    # plot comparison
    histories = [r["history"] for r in results]
    labels = [r["label"] for r in results]
    plot_training_curves(histories, os.path.join(PLOTS_DIR, f"{ablation}_{env}_curves.png"),
                          title=f"GRPO Ablation - {env}", labels=labels)
    plot_ablation_bars(results, os.path.join(PLOTS_DIR, f"{ablation}_{env}_bars.png"),
                        title=f"GRPO Best Reward - {env}", metric="best_fitness",
                        metric_label="Best Reward")
    print(f"\n  Saved plots: {PLOTS_DIR}/{ablation}_{env}_*.png")


def ablation_speciation(env: str, n_gens: int = 30, pop: int = 40) -> None:
    """Compare speciation policies: single, standard, purge_then_standard."""
    ablation = "speciation"
    print(f"\n{'='*60}\n  Ablation: speciation policies [{env}]\n{'='*60}")
    base_hp = {"pop_size": pop, "init_neurons": 1 if ENVS[env].n_inputs < 10 else 0,
               "threshold": 0.2}
    if ENVS[env].n_inputs >= 10:
        base_hp.update({"similarity_method": "standard", "threshold": 3.0,
                         "min_threshold": 0.5, "max_threshold": 10.0, "threshold_adjust": 0.5})
    variants = {
        "single": {"speciation_policy": "single"},
        "standard": {"speciation_policy": "standard"},
        "purge_then_standard": {"speciation_policy": "purge_then_standard",
                                  "target_species": 5},
    }
    results = []
    for vname, vhp in variants.items():
        hp = merge_hp({**base_hp, **vhp})
        label = f"spec {vname}"
        print(f"\n>>> {label}")
        # we need to override the Population's speciation_policy param.
        # build_population hardcodes it; let's monkeypatch via a custom call
        from benchmarks.harness import build_population as _bp
        import benchmarks.harness as _h
        orig_bp = _h.build_population
        def patched_bp(cfg, hp):
            pop = orig_bp(cfg, hp)
            # speciation_policy isn't a direct hp; we set it via the speciator
            if "speciation_policy" in hp:
                from neat.speciation import Speciator
                pop.speciator = Speciator(
                    policy=hp["speciation_policy"],
                    target_species=hp.get("target_species", 5),
                    threshold=hp.get("threshold", 0.25),
                    min_threshold=hp.get("min_threshold", 0.05),
                    max_threshold=hp.get("max_threshold", 0.5),
                    adjust=hp.get("threshold_adjust", 0.025),
                    similarity_method=hp.get("similarity_method", "percentage"),
                )
            return pop
        _h.build_population = patched_bp
        try:
            res = run_instrumented_round(
                env, 0, hp, n_gens=n_gens, n_avg=1,
                notes=f"{ablation}: {label}",
                record_video=False, save_genomes_dir=GENOMES_DIR, verbose=True,
            )
        finally:
            _h.build_population = orig_bp
        res["label"] = label
        res["variant"] = vname
        save_result(ablation, env, vname, res)
        results.append(res)
    histories = [r["history"] for r in results]
    labels = [r["label"] for r in results]
    plot_training_curves(histories, os.path.join(PLOTS_DIR, f"{ablation}_{env}_curves.png"),
                          title=f"Speciation Ablation - {env}", labels=labels)
    plot_ablation_bars(results, os.path.join(PLOTS_DIR, f"{ablation}_{env}_bars.png"),
                        title=f"Speciation Best Reward - {env}")


def ablation_similarity(env: str, n_gens: int = 30, pop: int = 40) -> None:
    """Compare percentage vs standard similarity."""
    ablation = "similarity"
    print(f"\n{'='*60}\n  Ablation: similarity methods [{env}]\n{'='*60}")
    base_hp = {"pop_size": pop, "init_neurons": 1 if ENVS[env].n_inputs < 10 else 0,
               "target_species": 5}
    variants = {
        "percentage": {"similarity_method": "percentage", "threshold": 0.2,
                        "min_threshold": 0.05, "max_threshold": 0.5, "threshold_adjust": 0.025},
        "standard": {"similarity_method": "standard", "threshold": 3.0,
                      "min_threshold": 0.5, "max_threshold": 10.0, "threshold_adjust": 0.5},
    }
    results = []
    for vname, vhp in variants.items():
        hp = merge_hp({**base_hp, **vhp})
        label = f"sim {vname}"
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
                          title=f"Similarity Ablation - {env}", labels=labels)
    plot_ablation_bars(results, os.path.join(PLOTS_DIR, f"{ablation}_{env}_bars.png"),
                        title=f"Similarity Best Reward - {env}")


def ablation_mutation_rates(env: str, n_gens: int = 30, pop: int = 40) -> None:
    """Compare mutation rate combinations."""
    ablation = "mutation"
    print(f"\n{'='*60}\n  Ablation: mutation rates [{env}]\n{'='*60}")
    base_hp = {"pop_size": pop, "init_neurons": 1 if ENVS[env].n_inputs < 10 else 0,
               "target_species": 5, "threshold": 0.2}
    if ENVS[env].n_inputs >= 10:
        base_hp.update({"similarity_method": "standard", "threshold": 3.0,
                         "min_threshold": 0.5, "max_threshold": 10.0, "threshold_adjust": 0.5})
    variants = {
        "low": {"weight_prob": 0.5, "connection_prob": 0.05, "neuron_prob": 0.02, "pruning_prob": 0.01},
        "default": {"weight_prob": 0.9, "connection_prob": 0.15, "neuron_prob": 0.08, "pruning_prob": 0.03},
        "high": {"weight_prob": 1.0, "connection_prob": 0.3, "neuron_prob": 0.15, "pruning_prob": 0.05},
    }
    results = []
    for vname, vhp in variants.items():
        hp = merge_hp({**base_hp, **vhp})
        label = f"mut {vname}"
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
                          title=f"Mutation Rate Ablation - {env}", labels=labels)
    plot_ablation_bars(results, os.path.join(PLOTS_DIR, f"{ablation}_{env}_bars.png"),
                        title=f"Mutation Rate Best Reward - {env}")


def ablation_pop_size(env: str, n_gens: int = 30) -> None:
    """Compare population sizes."""
    ablation = "popsize"
    print(f"\n{'='*60}\n  Ablation: population size [{env}]\n{'='*60}")
    base_hp = {"init_neurons": 1 if ENVS[env].n_inputs < 10 else 0,
               "target_species": 5, "threshold": 0.2}
    if ENVS[env].n_inputs >= 10:
        base_hp.update({"similarity_method": "standard", "threshold": 3.0,
                         "min_threshold": 0.5, "max_threshold": 10.0, "threshold_adjust": 0.5})
    variants = {
        "pop20": {"pop_size": 20},
        "pop40": {"pop_size": 40},
        "pop80": {"pop_size": 80},
    }
    results = []
    for vname, vhp in variants.items():
        hp = merge_hp({**base_hp, **vhp})
        label = vname
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
                          title=f"Population Size Ablation - {env}", labels=labels)
    plot_ablation_bars(results, os.path.join(PLOTS_DIR, f"{ablation}_{env}_bars.png"),
                        title=f"Pop Size Best Reward - {env}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=["grpo", "speciation", "similarity", "mutation", "popsize", "all"], default="all")
    p.add_argument("--env", default="all", help="env name or 'all'")
    p.add_argument("--gens", type=int, default=25)
    p.add_argument("--pop", type=int, default=40)
    args = p.parse_args()

    ensure_dirs()
    envs = list(ENVS.keys()) if args.env == "all" else [args.env]

    if args.ablation in ("grpo", "all"):
        for env in envs:
            ablation_grpo(env, n_gens=args.gens, pop=args.pop)
    if args.ablation in ("speciation", "all"):
        for env in envs:
            ablation_speciation(env, n_gens=args.gens, pop=args.pop)
    if args.ablation in ("similarity", "all"):
        for env in envs:
            ablation_similarity(env, n_gens=args.gens, pop=args.pop)
    if args.ablation in ("mutation", "all"):
        for env in envs:
            ablation_mutation_rates(env, n_gens=args.gens, pop=args.pop)
    if args.ablation in ("popsize", "all"):
        for env in envs:
            ablation_pop_size(env, n_gens=args.gens)

    print(f"\n{'='*60}\n  All ablations done. Results in {RESULTS_FILE}\n{'='*60}")


if __name__ == "__main__":
    main()
