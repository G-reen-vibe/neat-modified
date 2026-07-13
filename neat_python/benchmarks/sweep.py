"""
Hyperparameter sweep runner.

Manages rounds of hyperparameter optimization, persists results to a JSON
file, and provides utilities for analyzing results across rounds.

Each round has a unique ID (per env) and stores:
  - hyperparameters used
  - best/mean fitness over generations
  - solved_gen (if solved)
  - notes (what we tried, what we observed)
  - history (per-generation stats)

Usage:
    python3 sweep.py --env MountainCar-v0 --round 1 --gens 30 --notes "baseline"
    python3 sweep.py --analyze MountainCar-v0
    python3 sweep.py --analyze-all
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.harness import ENVS, run_round, RoundResult

RESULTS_FILE = "/home/z/my-project/download/sweep-results.json"


def load_results() -> Dict:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {"rounds": {}}


def save_results(data: Dict) -> None:
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def add_result(result: RoundResult) -> None:
    data = load_results()
    env = result.env
    if env not in data["rounds"]:
        data["rounds"][env] = []
    data["rounds"][env].append({
        "round_id": result.round_id,
        "hyperparams": result.hyperparams,
        "best_fitness": result.best_fitness,
        "mean_fitness": result.mean_fitness,
        "final_best": result.final_best,
        "final_mean": result.final_mean,
        "n_species": result.n_species,
        "n_gens": result.n_gens,
        "solved_gen": result.solved_gen,
        "avg_conns": result.avg_conns,
        "avg_nodes": result.avg_nodes,
        "time_s": result.time_s,
        "history": result.history,
        "notes": result.notes,
        "timestamp": time.time(),
    })
    save_results(data)


def next_round_id(env: str) -> int:
    data = load_results()
    rounds = data["rounds"].get(env, [])
    return max((r["round_id"] for r in rounds), default=0) + 1


# ----------------------------------------------------- default hyperparams -
DEFAULT_HP: Dict[str, Any] = {
    "pop_size": 60,
    "weight_prob": 0.9,
    "weight_pct": 0.5,
    "weight_std": 0.1,
    "connection_prob": 0.15,
    "conn_std": 0.5,
    "neuron_prob": 0.08,
    "pruning_prob": 0.03,
    "init_mult": 2.0,
    "init_neurons": 1,
    "target_species": 5,
    "threshold": 0.25,
    "elitism": 2,
    "cull_pct": 0.5,
    "n_interspecies": 1,
    "asexual_pct": 0.8,
    "crossover_pct": 0.2,
    "optimizer_enabled": False,
    "opt_lr": 0.02,
    "opt_method": "adam",
    "seed": 0,
}


def merge_hp(overrides: Dict) -> Dict:
    hp = copy.deepcopy(DEFAULT_HP)
    hp.update(overrides)
    return hp


# ----------------------------------------------------------- analysis ------
def analyze_env(env: str) -> None:
    data = load_results()
    rounds = data["rounds"].get(env, [])
    cfg = ENVS[env]
    print(f"\n{'='*72}")
    print(f"  {env}  (target: {cfg.solved_reward}, max_steps: {cfg.max_steps})")
    print(f"{'='*72}")
    print(f"{'R':>3} {'Best':>10} {'FinalBest':>10} {'Mean':>10} {'Solved':>7} {'Time':>7}  Notes")
    print(f"{'-'*3} {'-'*10} {'-'*10} {'-'*10} {'-'*7} {'-'*7}  {'-'*30}")
    best_round = None
    best_score = -float("inf")
    for r in rounds:
        score = r["best_fitness"]
        if score > best_score:
            best_score = score
            best_round = r
        solved = f"gen{r['solved_gen']}" if r["solved_gen"] is not None else "-"
        print(f"{r['round_id']:>3d} {r['best_fitness']:>10.2f} {r['final_best']:>10.2f} "
              f"{r['mean_fitness']:>10.2f} {solved:>7} {r['time_s']:>6.1f}s  {r['notes'][:50]}")
    if best_round:
        print(f"\n  Best round: R{best_round['round_id']} = {best_round['best_fitness']:.2f}")
        # show key hyperparams that differ from default
        hp = best_round["hyperparams"]
        diffs = {k: v for k, v in hp.items() if DEFAULT_HP.get(k) != v}
        if diffs:
            print(f"  Non-default HPs: {json.dumps(diffs)}")
    print(f"  Total rounds: {len(rounds)}")


def analyze_all() -> None:
    data = load_results()
    print("\n" + "=" * 72)
    print("  SWEEP SUMMARY (all envs)")
    print("=" * 72)
    print(f"{'Env':<22} {'Rounds':>6} {'Best':>10} {'Solved?':>8}")
    print(f"{'-'*22} {'-'*6} {'-'*10} {'-'*8}")
    for env in ENVS:
        rounds = data["rounds"].get(env, [])
        if not rounds:
            print(f"{env:<22} {0:>6} {'-':>10} {'-':>8}")
            continue
        best = max(r["best_fitness"] for r in rounds)
        any_solved = any(r["solved_gen"] is not None for r in rounds)
        print(f"{env:<22} {len(rounds):>6} {best:>10.2f} {'YES' if any_solved else 'no':>8}")
    for env in ENVS:
        analyze_env(env)


# --------------------------------------------------------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(ENVS.keys()), help="env to run")
    p.add_argument("--round", type=int, default=None, help="round id (default: auto)")
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-avg", type=int, default=1)
    p.add_argument("--notes", type=str, default="")
    # HP overrides
    p.add_argument("--weight-prob", type=float, default=None)
    p.add_argument("--weight-pct", type=float, default=None)
    p.add_argument("--weight-std", type=float, default=None)
    p.add_argument("--connection-prob", type=float, default=None)
    p.add_argument("--conn-std", type=float, default=None)
    p.add_argument("--neuron-prob", type=float, default=None)
    p.add_argument("--pruning-prob", type=float, default=None)
    p.add_argument("--init-neurons", type=int, default=None)
    p.add_argument("--init-mult", type=float, default=None)
    p.add_argument("--target-species", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--elitism", type=int, default=None)
    p.add_argument("--cull-pct", type=float, default=None)
    p.add_argument("--optimizer", action="store_true", default=None)
    p.add_argument("--no-optimizer", action="store_true", default=None)
    p.add_argument("--opt-lr", type=float, default=None)
    p.add_argument("--opt-method", type=str, default=None, choices=["sgd", "momentum", "rmsprop", "adam"])
    # actions
    p.add_argument("--analyze", type=str, default=None, help="analyze one env and exit")
    p.add_argument("--analyze-all", action="store_true")
    args = p.parse_args()

    if args.analyze_all:
        analyze_all()
        return
    if args.analyze:
        analyze_env(args.analyze)
        return
    if not args.env:
        p.print_help()
        return

    # build HP overrides
    overrides = {}
    if args.pop is not None: overrides["pop_size"] = args.pop
    if args.weight_prob is not None: overrides["weight_prob"] = args.weight_prob
    if args.weight_pct is not None: overrides["weight_pct"] = args.weight_pct
    if args.weight_std is not None: overrides["weight_std"] = args.weight_std
    if args.connection_prob is not None: overrides["connection_prob"] = args.connection_prob
    if args.conn_std is not None: overrides["conn_std"] = args.conn_std
    if args.neuron_prob is not None: overrides["neuron_prob"] = args.neuron_prob
    if args.pruning_prob is not None: overrides["pruning_prob"] = args.pruning_prob
    if args.init_neurons is not None: overrides["init_neurons"] = args.init_neurons
    if args.init_mult is not None: overrides["init_mult"] = args.init_mult
    if args.target_species is not None: overrides["target_species"] = args.target_species
    if args.threshold is not None: overrides["threshold"] = args.threshold
    if args.elitism is not None: overrides["elitism"] = args.elitism
    if args.cull_pct is not None: overrides["cull_pct"] = args.cull_pct
    if args.optimizer: overrides["optimizer_enabled"] = True
    if args.no_optimizer: overrides["optimizer_enabled"] = False
    if args.opt_lr is not None: overrides["opt_lr"] = args.opt_lr
    if args.opt_method is not None: overrides["opt_method"] = args.opt_method
    overrides["seed"] = args.seed

    hp = merge_hp(overrides)
    round_id = args.round if args.round is not None else next_round_id(args.env)
    print(f"\n>>> Running {args.env} round {round_id} ({args.gens} gens, pop={hp['pop_size']})")
    if args.notes:
        print(f"    Notes: {args.notes}")
    print(f"    HP overrides: {overrides}")
    print()

    res = run_round(args.env, round_id, hp, n_gens=args.gens, n_avg=args.n_avg, notes=args.notes)
    add_result(res)
    print(f"\n=== Result: best={res.best_fitness:.2f}, final_best={res.final_best:.2f}, "
          f"solved_gen={res.solved_gen}, time={res.time_s:.1f}s ===")
    print(f"Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
