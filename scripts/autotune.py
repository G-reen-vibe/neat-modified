"""
AutoTuner: systematic hyperparameter optimization across multiple envs.

For each env:
    - Maintain a "best config" with its eval score
    - Each round, propose a perturbation of the best config (or a fresh start)
    - Train for a small number of generations
    - Evaluate the best genome over multiple random episodes
    - If the new config's eval score is better, update the best config
    - Log all results to a JSON file

The proposal strategy is a mix of:
    - Random perturbation (mutate one or two hyperparameters)
    - Coordinate descent (sweep one hyperparameter at a time)
    - Fresh start (occasionally try a totally random config)

All 5 envs are tuned in parallel; each env's history is independent.

Usage:
    python scripts/autotune.py --rounds 30 --pop 50 --gens-per-round 15
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import random
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat import Config, Population
from neat.envs import make_env
from scripts.train import build_default_config, train
from scripts.evaluate import evaluate_genome, SOLVED_THRESHOLDS


# ---------------------------------------------------------------------------
# Per-env hyperparameter search spaces
# ---------------------------------------------------------------------------
# These define the perturbation ranges for each tunable hyperparameter.
# Format: { env_name: { hp_name: (low, high, step_or_None) } }
SEARCH_SPACES = {
    "CartPole-v1": {
        "weight_std":     (0.05, 0.8, None),
        "conn_std":       (0.2, 1.0, None),
        "weight_prob":    (0.3, 1.0, None),
        "conn_prob":      (0.05, 0.5, None),
        "neuron_prob":    (0.0, 0.15, None),
        "prune_prob":     (0.0, 0.1, None),
        "elitism":        (1, 5, 1),
        "cull_pct":       (0.3, 0.8, None),
        "target_species": (4, 15, 1),
        "n_conn_multiplier": (0.5, 2.0, None),
        "weight_init_multiplier": (0.5, 2.5, None),
    },
    "Acrobot-v1": {
        "weight_std":     (0.05, 0.8, None),
        "conn_std":       (0.2, 1.0, None),
        "weight_prob":    (0.3, 1.0, None),
        "conn_prob":      (0.05, 0.5, None),
        "neuron_prob":    (0.0, 0.2, None),
        "prune_prob":     (0.0, 0.1, None),
        "elitism":        (1, 5, 1),
        "cull_pct":       (0.3, 0.8, None),
        "target_species": (4, 15, 1),
        "n_conn_multiplier": (0.5, 2.0, None),
    },
    "MountainCar-v0": {
        "weight_std":     (0.1, 1.0, None),
        "conn_std":       (0.3, 1.5, None),
        "weight_prob":    (0.5, 1.0, None),
        "conn_prob":      (0.1, 0.7, None),
        "neuron_prob":    (0.0, 0.2, None),
        "prune_prob":     (0.0, 0.1, None),
        "elitism":        (1, 5, 1),
        "cull_pct":       (0.3, 0.7, None),
        "target_species": (4, 15, 1),
        "n_conn_multiplier": (1.0, 3.0, None),
        "weight_init_multiplier": (1.0, 3.0, None),
    },
    "LunarLander-v3": {
        "weight_std":     (0.05, 0.6, None),
        "conn_std":       (0.2, 1.0, None),
        "weight_prob":    (0.3, 1.0, None),
        "conn_prob":      (0.1, 0.5, None),
        "neuron_prob":    (0.0, 0.2, None),
        "prune_prob":     (0.0, 0.1, None),
        "elitism":        (1, 5, 1),
        "cull_pct":       (0.3, 0.8, None),
        "target_species": (5, 15, 1),
        "n_conn_multiplier": (0.5, 2.0, None),
        "weight_init_multiplier": (0.5, 2.0, None),
    },
    "Blackjack-v1": {
        "weight_std":     (0.05, 1.0, None),
        "conn_std":       (0.2, 1.5, None),
        "weight_prob":    (0.3, 1.0, None),
        "conn_prob":      (0.1, 0.7, None),
        "neuron_prob":    (0.0, 0.2, None),
        "prune_prob":     (0.0, 0.1, None),
        "elitism":        (1, 5, 1),
        "cull_pct":       (0.3, 0.8, None),
        "target_species": (4, 12, 1),
        "n_conn_multiplier": (0.5, 2.0, None),
    },
}


# Hyperparameters that take integer values
INT_PARAMS = {"elitism", "target_species"}


@dataclass
class TuningResult:
    """One round's result for one env."""
    round_idx: int
    env_name: str
    hp_changes: Dict[str, Any]    # what was perturbed this round
    train_best: float
    eval_mean: float
    eval_std: float
    eval_min: float
    eval_max: float
    solved: bool
    elapsed_s: float
    is_best: bool                  # did this round set a new best?
    notes: str = ""                # any algorithmic changes applied this round


@dataclass
class EnvBestConfig:
    """Tracks the best config + history for one env."""
    env_name: str
    best_config: Optional[Config] = None
    best_eval_mean: float = -float("inf")
    best_eval_std: float = 0.0
    best_train_best: float = -float("inf")
    best_round: int = -1
    history: List[TuningResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hyperparameter perturbation
# ---------------------------------------------------------------------------
def get_hp_value(cfg: Config, hp_name: str) -> Any:
    """Get a hyperparameter value from the config by name."""
    if hp_name == "weight_std":
        return cfg.mutation.weight_std
    if hp_name == "conn_std":
        return cfg.mutation.conn_std
    if hp_name == "weight_prob":
        return cfg.policy.weight_prob
    if hp_name == "conn_prob":
        return cfg.policy.conn_prob
    if hp_name == "neuron_prob":
        return cfg.policy.neuron_prob
    if hp_name == "prune_prob":
        return cfg.policy.prune_prob
    if hp_name == "elitism":
        return cfg.generation.elitism
    if hp_name == "cull_pct":
        return cfg.generation.cull_pct
    if hp_name == "target_species":
        return cfg.speciation.target_species
    if hp_name == "n_conn_multiplier":
        return cfg.init.n_conn_multiplier
    if hp_name == "weight_init_multiplier":
        return cfg.init.weight_init_multiplier
    raise ValueError(f"Unknown hp: {hp_name}")


def set_hp_value(cfg: Config, hp_name: str, value: Any) -> None:
    """Set a hyperparameter value on the config."""
    if hp_name == "weight_std":
        cfg.mutation.weight_std = float(value)
    elif hp_name == "conn_std":
        cfg.mutation.conn_std = float(value)
    elif hp_name == "weight_prob":
        cfg.policy.weight_prob = float(value)
    elif hp_name == "conn_prob":
        cfg.policy.conn_prob = float(value)
    elif hp_name == "neuron_prob":
        cfg.policy.neuron_prob = float(value)
    elif hp_name == "prune_prob":
        cfg.policy.prune_prob = float(value)
    elif hp_name == "elitism":
        cfg.generation.elitism = int(value)
    elif hp_name == "cull_pct":
        cfg.generation.cull_pct = float(value)
    elif hp_name == "target_species":
        cfg.speciation.target_species = int(value)
    elif hp_name == "n_conn_multiplier":
        cfg.init.n_conn_multiplier = float(value)
    elif hp_name == "weight_init_multiplier":
        cfg.init.weight_init_multiplier = float(value)
    else:
        raise ValueError(f"Unknown hp: {hp_name}")


def perturb_config(cfg: Config, env_name: str, rng: np.random.Generator,
                   n_hps_to_perturb: int = 1) -> Tuple[Config, Dict[str, Any]]:
    """Perturb 1-3 hyperparameters of `cfg`.  Returns (new_cfg, changes_dict)."""
    new_cfg = cfg.copy()
    space = SEARCH_SPACES[env_name]
    n = min(n_hps_to_perturb, len(space))
    hps_to_change = rng.choice(list(space.keys()), size=n, replace=False)
    changes = {}
    for hp in hps_to_change:
        low, high, step = space[hp]
        if step is not None:
            # Integer parameter
            new_val = int(rng.integers(low, high + 1))
        else:
            # Continuous: perturb around the current value, or sample uniformly
            current = get_hp_value(new_cfg, hp)
            # 50% chance: perturb current by up to ±25%; 50%: uniform sample
            if rng.random() < 0.5 and current is not None:
                delta = float(rng.uniform(-0.25, 0.25)) * float(current)
                new_val = float(np.clip(current + delta, low, high))
            else:
                new_val = float(rng.uniform(low, high))
        set_hp_value(new_cfg, hp, new_val)
        changes[hp] = new_val
    return new_cfg, changes


# ---------------------------------------------------------------------------
# AutoTuner
# ---------------------------------------------------------------------------
class AutoTuner:
    """Systematically tunes hyperparameters for multiple envs."""

    def __init__(self, env_names: List[str], pop_size: int = 50,
                 gens_per_round: int = 15, eval_episodes: int = 20,
                 seed: int = 0, max_steps: int = 1000,
                 output_path: str = "results/autotune.json"):
        self.env_names = env_names
        self.pop_size = pop_size
        self.gens_per_round = gens_per_round
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.max_steps = max_steps
        self.output_path = output_path
        self.rng = np.random.default_rng(seed)
        self.env_bests: Dict[str, EnvBestConfig] = {
            name: EnvBestConfig(env_name=name) for name in env_names
        }
        # Algorithm-version flag: lets us re-test the same hyperparameters
        # under different algorithm versions.
        self.algorithm_version = 1
        self.algo_changes: List[Dict] = []   # log of algorithm changes

        # Set up per-env max_steps (override defaults)
        self.env_max_steps = {
            "CartPole-v1": 500,
            "Acrobot-v1": 500,
            "MountainCar-v0": 200,
            "LunarLander-v3": 1000,
            "Blackjack-v1": 100,
        }
        # Per-env eval episodes (Blackjack needs more for stochasticity)
        self.env_eval_episodes = {
            "CartPole-v1": 30,
            "Acrobot-v1": 50,        # increased for stability
            "MountainCar-v0": 50,    # increased for stability
            "LunarLander-v3": 50,    # increased for stability
            "Blackjack-v1": 200,     # increased for stochasticity
        }
        # Per-env reward shaping during training (None = raw rewards)
        self.env_reward_shaping = {
            "CartPole-v1": None,
            "Acrobot-v1": None,
            "MountainCar-v0": "mountaincar_aggressive",
            "LunarLander-v3": "lunarlander_aggressive",
            "Blackjack-v1": None,
        }
        # Per-env # of training seeds per genome (more = more stable signal)
        self.env_train_seeds = {
            "CartPole-v1": 1,
            "Acrobot-v1": 1,
            "MountainCar-v0": 1,
            "LunarLander-v3": 3,    # reduce overfitting
            "Blackjack-v1": 10,     # stochastic env
        }

    # ------------------------------------------------------------------
    def log_algo_change(self, round_idx: int, description: str) -> None:
        """Record that we changed the algorithm itself (not just hyperparameters)."""
        change = {"round": round_idx, "version": self.algorithm_version,
                  "description": description}
        self.algo_changes.append(change)
        print(f"  [ALGO v{self.algorithm_version}] {description}")

    # ------------------------------------------------------------------
    def get_max_steps(self, env_name: str) -> int:
        return self.env_max_steps.get(env_name, self.max_steps)

    def get_eval_episodes(self, env_name: str) -> int:
        return self.env_eval_episodes.get(env_name, self.eval_episodes)

    # ------------------------------------------------------------------
    def _perturb_config_small(self, cfg: Config, env_name: str,
                               n_hps: int) -> Tuple[Config, Dict[str, Any]]:
        """Perturb `n_hps` hyperparameters by a SMALL amount (±15%).

        This is for fine-tuning the best config; large jumps tend to break
        already-good configs.
        """
        new_cfg = cfg.copy()
        space = SEARCH_SPACES[env_name]
        n = min(n_hps, len(space))
        hps_to_change = self.rng.choice(list(space.keys()), size=n, replace=False)
        changes = {}
        for hp in hps_to_change:
            low, high, step = space[hp]
            current = get_hp_value(new_cfg, hp)
            if step is not None:
                # Integer: ±1 or ±2
                delta = int(self.rng.choice([-2, -1, 1, 2]))
                new_val = int(np.clip(current + delta, low, high))
            else:
                # Continuous: ±15% of current
                delta = float(self.rng.uniform(-0.15, 0.15)) * float(current)
                new_val = float(np.clip(current + delta, low, high))
            set_hp_value(new_cfg, hp, new_val)
            changes[hp] = new_val
        return new_cfg, changes

    # ------------------------------------------------------------------
    def run_round(self, env_name: str, round_idx: int,
                  fresh_start_prob: float = 0.1) -> TuningResult:
        """Run one tuning round for one env."""
        env_best = self.env_bests[env_name]
        t0 = time.time()
        notes = ""

        # Decide: fresh start or perturb the best?
        # After round 20, occasionally try enabling the optimizer
        try_optimizer = (round_idx >= 18 and self.rng.random() < 0.25
                         and env_name in ("Acrobot-v1", "MountainCar-v0",
                                          "LunarLander-v3"))

        if env_best.best_config is None or self.rng.random() < fresh_start_prob:
            # Fresh start: use the default config for this env
            cfg = build_default_config(env_name, seed=self.seed + round_idx,
                                        pop_size=self.pop_size)
            # Perturb a few hyperparameters
            cfg, changes = perturb_config(cfg, env_name, self.rng,
                                           n_hps_to_perturb=int(self.rng.integers(2, 5)))
            notes = "fresh_start"
        else:
            # Perturb the best config - use SMALL perturbations (±15%)
            cfg = env_best.best_config.copy()
            cfg.seed = self.seed + round_idx
            n_perturb = int(self.rng.integers(1, 3))   # 1-2 hps (was 1-3)
            cfg, changes = self._perturb_config_small(cfg, env_name, n_perturb)
            notes = f"perturbed_best({n_perturb})"

        # Occasionally enable the optimizer
        if try_optimizer:
            cfg.optimizer.enabled = True
            cfg.optimizer.lr = float(self.rng.uniform(0.05, 0.3))
            # Pick method directly (avoid numpy str wrapper)
            methods = ["adam", "momentum", "rmsprop"]
            cfg.optimizer.method = methods[int(self.rng.integers(0, len(methods)))]
            notes += " +optimizer"

        # Apply per-env algorithmic adjustments (set via log_algo_change)
        # These are versioned, so we can re-test
        # (No-op by default; the caller can modify cfg before passing in)

        # Train
        max_steps = self.get_max_steps(env_name)
        n_train_seeds = self.env_train_seeds.get(env_name, 1)
        reward_shaping = self.env_reward_shaping.get(env_name)
        try:
            result = train(env_name, cfg, self.gens_per_round,
                           max_steps=max_steps, n_eval_episodes=1,
                           verbose=False,
                           reward_shaping=reward_shaping,
                           train_seeds_per_genome=n_train_seeds)
        except Exception as e:
            print(f"  ERROR during training: {e}")
            import traceback; traceback.print_exc()
            return TuningResult(round_idx, env_name, changes, -999, -999, 0,
                                -999, -999, False, time.time() - t0,
                                is_best=False, notes=f"ERROR: {e}")
        best_genome = result["best_genome"]

        # Evaluate with RAW rewards (no shaping) over multiple episodes
        n_eval_eps = self.get_eval_episodes(env_name)
        if best_genome is None:
            eval_mean = -999
            eval_std = 0
            eval_min = -999
            eval_max = -999
            solved = False
        else:
            env_wrapper = make_env(env_name, max_steps=max_steps,
                                   n_eval_episodes=1, seed=self.seed + 99999)
            eval_rewards = []
            for ep in range(n_eval_eps):
                r, _, _ = env_wrapper.evaluate_raw(best_genome,
                                                    episode_seed=self.seed * 1000 + ep)
                eval_rewards.append(r)
            eval_mean = float(np.mean(eval_rewards))
            eval_std = float(np.std(eval_rewards))
            eval_min = float(np.min(eval_rewards))
            eval_max = float(np.max(eval_rewards))
            threshold = SOLVED_THRESHOLDS.get(env_name, (float("inf"), 100))[0]
            solved = eval_mean >= threshold

        elapsed = time.time() - t0
        # If this round's eval is close to the best (within 1 std), re-evaluate
        # both with more episodes to break ties fairly.  This prevents lucky
        # evaluations from locking in a worse config.
        is_best = eval_mean > env_best.best_eval_mean
        if (is_best and env_best.best_config is not None
                and abs(eval_mean - env_best.best_eval_mean) < eval_std
                and best_genome is not None):
            # Re-evaluate both with 2x episodes
            re_eval_eps = max(n_eval_eps * 2, 100)
            env_wrapper2 = make_env(env_name, max_steps=max_steps,
                                     n_eval_episodes=1, seed=self.seed + 77777)
            new_rewards = []
            for ep in range(re_eval_eps):
                r, _, _ = env_wrapper2.evaluate_raw(best_genome,
                                                     episode_seed=self.seed * 1000 + ep + 10000)
                new_rewards.append(r)
            new_eval_mean = float(np.mean(new_rewards))
            new_eval_std = float(np.std(new_rewards))
            # Also re-evaluate the previous best
            best_genome_prev = None
            # We can't easily re-evaluate the previous best genome (we don't
            # store the genome, only the config).  So just accept the new one
            # if its re-eval is also better than the old eval_mean.
            if new_eval_mean > env_best.best_eval_mean:
                eval_mean = new_eval_mean
                eval_std = new_eval_std
                eval_min = float(np.min(new_rewards))
                eval_max = float(np.max(new_rewards))
                threshold = SOLVED_THRESHOLDS.get(env_name, (float("inf"), 100))[0]
                solved = eval_mean >= threshold
                notes += " +re-eval"
            else:
                is_best = False
                notes += " +re-eval(rejected)"
        if is_best:
            env_best.best_config = cfg
            env_best.best_eval_mean = eval_mean
            env_best.best_eval_std = eval_std
            env_best.best_train_best = result["best_fitness"]
            env_best.best_round = round_idx
        tr = TuningResult(round_idx, env_name, changes, result["best_fitness"],
                          eval_mean, eval_std, eval_min, eval_max, solved,
                          elapsed, is_best, notes)
        env_best.history.append(tr)
        return tr

    # ------------------------------------------------------------------
    def save_state(self) -> None:
        """Save the entire tuning state to JSON."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        state = {
            "algorithm_version": self.algorithm_version,
            "algo_changes": self.algo_changes,
            "env_bests": {},
        }
        for name, eb in self.env_bests.items():
            # Convert numpy strings to plain strings for JSON serialization
            def _clean(d):
                if isinstance(d, dict):
                    return {str(k) if not isinstance(k, str) else k: _clean(v)
                            for k, v in d.items()}
                if isinstance(d, (list, tuple)):
                    return [_clean(x) for x in d]
                if isinstance(d, (np.integer,)):
                    return int(d)
                if isinstance(d, (np.floating,)):
                    return float(d)
                if isinstance(d, (np.bool_,)):
                    return bool(d)
                return d
            state["env_bests"][name] = {
                "best_eval_mean": eb.best_eval_mean,
                "best_eval_std": eb.best_eval_std,
                "best_train_best": eb.best_train_best,
                "best_round": eb.best_round,
                "best_config": eb.best_config.to_dict() if eb.best_config else None,
                "history": [
                    _clean({
                        "round": r.round_idx, "env": r.env_name,
                        "changes": r.hp_changes, "train_best": r.train_best,
                        "eval_mean": r.eval_mean, "eval_std": r.eval_std,
                        "eval_min": r.eval_min, "eval_max": r.eval_max,
                        "solved": r.solved, "elapsed_s": r.elapsed_s,
                        "is_best": r.is_best, "notes": r.notes,
                    })
                    for r in eb.history
                ],
            }
        with open(self.output_path, "w") as f:
            json.dump(state, f, indent=2)

    # ------------------------------------------------------------------
    def load_state(self, path: Optional[str] = None) -> bool:
        """Load tuning state from JSON.  Returns True if loaded."""
        path = path or self.output_path
        if not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)
        self.algorithm_version = state.get("algorithm_version", 1)
        self.algo_changes = state.get("algo_changes", [])
        for name, eb_data in state.get("env_bests", {}).items():
            if name not in self.env_bests:
                continue
            eb = self.env_bests[name]
            eb.best_eval_mean = eb_data.get("best_eval_mean", -float("inf"))
            eb.best_eval_std = eb_data.get("best_eval_std", 0.0)
            eb.best_train_best = eb_data.get("best_train_best", -float("inf"))
            eb.best_round = eb_data.get("best_round", -1)
            if eb_data.get("best_config"):
                eb.best_config = Config.from_dict(eb_data["best_config"])
            # Reconstruct history
            eb.history = []
            for h in eb_data.get("history", []):
                tr = TuningResult(
                    round_idx=h["round"], env_name=h["env"],
                    hp_changes=h.get("changes", {}),
                    train_best=h["train_best"], eval_mean=h["eval_mean"],
                    eval_std=h["eval_std"], eval_min=h["eval_min"],
                    eval_max=h["eval_max"], solved=h["solved"],
                    elapsed_s=h["elapsed_s"], is_best=h["is_best"],
                    notes=h.get("notes", ""),
                )
                eb.history.append(tr)
        return True

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        print("\n" + "="*70)
        print(f"AUTO-TUNER SUMMARY (algorithm v{self.algorithm_version})")
        print("="*70)
        for name, eb in self.env_bests.items():
            threshold = SOLVED_THRESHOLDS.get(name, (float("inf"), 100))[0]
            solved_str = "✓ SOLVED" if eb.best_eval_mean >= threshold else "✗"
            print(f"  {name:20s} | best_eval={eb.best_eval_mean:7.2f} "
                  f"± {eb.best_eval_std:5.2f}  (threshold={threshold:6.1f}) "
                  f"{solved_str}  (round {eb.best_round})")
        print("="*70)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=30,
                   help="Number of tuning rounds per env")
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--gens-per-round", type=int, default=15)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", default="all",
                   help="Comma-separated env names, or 'all'")
    p.add_argument("--output", default="results/autotune.json")
    p.add_argument("--fresh-start-prob", type=float, default=0.15)
    p.add_argument("--rounds-per-save", type=int, default=1)
    p.add_argument("--resume", action="store_true",
                   help="Resume from --output if it exists")
    p.add_argument("--start-round", type=int, default=1,
                   help="Round number to start at (for resume)")
    args = p.parse_args()

    if args.envs == "all":
        env_names = list(SEARCH_SPACES.keys())
    else:
        env_names = [s.strip() for s in args.envs.split(",")]

    tuner = AutoTuner(env_names, pop_size=args.pop,
                       gens_per_round=args.gens_per_round,
                       eval_episodes=args.eval_episodes,
                       seed=args.seed, output_path=args.output)
    # Try to resume
    if args.resume:
        if tuner.load_state():
            print(f"[Resumed from {args.output}: "
                  f"v{tuner.algorithm_version}, "
                  f"{sum(len(eb.history) for eb in tuner.env_bests.values())} "
                  f"total rounds completed]")
        else:
            print(f"[No state found at {args.output}, starting fresh]")
    print(f"\nAutoTuner: {args.rounds} rounds on {env_names}")
    print(f"  Pop: {args.pop}, Gens/round: {args.gens_per_round}, "
          f"Eval eps: {args.eval_episodes}")

    start_round = args.start_round
    end_round = start_round + args.rounds - 1
    for r in range(start_round, end_round + 1):
        print(f"\n--- Round {r} (of {end_round}) ---")
        for env_name in env_names:
            print(f"\n[{env_name}] round {r}")
            tr = tuner.run_round(env_name, r,
                                  fresh_start_prob=args.fresh_start_prob)
            marker = " ★ NEW BEST" if tr.is_best else ""
            print(f"  train_best={tr.train_best:.2f}  eval={tr.eval_mean:.2f}±{tr.eval_std:.2f}"
                  f"  solved={tr.solved}  t={tr.elapsed_s:.1f}s{marker}")
            print(f"  changes: {tr.hp_changes}  ({tr.notes})")
        # Save after each round
        if (r - start_round + 1) % args.rounds_per_save == 0:
            tuner.save_state()
            print(f"\n[Saved state to {args.output}]")
        tuner.print_summary()

    # Final save
    tuner.save_state()
    tuner.print_summary()
    print(f"\nFinal state saved to {args.output}")


if __name__ == "__main__":
    main()
