"""
Main training entrypoint.

Usage:
    python scripts/train.py --env CartPole-v1 --pop 100 --gens 50 --seed 0
    python scripts/train.py --config configs/cartpole.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
from typing import Optional

# Make `neat` importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neat import Config, Population
from neat.envs import make_env


def build_default_config(env_name: str, seed: int = 0, pop_size: int = 100,
                          optimizer: bool = False) -> Config:
    """Sensible defaults per env."""
    cfg = Config(seed=seed)
    cfg.generation.pop_size = pop_size

    if env_name == "CartPole-v1":
        cfg.n_inputs = 4
        cfg.n_outputs = 2
        cfg.output_activation = "tanh"
        cfg.hidden_activation = "tanh"
        cfg.init.n_conn_multiplier = 1.0
        cfg.init.weight_init_multiplier = 1.0
        cfg.init.n_neurons_range = (0, 1)
        cfg.mutation.weight_std = 0.3
        cfg.mutation.conn_std = 0.5
        cfg.policy.kind = "per_type"
        cfg.policy.weight_prob = 0.8       # was 1.0 (100%)
        cfg.policy.conn_prob = 0.25
        cfg.policy.neuron_prob = 0.05
        cfg.policy.prune_prob = 0.03
        cfg.generation.elitism = 2          # was 1
        cfg.generation.cull_pct = 0.5
        cfg.speciation.target_species = 8
        cfg.speciation.initial_kind = "purge"
        cfg.speciation.subsequent_kind = "standard"
        cfg.speciation.similarity = "percentage"
        cfg.speciation.min_threshold = 0.05
        cfg.speciation.max_threshold = 0.6
    elif env_name == "LunarLander-v3":
        cfg.n_inputs = 8
        cfg.n_outputs = 4
        cfg.output_activation = "tanh"
        cfg.hidden_activation = "tanh"
        cfg.init.n_conn_multiplier = 1.0
        cfg.init.weight_init_multiplier = 0.7
        cfg.init.n_neurons_range = (0, 2)
        cfg.mutation.weight_std = 0.2
        cfg.mutation.conn_std = 0.4
        cfg.policy.conn_prob = 0.3
        cfg.policy.neuron_prob = 0.08
        cfg.policy.prune_prob = 0.03
        cfg.speciation.target_species = 10
        cfg.speciation.initial_kind = "purge"
        cfg.speciation.subsequent_kind = "standard"
    elif env_name == "Acrobot-v1":
        cfg.n_inputs = 6
        cfg.n_outputs = 3
        cfg.output_activation = "tanh"
        cfg.hidden_activation = "tanh"
    elif env_name == "MountainCar-v0":
        cfg.n_inputs = 2
        cfg.n_outputs = 3
        cfg.output_activation = "tanh"
        cfg.hidden_activation = "tanh"
        cfg.init.n_conn_multiplier = 2.0
        cfg.policy.conn_prob = 0.5
    else:
        import gymnasium as gym
        env = gym.make(env_name)
        cfg.n_inputs = int(np.prod(env.observation_space.shape))
        cfg.n_outputs = env.action_space.n
        env.close()

    if optimizer:
        cfg.optimizer.enabled = True
    return cfg


def train(env_name: str, cfg: Config, n_generations: int,
          max_steps: int = 1000, n_eval_episodes: int = 1,
          log_every: int = 1, verbose: bool = True,
          checkpoint_dir: Optional[str] = None,
          checkpoint_every: int = 10) -> dict:
    """Train and return final stats + best genome."""
    env_wrapper = make_env(env_name, max_steps=max_steps,
                           n_eval_episodes=n_eval_episodes,
                           seed=cfg.seed)
    assert cfg.n_inputs == env_wrapper.n_inputs, \
        f"Config n_inputs={cfg.n_inputs} != env n_inputs={env_wrapper.n_inputs}"
    assert cfg.n_outputs == env_wrapper.n_outputs

    pop = Population(cfg)
    history = []
    best_ever_fitness = -float("inf")
    best_ever_genome = None
    t0 = time.time()

    for gen in range(n_generations):
        ep_seed = cfg.seed * 10000 + gen
        def eval_fn(g, _ep_seed=ep_seed):
            r, _, _ = env_wrapper.evaluate(g, episode_seed=_ep_seed)
            return r
        pop.evaluate(eval_fn)
        gen_best = pop.best()
        if gen_best and gen_best.fitness > best_ever_fitness:
            best_ever_fitness = gen_best.fitness
            best_ever_genome = gen_best.clone()
        stats = pop.step()
        history.append(stats)
        if verbose and (gen % log_every == 0 or gen == n_generations - 1):
            elapsed = time.time() - t0
            print(f"gen {gen+1:3d}/{n_generations} | "
                  f"max={stats['fitness_max']:7.2f} mean={stats['fitness_mean']:7.2f} "
                  f"min={stats['fitness_min']:7.2f} | "
                  f"species={stats['n_species']:2d} "
                  f"conns={stats['avg_conns']:5.1f} nodes={stats['avg_nodes']:4.1f} "
                  f"| best={best_ever_fitness:7.2f} t={elapsed:.1f}s")
        if checkpoint_dir and (gen + 1) % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"gen_{gen+1:04d}.json")
            save_checkpoint(pop, best_ever_genome, history, ckpt_path)

    return {
        "history": history,
        "best_fitness": best_ever_fitness,
        "best_genome": best_ever_genome,
        "elapsed_s": time.time() - t0,
    }


def save_checkpoint(pop: Population, best_genome, history, path: str) -> None:
    import pickle
    data = {
        "cfg": pop.cfg.to_dict(),
        "generation": pop.generation,
        "best_genome": best_genome.to_dict() if best_genome else None,
        "history": history,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--optimizer", action="store_true")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--save-config", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg = build_default_config(args.env, seed=args.seed, pop_size=args.pop,
                                    optimizer=args.optimizer)
    env_name = args.env

    if args.save_config:
        cfg.save(args.save_config)
        print(f"Saved config to {args.save_config}")
        return

    result = train(env_name, cfg, args.gens,
                   max_steps=args.max_steps,
                   n_eval_episodes=args.eval_episodes,
                   verbose=not args.quiet,
                   checkpoint_dir=args.checkpoint_dir,
                   checkpoint_every=args.checkpoint_every)
    print(f"\nBest fitness: {result['best_fitness']:.2f}")
    print(f"Elapsed: {result['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
