"""
Capture agent behavior at multiple training checkpoints to show how
the agent improves over generations.

For a chosen env, trains for N generations and every K generations captures
a GIF of the best genome's behavior. Produces a "training progression"
visualization.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat import Config, Population
from neat.envs import make_env
from neat.analysis import TrainingStats, capture_training_trajectory, visualize_genome
from scripts.train import build_default_config
from scripts.evaluate import SOLVED_THRESHOLDS


def train_with_checkpoints(env_name: str, cfg: Config, n_gens: int,
                            checkpoint_gens: list, max_steps: int = 500,
                            output_dir: str = "results/training_progress"):
    """Train and capture GIF + genome viz at specified generation checkpoints."""
    os.makedirs(output_dir, exist_ok=True)
    env_wrapper = make_env(env_name, max_steps=max_steps, n_eval_episodes=1, seed=cfg.seed)
    pop = Population(cfg)
    stats = TrainingStats(snapshot_every=1)
    best_ever_fitness = -float("inf")
    best_ever_genome = None

    # Per-env train seeds
    n_train_seeds = 1
    if env_name == "Blackjack-v1":
        n_train_seeds = 10
    elif env_name == "MountainCar-v0":
        n_train_seeds = 5
    elif env_name in ("Acrobot-v1", "LunarLander-v3"):
        n_train_seeds = 3

    for gen in range(n_gens):
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

        # Capture checkpoint
        if gen in checkpoint_gens and best_ever_genome:
            print(f"  Capturing checkpoint at gen {gen}...")
            gif_path = os.path.join(output_dir, f"{env_name}_gen{gen:03d}.gif")
            try:
                capture_training_trajectory(best_ever_genome, env_name, gif_path,
                                              max_steps=max_steps, seed=cfg.seed + 42)
            except Exception as e:
                print(f"    GIF capture failed: {e}")
            genome_path = os.path.join(output_dir, f"{env_name}_gen{gen:03d}_genome.png")
            try:
                visualize_genome(best_ever_genome, genome_path,
                                  title=f"{env_name} gen {gen} (fit={best_ever_fitness:.1f})")
            except Exception as e:
                print(f"    Genome viz failed: {e}")
        pop.step()

    # Save stats
    stats.save(os.path.join(output_dir, f"{env_name}_stats.json"))
    return best_ever_fitness, best_ever_genome, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--pop", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--output", default="results/training_progress")
    args = p.parse_args()

    # CartPole default config
    cfg = build_default_config(args.env, seed=0, pop_size=args.pop)
    # Checkpoints: early, middle, late
    checkpoints = [0, 2, 5, 10, 15, 20, 25, 29] if args.gens >= 30 else \
                   [int(args.gens * f) for f in [0, 0.1, 0.25, 0.5, 0.75, 0.99]]
    checkpoints = [c for c in checkpoints if c < args.gens]
    print(f"Training {args.env} for {args.gens} gens, checkpoints at {checkpoints}")
    best_fit, best_g, stats = train_with_checkpoints(
        args.env, cfg, args.gens, checkpoints, max_steps=args.max_steps,
        output_dir=args.output)
    print(f"Best fitness: {best_fit:.2f}")
    # Plot the training curves
    plots_dir = os.path.join(args.output, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    stats.plot_fitness(os.path.join(plots_dir, f"{args.env}_fitness.png"))
    stats.plot_genome_size(os.path.join(plots_dir, f"{args.env}_genome_size.png"))
    stats.plot_species(os.path.join(plots_dir, f"{args.env}_species.png"))
    print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
