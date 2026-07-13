"""Final evaluation script: re-train each env's best config and evaluate
over many episodes to confirm the solved status.

Usage:
    python scripts/final_eval.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat import Config
from scripts.train import build_default_config, train
from scripts.evaluate import evaluate_genome, SOLVED_THRESHOLDS


def main():
    # Load the autotune state
    with open("results/autotune_v2.json") as f:
        state = json.load(f)

    print("\n" + "="*80)
    print("FINAL EVALUATION - Re-training best configs and evaluating")
    print("="*80)

    results = {}
    for env_name, eb in state["env_bests"].items():
        if not eb.get("best_config"):
            continue
        print(f"\n=== {env_name} ===")
        cfg = Config.from_dict(eb["best_config"])
        cfg.seed = 100  # different seed for final eval (no overfitting)
        # Set per-env training params
        max_steps_map = {"CartPole-v1": 500, "Acrobot-v1": 500, "MountainCar-v0": 200,
                         "LunarLander-v3": 1000, "Blackjack-v1": 100}
        train_seeds_map = {"CartPole-v1": 1, "Acrobot-v1": 3, "MountainCar-v0": 8,
                           "LunarLander-v3": 3, "Blackjack-v1": 10}
        n_gens_map = {"CartPole-v1": 30, "Acrobot-v1": 30, "MountainCar-v0": 30,
                      "LunarLander-v3": 30, "Blackjack-v1": 30}
        max_steps = max_steps_map.get(env_name, 500)
        train_seeds = train_seeds_map.get(env_name, 1)
        n_gens = n_gens_map.get(env_name, 30)
        # Shape reward during training (per env)
        shaping_map = {"LunarLander-v3": "lunarlander_aggressive"}
        shaping = shaping_map.get(env_name)
        print(f"  Training: pop={cfg.generation.pop_size}, gens={n_gens}, "
              f"train_seeds={train_seeds}, shaping={shaping}")
        t0 = time.time()
        result = train(env_name, cfg, n_generations=n_gens, max_steps=max_steps,
                       n_eval_episodes=1, verbose=False,
                       train_seeds_per_genome=train_seeds,
                       reward_shaping=shaping)
        train_time = time.time() - t0
        print(f"  Train best (shaped): {result['best_fitness']:.2f}, time={train_time:.1f}s")
        # Evaluate with raw rewards over 100 episodes
        n_eval = 100 if env_name != "Blackjack-v1" else 500
        eval_res = evaluate_genome(result["best_genome"], env_name,
                                    n_episodes=n_eval, max_steps=max_steps,
                                    seed=99999)
        threshold = SOLVED_THRESHOLDS.get(env_name, (float("inf"), 100))[0]
        solved = eval_res["mean_reward"] >= threshold
        marker = "✓ SOLVED" if solved else "✗"
        print(f"  Eval over {n_eval} eps: mean={eval_res['mean_reward']:.2f} "
              f"± {eval_res['std_reward']:.2f}  (threshold={threshold:.1f})  {marker}")
        results[env_name] = {
            "train_best": result["best_fitness"],
            "eval_mean": eval_res["mean_reward"],
            "eval_std": eval_res["std_reward"],
            "eval_min": eval_res["min_reward"],
            "eval_max": eval_res["max_reward"],
            "solved": solved,
            "threshold": threshold,
            "n_eval_episodes": n_eval,
            "train_time_s": train_time,
        }

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Env':20s} | {'Eval Mean':>10s} | {'Std':>6s} | {'Threshold':>10s} | "
          f"{'Solved?':>8s} | {'Time':>6s}")
    print("-"*80)
    n_solved = 0
    for env_name, r in results.items():
        if r["solved"]:
            n_solved += 1
        print(f"{env_name:20s} | {r['eval_mean']:10.2f} | {r['eval_std']:6.2f} | "
              f"{r['threshold']:10.1f} | {'✓' if r['solved'] else '✗':>8s} | "
              f"{r['train_time_s']:5.1f}s")
    print("-"*80)
    print(f"Solved: {n_solved}/{len(results)}")
    # Save results
    with open("results/final_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/final_eval.json")


if __name__ == "__main__":
    main()
