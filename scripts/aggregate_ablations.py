"""Aggregate all individual ablation results into a single summary file."""
import os, json, glob

summary = {}
for path in glob.glob("results/ablations/*.json"):
    if path.endswith("summary.json") or path.endswith("_stats.json"):
        continue
    name = os.path.basename(path).replace(".json", "")
    with open(path) as f:
        data = json.load(f)
    if "name" not in data:
        continue  # not an ablation result file
    env = data.get("env", "unknown")
    if env not in summary:
        summary[env] = []
    summary[env].append({
        "name": data["name"],
        "description": data.get("description", ""),
        "best_fitness_train": data["best_fitness_train"],
        "eval_mean": data["eval_mean"],
        "eval_std": data["eval_std"],
        "solved": data["solved"],
        "threshold": data.get("threshold", 0),
        "elapsed_s": data["elapsed_s"],
        "best_genome_summary": data.get("best_genome_summary"),
    })

# Print summary
print("="*80)
print("AGGREGATED ABLATION SUMMARY")
print("="*80)
total = 0
for env, results in summary.items():
    threshold = results[0]["threshold"] if results else 0
    print(f"\n  [{env}] (threshold={threshold})  -  {len(results)} ablations")
    total += len(results)
    for r in sorted(results, key=lambda x: -x["eval_mean"]):
        marker = "✓" if r["solved"] else "✗"
        print(f"    {marker} {r['name']:40s}  eval={r['eval_mean']:7.2f} "
              f"± {r['eval_std']:5.2f}  t={r['elapsed_s']:.1f}s")

# Save
with open("results/ablations/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved aggregated summary to results/ablations/summary.json")
print(f"Total ablations: {total}")

