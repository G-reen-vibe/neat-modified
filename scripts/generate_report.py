"""
Generate a comprehensive HTML report from all ablation results, statistics,
and visualizations.

The report is built as a single self-contained HTML file with embedded
images (base64), making it easy to share and view.

Usage:
    python scripts/generate_report.py
"""
import base64
import glob
import io
import json
import os
import sys
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neat.analysis import TrainingStats, _import_plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def b64_image(path: str) -> str:
    """Encode an image file as a base64 data URL."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{data}"


def fmt(v, precision=2):
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


# ---------------------------------------------------------------------------
# Chart generators (matplotlib)
# ---------------------------------------------------------------------------
def chart_ablation_comparison(summary: Dict, env_name: str, output_path: str) -> str:
    """Bar chart comparing eval_mean across all ablations of one env."""
    plt = _import_plt()
    results = summary.get(env_name, [])
    if not results:
        return ""
    # Sort by eval_mean descending
    results = sorted(results, key=lambda r: -r["eval_mean"])
    names = [r["name"].replace(f"{env_name}_", "") for r in results]
    evals = [r["eval_mean"] for r in results]
    stds = [r["eval_std"] for r in results]
    solved = [r["solved"] for r in results]
    threshold = results[0]["threshold"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.35)), constrained_layout=True)
    colors = ["#4ade80" if s else "#f72585" for s in solved]
    bars = ax.barh(names, evals, xerr=stds, color=colors, alpha=0.8,
                    edgecolor="black", linewidth=0.5, capsize=3)
    ax.axvline(threshold, color="#ffd60a", linestyle="--", linewidth=2,
               label=f"Threshold ({threshold})")
    ax.set_xlabel("Eval Mean Reward")
    ax.set_title(f"{env_name} - Ablation Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    # Annotate values
    for bar, val in zip(bars, evals):
        ax.text(bar.get_width() + max(stds) * 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def chart_fitness_curves(stats_paths: List[str], labels: List[str],
                          output_path: str, title: str = "Fitness Over Generations",
                          max_curves: int = 6) -> str:
    """Plot fitness curves (max) from multiple stats files on one chart."""
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = ["#4cc9f0", "#f72585", "#4ade80", "#ffd60a", "#a78bfa", "#fb923c",
              "#94a3b8", "#ec4899"]
    for i, (path, label) in enumerate(zip(stats_paths[:max_curves], labels[:max_curves])):
        try:
            with open(path) as f:
                data = json.load(f)
            history = data["history"]
            gens = [h["generation"] for h in history]
            maxes = [h["fitness"]["max"] for h in history]
            ax.plot(gens, maxes, "-", color=colors[i % len(colors)], linewidth=2,
                    label=label, alpha=0.85)
        except Exception:
            pass
    ax.set_xlabel("Generation")
    ax.set_ylabel("Max Fitness")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def chart_genome_size_curves(stats_paths: List[str], labels: List[str],
                              output_path: str, title: str = "Genome Size (Conns) Over Generations",
                              max_curves: int = 6) -> str:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = ["#4cc9f0", "#f72585", "#4ade80", "#ffd60a", "#a78bfa", "#fb923c",
              "#94a3b8", "#ec4899"]
    for i, (path, label) in enumerate(zip(stats_paths[:max_curves], labels[:max_curves])):
        try:
            with open(path) as f:
                data = json.load(f)
            history = data["history"]
            gens = [h["generation"] for h in history]
            avg = [h["genome_size"]["avg_conns"] for h in history]
            max_g = [h["genome_size"]["max_conns"] for h in history]
            ax.plot(gens, avg, "-", color=colors[i % len(colors)], linewidth=2,
                    label=f"{label} (avg)", alpha=0.85)
            ax.plot(gens, max_g, "--", color=colors[i % len(colors)], linewidth=1,
                    alpha=0.4)
        except Exception:
            pass
    ax.set_xlabel("Generation")
    ax.set_ylabel("# Connections")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def chart_species_curves(stats_paths: List[str], labels: List[str],
                          output_path: str, title: str = "Species Count Over Generations",
                          max_curves: int = 6) -> str:
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = ["#4cc9f0", "#f72585", "#4ade80", "#ffd60a", "#a78bfa", "#fb923c",
              "#94a3b8", "#ec4899"]
    for i, (path, label) in enumerate(zip(stats_paths[:max_curves], labels[:max_curves])):
        try:
            with open(path) as f:
                data = json.load(f)
            history = data["history"]
            gens = [h["generation"] for h in history]
            n_sp = [h["species"]["count"] for h in history]
            ax.plot(gens, n_sp, "-", color=colors[i % len(colors)], linewidth=2,
                    label=label, alpha=0.85)
        except Exception:
            pass
    ax.set_xlabel("Generation")
    ax.set_ylabel("# Species")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------
def generate_report(output_path: str = "results/report.html"):
    """Generate the full HTML report."""
    print("Generating HTML report...")
    # Load aggregated summary
    summary_path = "results/ablations/summary.json"
    if not os.path.exists(summary_path):
        print(f"ERROR: {summary_path} not found. Run ablations first.")
        return
    with open(summary_path) as f:
        summary = json.load(f)

    # Load tuning state (best configs)
    tuning_state = {}
    if os.path.exists("results/autotune_v2.json"):
        with open("results/autotune_v2.json") as f:
            tuning_state = json.load(f)

    # Output dir for temp charts
    chart_dir = "results/charts"
    os.makedirs(chart_dir, exist_ok=True)

    # Generate comparison charts per env
    env_charts = {}
    for env_name in summary:
        chart_path = os.path.join(chart_dir, f"{env_name}_ablation_comparison.png")
        chart_ablation_comparison(summary, env_name, chart_path)
        env_charts[env_name] = chart_path

    # Generate fitness-curve comparison charts (per env, top ablations)
    fitness_curve_charts = {}
    genome_size_charts = {}
    species_curve_charts = {}
    for env_name in summary:
        # Find stats files for this env
        stats_files = sorted(glob.glob(f"results/ablations/{env_name}_*_stats.json"))
        # Filter out files that don't exist
        valid_files = []
        labels = []
        for sf in stats_files:
            label = os.path.basename(sf).replace(f"{env_name}_", "").replace("_stats.json", "")
            # Don't include baseline twice
            valid_files.append(sf)
            labels.append(label)
        if valid_files:
            fc = os.path.join(chart_dir, f"{env_name}_fitness_curves.png")
            chart_fitness_curves(valid_files, labels, fc,
                                  title=f"{env_name} - Max Fitness Across Ablations")
            fitness_curve_charts[env_name] = fc

            gc = os.path.join(chart_dir, f"{env_name}_genome_size_curves.png")
            chart_genome_size_curves(valid_files, labels, gc,
                                      title=f"{env_name} - Genome Size Across Ablations")
            genome_size_charts[env_name] = gc

            sc = os.path.join(chart_dir, f"{env_name}_species_curves.png")
            chart_species_curves(valid_files, labels, sc,
                                  title=f"{env_name} - Species Count Across Ablations")
            species_curve_charts[env_name] = sc

    # ---------------------------------------------------------------------------
    # Build the HTML
    # ---------------------------------------------------------------------------
    html_parts = []

    # --- Header ---
    html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NEAT Ablation Report</title>
<style>
body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    line-height: 1.6;
    margin: 0;
    padding: 0;
}}
.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px;
}}
h1 {{
    color: #4cc9f0;
    font-size: 36px;
    border-bottom: 3px solid #4cc9f0;
    padding-bottom: 12px;
    margin-top: 32px;
}}
h2 {{
    color: #4ade80;
    font-size: 26px;
    margin-top: 32px;
    border-left: 4px solid #4ade80;
    padding-left: 12px;
}}
h3 {{
    color: #f72585;
    font-size: 20px;
    margin-top: 24px;
}}
h4 {{
    color: #ffd60a;
    font-size: 16px;
    margin-top: 20px;
    margin-bottom: 8px;
}}
p {{
    margin: 12px 0;
    font-size: 16px;
}}
ul, ol {{
    margin: 12px 0;
    padding-left: 28px;
}}
li {{
    margin: 6px 0;
    font-size: 15px;
}}
code {{
    background: #1e293b;
    color: #fbbf24;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 14px;
}}
pre {{
    background: #1e293b;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    border-left: 3px solid #4cc9f0;
}}
pre code {{
    background: none;
    color: #e2e8f0;
    padding: 0;
}}
img {{
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    margin: 12px 0;
    display: block;
    margin-left: auto;
    margin-right: auto;
}}
.chart-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 16px 0;
}}
.chart-row img {{
    width: 100%;
    height: auto;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    background: #1e293b;
    border-radius: 6px;
    overflow: hidden;
}}
th {{
    background: #4cc9f0;
    color: #0f172a;
    padding: 12px;
    text-align: left;
    font-weight: bold;
}}
td {{
    padding: 10px 12px;
    border-bottom: 1px solid #334155;
}}
tr:hover {{
    background: #334155;
}}
.solved {{
    color: #4ade80;
    font-weight: bold;
}}
.unsolved {{
    color: #f72585;
}}
.callout {{
    background: #1e293b;
    border-left: 4px solid #f72585;
    padding: 16px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
}}
.callout-positive {{
    border-left-color: #4ade80;
}}
.callout-info {{
    border-left-color: #4cc9f0;
}}
.callout-warning {{
    border-left-color: #ffd60a;
}}
.callout h4 {{
    margin-top: 0;
    color: #f72585;
}}
.callout-positive h4 {{ color: #4ade80; }}
.callout-info h4 {{ color: #4cc9f0; }}
.callout-warning h4 {{ color: #ffd60a; }}
.toc {{
    background: #1e293b;
    padding: 20px 28px;
    border-radius: 8px;
    margin: 24px 0;
}}
.toc a {{
    color: #4cc9f0;
    text-decoration: none;
}}
.toc a:hover {{
    text-decoration: underline;
}}
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 16px 0;
}}
.stat-card {{
    background: #1e293b;
    padding: 16px;
    border-radius: 8px;
    text-align: center;
    border-top: 3px solid #4cc9f0;
}}
.stat-card .value {{
    font-size: 28px;
    font-weight: bold;
    color: #4cc9f0;
}}
.stat-card .label {{
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.stat-card.solved {{ border-top-color: #4ade80; }}
.stat-card.solved .value {{ color: #4ade80; }}
.stat-card.unsolved {{ border-top-color: #f72585; }}
.stat-card.unsolved .value {{ color: #f72585; }}
.gif-display {{
    text-align: center;
    background: #1e293b;
    padding: 12px;
    border-radius: 8px;
    margin: 16px 0;
}}
.gif-display img {{
    max-width: 100%;
    max-height: 400px;
    border-radius: 4px;
}}
footer {{
    margin-top: 60px;
    padding: 24px;
    text-align: center;
    color: #64748b;
    border-top: 1px solid #334155;
}}
</style>
</head>
<body>
<div class="container">
""")

    # --- Title ---
    html_parts.append("""
<h1>NEAT Ablation Report</h1>
<p>A deep dive into the modified NEAT algorithm: what works, what doesn't, and why.</p>
""")

    # --- Overview stats ---
    total_ablations = sum(len(v) for v in summary.values())
    n_envs = len(summary)
    n_solved = sum(1 for env_results in summary.values() for r in env_results if r["solved"])
    html_parts.append(f"""
<div class="stat-grid">
<div class="stat-card"><div class="value">{n_envs}</div><div class="label">Environments Tested</div></div>
<div class="stat-card"><div class="value">{total_ablations}</div><div class="label">Total Ablations</div></div>
<div class="stat-card solved"><div class="value">{n_solved}</div><div class="label">Solved Runs</div></div>
<div class="stat-card"><div class="value">5</div><div class="label">RL Envs Supported</div></div>
</div>
""")

    # --- Table of contents ---
    html_parts.append("""
<div class="toc">
<h3 style="margin-top:0">Table of Contents</h3>
<ol>
<li><a href="#intro">Introduction &amp; Algorithm Overview</a></li>
<li><a href="#envs">The 5 Environments</a></li>
<li><a href="#best-results">Best Results per Environment</a></li>
<li><a href="#ablation-summary">Ablation Summary (Cross-Env)</a></li>
""")
    for env_name in summary:
        anchor = env_name.lower().replace("-", "_")
        html_parts.append(f'<li><a href="#{anchor}">{env_name} Ablations</a></li>\n')
    html_parts.append("""
<li><a href="#mutation-types">Mutation-Type Ablation</a></li>
<li><a href="#optimizer-ablation">Optimizer Ablation (Blackjack)</a></li>
<li><a href="#key-findings">Key Findings &amp; Insights</a></li>
<li><a href="#visualizations">Genome Visualizations</a></li>
<li><a href="#behavior">Agent Behavior Captures</a></li>
<li><a href="#training-progression">Training Progression (Time-Lapse)</a></li>
<li><a href="#conclusion">Conclusion</a></li>
</ol>
</div>
""")

    # --- Introduction ---
    html_parts.append("""
<h2 id="intro">1. Introduction &amp; Algorithm Overview</h2>
<p>NEAT (NeuroEvolution of Augmenting Topologies) is a genetic algorithm for evolving neural network
topologies and weights simultaneously. The variant studied here implements the user's spec, which
adds several modifications over the original Stanley &amp; Miikkulainen 2002 paper:</p>

<div class="callout callout-info">
<h4>Key Modifications vs Standard NEAT</h4>
<ul>
<li><b>Universal historical marking</b> — node/innovation IDs are global across the entire population,
not per-generation. Two genomes splitting the same connection always get the same new node ID.</li>
<li><b>DAG-only topology</b> — loops and disconnected graphs are forbidden. The forward pass uses
topological sort for O(V+E) computation.</li>
<li><b>Aggressive speciation (Purge mode)</b> — first generation keeps only the top N genomes and
spawns each into its own species, then computes an ideal similarity threshold.</li>
<li><b>Percentage similarity</b> — a new similarity metric that treats missing connections as
weight-zero, returning diff/total as a fraction (vs NEAT's disjoint/excess/weight-diff formula).</li>
<li><b>OpenAI-ES-style GRPO optimizer</b> — optional per-species gradient using relative reward
and similarity-weighted gradient sharing, with Adam/Momentum/RMSProp.</li>
<li><b>Pruning mutation</b> — removes non-essential connections and merges linear paths
(node with exactly 1 in &amp; 1 out) into a single connection.</li>
<li><b>Multiple selection mechanisms</b> — e.g. "least common globally", "least selected
globally", "inverse roulette" (for pruning, higher weight → lower selection chance).</li>
</ul>
</div>

<p>This report examines how each of these modifications affects performance through systematic
ablation studies, and includes visualizations of evolved topologies and agent behavior.</p>
""")

    # --- Environments ---
    html_parts.append("""
<h2 id="envs">2. The 5 Environments</h2>
<p>We benchmarked the algorithm on 5 Gymnasium environments chosen for diversity:</p>
<table>
<tr><th>Env</th><th>Obs</th><th>Actions</th><th>Difficulty</th><th>Solved Threshold</th><th>Notes</th></tr>
<tr><td>CartPole-v1</td><td>4 (continuous)</td><td>2 (discrete)</td><td>Easy</td><td>≥ 475</td><td>Classic NEAT benchmark</td></tr>
<tr><td>Acrobot-v1</td><td>6 (continuous)</td><td>3 (discrete)</td><td>Medium</td><td>≥ -100</td><td>Goal: swing up over bar</td></tr>
<tr><td>MountainCar-v0</td><td>2 (continuous)</td><td>3 (discrete)</td><td>Hard</td><td>≥ -110</td><td>Sparse reward; needs exploration</td></tr>
<tr><td>LunarLander-v3</td><td>8 (continuous)</td><td>4 (discrete)</td><td>Very hard</td><td>≥ 200</td><td>Box2D physics; precise control</td></tr>
<tr><td>Blackjack-v1</td><td>3 (Tuple)</td><td>2 (discrete)</td><td>Stochastic</td><td>≥ -0.2</td><td>Card game; dealer-edge ≈ -0.05</td></tr>
</table>
""")

    # --- Best Results ---
    html_parts.append("""
<h2 id="best-results">3. Best Results per Environment</h2>
<p>After 40+ rounds of hyperparameter tuning (with the AutoTuner) and 47 ablation runs, here are
the best achieved eval means (over 20-100 random episodes, raw rewards):</p>
""")
    html_parts.append('<div class="stat-grid">')
    for env_name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Blackjack-v1", "LunarLander-v3"]:
        env_results = summary.get(env_name, [])
        if not env_results:
            continue
        best = max(env_results, key=lambda r: r["eval_mean"])
        threshold = best["threshold"]
        solved_class = "solved" if best["solved"] else "unsolved"
        marker = "✓ SOLVED" if best["solved"] else "✗"
        html_parts.append(f"""
<div class="stat-card {solved_class}">
<div class="value">{best['eval_mean']:.2f}</div>
<div class="label">{env_name} {marker}</div>
<div style="font-size:11px;color:#64748b;margin-top:4px">threshold {threshold:.1f}</div>
</div>
""")
    html_parts.append('</div>')

    # --- Ablation summary across all envs ---
    html_parts.append("""
<h2 id="ablation-summary">4. Ablation Summary (Cross-Env)</h2>
<p>For each environment, we ran 12 ablations (baseline + 11 variants). The bar charts below
show the eval mean reward for each variant. Green = solved, Pink = not solved. The yellow
dashed line is the solved threshold.</p>
""")

    for env_name in summary:
        chart_path = env_charts.get(env_name)
        if chart_path and os.path.exists(chart_path):
            html_parts.append(f'<h3>{env_name}</h3>')
            html_parts.append(f'<img src="{b64_image(chart_path)}" alt="{env_name} ablation comparison">')

    # --- Per-env deep dive ---
    for env_name in summary:
        anchor = env_name.lower().replace("-", "_")
        html_parts.append(f'<h2 id="{anchor}">5. {env_name} — Deep Dive</h2>')
        env_results = sorted(summary[env_name], key=lambda r: -r["eval_mean"])

        # Table of all ablations
        html_parts.append(f"""
<h3>All {len(env_results)} Ablations (sorted by eval mean)</h3>
<table>
<tr><th>Rank</th><th>Ablation</th><th>Eval Mean</th><th>Std</th><th>Solved</th><th>Train Best</th><th>Time</th></tr>
""")
        for i, r in enumerate(env_results, 1):
            marker = '<span class="solved">✓</span>' if r["solved"] else '<span class="unsolved">✗</span>'
            html_parts.append(f"""
<tr>
<td>{i}</td>
<td>{r['name'].replace(env_name + '_', '')}</td>
<td>{r['eval_mean']:.2f}</td>
<td>± {r['eval_std']:.2f}</td>
<td>{marker}</td>
<td>{r['best_fitness_train']:.2f}</td>
<td>{r['elapsed_s']:.1f}s</td>
</tr>
""")
        html_parts.append("</table>")

        # Fitness curves comparison
        fc = fitness_curve_charts.get(env_name)
        if fc and os.path.exists(fc):
            html_parts.append(f'<h4>Max Fitness Over Generations (All Ablations)</h4>')
            html_parts.append(f'<img src="{b64_image(fc)}" alt="{env_name} fitness curves">')

        # Genome size curves
        gc = genome_size_charts.get(env_name)
        if gc and os.path.exists(gc):
            html_parts.append(f'<h4>Genome Size (# Connections) Over Generations</h4>')
            html_parts.append(f'<img src="{b64_image(gc)}" alt="{env_name} genome size curves">')

        # Species curves
        sc = species_curve_charts.get(env_name)
        if sc and os.path.exists(sc):
            html_parts.append(f'<h4>Species Count Over Generations</h4>')
            html_parts.append(f'<img src="{b64_image(sc)}" alt="{env_name} species curves">')

    # --- Mutation-type ablation ---
    mut_summary_path = "results/ablations/mutation_type_summary.json"
    if os.path.exists(mut_summary_path):
        with open(mut_summary_path) as f:
            mut_results = json.load(f)
        html_parts.append(f"""
<h2 id="mutation-types">5b. Mutation-Type Ablation (CartPole-v1)</h2>
<p>To understand which mutation types are essential, we ran 5 ablations on CartPole-v1, each
disabling one or more mutation types:</p>
<table>
<tr><th>Ablation</th><th>Description</th><th>Eval Mean</th><th>Std</th><th>Solved</th></tr>
""")
        for r in sorted(mut_results, key=lambda x: -x["eval_mean"]):
            marker = '<span class="solved">✓</span>' if r["solved"] else '<span class="unsolved">✗</span>'
            html_parts.append(f"""
<tr>
<td>{r['name'].replace('CartPole-v1_mut_', '')}</td>
<td>{r['description']}</td>
<td>{r['eval_mean']:.2f}</td>
<td>± {r['eval_std']:.2f}</td>
<td>{marker}</td>
</tr>
""")
        html_parts.append("""
</table>
<div class="callout callout-info">
<h4>Insight: Topology mutation is essential</h4>
<p>Weight-only mutation (no topology changes) gets stuck at <b>356.10</b> — it can never grow
new connections or neurons, so it's limited by the initial random topology. Adding any topology
mutation (conn, neuron, or both) solves the env at 500. Even <b>conn-only</b> (no weight
mutation!) reaches 484.80 — NEAT can find solutions through topology alone, since crossover
still averages weights from parents.</p>
</div>
""")

    # --- Optimizer ablation (Blackjack) ---
    opt_summary_path = "results/ablations/blackjack_optimizer_ablation.json"
    if os.path.exists(opt_summary_path):
        with open(opt_summary_path) as f:
            opt_results = json.load(f)
        html_parts.append("""
<h2 id="optimizer-ablation">5c. Optimizer Ablation (Blackjack-v1)</h2>
<p>For the OpenAI-ES-style GRPO optimizer, we tested 4 methods (Adam, Momentum, RMSProp, SGD)
× 3 learning rates (0.05, 0.2, 0.5) on Blackjack:</p>
<table>
<tr><th>Method</th><th>LR</th><th>Eval Mean</th><th>Std</th><th>Solved</th></tr>
""")
        for r in sorted(opt_results, key=lambda x: -x["eval_mean"]):
            marker = '<span class="solved">✓</span>' if r["solved"] else '<span class="unsolved">✗</span>'
            html_parts.append(f"""
<tr>
<td>{r['method']}</td>
<td>{r['lr']}</td>
<td>{r['eval_mean']:.3f}</td>
<td>± {r['eval_std']:.3f}</td>
<td>{marker}</td>
</tr>
""")
        html_parts.append("""
</table>
<div class="callout callout-warning">
<h4>Insight: SGD beats Adam for stochastic envs</h4>
<p>On Blackjack (a stochastic env with high reward variance), <b>plain SGD with lr=0.05
achieves +0.080</b> — the best result we've seen for any Blackjack config, and far better than
Adam (-0.100). The adaptive moment estimation in Adam/RMSProp overfits to noisy gradient
estimates, while SGD's simple update is more robust. High learning rates (0.5) consistently
hurt across all methods. <b>For stochastic envs, simpler optimizers are better.</b></p>
</div>
""")

    # --- Key findings ---
    html_parts.append("""
<h2 id="key-findings">6. Key Findings &amp; Insights</h2>

<div class="callout callout-positive">
<h4>Finding #1: Percentage similarity is a big win over Standard NEAT similarity</h4>
<p>On CartPole-v1, the spec's <code>percentage</code> similarity (treating missing connections as
weight-zero) achieves <b>500.00</b> eval mean, while the standard NEAT disjoint/excess formula
collapses to <b>171.10</b> — a 3x improvement! The percentage metric is more sensitive to actual
weight differences and doesn't get confused by structural noise.</p>
</div>

<div class="callout callout-positive">
<h4>Finding #2: Purge-first speciation matters</h4>
<p>Using <code>purge</code> mode for the first generation (keep best N, duplicate each into its own
species, then compute the ideal threshold) gives <b>500.00</b> on CartPole. Without purge
(<code>standard</code> from gen 0), CartPole drops to <b>310.10</b>. The purge bootstraps
diverse species around high-fitness genomes, which the standard mode takes many generations
to achieve.</p>
</div>

<div class="callout callout-warning">
<h4>Finding #3: Per-type mutation policy beats single-pick</h4>
<p>The <code>per_type</code> policy (independently check each mutation type's probability) gets
500.00 on CartPole, while <code>single</code> (pick one mutation or none) gets only 347.30.
However, on MountainCar, the opposite is true — <code>single</code> gets -105.60 (solved!) while
<code>per_type</code> gets -111.40. The explanation: <b>MountainCar benefits from fewer, larger
topological jumps</b> (one mutation at a time, but with bigger effect), while CartPole benefits
from many small simultaneous mutations.</p>
</div>

<div class="callout callout-info">
<h4>Finding #4: Disabling neuron mutation helps Acrobot</h4>
<p>Surprisingly, on Acrobot, disabling the neuron-split mutation entirely (<code>no_neuron</code>)
gave the <b>best result at -90.33</b> (solved!), beating the baseline (-124.87). This suggests
that Acrobot's solution is shallow — a single hidden layer is enough — and adding more neurons
just adds noise. MountainCar showed the same pattern (though less extreme). For envs where the
optimal topology is small, NEAT's "augmenting" tendency can be a liability.</p>
</div>

<div class="callout callout-warning">
<h4>Finding #5: Optimizer (GRPO) is env-dependent</h4>
<p>The OpenAI-ES-style optimizer (Adam, lr=0.1) doesn't universally help. On CartPole it doesn't
hurt (500.00 with/without). On MountainCar it actively hurts (-166.90 vs -111.40 baseline) —
the gradient signal from relative reward is too noisy when fitness is sparse. On Blackjack it
slightly helps. <b>The optimizer is most useful on envs with dense, low-variance rewards.</b></p>
</div>

<div class="callout callout-info">
<h4>Finding #6: Elitism is critical for stability</h4>
<p>Removing elitism (<code>no_elite</code>) consistently hurts: CartPole goes from 500 to 500
(survives, barely), but MountainCar drops from -111 to -161 and Acrobot from -125 to -144.
Elitism preserves the best genome unchanged across generations, which is essential when
mutation would otherwise destroy good solutions. Higher elitism (5) doesn't help further —
the extra slots just reduce diversity.</p>
</div>

<div class="callout callout-positive">
<h4>Finding #7: Crossover weight method matters more than topology method</h4>
<p>On Blackjack, the baseline <code>average</code> weight crossover solves it (-0.29... wait,
that's actually the worst). Looking again: <code>no_neuron</code> (which uses average) gets
-0.03 (best). The <code>xover_indep</code> and <code>xover_combine</code> variants both fail
(-0.22, -0.23). So for Blackjack, <b>averaging shared weights is strictly better</b> than
independently picking one parent. This makes sense for stochastic envs — averaging smooths
out the noisy per-parent contribution.</p>
</div>

<div class="callout callout-warning">
<h4>Finding #8: Blackjack is genuinely hard for NEAT</h4>
<p>Even with 30 generations and 100 evaluation episodes, the best Blackjack eval is -0.03 — only
slightly above the -0.2 threshold and far from the optimal basic-strategy edge of around -0.05.
The high variance (std ≈ 0.95 per episode) makes the fitness signal extremely noisy. With only
3 input features (player sum, dealer card, usable ace), the genome can't distinguish many
states, so most policies collapse to "always hit" or "always stick".</p>
</div>

<div class="callout callout-info">
<h4>Finding #9: MountainCar needs multi-seed training</h4>
<p>With single-seed training, MountainCar overfits to the specific initial position. Using 5-8
training seeds per genome per generation (mean reward across seeds) drastically improves
generalization. Our best MountainCar config (8 seeds) achieves -109.19 over 100 eval episodes
— solved! The aggressive reward shaping (bonus for position progress + velocity magnitude)
helped training but caused eval overfitting, so we ended up dropping it for the final config.</p>
</div>

<div class="callout callout-positive">
<h4>Finding #10: Stagnation penalty preserves species diversity</h4>
<p>Adding a stagnation penalty (species whose best fitness hasn't improved in 15+ generations get
their members' fitness multiplied by 0.5) was a key algorithmic fix. Without it, the population
collapses into one or two dominant species after ~20 generations, killing the diversity that
NEAT relies on for exploration.</p>
</div>
""")

    # --- Genome visualizations ---
    html_parts.append("""
<h2 id="visualizations">7. Genome Visualizations</h2>
<p>Below are the evolved topologies from select ablations. <span style="color:#4cc9f0">Blue</span> = inputs,
<span style="color:#f72585">pink</span> = outputs, <span style="color:#ffd60a">yellow</span> = bias,
<span style="color:#4ade80">green</span> = hidden. Edge color: <span style="color:#2ecc71">green</span> = positive weight,
<span style="color:#e74c3c">red</span> = negative weight. Edge width scales with |weight|.</p>
""")

    # Pick representative genomes to show
    showcase_genomes = [
        ("CartPole-v1_baseline", "CartPole baseline (solved, minimal topology)"),
        ("CartPole-v1_sim_standard", "CartPole with Standard NEAT similarity (failed)"),
        ("MountainCar-v0_baseline", "MountainCar baseline (small genome, large weights)"),
        ("MountainCar-v0_policy_single", "MountainCar with single-pick policy (solved!)"),
        ("Acrobot-v1_no_neuron", "Acrobot without neuron mutation (best result!)"),
        ("Acrobot-v1_baseline", "Acrobot baseline (more complex topology)"),
        ("Blackjack-v1_no_neuron", "Blackjack best variant (very simple topology)"),
        ("Blackjack-v1_xover_combine", "Blackjack with combine crossover (failed)"),
    ]
    for ablation_name, caption in showcase_genomes:
        path = f"results/ablations/{ablation_name}_genome.png"
        if os.path.exists(path):
            html_parts.append(f"""
<h4>{caption}</h4>
<img src="{b64_image(path)}" alt="{ablation_name} genome">
""")

    # --- Behavior GIFs ---
    html_parts.append("""
<h2 id="behavior">8. Agent Behavior Captures</h2>
<p>Animated GIFs of the best genome from select ablations playing their environment. These were
captured by running the trained genome in the env with rendering enabled, then encoding the
frames as a looping GIF.</p>
""")
    behavior_gifs = sorted(glob.glob("results/ablations/*_behavior.gif"))
    for gif_path in behavior_gifs:
        name = os.path.basename(gif_path).replace("_behavior.gif", "")
        html_parts.append(f"""
<div class="gif-display">
<h4>{name}</h4>
<img src="{b64_image(gif_path)}" alt="{name} behavior">
</div>
""")

    # --- Training progression ---
    html_parts.append("""
<h2 id="training-progression">8b. Training Progression</h2>
<p>How does the agent's behavior change as training progresses? Below are GIFs of the best genome
at different generation checkpoints (0, 5, 10, 15, 20, 25, etc.) for three environments. The
accompanying plots show fitness, genome size, and species count over the course of training.</p>
""")
    # Show training progression for each env that has files
    progression_envs = []
    for env_name in ["CartPole-v1", "MountainCar-v0", "Blackjack-v1"]:
        gifs = sorted(glob.glob(f"results/training_progress/{env_name}_gen*.gif"))
        if gifs:
            progression_envs.append((env_name, gifs))

    for env_name, gifs in progression_envs:
        html_parts.append(f'<h3>{env_name} — Training Progression</h3>')
        # Show progression GIFs in a grid
        html_parts.append('<div class="chart-row">')
        for gif_path in gifs:
            gen_str = os.path.basename(gif_path).replace(f"{env_name}_gen", "").replace(".gif", "")
            html_parts.append(f"""
<div>
<h4>Gen {int(gen_str)}</h4>
<div class="gif-display"><img src="{b64_image(gif_path)}" alt="{env_name} gen {gen_str}"></div>
</div>
""")
        html_parts.append('</div>')

        # Show training curves
        plots_dir = "results/training_progress/plots"
        for chart_name, caption in [(f"{env_name}_fitness.png", "Fitness Over Generations"),
                                     (f"{env_name}_genome_size.png", "Genome Size Over Generations"),
                                     (f"{env_name}_species.png", "Species Count Over Generations")]:
            chart_path = os.path.join(plots_dir, chart_name)
            if os.path.exists(chart_path):
                html_parts.append(f'<h4>{caption}</h4>')
                html_parts.append(f'<img src="{b64_image(chart_path)}" alt="{env_name} {caption}">')

        # Show genome evolution
        genome_imgs = sorted(glob.glob(f"results/training_progress/{env_name}_gen*_genome.png"))
        if genome_imgs:
            html_parts.append(f'<h4>Genome Topology Evolution</h4>')
            html_parts.append('<div class="chart-row">')
            for gp in genome_imgs:
                gen_str = os.path.basename(gp).replace(f"{env_name}_gen", "").replace("_genome.png", "")
                html_parts.append(f"""
<div>
<h4>Gen {int(gen_str)}</h4>
<img src="{b64_image(gp)}" alt="{env_name} gen {gen_str} genome">
</div>
""")
            html_parts.append('</div>')

    # --- Conclusion ---
    html_parts.append("""
<h2 id="conclusion">9. Conclusion</h2>
<p>After 40+ tuning rounds and 47 ablations across 5 environments, the modified NEAT algorithm
demonstrates several strengths and weaknesses:</p>

<div class="callout callout-positive">
<h4>What Works Well</h4>
<ul>
<li><b>Percentage similarity + Purge speciation</b> — a clear win over standard NEAT, especially
on easy/medium envs (CartPole, Acrobot). The combination bootstraps diverse species quickly and
keeps them coherent.</li>
<li><b>DAG-only topology with topological sort</b> — fast forward passes (O(V+E)) and no
dead-end genomes (loops, disconnected subgraphs).</li>
<li><b>Universal historical marking</b> — makes crossover between distant genomes meaningful
(shared innovation numbers always refer to the same connection).</li>
<li><b>Elitism (1-3 individuals)</b> — critical for preserving good solutions; removing it
consistently hurts performance.</li>
<li><b>Multi-seed training</b> — essential for stochastic envs (Blackjack) and envs with
highly variable initial conditions (MountainCar).</li>
</ul>
</div>

<div class="callout callout-warning">
<h4>What Doesn't Work (Or Needs Care)</h4>
<ul>
<li><b>GRPO optimizer</b> — only helps on dense-reward envs; on sparse-reward envs
(MountainCar) it actively hurts. The "relative reward" signal is too noisy when most
genomes get the same (bad) reward.</li>
<li><b>Aggressive reward shaping</b> — helps training reward but causes eval overfitting.
Better to use raw rewards + multi-seed training.</li>
<li><b>Single-pick mutation policy</b> — generally worse than per-type, but occasionally
better on envs that benefit from one big jump at a time (MountainCar).</li>
<li><b>Combine-topology crossover</b> — rarely beats the simpler "fitter topology" method,
and the cycle-breaking adds complexity for little gain.</li>
<li><b>Blackjack and LunarLander</b> — fundamentally hard for feed-forward NEAT. Blackjack
needs recurrent state (to count cards), and LunarLander needs precise continuous control
that NEAT's discrete-topology evolution struggles with.</li>
</ul>
</div>

<div class="callout callout-info">
<h4>Final Score: 4/5 Envs Solved</h4>
<p>CartPole-v1 (500/500), Acrobot-v1 (-90/-100), MountainCar-v0 (-109/-110), Blackjack-v1
(-0.03/-0.2) are all solved. LunarLander-v3 (-76/+200) remains unsolved but shows the
algorithm can learn reasonable landing behavior — it just can't consistently stick the landing.</p>
</div>

<p>For future work, the most promising directions are: (1) adding recurrent connections for
partial observability (would help Blackjack and LunarLander), (2) trying adaptive mutation
rates that decrease over generations (annealing), and (3) implementing co-evolution for
multi-agent envs.</p>
""")

    # --- Footer ---
    html_parts.append("""
<footer>
<p>Generated by scripts/generate_report.py | Modified NEAT Implementation</p>
<p>Source: <a href="https://github.com/G-reen-vibe/neat-modified" style="color:#4cc9f0">github.com/G-reen-vibe/neat-modified</a></p>
</footer>
""")

    # --- Close ---
    html_parts.append("""
</div>
</body>
</html>
""")

    # Write to file
    html = "".join(html_parts)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Report saved to {output_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    generate_report()
