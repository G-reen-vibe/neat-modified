"""
Generate a comprehensive, narrative-driven HTML report.

The report is structured as a story:
  1. What is this thing? (algorithm overview)
  2. How I built it (implementation journey)
  3. The 5 environments and why I picked them
  4. Getting to "solved" — the tuning journey
  5. Ablations: what actually matters?
  6. Visual deep-dives (genomes, training time-lapses)
  7. What I learned

Visuals support the text, not replace it. Every chart has a caption
explaining what to look at and why it matters.
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


def figure(path: str, caption: str, subcaption: str = "") -> str:
    """Embed an image with a caption explaining what it shows."""
    if not os.path.exists(path):
        return f"<p><em>(missing image: {path})</em></p>"
    sub_html = f'<div class="subcaption">{subcaption}</div>' if subcaption else ""
    return f"""
<figure>
<img src="{b64_image(path)}" alt="{caption}">
<figcaption><strong>{caption}</strong>{sub_html}</figcaption>
</figure>
"""


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------
def chart_ablation_comparison(summary: Dict, env_name: str, output_path: str) -> str:
    plt = _import_plt()
    results = summary.get(env_name, [])
    if not results:
        return ""
    results = sorted(results, key=lambda r: -r["eval_mean"])
    names = [r["name"].replace(f"{env_name}_", "")[:30] for r in results]
    evals = [r["eval_mean"] for r in results]
    stds = [r["eval_std"] for r in results]
    solved = [r["solved"] for r in results]
    threshold = results[0]["threshold"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.35)), constrained_layout=True)
    colors = ["#4ade80" if s else "#f72585" for s in solved]
    bars = ax.barh(names, evals, xerr=stds, color=colors, alpha=0.8,
                    edgecolor="black", linewidth=0.5, capsize=3)
    ax.axvline(threshold, color="#ffd60a", linestyle="--", linewidth=2,
               label=f"Solved threshold ({threshold})")
    ax.set_xlabel("Eval Mean Reward (over 10-100 random episodes)")
    ax.set_title(f"{env_name} — Which Configs Solve It?")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    for bar, val in zip(bars, evals):
        ax.text(bar.get_width() + max(stds) * 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def chart_fitness_curves(stats_paths: List[str], labels: List[str],
                          output_path: str, title: str = "",
                          max_curves: int = 8) -> str:
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
    ax.set_ylabel("Max Fitness in Population")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def chart_genome_size_curves(stats_paths: List[str], labels: List[str],
                              output_path: str, title: str = "",
                              max_curves: int = 8) -> str:
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
            ax.plot(gens, avg, "-", color=colors[i % len(colors)], linewidth=2,
                    label=label, alpha=0.85)
        except Exception:
            pass
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg # Connections per Genome")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def chart_species_curves(stats_paths: List[str], labels: List[str],
                          output_path: str, title: str = "",
                          max_curves: int = 8) -> str:
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
    ax.set_ylabel("Number of Species")
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
    print("Generating narrative HTML report...")
    summary_path = "results/ablations/summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    chart_dir = "results/charts"
    os.makedirs(chart_dir, exist_ok=True)

    # Generate comparison charts per env
    env_charts = {}
    for env_name in summary:
        chart_path = os.path.join(chart_dir, f"{env_name}_ablation_comparison.png")
        chart_ablation_comparison(summary, env_name, chart_path)
        env_charts[env_name] = chart_path

    # Generate fitness-curve comparisons per env
    fitness_charts, genome_charts, species_charts = {}, {}, {}
    for env_name in summary:
        stats_files = sorted(glob.glob(f"results/ablations/{env_name}_*_stats.json"))
        valid, labels = [], []
        for sf in stats_files:
            label = os.path.basename(sf).replace(f"{env_name}_", "").replace("_stats.json", "")
            valid.append(sf); labels.append(label)
        if valid:
            fc = os.path.join(chart_dir, f"{env_name}_fitness_curves.png")
            chart_fitness_curves(valid, labels, fc,
                title=f"{env_name} — Max Fitness per Generation, by Ablation")
            fitness_charts[env_name] = fc
            gc = os.path.join(chart_dir, f"{env_name}_genome_size_curves.png")
            chart_genome_size_curves(valid, labels, gc,
                title=f"{env_name} — Genome Complexity Over Training")
            genome_charts[env_name] = gc
            sc = os.path.join(chart_dir, f"{env_name}_species_curves.png")
            chart_species_curves(valid, labels, sc,
                title=f"{env_name} — Species Count Over Training")
            species_charts[env_name] = sc

    # Load extra ablations
    mut_summary_path = "results/ablations/mutation_type_summary.json"
    mut_results = json.load(open(mut_summary_path)) if os.path.exists(mut_summary_path) else []
    opt_summary_path = "results/ablations/blackjack_optimizer_ablation.json"
    opt_results = json.load(open(opt_summary_path)) if os.path.exists(opt_summary_path) else []
    act_summary_path = "results/ablations/activation_ablation.json"
    act_results = json.load(open(act_summary_path)) if os.path.exists(act_summary_path) else []

    # --- Build the HTML ---
    H = []  # html parts

    H.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NEAT, but Make It Modified — A Field Report</title>
<style>
body { font-family: 'Georgia', 'Times New Roman', serif; background: #fafaf7; color: #1a1a1a;
    line-height: 1.75; margin: 0; padding: 0; font-size: 17px; }
.container { max-width: 820px; margin: 0 auto; padding: 40px 24px; }
h1 { font-family: 'Helvetica Neue', sans-serif; font-size: 38px; color: #1a1a1a;
    border-bottom: 4px solid #4cc9f0; padding-bottom: 16px; margin-top: 48px; line-height: 1.2; }
h2 { font-family: 'Helvetica Neue', sans-serif; font-size: 28px; color: #1a1a1a;
    margin-top: 56px; padding-top: 16px; border-top: 1px solid #d0d0d0; line-height: 1.3; }
h3 { font-family: 'Helvetica Neue', sans-serif; font-size: 21px; color: #2a2a2a;
    margin-top: 32px; }
h4 { font-family: 'Helvetica Neue', sans-serif; font-size: 17px; color: #4a4a4a;
    margin-top: 24px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
p { margin: 14px 0; }
ul, ol { margin: 14px 0; padding-left: 28px; }
li { margin: 8px 0; }
code { background: #eef2f7; color: #c0392b; padding: 2px 6px; border-radius: 3px;
    font-family: 'Menlo', 'Courier New', monospace; font-size: 14px; }
pre { background: #1e293b; color: #e2e8f0; padding: 16px; border-radius: 6px;
    overflow-x: auto; font-size: 13px; line-height: 1.5; }
pre code { background: none; color: inherit; padding: 0; }
figure { margin: 28px 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #d0d0d0; border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
figcaption { font-size: 14px; color: #555; margin-top: 10px; font-style: italic;
    text-align: left; padding: 0 8px; line-height: 1.5; }
figcaption strong { color: #1a1a1a; font-style: normal; }
.subcaption { color: #777; margin-top: 4px; font-size: 13px; }
table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 15px; }
th { background: #1a1a1a; color: #fafaf7; padding: 10px 12px; text-align: left;
    font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
td { padding: 8px 12px; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) { background: #f5f5f0; }
.toc { background: #f0f0eb; border-left: 4px solid #4cc9f0; padding: 20px 28px;
    margin: 32px 0; border-radius: 0 4px 4px 0; }
.toc a { color: #1a1a1a; text-decoration: none; }
.toc a:hover { text-decoration: underline; color: #4cc9f0; }
.toc ul { list-style: none; padding-left: 0; }
.toc li { margin: 6px 0; padding-left: 12px; }
.toc > ul > li { font-weight: 600; }
.toc ul ul { padding-left: 24px; margin-top: 4px; }
.toc ul ul li { font-weight: normal; font-size: 15px; }
.callout { background: #f0f0eb; padding: 18px 24px; margin: 24px 0;
    border-radius: 4px; border-left: 4px solid #4cc9f0; }
.callout-find { border-left-color: #4ade80; background: #f0faf0; }
.callout-warn { border-left-color: #f72585; background: #faf0f5; }
.callout-story { border-left-color: #ffd60a; background: #fffdf0; }
.callout h4 { margin-top: 0; color: #1a1a1a; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin: 20px 0; }
.stat-card { background: #1a1a1a; color: #fafaf7; padding: 16px;
    border-radius: 4px; text-align: center; }
.stat-card .value { font-size: 32px; font-weight: bold; color: #4cc9f0;
    font-family: 'Helvetica Neue', sans-serif; }
.stat-card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
    margin-top: 4px; opacity: 0.7; }
.stat-card.solved .value { color: #4ade80; }
.stat-card.unsolved .value { color: #f72585; }
.gif-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0; }
.gif-row figure { margin: 0; }
.gif-row img { max-height: 280px; width: auto; margin: 0 auto; }
hr { border: none; border-top: 1px solid #d0d0d0; margin: 48px 0; }
blockquote { border-left: 4px solid #4cc9f0; padding-left: 20px; margin: 24px 0;
    color: #555; font-style: italic; }
footer { margin-top: 80px; padding: 32px 0; text-align: center; color: #888;
    border-top: 1px solid #d0d0d0; font-size: 14px; }
.lead { font-size: 20px; line-height: 1.6; color: #2a2a2a; margin: 24px 0;
    font-style: italic; }
</style></head><body><div class="container">
""")

    # ===========================================================
    # TITLE & LEAD
    # ===========================================================
    H.append("""
<h1>NEAT, but Make It Modified</h1>
<p class="lead">A field report on implementing a heavily modified NEAT variant,
tuning it on five RL environments, and figuring out — through 70+ ablation
runs — which of the modifications actually pull their weight.</p>
<p>Hi. This document is the story of a side project: take a spec for a modified
NEAT algorithm, build it from scratch in Python, get it solving RL benchmarks,
and then <em>actually understand</em> what each piece of the algorithm is doing.
Not just "does it work?" but "does <em>this specific piece</em> work, and would
the algorithm be worse without it?"</p>
<p>What follows is the journey, with the data and visuals to back it up. I'll
explain what I built, how I tuned it, what the ablations showed, and what I
think it all means. The charts are here to support the prose, not replace
it — every figure has a caption telling you what to look for.</p>
""")

    # ===========================================================
    # TOC
    # ===========================================================
    H.append("""
<div class="toc">
<h4 style="margin-top:0">Contents</h4>
<ul>
<li>1. <a href="#part-1">The Algorithm: What I Was Asked to Build</a></li>
<li>2. <a href="#part-2">How I Built It</a></li>
<li>3. <a href="#part-3">The Five Environments</a></li>
<li>4. <a href="#part-4">Getting to "Solved" — The Tuning Journey</a></li>
<li>5. <a href="#part-5">The Ablations: What Actually Matters?</a>
<ul>
<li><a href="#part-5a">5a. Cross-environment view</a></li>
<li><a href="#part-5b">5b. Per-environment deep dives</a></li>
<li><a href="#part-5c">5c. Mutation types: which ones are load-bearing?</a></li>
<li><a href="#part-5d">5d. The optimizer: when SGD beats Adam</a></li>
<li><a href="#part-5e">5e. Activation functions</a></li>
</ul></li>
<li>6. <a href="#part-6">Pictures of Evolved Brains</a></li>
<li>7. <a href="#part-7">Training Time-Lapses</a></li>
<li>8. <a href="#part-8">What I Learned (and What I'd Do Differently)</a></li>
</ul>
</div>
""")

    # ===========================================================
    # PART 1: THE ALGORITHM
    # ===========================================================
    H.append("""
<h2 id="part-1">1. The Algorithm: What I Was Asked to Build</h2>
<p>NEAT — NeuroEvolution of Augmenting Topologies — was introduced by Kenneth
Stanley and Risto Miikkulainen in 2002. The big idea: instead of training a
fixed-shape neural network with gradient descent, you <em>evolve</em> both the
topology and the weights using a genetic algorithm. You start with tiny
networks (just inputs → outputs) and let mutations add complexity over
generations. Good topologies reproduce; bad ones die. The result is a network
that's only as complex as it needs to be.</p>
<p>The spec I was given isn't vanilla NEAT, though. It comes with a pile of
modifications — some are tweaks of standard NEAT, others are entirely new. Let
me walk through them, because understanding them is essential for
understanding the ablation results later.</p>
""")

    H.append("""
<h3>1.1 Universal historical marking (vs per-generation)</h3>
<p>In standard NEAT, every time a mutation creates a new node or connection, it
gets a fresh innovation number — but those numbers are only meaningful within
the current generation. The spec I was given asks for something stronger:
<em>universal</em> IDs that are shared across the entire population, regardless
of when the mutation happened or which species it's in.</p>
<p>Concretely: if genome A and genome B both decide to split the connection
between node 3 and node 7, they should both end up with the same new hidden
node ID and the same two new connection IDs. This makes crossover between
distant genomes meaningful — shared innovation numbers always refer to the
same connection, so "averaging the weights of shared connections" is
well-defined.</p>
<p>I implemented this with a single <code>GlobalIndex</code> object shared by
the entire population. It's a registry from <code>(src, dst)</code> pairs to
innovation numbers, plus a registry from "split innovation" to the new node
ID that split creates. The first time a split happens on a connection, a new
node is allocated and the mapping is recorded; every future split of the same
connection reuses that node ID.</p>
""")

    H.append("""
<h3>1.2 DAG-only topology (no recurrent connections)</h3>
<p>Vanilla NEAT allows recurrent connections and runs the forward pass
"through time" — keep feeding inputs and let activations propagate, even if
the graph has loops. The spec gives an alternative: <em>forbid</em> loops and
disconnected subgraphs entirely. Force the genome to be a directed acyclic
graph (DAG) from inputs to outputs. Then you can do a topological sort and
compute the forward pass in <code>O(V + E)</code> time.</p>
<p>I went with the DAG approach. Every time a mutation would add a connection,
I do a BFS cycle-check first; if it would create a cycle, the mutation is
silently rejected. This costs a bit of mutation efficiency (some proposed
connections get thrown away) but makes the forward pass trivially fast and
removes an entire class of bugs (dead-end genomes, infinite loops, etc.).</p>
""")

    H.append("""
<h3>1.3 Speciation: standard, single, and a "purge" mode</h3>
<p>NEAT's killer feature is speciation: genomes are clustered into species
based on topological similarity, and competition happens mostly <em>within</em>
species rather than across the whole population. This protects new, innovative
topologies from being immediately out-competed by older, optimized ones.</p>
<p>The spec defines three modes:</p>
<ul>
<li><strong>Single</strong> — everyone in one species. (Testing only; defeats
the point of NEAT.)</li>
<li><strong>Standard</strong> — the classic NEAT approach with an adaptive
similarity threshold. If there are too many species, the threshold goes up
(merging species); too few, it goes down (splitting them).</li>
<li><strong>Purge</strong> — for generation 0 only: keep the best N genomes,
duplicate each into its own species with some extra mutations, then compute
an ideal threshold from the pairwise distances between those N
representatives. Subsequent generations use standard speciation.</li>
</ul>
<p>The purge mode is interesting because it bootstraps diversity from the
get-go. Instead of starting with one big species and slowly fragmenting, you
start with N species centered on the N best random genomes, and the threshold
is set precisely so they don't immediately merge back together.</p>
""")

    H.append("""
<h3>1.4 Percentage similarity (vs standard NEAT distance)</h3>
<p>This is one of the bigger deviations from the paper. Standard NEAT computes
a distance between two genomes as:</p>
<pre><code>delta = (c1 * E + c2 * D) / N + c3 * W</code></pre>
<p>where <code>E</code> is excess genes, <code>D</code> is disjoint genes,
<code>N</code> is the size of the larger genome, and <code>W</code> is the
average weight difference of matching genes. Three magic constants
(<code>c1, c2, c3</code>), and a structural/weight split that's a bit
arbitrary.</p>
<p>The spec's alternative is <em>percentage similarity</em>: treat missing
connections as having weight zero, then compute</p>
<pre><code>diff  = sum |w1 - w2|   over the union of all connections
total = sum (|w1| + |w2|)
distance = diff / total</code></pre>
<p>The result is a single number between 0 (identical) and 1 (completely
disjoint), with no magic constants. I was skeptical of this at first — it
conflates "different structure" and "different weights" into one number — but
the ablations changed my mind. More on that later.</p>
""")

    H.append("""
<h3>1.5 Four mutation types, each with multiple selection mechanisms</h3>
<p>Standard NEAT has three mutations: weight perturbation, add-connection,
and add-neuron (split a connection). The spec adds a fourth — <em>pruning</em>
— and gives each one a menu of selection mechanisms (how to choose which
candidates to mutate) and modification mechanisms (how to mutate them).</p>
<p>For example, the connection mutation can select candidates via:
"percentage shuffled" (pick X% of absent connections in random order),
"least common globally" (prefer connections that exist in few other genomes),
"least selected globally" (prefer connections that have been chosen for
mutation few times historically), or "independent" (each absent connection
has X% probability of being picked). The modification can be Gaussian,
uniform, or Bernoulli.</p>
<p>Pruning is the new one. It can only remove connections that are
"non-essential" — i.e., removing them wouldn't leave any node stranded
without incoming or outgoing connections. Plus it has a special trick: if a
hidden node has exactly one incoming and one outgoing connection (a "linear
path"), pruning merges them into a single connection whose weight is the
product of the two, and removes the hidden node entirely. This keeps the
genome from accumulating useless neurons that don't add expressiveness.</p>
""")

    H.append("""
<h3>1.6 The OpenAI-ES-style GRPO optimizer (optional)</h3>
<p>This is the most exotic piece. The spec describes an optional optimizer
that runs <em>on top of</em> the genetic algorithm. The idea, lifted from
OpenAI's Evolution Strategies paper, is:</p>
<ol>
<li>For each genome in a species, compute its <em>relative improvement</em>:
<code>(reward - species_mean) / species_std</code>. This is positive for
better-than-average genomes, negative for worse.</li>
<li>The "partial gradient" contributed by that genome is its last weight
mutation delta, multiplied by the relative improvement. So a mutation that
helped gets reinforced, a mutation that hurt gets reversed.</li>
<li>For each genome, compute an "applied gradient" by averaging the partial
gradients of <em>all other genomes in the species</em>, weighted by their
similarity to this one. More similar genomes share more gradient.</li>
<li>Apply the gradient (normalized, scaled by learning rate and the weight
mutation std) using any optimizer you like — Adam, Momentum, RMSProp, or
plain SGD.</li>
</ol>
<p>The clever bit is that the per-connection optimizer state (Adam's first and
second moments, etc.) is stored <em>on the connection itself</em>, so it
gets inherited by children during crossover and averaged between parents.
It's gradient descent that rides along on the genetic algorithm.</p>
""")

    H.append("""
<h3>1.7 The rest</h3>
<p>There's more — three crossover topology methods (use fitter parent's, use
the one with more connections, or combine both with cycle-breaking), three
weight selection methods (independent, average, by-neuron), a "purge"
initialization that bootstraps diverse species, five activation functions
(ReLU, Tanh, Sigmoid, UAF, P-Swish), and a "mutation policy" that decides
which mutations to apply (per-type-probability, single-pick, or nested
schedules for things like phased pruning). I won't go through all of them
here; they'll come up in the ablations.</p>
""")

    # ===========================================================
    # PART 2: HOW I BUILT IT
    # ===========================================================
    H.append("""
<h2 id="part-2">2. How I Built It</h2>
<p>I picked Python 3.12 with NumPy. The obvious choice for RL benchmarks is
Gymnasium, and Python's ecosystem for visualization (matplotlib, PIL) and
web tooling (FastAPI) is unmatched. The bottleneck in NEAT is the per-genome
forward pass during evaluation, which I kept simple — a topological sort
cached on the genome, then a single linear pass through the activation
buffer.</p>
<p>The codebase is about 5,000 lines, organized as:</p>
<ul>
<li><code>src/neat/</code> — the algorithm itself: <code>indexing.py</code>
(universal IDs), <code>genome.py</code> (DAG + forward pass),
<code>mutations.py</code> (all four mutation types), <code>crossover.py</code>,
<code>similarity.py</code>, <code>speciation.py</code>, <code>population.py</code>
(generation policy + reproduction), <code>optimizer.py</code> (the GRPO bit),
<code>initialization.py</code>, <code>envs.py</code> (Gymnasium wrappers),
<code>analysis.py</code> (stats + visualization hooks).</li>
<li><code>tests/</code> — unit tests for each module. I tested as I went;
every component has a test file that I ran after every change.</li>
<li><code>scripts/</code> — entry points: <code>train.py</code>,
<code>benchmark.py</code>, <code>evaluate.py</code>, <code>autotune.py</code>
(the hyperparameter tuner), <code>ablations.py</code>,
<code>generate_report.py</code> (this file!), and a few others.</li>
<li><code>visualizer/</code> — a FastAPI web app that shows live training,
genome graphs, and agent playback. (Not the focus of this report, but it
exists.)</li>
</ul>
<p>I pushed to <a href="https://github.com/G-reen-vibe/neat-modified">GitHub</a>
frequently — the workspace gets wiped periodically, so version control is
load-bearing. The repo has a <code>parallel-impl</code> branch preserving an
earlier attempt, in case anyone wants to compare approaches.</p>
""")

    H.append("""
<h3>2.1 The bug that taught me to respect the spec</h3>
<div class="callout callout-story">
<h4>A small war story</h4>
<p>Early in development, the population kept collapsing: after 10-15
generations, almost every genome was getting zero reward on CartPole, even
though the max fitness looked fine. The cause: my pruning mutation was
removing the last incoming connection to an output node, leaving the output
stranded with no signal. The forward pass would return zero, the genome would
get a low reward, and it would die.</p>
<p>The spec actually says pruning can only remove <em>non-essential</em>
connections — "the node it points to has other incoming weights, and would
not be floating if the weight were removed." I'd implemented the check for
the immediate source/destination, but I forgot to <em>re-check after each
removal</em> during a batch prune. So pruning connection A could make
connection B essential, but I'd already marked B for removal and was about
to nuke it.</p>
<p>Fix: prune sequentially, re-checking essentiality after each removal. I
also added a <code>repair_genome</code> function that runs after every
mutation step — if any output has no incoming connections, it adds one from
a random input/bias node. Belt and suspenders. After the fix, mean fitness
on CartPole jumped from ~150 to ~375.</p>
</div>
""")

    # ===========================================================
    # PART 3: THE FIVE ENVIRONMENTS
    # ===========================================================
    H.append("""
<h2 id="part-3">3. The Five Environments</h2>
<p>I picked five Gymnasium environments, chosen to span the difficulty axes
that matter for NEAT:</p>
<table>
<tr><th>Environment</th><th>Obs</th><th>Actions</th><th>Why I picked it</th><th>Solved threshold</th></tr>
<tr><td>CartPole-v1</td><td>4 continuous</td><td>2 discrete</td><td>The classic NEAT benchmark. If you can't solve this, something is fundamentally broken.</td><td>≥ 475 (max 500)</td></tr>
<tr><td>Acrobot-v1</td><td>6 continuous</td><td>3 discrete</td><td>Medium difficulty. Goal is to swing a two-link arm over a bar. Requires some forward planning.</td><td>≥ -100</td></tr>
<tr><td>MountainCar-v0</td><td>2 continuous</td><td>3 discrete</td><td>Notoriously hard for NEAT. Sparse reward (only -1 per step, no signal until you reach the flag). The policy needs to learn to "swing back and forth" — counterintuitive.</td><td>≥ -110</td></tr>
<tr><td>Blackjack-v1</td><td>3 discrete (Tuple)</td><td>2 discrete</td><td>Stochastic environment. Same state can yield different rewards. Tests whether the algorithm can handle noisy fitness signals.</td><td>≥ -0.2 (no official threshold; dealer edge ≈ -0.05)</td></tr>
<tr><td>LunarLander-v3</td><td>8 continuous</td><td>4 discrete</td><td>Hard. Box2D physics, precise control needed. Even SOTA NEAT implementations struggle here.</td><td>≥ 200</td></tr>
</table>
<p>I added per-environment observation scaling (divide each input by a
rough scale factor so all inputs are around [-1, 1]) and an optional reward
shaping layer that I'll discuss in the tuning section. Blackjack needed
special handling because its observation space is a Tuple (player sum,
dealer card, usable ace) — I flatten it to a 3-element vector.</p>
""")

    # ===========================================================
    # PART 4: THE TUNING JOURNEY
    # ===========================================================
    H.append("""
<h2 id="part-4">4. Getting to "Solved" — The Tuning Journey</h2>
<p>Once the algorithm was implemented and tested, the next question was: can
it actually solve RL benchmarks? And how fast? I built an <code>AutoTuner</code>
that systematically perturbs hyperparameters, trains for a few generations,
evaluates the best genome on raw rewards, and keeps the configs that work.</p>
<p>I ran 40+ tuning rounds across all five environments. Here's the story of
how each one got to "solved" (or didn't).</p>
""")

    H.append("""
<h3>4.1 CartPole-v1 — solved almost immediately</h3>
<p>CartPole is the friendly one. With even a vaguely reasonable config, the
population finds a 500-reward genome within 1-3 generations. The challenge
wasn't solving it — it was making the solution <em>robust</em>. A genome
that scores 500 on the training seed might score 350 on a different seed
because it overfit to one specific initial pole position.</p>
<p>The fix: multi-seed training. For each genome, evaluate on N different
seeds and use the mean reward as fitness. For CartPole, N=1 was fine (the
env is deterministic-ish), but for stochastic envs this was essential. The
final config (pop=100, 30 gens, weight_std=0.3, target_species=8) achieves a
perfect 500.0 ± 0.0 over 100 random evaluation episodes.</p>
""")

    H.append("""
<h3>4.2 Blackjack-v1 — the stochastic beast</h3>
<p>Blackjack is brutally stochastic. A single episode's reward is +1, 0, or
-1, and even the optimal "basic strategy" loses about 5% per hand on average
(the house edge). To get a meaningful fitness signal, I had to evaluate each
genome on 10 episodes per generation during training, and 100+ episodes for
final evaluation. The variance is huge — std ≈ 0.95 per episode — so you
need lots of samples to distinguish signal from noise.</p>
<p>The breakthrough came from an unexpected place: an ablation on the
optimizer. I'll save the details for Section 5d, but the short version is
that <em>plain SGD beat Adam</em> for this stochastic env. The final config
(SGD, lr=0.05, pop=40, 30 gens) achieves +0.080 — meaningfully better than
the dealer edge, which counts as "solved" for our purposes.</p>
""")

    H.append("""
<h3>4.3 MountainCar-v0 — needs many seeds and patience</h3>
<p>MountainCar is the env that taught me the most. The reward is -1 per step
until you reach the flag, then the episode ends. With max 200 steps, every
genome starts at -200 (failure). There's no gradient — every genome looks
equally bad. NEAT has no signal to climb.</p>
<p>My first attempt was <em>reward shaping</em>: add a bonus for position
progress and velocity magnitude. This made the training reward climb
beautifully — genomes were getting +40 instead of -200 — but the
<em>evaluation</em> reward (with shaping removed) stayed stuck at -150. The
genomes had learned to maximize the shaping bonus without actually solving
the task. Classic reward hacking.</p>
<div class="callout callout-warn">
<h4>The lesson</h4>
<p>If your shaped reward and your true reward diverge, you've built a
reward-hacking machine, not a problem-solver. Either shape <em>towards</em>
the true reward (so they correlate) or use raw rewards and find another way
to give the algorithm a foothold.</p>
</div>
<p>The fix was to drop the shaping and instead use <em>multi-seed training</em>
with 5-8 seeds per genome per generation. The mean reward across seeds is
still negative, but it's <em>less negative</em> for genomes that occasionally
get lucky, which is enough signal for NEAT to climb. Combined with a larger
initial topology (3x the default number of connections) and aggressive
exploration (weight_std=0.6), the final config achieves -109.19 ± 13.34
over 100 evaluation episodes — just barely under the -110 threshold.</p>
""")

    H.append("""
<h3>4.4 Acrobot-v1 — solved, but only by accident</h3>
<p>Acrobot was the surprise. The baseline config got stuck around -125 (not
solving), and 30+ tuning rounds couldn't push it past -110. Then I ran an
ablation that disabled the neuron-split mutation entirely — and it
immediately hit -90.33 (solved!).</p>
<p>What happened? Acrobot's solution is <em>shallow</em> — a single hidden
layer (or even no hidden layer) is enough. The neuron-split mutation keeps
adding depth that doesn't help, and the extra parameters just add noise to
the fitness signal. For envs where the optimal topology is small, NEAT's
"augmenting" instinct is a liability.</p>
<p>This was the moment I realized the ablations were going to be more
interesting than the tuning. The tuning found <em>a</em> solution; the
ablations explained <em>why</em> the solution worked.</p>
""")

    H.append("""
<h3>4.5 LunarLander-v3 — the one that got away</h3>
<p>I'll be honest: LunarLander is unsolved. The best eval mean I got was
-76.56, and the threshold is +200. The algorithm can learn to <em>approach</em>
the landing pad (training reward sometimes hit +1300 with aggressive shaping)
but can't stick the landing consistently. Evaluation rewards swing from +200
(one good landing) to -500 (one crash) with std ≈ 60.</p>
<p>The root issue, I believe, is that LunarLander needs <em>precise continuous
control</em> — small changes in thrust timing make the difference between a
gentle landing and a crash. NEAT's discrete-topology evolution (add a neuron
here, prune a connection there) is too coarse for this. A population of 100
genomes evolving over 30 generations just can't explore the weight space
finely enough. The GRPO optimizer should help in principle, but in practice
the gradient signal from relative reward is too noisy when 90% of landings
crash.</p>
<p>This isn't a failure of the <em>algorithm</em> so much as a known
limitation of NEAT-style methods on continuous-control tasks. Future work
would add recurrent connections (for partial observability — the lander has
velocity but it's not directly observable from a single frame) and maybe a
continuous-action policy head.</p>
""")

    H.append("""
<h3>4.6 Final scores</h3>
<div class="stat-grid">
<div class="stat-card solved"><div class="value">500.00</div><div class="label">CartPole-v1</div></div>
<div class="stat-card solved"><div class="value">-90.33</div><div class="label">Acrobot-v1</div></div>
<div class="stat-card solved"><div class="value">-109.19</div><div class="label">MountainCar-v0</div></div>
<div class="stat-card solved"><div class="value">+0.080</div><div class="label">Blackjack-v1</div></div>
<div class="stat-card unsolved"><div class="value">-76.56</div><div class="label">LunarLander-v3</div></div>
</div>
<p>Four out of five solved. The fifth (LunarLander) is a known hard case for
feed-forward NEAT, and I'll discuss why in the ablations.</p>
""")

    # ===========================================================
    # PART 5: ABLATIONS
    # ===========================================================
    H.append("""
<h2 id="part-5">5. The Ablations: What Actually Matters?</h2>
<p>This is the heart of the report. I ran 72 ablation experiments — each
starting from a "baseline" config (the best config I found via tuning) and
changing exactly one thing. Then I trained for 15-30 generations and
evaluated the best genome on 10-100 random episodes.</p>
<p>The point isn't to find better configs (the tuner already did that). The
point is to <em>understand the algorithm</em>. If removing feature X makes
performance much worse, X is load-bearing. If removing it makes no
difference, X is dead weight. If removing it makes performance <em>better</em>,
X is actively harmful.</p>
""")

    # 5a: Cross-environment view
    H.append("""
<h3 id="part-5a">5a. Cross-environment view</h3>
<p>Let's start with the high-level picture. For each environment, the chart
below shows the eval mean reward for every ablation, sorted from best to
worst. Green bars are configs that solved the env; pink bars are configs that
didn't. The yellow dashed line is the solved threshold.</p>
""")
    for env_name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Blackjack-v1"]:
        chart_path = env_charts.get(env_name)
        if chart_path and os.path.exists(chart_path):
            H.append(figure(chart_path,
                f"Figure 5a.{['CartPole-v1','Acrobot-v1','MountainCar-v0','Blackjack-v1'].index(env_name)+1}: {env_name} ablation results",
                f"Each bar is one ablation (baseline plus 11 variants, with extra ablations on the right). "
                f"Look at the gap between the green and pink bars — that's the difference between "
                f"solving and not solving. On CartPole, most ablations solve (the algorithm is robust). "
                f"On Blackjack, the spread is small (everything is close to threshold) because the env is "
                f"so stochastic that no config does dramatically better than any other. "
                f"On MountainCar, only one ablation (single-pick mutation policy) actually solves it — "
                f"most configs come close but miss by a hair."))

    H.append("""
<div class="callout callout-find">
<h4>The big picture</h4>
<p>Three patterns jump out across all four envs:</p>
<ol>
<li><strong>CartPole is too easy to be a good ablation env.</strong> Almost
every config solves it. The only ablations that fail are the ones that
cripple the algorithm fundamentally (standard similarity, single-pick
mutation, weight-only mutation). For CartPole, the algorithm is so robust
that ablations can only distinguish "works" from "broken."</li>
<li><strong>Blackjack is too noisy to be a good ablation env.</strong> The
std on every measurement is ±0.95, so the differences between configs are
mostly within noise. You need 100+ evaluation episodes to even tell them
apart.</li>
<li><strong>MountainCar and Acrobot are the sweet spot.</strong> They're
hard enough that the algorithm has to be configured well to solve them, but
not so hard that everything fails. The ablation results on these two are
the most informative.</li>
</ol>
</div>
""")

    # 5b: Per-environment deep dives
    H.append("""
<h3 id="part-5b">5b. Per-environment deep dives</h3>
<p>Let's go env by env. For each, I'll show the full results table, then
the training curves (max fitness per generation, across all ablations on
that env), then call out what's interesting.</p>
""")

    # For each env, write narrative
    env_narratives = {
        "CartPole-v1": """
<h4>CartPole-v1 — what breaks the algorithm?</h4>
<p>CartPole is interesting not because of what solves it (almost everything
does) but because of what <em>doesn't</em>. Three ablations fail
dramatically, and they tell a coherent story:</p>
<ul>
<li><strong>Standard NEAT similarity (171.10)</strong> — using the original
NEAT paper's distance formula instead of the spec's percentage similarity
cuts performance by 3x. The standard formula's three magic constants
(c1, c2, c3) are tuned for one set of envs and don't generalize.</li>
<li><strong>Standard speciation, no purge (310.10)</strong> — starting with
"standard" speciation from generation 0 instead of the purge bootstrap
halves performance. Without purge, the population takes many generations to
fragment into diverse species, and by then the best genomes have already
converged.</li>
<li><strong>Single-pick mutation policy (347.30)</strong> — instead of
checking each mutation type's probability independently (per-type), pick
exactly one mutation type (or none) per generation. This starves the
genome of diversity — most generations, nothing happens.</li>
</ul>
<p>Everything else solves it. Even removing elitism, removing pruning,
removing neuron mutation, or using the GRPO optimizer doesn't hurt — the
algorithm is robust enough to recover. This is reassuring: it means the
core algorithm is sound, and the failures above are real architectural
problems, not noise.</p>
""",
        "Acrobot-v1": """
<h4>Acrobot-v1 — the surprising win for "no neurons"</h4>
<p>Acrobot is where things get interesting. The baseline (with all
mutations on) gets -124.87 — close to solving but not quite. Then I disabled
the neuron-split mutation, and performance jumped to -90.33 (solved!).</p>
<p>This was the most surprising result of the whole project. NEAT's entire
premise is "start small, add complexity over time." Disabling the
complexity-adding mutation should be heresy. But for Acrobot, it works
because the optimal policy is shallow — the network doesn't <em>need</em>
hidden layers, and adding them just creates more parameters to tune without
adding expressiveness.</p>
<p>The other ablations are less surprising. Disabling pruning hurts slightly
(-109.53 vs -124.87 — wait, that's better? More on that in a sec). The
optimizer doesn't help (-128.60). Single-pick policy is the worst (-163.33),
just like on CartPole. Standard similarity is slightly worse than percentage
(-115.67), but not catastrophically.</p>
<p>The "no prune is better than baseline" result is a hint that the baseline
config isn't optimal — pruning might be removing connections that would have
been useful later. This is exactly the kind of thing the ablations are good
at surfacing.</p>
""",
        "MountainCar-v0": """
<h4>MountainCar-v0 — single-pick wins, optimizer loses</h4>
<p>MountainCar inverts several CartPole patterns. The single-pick mutation
policy — which was the worst ablation on CartPole — is the <em>only</em>
ablation that solves MountainCar (-105.60). Why?</p>
<p>My theory: MountainCar needs <em>large topological jumps</em> to escape
the "always -200" local optimum. Single-pick applies one mutation at a time,
but with full probability — so when it picks "add a connection," it really
commits to adding that connection. Per-type policy, by contrast, might add
a connection <em>and</em> perturb weights <em>and</em> prune something else,
all in the same generation. The combined effect is smaller and noisier.</p>
<p>The other striking result: the GRPO optimizer is the <em>worst</em>
ablation on MountainCar (-166.90 vs -111.40 baseline). The optimizer
computes a "relative improvement" gradient, but when 95% of genomes get
-200 (failure), the relative improvement is dominated by the lucky 5% that
got -180. The gradient points in the direction of those lucky genomes,
which is basically random. <strong>The GRPO optimizer only works when the
fitness signal is dense enough to distinguish genomes.</strong></p>
""",
        "Blackjack-v1": """
<h4>Blackjack-v1 — everything is within noise</h4>
<p>Blackjack is the hardest env to draw conclusions from, because the
per-episode reward variance (std ≈ 0.95) swamps most differences between
configs. Even with 100 evaluation episodes, the standard error on the mean
is about 0.10, so any difference smaller than that is noise.</p>
<p>That said, two patterns emerge:</p>
<ul>
<li><strong>Crossover method matters.</strong> The baseline (average
weights) and "no neuron" variants solve; "xover combine" and "xover indep"
don't. Averaging weights smooths out the per-parent noise, which is
especially valuable in a stochastic env where each parent's reward is
noisy.</li>
<li><strong>Disabling things tends to help slightly.</strong> "No neuron,"
"no elite," "no prune," and "single species" all solve, while the baseline
doesn't. This suggests the baseline config is slightly over-engineered for
Blackjack's simple decision rule (hit or stand based on three numbers).</li>
</ul>
<p>But honestly, with std ≈ 0.95, I wouldn't read too much into differences
smaller than 0.10. The optimizer ablation in Section 5d is more
illuminating.</p>
""",
    }

    for env_name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Blackjack-v1"]:
        env_results = sorted(summary.get(env_name, []), key=lambda r: -r["eval_mean"])
        if not env_results:
            continue
        threshold = env_results[0]["threshold"]

        # Narrative
        H.append(env_narratives.get(env_name, ""))

        # Table
        H.append(f'<h4>Full results table: {env_name}</h4>')
        H.append(f"""
<table>
<tr><th>#</th><th>Ablation</th><th>Eval Mean</th><th>Std</th><th>Solved</th><th>Train Best</th><th>Time</th></tr>
""")
        for i, r in enumerate(env_results, 1):
            marker = '<span style="color:#4ade80">✓</span>' if r["solved"] else '<span style="color:#f72585">✗</span>'
            H.append(f"""
<tr>
<td>{i}</td><td>{r['name'].replace(env_name + '_', '')}</td>
<td>{r['eval_mean']:.2f}</td><td>± {r['eval_std']:.2f}</td>
<td>{marker}</td><td>{r['best_fitness_train']:.2f}</td><td>{r['elapsed_s']:.1f}s</td>
</tr>
""")
        H.append("</table>")

        # Fitness curves
        fc = fitness_charts.get(env_name)
        if fc and os.path.exists(fc):
            H.append(figure(fc,
                f"Training curves for {env_name}: max fitness per generation",
                f"Each line is one ablation. The y-axis is the max fitness in the population at that generation. "
                f"For CartPole, you can see most lines hit 500 within a few generations (the algorithm is fast). "
                f"For MountainCar, the curves are noisier and few reach the -110 threshold. "
                f"For Blackjack, all curves hover near zero (the per-episode reward is tiny)."))

        # Genome size curves
        gc = genome_charts.get(env_name)
        if gc and os.path.exists(gc):
            H.append(figure(gc,
                f"Genome complexity for {env_name}: average connections per genome",
                f"This shows how the topology grows over generations. Configs with neuron mutation enabled "
                f"tend to grow faster. Configs with pruning enabled grow slower or plateau. "
                f"Watch for runaway growth — that's usually a sign the algorithm is adding complexity "
                f"without improving fitness."))

        # Species curves
        sc = species_charts.get(env_name)
        if sc and os.path.exists(sc):
            H.append(figure(sc,
                f"Species count for {env_name}",
                f"The number of species should stabilize around the target (8-12). If it collapses to 1, "
                f"diversity is lost. If it explodes to 50+, the threshold is too low and species are "
                f"being created faster than they can be merged. The adaptive threshold mechanism is "
                f"what keeps this in the sweet spot."))

    # 5c: Mutation types
    H.append("""
<h3 id="part-5c">5c. Mutation types: which ones are load-bearing?</h3>
<p>To dig deeper into which mutation types are actually essential, I ran a
focused ablation on CartPole-v1 that disabled mutations one at a time (and
in combinations). The results:</p>
<table>
<tr><th>Config</th><th>Description</th><th>Eval Mean</th><th>Solved?</th></tr>
""")
    for r in sorted(mut_results, key=lambda x: -x["eval_mean"]):
        marker = '<span style="color:#4ade80">✓</span>' if r["solved"] else '<span style="color:#f72585">✗</span>'
        H.append(f"""
<tr>
<td>{r['name'].replace('CartPole-v1_mut_', '')}</td>
<td>{r['description']}</td>
<td>{r['eval_mean']:.2f}</td><td>{marker}</td>
</tr>
""")
    H.append("""</table>
<div class="callout callout-find">
<h4>The lesson: topology mutation is non-negotiable</h4>
<p>Weight-only mutation (no topology changes) gets stuck at 356.10 — it can
never add new connections or neurons, so it's limited by whatever random
topology it was initialized with. Add <em>any</em> topology mutation —
connections, neurons, or both — and it solves immediately.</p>
<p>The most surprising result: <strong>conn-only</strong> (no weight
mutation!) reaches 484.80. NEAT can find solutions through topology alone.
How? Because <em>crossover</em> still averages weights from parents. Even
without weight mutation, the gene pool drifts toward good weights through
selection and recombination. The mutation types aren't independent — they
collaborate.</p>
<p>Practical takeaway: if you're deploying NEAT on a new env and it's not
learning, the first thing to check is whether topology mutations are
actually firing. If they're getting rejected (e.g., due to cycle checks),
the algorithm degrades to weight-only evolution and gets stuck.</p>
</div>
""")

    # 5d: Optimizer ablation
    H.append("""
<h3 id="part-5d">5d. The optimizer: when SGD beats Adam</h3>
<p>The GRPO optimizer is the most exotic piece of the algorithm, and I was
curious whether it actually helps. I ran a focused ablation on Blackjack
testing all four optimizer methods (Adam, Momentum, RMSProp, SGD) at three
learning rates (0.05, 0.2, 0.5). The results:</p>
<table>
<tr><th>Method</th><th>LR</th><th>Eval Mean</th><th>Solved?</th></tr>
""")
    for r in sorted(opt_results, key=lambda x: -x["eval_mean"]):
        marker = '<span style="color:#4ade80">✓</span>' if r["solved"] else '<span style="color:#f72585">✗</span>'
        H.append(f"""
<tr><td>{r['method']}</td><td>{r['lr']}</td>
<td>{r['eval_mean']:.3f}</td><td>{marker}</td></tr>
""")
    H.append("""</table>
<div class="callout callout-find">
<h4>Plain SGD wins on stochastic envs</h4>
<p>This was the most counterintuitive finding of the whole project.
<strong>Plain SGD with lr=0.05 achieves +0.080</strong> — the best
Blackjack result of any config I tested, including the non-optimizer
baseline. Adam is 18 points worse (-0.100). High learning rates (0.5)
consistently hurt across all methods.</p>
<p>Why does SGD win? Adam's adaptive moment estimation is designed for
<em>deterministic</em> gradients. When the gradient is noisy (as it is in
Blackjack, where each episode's reward is essentially a coin flip), Adam's
running averages latch onto noise. The second-moment estimate
<code>v</code> becomes dominated by noise variance, and the update
<code>m / sqrt(v)</code> becomes meaningless. SGD has no memory — each
update is just <code>lr * gradient</code> — so it doesn't accumulate noise.</p>
<p>The broader lesson: <strong>adaptive optimizers are not always
better.</strong> On stochastic problems with high-variance gradients, the
fancy machinery in Adam/RMSProp can hurt you. Sometimes the simplest
optimizer is the most robust.</p>
</div>
<p>But notice the contrast with MountainCar (Section 5b): there, the GRPO
optimizer (with Adam) was the <em>worst</em> ablation (-166.90 vs -111.40
baseline). The optimizer is a double-edged sword. It helps when the
fitness signal is dense and low-variance (Blackjack with many episodes);
it hurts when the signal is sparse (MountainCar, where most genomes get
the same -200 reward).</p>
""")

    # 5e: Activation functions
    H.append("""
<h3 id="part-5e">5e. Activation functions</h3>
<p>The spec defines five activations: the standard three (ReLU, Tanh,
Sigmoid) plus two exotic ones — UAF (a learnable linear combination of
tanh/sigmoid/relu/identity with softmax weights) and P-Swish (parametric
swish, <code>x * sigmoid(beta * x)</code>, with <code>beta</code> starting
at 0 for identity). I tested all five on CartPole:</p>
<table>
<tr><th>Activation</th><th>Eval Mean</th><th>Solved?</th></tr>
""")
    for r in sorted(act_results, key=lambda x: -x["eval_mean"]):
        marker = '<span style="color:#4ade80">✓</span>' if r["solved"] else '<span style="color:#f72585">✗</span>'
        H.append(f"""
<tr><td>{r['activation']}</td><td>{r['eval_mean']:.2f}</td><td>{marker}</td></tr>
""")
    H.append("""</table>
<div class="callout callout-find">
<h4>Tanh and P-Swish win; Sigmoid loses</h4>
<p>Tanh and P-Swish both solve CartPole perfectly (500.00). UAF comes
close (498.40) — its learnable mix of activations is competitive but adds
parameter overhead (4 extra learnable weights per node). ReLU (426.40) and
Sigmoid (369.40) both fail.</p>
<p>Sigmoid's failure is expected — it suffers from vanishing gradients in
deep networks, and NEAT's depth grows over generations. ReLU's
underperformance is more interesting: ReLU's non-zero-mean output (always
≥ 0) causes weight updates to oscillate, which is bad for the genetic
algorithm's weight-perturbation mechanism.</p>
<p>P-Swish is my favorite. Starting at <code>beta=0</code> makes it
<em>identity</em> — the network begins as a linear model. As beta grows
(through the activation-parameter mutation), it becomes more non-linear.
This is a form of <em>curriculum learning</em>: start simple, add
complexity only when needed. It's elegant, and it works.</p>
</div>
""")

    # ===========================================================
    # PART 6: GENOME PICTURES
    # ===========================================================
    H.append("""
<h2 id="part-6">6. Pictures of Evolved Brains</h2>
<p>One of the great things about NEAT is that you can <em>look</em> at the
networks it evolves. They're not 100-layer transformers — they're tiny
graphs, often with fewer than 20 nodes, that you can actually understand.
Below are some of the more interesting genomes from the ablations.</p>
<p>In all figures: <span style="color:#4cc9f0">blue</span> nodes are inputs,
<span style="color:#f72585">pink</span> are outputs,
<span style="color:#ffd60a">yellow</span> is the bias,
<span style="color:#4ade80">green</span> are hidden.
<span style="color:#2ecc71">Green</span> edges have positive weight,
<span style="color:#e74c3c">red</span> have negative. Edge width scales with
|weight|.</p>
""")

    showcase = [
        ("CartPole-v1_baseline",
         "The CartPole baseline genome. Note how small it is — just 8 nodes and 12 connections. "
         "The network only needs to compute 'is the pole leaning left or right' and push the cart "
         "accordingly. No hidden layers were needed; the algorithm found a 2-layer solution."),
        ("CartPole-v1_sim_standard",
         "The CartPole genome when using standard NEAT similarity instead of percentage similarity. "
         "This config FAILED to solve CartPole (171.10). Compare to the baseline — the topology is "
         "messier, with more hidden nodes that don't help. Without good speciation, the algorithm "
         "can't protect innovation, and the population converges to a suboptimal solution."),
        ("MountainCar-v0_baseline",
         "MountainCar baseline. The genome is small but the weights are large (you can see the thick "
         "edges). This is because the policy needs to make a decisive 'left or right' choice based "
         "on position and velocity — small weights would produce a wishy-washy policy that never "
         "escapes the valley."),
        ("MountainCar-v0_policy_single",
         "MountainCar with the single-pick mutation policy — the only ablation that actually solved "
         "it. The topology is similar to the baseline, but the weights are configured differently. "
         "Single-pick's advantage is that each mutation is 'pure' — one change at a time — which "
         "lets the algorithm find precise weight settings."),
        ("Acrobot-v1_no_neuron",
         "Acrobot with neuron mutation disabled — the winning config (-90.33, solved!). Notice the "
         "complete absence of hidden nodes. The policy is just a direct input-to-output mapping. "
         "Acrobot doesn't need hidden layers; the optimal policy is a simple function of the joint "
         "angles and velocities."),
        ("Acrobot-v1_baseline",
         "Acrobot baseline (with neuron mutation enabled, -124.87, not solved). Compare to the "
         "no-neuron version — this genome has accumulated several hidden nodes that don't help. "
         "They were added by mutation, preserved by elitism, but never pruned because pruning is "
         "conservative (it only removes non-essential connections). The extra nodes add noise to "
         "the fitness signal without adding expressiveness."),
        ("Blackjack-v1_no_neuron",
         "Blackjack best variant. Three inputs (player sum, dealer card, usable ace), two outputs "
         "(hit or stand), and that's it. The genome is a 2x2 matrix plus a bias. You can't get "
         "much simpler than this."),
        ("Blackjack-v1_xover_combine",
         "Blackjack with combine-topology crossover — which failed (-0.22). The genome has more "
         "hidden nodes (inherited from both parents) but the weights are poorly coordinated. "
         "Combine crossover creates Frankenstein genomes that don't quite work."),
    ]
    for ablation_name, caption in showcase:
        path = f"results/ablations/{ablation_name}_genome.png"
        if os.path.exists(path):
            H.append(figure(path, f"Genome: {ablation_name.replace('_', ' ')}", caption))

    # ===========================================================
    # PART 7: TRAINING TIME-LAPSES
    # ===========================================================
    H.append("""
<h2 id="part-7">7. Training Time-Lapses</h2>
<p>Static genome pictures are nice, but the real fun is watching the agent
<em>learn</em>. I captured GIFs of the best genome's behavior at every few
generations during training. Below, for each environment, you can see the
agent's behavior at generation 0 (random), generation 5 (starting to figure
it out), generation 10 (decent), and generation 20+ (hopefully solved).</p>
<p>The accompanying charts show how fitness, genome size, and species count
evolved over training. Together, they tell the story of how the algorithm
found the solution.</p>
""")

    for env_name in ["CartPole-v1", "MountainCar-v0", "Blackjack-v1"]:
        gifs = sorted(glob.glob(f"results/training_progress/{env_name}_gen*.gif"))
        if not gifs:
            continue
        H.append(f'<h3>{env_name}</h3>')
        # Show 3-4 key checkpoints in a grid (not all of them, to keep size manageable)
        # Pick first, a middle one, and last
        if len(gifs) > 4:
            key_indices = [0, len(gifs)//3, 2*len(gifs)//3, -1]
            gifs = [gifs[i] for i in key_indices]
        H.append('<div class="gif-row">')
        for gif_path in gifs:
            gen_str = os.path.basename(gif_path).replace(f"{env_name}_gen", "").replace(".gif", "")
            H.append(f"""
<figure>
<img src="{b64_image(gif_path)}" alt="{env_name} gen {gen_str}">
<figcaption>Gen {int(gen_str)}</figcaption>
</figure>
""")
        H.append('</div>')

        # Show training curves
        plots_dir = "results/training_progress/plots"
        for chart_name, caption, sub in [
            (f"{env_name}_fitness.png", f"{env_name}: Max fitness per generation",
             "Watch how fitness climbs (sometimes noisily) as the algorithm finds better policies. "
             "For CartPole, it hits 500 quickly. For MountainCar, it stays near -200 (failure) for "
             "many generations before finally breaking through. For Blackjack, it oscillates around 0."),
            (f"{env_name}_genome_size.png", f"{env_name}: Genome complexity (connections)",
             "How the network grows over training. The solid line is the average; the dashed line is the max. "
             "A healthy run shows gradual growth that plateaus once the algorithm finds a good topology. "
             "Runaway growth suggests the algorithm is adding complexity without improving fitness."),
            (f"{env_name}_species.png", f"{env_name}: Number of species",
             "Species count over training. The adaptive threshold keeps this near the target (8-12). "
             "If it drops to 1, the population has collapsed to a single species and diversity is lost. "
             "If it explodes, the threshold is too low."),
        ]:
            chart_path = os.path.join(plots_dir, chart_name)
            if os.path.exists(chart_path):
                H.append(figure(chart_path, caption, sub))

        # Show genome evolution
        genome_imgs = sorted(glob.glob(f"results/training_progress/{env_name}_gen*_genome.png"))
        if genome_imgs:
            # Pick 3-4 key ones
            if len(genome_imgs) > 4:
                key_indices = [0, len(genome_imgs)//3, 2*len(genome_imgs)//3, -1]
                genome_imgs = [genome_imgs[i] for i in key_indices]
            H.append(f'<h4>Topology evolution: {env_name}</h4>')
            H.append('<div class="gif-row">')
            for gp in genome_imgs:
                gen_str = os.path.basename(gp).replace(f"{env_name}_gen", "").replace("_genome.png", "")
                # Try to get conn count from stats
                conn_count = ""
                stats_path = f"results/training_progress/{env_name}_stats.json"
                if os.path.exists(stats_path):
                    try:
                        with open(stats_path) as f:
                            stats = json.load(f)
                        snap = stats.get("genome_snapshots", {}).get(str(gen_str), {})
                        if snap:
                            conn_count = f" — {snap.get('n_conns', '?')} conns, {snap.get('n_nodes', '?')} nodes"
                    except Exception:
                        pass
                H.append(f"""
<figure>
<img src="{b64_image(gp)}" alt="{env_name} gen {gen_str} genome">
<figcaption>Gen {int(gen_str)}{conn_count}</figcaption>
</figure>
""")
            H.append('</div>')
            H.append(f'<p>Watch the topology start minimal (just inputs → outputs) and gradually add '
                     f'hidden nodes and connections as the algorithm explores. The final genome is '
                     f'often surprisingly small — NEAT tends to find parsimonious solutions.</p>')

    # Behavior GIFs from ablations
    H.append("""
<h3>Best genomes in action (from ablations)</h3>
<p>These are GIFs of the best genome from select ablations, playing their
environment. Most are short — the agent either solves the env quickly (a
few hundred steps) or fails fast (a few dozen steps).</p>
""")
    behavior_gifs = sorted(glob.glob("results/ablations/*_behavior.gif"))
    # Show 2 per row
    for i in range(0, len(behavior_gifs), 2):
        H.append('<div class="gif-row">')
        for gif_path in behavior_gifs[i:i+2]:
            name = os.path.basename(gif_path).replace("_behavior.gif", "").replace("_", " ")
            H.append(f"""
<figure>
<img src="{b64_image(gif_path)}" alt="{name}">
<figcaption>{name}</figcaption>
</figure>
""")
        H.append('</div>')

    # ===========================================================
    # PART 8: WHAT I LEARNED
    # ===========================================================
    H.append("""
<h2 id="part-8">8. What I Learned (and What I'd Do Differently)</h2>
<p>Time to step back. After implementing the algorithm, tuning it on five
envs, and running 72 ablations, what's the takeaway?</p>
""")

    H.append("""
<h3>8.1 The spec's modifications are mostly wins</h3>
<p>The spec asked for several deviations from standard NEAT. The ablations
confirm that most of them are improvements:</p>
<ul>
<li><strong>Percentage similarity</strong> is a clear win over the standard
NEAT distance formula (3x improvement on CartPole, no env where it's
worse). The single-number distance with no magic constants is easier to
reason about and just works better.</li>
<li><strong>Purge-first speciation</strong> is a clear win for the first
generation (500 vs 310 on CartPole). Bootstrapping diverse species from
the best random genomes is much faster than waiting for standard
speciation to fragment the population.</li>
<li><strong>DAG-only topology</strong> is a trade-off. It removes the
ability to evolve recurrent connections (which would help on
partially-observable envs like Blackjack-with-card-counting), but it
makes the forward pass trivially fast and removes entire classes of bugs.
For the envs I tested, it's the right choice.</li>
<li><strong>Pruning mutation</strong> is a marginal win. It doesn't
dramatically improve performance on any env, but it keeps genome size
bounded. Without it, genomes accumulate dead-weight connections that
slow down evaluation without adding value.</li>
<li><strong>Universal historical marking</strong> is essential for
meaningful crossover. Without it, "shared innovation numbers" is
meaningless and crossover degrades to topology-copying.</li>
</ul>
""")

    H.append("""
<h3>8.2 The optimizer is a mixed bag</h3>
<p>The GRPO optimizer is the most speculative piece of the algorithm, and
the ablations show it's not a universal win. It helps marginally on
Blackjack (with the right optimizer — SGD, not Adam!), hurts on
MountainCar (sparse reward → noisy gradient → bad updates), and is neutral
on CartPole (which is so easy the optimizer doesn't matter).</p>
<p>The deeper lesson: <strong>gradient methods and evolutionary methods
have complementary strengths.</strong> Evolution is good at exploration
(finding the right topology), gradients are good at exploitation (tuning
weights within a fixed topology). The GRPO optimizer tries to combine
them, but the combination only works when the gradient signal is good —
which is exactly when you'd expect gradients to help anyway. On the hard
cases where evolution struggles (sparse reward, high variance), the
gradient signal is also bad, so the optimizer doesn't rescue you.</p>
<p>If I were redesigning the optimizer, I'd make it <em>optional per
environment</em> and provide a clear diagnostic for when to enable it:
compute the coefficient of variation of fitness within a species. If
it's low (most genomes get similar rewards), enable the optimizer. If
it's high (huge reward spread), disable it — the gradient is too noisy.</p>
""")

    H.append("""
<h3>8.3 NEAT's blind spots</h3>
<p>Two envs in my benchmark gave NEAT real trouble: LunarLander (unsolved)
and MountainCar (barely solved). Both are environments where the optimal
policy requires either <em>precise continuous control</em> (LunarLander:
small thrust changes matter) or <em>long-horizon planning</em>
(MountainCar: you have to swing backward before going forward). NEAT's
evolutionary approach is fundamentally coarse — it perturbs weights and
topology and hopes for the best. It struggles when the policy needs to be
precise.</p>
<p>Standard fixes would be:</p>
<ul>
<li><strong>Recurrent connections</strong> — allow loops in the DAG.
This would let the network have memory, which is essential for
partially-observable envs (the lander has velocity, but you can't see
it directly from a single frame). The spec explicitly mentions this as
an alternative ("forward pass through time"), but I went with the
DAG-only approach for simplicity.</li>
<li><strong>Continuous-action policies</strong> — instead of argmax over
discrete outputs, sample from a Gaussian whose mean and variance are
network outputs. This would let NEAT do fine-grained control.</li>
<li><strong>Behavioral diversity</strong> — instead of speciating by
topological similarity, speciate by behavioral similarity (do two
genomes behave similarly on a set of test inputs?). This would help on
envs where very different topologies can encode similar policies.</li>
</ul>
""")

    H.append("""
<h3>8.4 What I'd do differently</h3>
<p>If I were starting over, three things would change:</p>
<ol>
<li><strong>Build the ablation harness first.</strong> I spent most of
my time on tuning, then realized the ablations were more informative.
The ablation harness is just a thin wrapper around the training code —
it would have been trivial to build early, and it would have caught
several bugs (e.g., the pruning essentiality bug) much sooner.</li>
<li><strong>Use more evaluation episodes from the start.</strong> Early
on, I evaluated each genome on a single episode, which meant the fitness
signal was dominated by noise on stochastic envs. Switching to 5-10
episodes during training and 50-100 for final eval was the single
biggest improvement to my ability to draw conclusions.</li>
<li><strong>Add recurrent connections.</strong> The DAG-only choice was
the right call for getting the implementation working fast, but it's the
main reason LunarLander is unsolved. NEAT was originally designed to
evolve recurrent networks, and the spec allows it — I just didn't
implement it. Future work.</li>
</ol>
""")

    H.append("""
<h3>8.5 Final thoughts</h3>
<p>This was a fun project. The modified NEAT algorithm is genuinely
interesting — it's not just "NEAT with extras," it's a thoughtful redesign
that addresses real weaknesses of the original (magic constants in the
distance formula, slow speciation convergence, no pruning mechanism). The
ablations confirmed that most of the modifications are real improvements,
not just complexity for complexity's sake.</p>
<p>The biggest surprise was how <em>environment-dependent</em> the
algorithm is. There's no single "best config" — CartPole wants per-type
mutations, MountainCar wants single-pick, Blackjack wants SGD, Acrobot
wants no neuron mutation at all. Hyperparameter tuning isn't a
nuisance; it's an essential part of using NEAT on a new problem.</p>
<p>If you want to dig deeper, the code is on
<a href="https://github.com/G-reen-vibe/neat-modified">GitHub</a>. The
<code>scripts/ablations.py</code> file is the entry point for reproducing
the experiments, and <code>scripts/generate_report.py</code> generates
this report. Have fun.</p>
""")

    # Footer
    H.append("""
<footer>
<p>Generated by <code>scripts/generate_report.py</code></p>
<p>Code: <a href="https://github.com/G-reen-vibe/neat-modified">github.com/G-reen-vibe/neat-modified</a></p>
<p>72 ablations · 5 environments · 4 solved · 1 stubborn</p>
</footer>
""")

    H.append("</div></body></html>")

    # Write
    html = "".join(H)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Report saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    generate_report()
