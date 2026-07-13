"""
Visualization tools for NEAT training analysis.

- plot_genome_graph: render a genome as a network diagram PNG
- plot_training_curves: plot best/mean fitness over generations
- plot_species_composition: stacked area chart of species sizes
- plot_weight_distribution: histogram of weights
- plot_topology_growth: avg conns/nodes over generations
- plot_activation_usage: bar chart of activation function usage
- plot_ablation_comparison: compare multiple rounds side-by-side
"""
from __future__ import annotations

import os
import sys
import json
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf')
except Exception:
    pass
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Color palette for species (consistent across plots)
SPECIES_COLORS = [
    "#22d3ee", "#a78bfa", "#fb923c", "#34d399", "#f472b6",
    "#facc15", "#60a5fa", "#fb7185", "#2dd4bf", "#c084fc",
    "#fdba74", "#86efac", "#f9a8d4", "#fde68a", "#93c5fd",
]

ACTIVATION_COLORS = {
    "identity": "#94a3b8",
    "sigmoid": "#22d3ee",
    "tanh": "#a78bfa",
    "relu": "#fb923c",
    "p_swish": "#34d399",
    "uaf": "#f472b6",
}


# -------------------------------------------------------- genome graph ----
def plot_genome_graph(
    genome_dict: Dict,
    output_path: str,
    title: str = "",
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """Render a genome as a network diagram."""
    nodes = genome_dict["nodes"]
    conns = genome_dict["connections"]
    n_in = genome_dict["n_inputs"]
    n_out = genome_dict["n_outputs"]

    inputs = [n for n in nodes if n["kind"] == "input"]
    outputs = [n for n in nodes if n["kind"] == "output"]
    hidden = [n for n in nodes if n["kind"] == "hidden"]
    bias = [n for n in nodes if n["kind"] == "bias"]

    fig, ax = plt.subplots(figsize=figsize, facecolor='#0f172a')
    ax.set_facecolor('#0f172a')

    W = 10.0
    H = figsize[1]
    pad_x = 1.0
    pad_y = 0.6

    def place_column(arr, x):
        if not arr:
            return {}
        if len(arr) == 1:
            return {arr[0]["id"]: (x, H / 2)}
        return {arr[i]["id"]: (x, pad_y + i * (H - 2 * pad_y) / (len(arr) - 1)) for i in range(len(arr))}

    positions = {}
    positions.update(place_column(inputs, pad_x))
    positions.update(place_column(bias, pad_x + 0.3))
    positions.update(place_column(hidden, W / 2))
    positions.update(place_column(outputs, W - pad_x))

    max_abs_w = max([abs(c["weight"]) for c in conns], default=1.0) or 1.0

    # draw edges
    for c in conns:
        if c["in"] not in positions or c["out"] not in positions:
            continue
        x1, y1 = positions[c["in"]]
        x2, y2 = positions[c["out"]]
        color = "#38bdf8" if c["weight"] >= 0 else "#f87171"
        alpha = 0.3 + 0.7 * (abs(c["weight"]) / max_abs_w)
        lw = 0.5 + 3 * (abs(c["weight"]) / max_abs_w)
        ls = "-" if c["enabled"] else "--"
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=lw, linestyle=ls, zorder=1)

    # draw nodes
    for n in nodes:
        if n["id"] not in positions:
            continue
        x, y = positions[n["id"]]
        color = ACTIVATION_COLORS.get(n["activation"], "#cbd5e1")
        edge = "#475569" if n["kind"] == "input" else "#0f766e" if n["kind"] == "output" else "#334155"
        ax.scatter([x], [y], s=300, c=color, edgecolors=edge, linewidths=2, zorder=3)
        ax.text(x, y, str(n["id"]), ha="center", va="center", fontsize=8, fontweight="bold", color="#0f172a", zorder=4)
        ax.text(x, y - 0.35, n["activation"], ha="center", va="top", fontsize=6, color="#94a3b8", zorder=4)

    # column labels
    ax.text(pad_x, H + 0.1, "INPUTS", ha="center", fontsize=8, color="#64748b", fontweight="bold")
    if hidden:
        ax.text(W / 2, H + 0.1, "HIDDEN", ha="center", fontsize=8, color="#64748b", fontweight="bold")
    ax.text(W - pad_x, H + 0.1, "OUTPUTS", ha="center", fontsize=8, color="#64748b", fontweight="bold")

    ax.set_xlim(-0.5, W + 0.5)
    ax.set_ylim(-0.5, H + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color="#e2e8f0", fontsize=11, pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, facecolor='#0f172a', bbox_inches='tight')
    plt.close()


# ----------------------------------------------------- training curves ----
def plot_training_curves(
    histories: List[Dict],
    output_path: str,
    title: str = "Training Curves",
    labels: Optional[List[str]] = None,
) -> None:
    """Plot best/mean fitness over generations for one or more runs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#0f172a')
    for ax in axes:
        ax.set_facecolor('#0f172a')

    for i, h in enumerate(histories):
        gens = [entry["gen"] for entry in h]
        best = [entry["best"] for entry in h]
        mean = [entry["mean"] for entry in h]
        label = labels[i] if labels else f"Run {i+1}"
        axes[0].plot(gens, best, label=label, linewidth=2)
        axes[1].plot(gens, mean, label=label, linewidth=2, linestyle="--")

    axes[0].set_title("Best Fitness", color="#e2e8f0", fontsize=12)
    axes[0].set_xlabel("Generation", color="#94a3b8")
    axes[0].set_ylabel("Best Reward", color="#94a3b8")
    axes[0].grid(True, alpha=0.2, color="#334155")
    axes[0].legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    axes[0].tick_params(colors="#94a3b8")

    axes[1].set_title("Mean Fitness", color="#e2e8f0", fontsize=12)
    axes[1].set_xlabel("Generation", color="#94a3b8")
    axes[1].set_ylabel("Mean Reward", color="#94a3b8")
    axes[1].grid(True, alpha=0.2, color="#334155")
    axes[1].legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    axes[1].tick_params(colors="#94a3b8")

    fig.suptitle(title, color="#e2e8f0", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, facecolor='#0f172a', bbox_inches='tight')
    plt.close()


# ------------------------------------------- species composition -----------
def plot_species_composition(
    stats_history: List[Dict],
    output_path: str,
    title: str = "Species Composition Over Time",
) -> None:
    """Stacked area chart of species sizes."""
    if not stats_history:
        return
    gens = [s["generation"] for s in stats_history]
    # collect all species sizes per gen
    max_species = max(len(s["species_sizes"]) for s in stats_history) if stats_history else 0
    if max_species == 0:
        return
    # build matrix: rows=gen, cols=species_idx
    matrix = np.zeros((len(gens), max_species))
    for i, s in enumerate(stats_history):
        sizes = s["species_sizes"]
        for j in range(min(len(sizes), max_species)):
            matrix[i, j] = sizes[j]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    colors = [SPECIES_COLORS[i % len(SPECIES_COLORS)] for i in range(max_species)]
    ax.stackplot(gens, matrix.T, colors=colors, alpha=0.85)
    ax.set_xlabel("Generation", color="#94a3b8")
    ax.set_ylabel("Genomes per Species", color="#94a3b8")
    ax.set_title(title, color="#e2e8f0", fontsize=12)
    ax.grid(True, alpha=0.2, color="#334155")
    ax.tick_params(colors="#94a3b8")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, facecolor='#0f172a', bbox_inches='tight')
    plt.close()


# ----------------------------------------- weight distribution --------------
def plot_weight_distribution(
    stats_history: List[Dict],
    output_path: str,
    title: str = "Weight Distribution Over Time",
) -> None:
    """Plot weight mean/std range over generations."""
    if not stats_history:
        return
    gens = [s["generation"] for s in stats_history]
    means = [s["weight_mean"] for s in stats_history]
    stds = [s["weight_std"] for s in stats_history]
    mins = [s["weight_min"] for s in stats_history]
    maxs = [s["weight_max"] for s in stats_history]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.fill_between(gens, [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)],
                    alpha=0.3, color="#22d3ee", label="mean ± std")
    ax.plot(gens, means, color="#22d3ee", linewidth=2, label="mean")
    ax.plot(gens, mins, color="#f87171", linewidth=1, alpha=0.5, label="min")
    ax.plot(gens, maxs, color="#34d399", linewidth=1, alpha=0.5, label="max")
    ax.set_xlabel("Generation", color="#94a3b8")
    ax.set_ylabel("Weight Value", color="#94a3b8")
    ax.set_title(title, color="#e2e8f0", fontsize=12)
    ax.grid(True, alpha=0.2, color="#334155")
    ax.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, facecolor='#0f172a', bbox_inches='tight')
    plt.close()


# ----------------------------------- topology growth -----------------------
def plot_topology_growth(
    stats_history: List[Dict],
    output_path: str,
    title: str = "Network Topology Growth",
) -> None:
    """Plot avg conns/nodes and species count over generations."""
    if not stats_history:
        return
    gens = [s["generation"] for s in stats_history]
    avg_conns = [s["avg_conns"] for s in stats_history]
    avg_nodes = [s["avg_nodes"] for s in stats_history]
    max_conns = [s["max_conns"] for s in stats_history]
    min_conns = [s["min_conns"] for s in stats_history]
    n_species = [s["n_species"] for s in stats_history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='#0f172a')
    for ax in axes:
        ax.set_facecolor('#0f172a')

    axes[0].plot(gens, avg_conns, color="#22d3ee", linewidth=2, label="avg conns")
    axes[0].fill_between(gens, min_conns, max_conns, alpha=0.2, color="#22d3ee")
    axes[0].set_title("Connections", color="#e2e8f0")
    axes[0].set_xlabel("Generation", color="#94a3b8")
    axes[0].grid(True, alpha=0.2, color="#334155")
    axes[0].tick_params(colors="#94a3b8")
    axes[0].legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    axes[1].plot(gens, avg_nodes, color="#a78bfa", linewidth=2, label="avg nodes")
    axes[1].set_title("Nodes", color="#e2e8f0")
    axes[1].set_xlabel("Generation", color="#94a3b8")
    axes[1].grid(True, alpha=0.2, color="#334155")
    axes[1].tick_params(colors="#94a3b8")
    axes[1].legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    axes[2].plot(gens, n_species, color="#fb923c", linewidth=2, label="species")
    axes[2].set_title("Species Count", color="#e2e8f0")
    axes[2].set_xlabel("Generation", color="#94a3b8")
    axes[2].grid(True, alpha=0.2, color="#334155")
    axes[2].tick_params(colors="#94a3b8")
    axes[2].legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    fig.suptitle(title, color="#e2e8f0", fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, facecolor='#0f172a', bbox_inches='tight')
    plt.close()


# ----------------------------------- activation usage ----------------------
def plot_activation_usage(
    stats_history: List[Dict],
    output_path: str,
    title: str = "Activation Function Usage",
) -> None:
    """Plot activation function usage over generations."""
    if not stats_history:
        return
    gens = [s["generation"] for s in stats_history]
    all_acts = set()
    for s in stats_history:
        all_acts.update(s["activation_counts"].keys())
    all_acts = sorted(all_acts)

    fig, ax = plt.subplots(figsize=(11, 5), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    for act in all_acts:
        counts = [s["activation_counts"].get(act, 0) for s in stats_history]
        color = ACTIVATION_COLORS.get(act, "#cbd5e1")
        ax.plot(gens, counts, color=color, linewidth=2, label=act)
    ax.set_xlabel("Generation", color="#94a3b8")
    ax.set_ylabel("Node Count", color="#94a3b8")
    ax.set_title(title, color="#e2e8f0", fontsize=12)
    ax.grid(True, alpha=0.2, color="#334155")
    ax.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, facecolor='#0f172a', bbox_inches='tight')
    plt.close()


# ----------------------------------- ablation bar chart --------------------
def plot_ablation_bars(
    results: List[Dict],
    output_path: str,
    title: str = "Ablation Comparison",
    metric: str = "best_fitness",
    metric_label: str = "Best Reward",
) -> None:
    """Bar chart comparing rounds by a metric."""
    fig, ax = plt.subplots(figsize=(11, 5), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    labels = [r.get("label", f"R{r['round_id']}") for r in results]
    values = [r[metric] for r in results]
    colors = [SPECIES_COLORS[i % len(SPECIES_COLORS)] for i in range(len(results))]
    bars = ax.bar(labels, values, color=colors, edgecolor="#334155", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{v:.0f}", ha="center", va="bottom", color="#e2e8f0", fontsize=9)
    ax.set_ylabel(metric_label, color="#94a3b8")
    ax.set_title(title, color="#e2e8f0", fontsize=12)
    ax.grid(True, alpha=0.2, color="#334155", axis="y")
    ax.tick_params(colors="#94a3b8")
    plt.xticks(rotation=20, ha="right", color="#e2e8f0")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, facecolor='#0f172a', bbox_inches='tight')
    plt.close()


# ----------------------------------- combined dashboard --------------------
def plot_dashboard(
    stats_history: List[Dict],
    history: List[Dict],
    output_path: str,
    title: str = "Training Dashboard",
) -> None:
    """Single dashboard image with 6 subplots."""
    if not stats_history:
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor='#0f172a')
    for ax in axes.flat:
        ax.set_facecolor('#0f172a')

    gens = [s["generation"] for s in stats_history]
    best = [h["best"] for h in history]
    mean = [h["mean"] for h in history]
    avg_conns = [s["avg_conns"] for s in stats_history]
    avg_nodes = [s["avg_nodes"] for s in stats_history]
    n_species = [s["n_species"] for s in stats_history]
    w_means = [s["weight_mean"] for s in stats_history]
    w_stds = [s["weight_std"] for s in stats_history]

    # 1. fitness
    axes[0,0].plot(gens, best, color="#34d399", linewidth=2, label="best")
    axes[0,0].plot(gens, mean, color="#fbbf24", linewidth=2, label="mean")
    axes[0,0].set_title("Fitness", color="#e2e8f0")
    axes[0,0].legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    axes[0,0].grid(True, alpha=0.2, color="#334155")
    axes[0,0].tick_params(colors="#94a3b8")

    # 2. species count
    axes[0,1].plot(gens, n_species, color="#fb923c", linewidth=2)
    axes[0,1].set_title("Species Count", color="#e2e8f0")
    axes[0,1].grid(True, alpha=0.2, color="#334155")
    axes[0,1].tick_params(colors="#94a3b8")

    # 3. topology
    axes[0,2].plot(gens, avg_conns, color="#22d3ee", linewidth=2, label="conns")
    axes[0,2].plot(gens, avg_nodes, color="#a78bfa", linewidth=2, label="nodes")
    axes[0,2].set_title("Topology", color="#e2e8f0")
    axes[0,2].legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    axes[0,2].grid(True, alpha=0.2, color="#334155")
    axes[0,2].tick_params(colors="#94a3b8")

    # 4. weight distribution
    axes[1,0].fill_between(gens, [m-s for m,s in zip(w_means,w_stds)],
                            [m+s for m,s in zip(w_means,w_stds)],
                            alpha=0.3, color="#22d3ee")
    axes[1,0].plot(gens, w_means, color="#22d3ee", linewidth=2)
    axes[1,0].set_title("Weight mean ± std", color="#e2e8f0")
    axes[1,0].grid(True, alpha=0.2, color="#334155")
    axes[1,0].tick_params(colors="#94a3b8")

    # 5. species sizes stacked
    max_species = max(len(s["species_sizes"]) for s in stats_history) if stats_history else 0
    if max_species > 0:
        matrix = np.zeros((len(gens), max_species))
        for i, s in enumerate(stats_history):
            sizes = s["species_sizes"]
            for j in range(min(len(sizes), max_species)):
                matrix[i, j] = sizes[j]
        colors = [SPECIES_COLORS[i % len(SPECIES_COLORS)] for i in range(max_species)]
        axes[1,1].stackplot(gens, matrix.T, colors=colors, alpha=0.85)
    axes[1,1].set_title("Species Sizes", color="#e2e8f0")
    axes[1,1].grid(True, alpha=0.2, color="#334155")
    axes[1,1].tick_params(colors="#94a3b8")

    # 6. activation usage
    all_acts = set()
    for s in stats_history:
        all_acts.update(s["activation_counts"].keys())
    all_acts = sorted(all_acts)
    for act in all_acts:
        counts = [s["activation_counts"].get(act, 0) for s in stats_history]
        color = ACTIVATION_COLORS.get(act, "#cbd5e1")
        axes[1,2].plot(gens, counts, color=color, linewidth=2, label=act)
    axes[1,2].set_title("Activations", color="#e2e8f0")
    axes[1,2].legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    axes[1,2].grid(True, alpha=0.2, color="#334155")
    axes[1,2].tick_params(colors="#94a3b8")

    fig.suptitle(title, color="#e2e8f0", fontsize=14, y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, facecolor='#0f172a', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # smoke test with a tiny run
    from benchmarks.instrument import run_instrumented_round
    res = run_instrumented_round(
        "MountainCar-v0", 1, {"pop_size": 20, "seed": 0},
        n_gens=10, verbose=False,
        save_genomes_dir="/home/z/my-project/download/genomes",
    )
    out_dir = "/home/z/my-project/download/plots/test"
    plot_dashboard(res["stats"]["history"], res["history"], os.path.join(out_dir, "dashboard.png"))
    # plot first genome snapshot
    if res["stats"]["best_genome_snapshots"]:
        plot_genome_graph(res["stats"]["best_genome_snapshots"][0],
                           os.path.join(out_dir, "genome.png"),
                           title="Best Genome (gen 0)")
    print(f"Smoke test OK. Plots in {out_dir}")
