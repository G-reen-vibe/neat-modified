"""
Analysis hooks for NEAT training.

Provides:
    - TrainingStats: collects per-generation statistics during training
      (fitness, genome size, species count, mutation effectiveness, etc.)
    - capture_genome_snapshot: save a genome's structure to JSON
    - capture_agent_video: record frames of an agent's behavior
    - make_gif: convert frames to an animated GIF

Usage:
    stats = TrainingStats()
    # Inside the training loop:
    stats.record_generation(pop, gen_idx, eval_fn)
    # After training:
    stats.save('results/stats.json')
    stats.plot_fitness('results/fitness.png')
"""
from __future__ import annotations
import os
import json
import time
import base64
import io
from typing import List, Dict, Any, Optional, Callable
import numpy as np

# Lazy imports for plotting (only when needed)
def _import_plt():
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
        fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    except Exception:
        pass
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


# ---------------------------------------------------------------------------
# TrainingStats: collects per-generation statistics
# ---------------------------------------------------------------------------
class TrainingStats:
    """Collects detailed per-generation statistics during training.

    Records:
        - Fitness: max, mean, min, std per generation
        - Genome size: avg #nodes, #conns, max #nodes, max #conns
        - Species: count, sizes, fitness per species
        - Mutation effectiveness (optional, requires delta tracking)
        - Best genome snapshot every N generations
    """

    def __init__(self, snapshot_every: int = 5):
        self.snapshot_every = snapshot_every
        self.history: List[Dict[str, Any]] = []
        self.genome_snapshots: Dict[int, Dict] = {}  # gen_idx -> genome dict
        self.start_time = time.time()

    def record_generation(self, pop, gen_idx: int, eval_fn: Optional[Callable] = None) -> Dict:
        """Record statistics for the current generation.

        Assumes pop.evaluate(eval_fn) has already been called.
        """
        genomes = pop.genomes
        fits = np.array([g.fitness for g in genomes])
        n_nodes = np.array([len(g.nodes) for g in genomes])
        n_conns = np.array([len(g.conns) for g in genomes])

        # Species info
        species_info = []
        for sid, s in pop.speciator.species.items():
            members = [g for g in genomes if g.id in s.members]
            if members:
                species_info.append({
                    "id": sid,
                    "n_members": len(members),
                    "best_fitness": float(max(g.fitness for g in members)),
                    "mean_fitness": float(np.mean([g.fitness for g in members])),
                    "age": pop.generation - s.last_improved,
                })

        stats = {
            "generation": gen_idx,
            "elapsed_s": time.time() - self.start_time,
            "fitness": {
                "max": float(fits.max()) if len(fits) else 0.0,
                "mean": float(fits.mean()) if len(fits) else 0.0,
                "min": float(fits.min()) if len(fits) else 0.0,
                "std": float(fits.std()) if len(fits) else 0.0,
            },
            "genome_size": {
                "avg_nodes": float(n_nodes.mean()) if len(n_nodes) else 0.0,
                "avg_conns": float(n_conns.mean()) if len(n_conns) else 0.0,
                "max_nodes": int(n_nodes.max()) if len(n_nodes) else 0,
                "max_conns": int(n_conns.max()) if len(n_conns) else 0,
                "min_nodes": int(n_nodes.min()) if len(n_nodes) else 0,
                "min_conns": int(n_conns.min()) if len(n_conns) else 0,
            },
            "species": {
                "count": len(pop.speciator.species),
                "details": species_info,
                "threshold": float(pop.speciator.threshold),
            },
            "index": pop.index.snapshot(),
        }
        self.history.append(stats)

        # Snapshot best genome
        if gen_idx % self.snapshot_every == 0:
            best = pop.best()
            if best is not None:
                self.genome_snapshots[gen_idx] = self._genome_to_dict(best)
        return stats

    def _genome_to_dict(self, g) -> Dict:
        """Convert a genome to a JSON-serializable dict (for snapshots)."""
        return {
            "id": g.id,
            "fitness": float(g.fitness),
            "n_nodes": len(g.nodes),
            "n_conns": len(g.conns),
            "nodes": [
                {"id": n.node_id, "kind": n.kind, "activation": n.activation.kind}
                for n in g.nodes.values()
            ],
            "conns": [
                {"innov": c.innov, "src": c.src, "dst": c.dst, "weight": float(c.weight)}
                for c in g.conns.values()
            ],
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "history": self.history,
                "genome_snapshots": self.genome_snapshots,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingStats":
        with open(path) as f:
            data = json.load(f)
        ts = cls()
        ts.history = data["history"]
        ts.genome_snapshots = {int(k): v for k, v in data["genome_snapshots"].items()}
        return ts

    # ------------------------------------------------------------------
    # Plotting methods
    # ------------------------------------------------------------------
    def plot_fitness(self, path: str, title: str = "Fitness Over Generations") -> str:
        plt = _import_plt()
        gens = [h["generation"] for h in self.history]
        maxes = [h["fitness"]["max"] for h in self.history]
        means = [h["fitness"]["mean"] for h in self.history]
        mins = [h["fitness"]["min"] for h in self.history]
        stds = [h["fitness"]["std"] for h in self.history]

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.fill_between(gens,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color="#4cc9f0", label="±1 std")
        ax.plot(gens, maxes, "-o", color="#4cc9f0", markersize=3, label="Max")
        ax.plot(gens, means, "-", color="#4ade80", linewidth=2, label="Mean")
        ax.plot(gens, mins, "-", color="#f72585", alpha=0.5, label="Min")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    def plot_genome_size(self, path: str, title: str = "Genome Size Over Generations") -> str:
        plt = _import_plt()
        gens = [h["generation"] for h in self.history]
        avg_nodes = [h["genome_size"]["avg_nodes"] for h in self.history]
        avg_conns = [h["genome_size"]["avg_conns"] for h in self.history]
        max_nodes = [h["genome_size"]["max_nodes"] for h in self.history]
        max_conns = [h["genome_size"]["max_conns"] for h in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        ax1.plot(gens, avg_nodes, "-o", color="#4cc9f0", markersize=3, label="Avg nodes")
        ax1.plot(gens, max_nodes, "--", color="#4cc9f0", alpha=0.5, label="Max nodes")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Nodes")
        ax1.set_title("Nodes")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(gens, avg_conns, "-o", color="#f72585", markersize=3, label="Avg conns")
        ax2.plot(gens, max_conns, "--", color="#f72585", alpha=0.5, label="Max conns")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Connections")
        ax2.set_title("Connections")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    def plot_species(self, path: str, title: str = "Species Count Over Generations") -> str:
        plt = _import_plt()
        gens = [h["generation"] for h in self.history]
        n_species = [h["species"]["count"] for h in self.history]
        thresholds = [h["species"]["threshold"] for h in self.history]

        fig, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax1.bar(gens, n_species, color="#4cc9f0", alpha=0.7, label="# species")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Species count", color="#4cc9f0")
        ax1.tick_params(axis='y', labelcolor="#4cc9f0")
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = ax1.twinx()
        ax2.plot(gens, thresholds, "-", color="#f72585", linewidth=2, label="threshold")
        ax2.set_ylabel("Similarity threshold", color="#f72585")
        ax2.tick_params(axis='y', labelcolor="#f72585")

        fig.suptitle(title)
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    def plot_species_distribution(self, path: str, title: str = "Species Distribution") -> str:
        """Stacked area chart of species sizes over generations."""
        plt = _import_plt()
        # Get all species ids ever seen
        all_sids = set()
        for h in self.history:
            for s in h["species"]["details"]:
                all_sids.add(s["id"])
        all_sids = sorted(all_sids)
        if not all_sids:
            # Make a placeholder image
            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
            ax.text(0.5, 0.5, "No species data", ha="center", va="center")
            fig.savefig(path, dpi=100)
            plt.close(fig)
            return path

        gens = [h["generation"] for h in self.history]
        # Build matrix: rows = gens, cols = sids
        mat = np.zeros((len(gens), len(all_sids)))
        for i, h in enumerate(self.history):
            sid_to_n = {s["id"]: s["n_members"] for s in h["species"]["details"]}
            for j, sid in enumerate(all_sids):
                mat[i, j] = sid_to_n.get(sid, 0)

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        # Stacked area
        ax.stackplot(gens, mat.T, labels=[f"Sp {s}" for s in all_sids], alpha=0.7)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Members")
        ax.set_title(title)
        # Only show legend if not too many species
        if len(all_sids) <= 12:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path


# ---------------------------------------------------------------------------
# Genome graph visualization
# ---------------------------------------------------------------------------
def visualize_genome(g, path: str, title: str = "Genome Topology") -> str:
    """Render a genome's network topology as a PNG.

    Nodes are colored by kind (input/output/bias/hidden) and positioned
    by kind (inputs left, outputs right, hidden in middle).
    Edges are colored by weight sign (green=positive, red=negative) and
    width scales with |weight|.
    """
    plt = _import_plt()
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=20)

    # Categorize nodes
    input_nodes = [n for n in g.nodes.values() if n.kind == "input"]
    output_nodes = [n for n in g.nodes.values() if n.kind == "output"]
    bias_nodes = [n for n in g.nodes.values() if n.kind == "bias"]
    hidden_nodes = [n for n in g.nodes.values() if n.kind == "hidden"]

    # Position nodes
    pos = {}
    # Inputs on the left, evenly spaced vertically
    n_in = max(len(input_nodes), 1)
    for i, n in enumerate(input_nodes):
        pos[n.node_id] = (-1.0, 0.8 - i * 1.6 / max(n_in - 1, 1))
    # Bias next to inputs
    for i, n in enumerate(bias_nodes):
        pos[n.node_id] = (-1.0, -1.0)
    # Outputs on the right
    n_out = max(len(output_nodes), 1)
    for i, n in enumerate(output_nodes):
        pos[n.node_id] = (1.0, 0.8 - i * 1.6 / max(n_out - 1, 1))
    # Hidden nodes in the middle, arranged in a grid
    n_hid = len(hidden_nodes)
    if n_hid > 0:
        # Try to arrange in columns based on topological depth
        # For simplicity, use a circular/grid arrangement
        cols = int(np.ceil(np.sqrt(n_hid)))
        rows = int(np.ceil(n_hid / cols))
        for i, n in enumerate(hidden_nodes):
            r = i // cols
            c = i % cols
            x = -0.5 + c * 1.0 / max(cols - 1, 1)
            y = 0.8 - r * 1.6 / max(rows, 1)
            pos[n.node_id] = (x, y)

    # Draw edges
    for c in g.conns.values():
        if c.src not in pos or c.dst not in pos:
            continue
        x1, y1 = pos[c.src]
        x2, y2 = pos[c.dst]
        color = "#2ecc71" if c.weight > 0 else "#e74c3c"
        lw = min(3.0, abs(c.weight) * 1.5 + 0.2)
        alpha = min(0.9, abs(c.weight) * 0.5 + 0.2)
        ax.plot([x1, x2], [y1, y2], "-", color=color, linewidth=lw, alpha=alpha)

    # Draw nodes
    colors = {"input": "#4cc9f0", "output": "#f72585", "bias": "#ffd60a", "hidden": "#4ade80"}
    for n in g.nodes.values():
        if n.node_id not in pos:
            continue
        x, y = pos[n.node_id]
        ax.scatter(x, y, s=500, c=colors[n.kind], edgecolors="black", linewidths=1.5, zorder=5)
        ax.text(x, y, str(n.node_id), ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#4cc9f0", label="Input"),
        mpatches.Patch(color="#f72585", label="Output"),
        mpatches.Patch(color="#ffd60a", label="Bias"),
        mpatches.Patch(color="#4ade80", label="Hidden"),
        mpatches.Patch(color="#2ecc71", label="Weight > 0"),
        mpatches.Patch(color="#e74c3c", label="Weight < 0"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    # Add fitness info
    ax.text(0.0, -1.15, f"Fitness: {g.fitness:.2f}  |  Nodes: {len(g.nodes)}  |  Conns: {len(g.conns)}",
            ha="center", va="top", fontsize=10, style="italic")

    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Agent behavior capture (video / GIF)
# ---------------------------------------------------------------------------
def capture_agent_behavior(g, env_name: str, max_steps: int = 500,
                            seed: int = 0, n_episodes: int = 1,
                            output_dir: str = "results/agent_frames",
                            tag: str = "") -> Dict[str, Any]:
    """Run the genome in the env and save frames as PNG + return metadata.

    Returns dict with:
        - frames: list of PNG paths
        - rewards: per-step rewards
        - total_reward: final reward
        - n_steps: episode length
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from neat.envs import make_env

    os.makedirs(output_dir, exist_ok=True)
    env = make_env(env_name, max_steps=max_steps, n_eval_episodes=1,
                   seed=seed, render_mode="rgb_array")
    result = env.rollout(g, episode_seed=seed, render=True)
    env.close()

    from PIL import Image
    frame_paths = []
    prefix = f"{tag}_ep{seed}" if tag else f"ep{seed}"
    for i, frame in enumerate(result["frames"]):
        img = Image.fromarray(frame)
        path = os.path.join(output_dir, f"{prefix}_frame_{i:04d}.png")
        img.save(path)
        frame_paths.append(path)

    return {
        "frames": frame_paths,
        "rewards": result["rewards"],
        "total_reward": result["total_reward"],
        "n_steps": result["steps"],
        "actions": result["actions"],
    }


def make_gif(frame_paths: List[str], output_path: str, fps: int = 20,
             duration: float = 0.05) -> str:
    """Combine a list of PNG paths into an animated GIF."""
    from PIL import Image
    if not frame_paths:
        return output_path
    images = [Image.open(p) for p in frame_paths]
    # Resize if large
    max_dim = 480
    if images[0].size[0] > max_dim or images[0].size[1] > max_dim:
        scale = max_dim / max(images[0].size)
        new_size = (int(images[0].size[0] * scale), int(images[0].size[1] * scale))
        images = [im.resize(new_size, Image.LANCZOS) for im in images]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=duration, loop=0, optimize=True)
    return output_path


def capture_training_trajectory(g, env_name: str, output_path: str,
                                 max_steps: int = 500, seed: int = 0) -> str:
    """Capture a single rollout as a GIF."""
    res = capture_agent_behavior(g, env_name, max_steps=max_steps, seed=seed,
                                  output_dir=os.path.dirname(output_path) or ".",
                                  tag=f"traj_{seed}")
    make_gif(res["frames"], output_path, duration=0.05)
    # Clean up individual frames
    for p in res["frames"]:
        try:
            os.remove(p)
        except Exception:
            pass
    return output_path


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from neat import Config, Population, Genome, GlobalIndex

    # Build a simple genome
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    idx = GlobalIndex(4, 2, bias_enabled=True)
    g = Genome(cfg, idx)
    g.add_conn(0, 4, 0.5)
    g.add_conn(1, 4, -0.3)
    g.add_conn(2, 5, 0.7)
    g.add_conn(3, 5, -0.2)
    g.add_conn(6, 4, 0.1)  # bias
    g.add_conn(6, 5, 0.1)
    # Add a hidden node
    h = g.add_hidden_node()
    g.add_conn(0, h.node_id, 0.4)
    g.add_conn(h.node_id, 4, 0.6)

    os.makedirs("results/test", exist_ok=True)
    visualize_genome(g, "results/test/genome_test.png", title="Test Genome")
    print("Saved genome visualization to results/test/genome_test.png")
