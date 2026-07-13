"""
Network similarity tests.

Standard - the NEAT paper formula:
  delta = (c1 * E + c2 * D) / N + c3 * W_avg_diff
  where E = excess, D = disjoint, N = max(|conns_a|, |conns_b|), W_avg_diff
  is the average weight difference over matching innovations.

Percentage - a percentage-based difference:
  treats missing connections as weight 0
  total   = sum(|w_a| + |w_b|) over all innovations seen in either
  diff    = sum(|w_a - w_b|) over all innovations seen in either (0 if missing in one)
  pct     = diff / total
  Smaller pct = more similar. We return *similarity* = 1 - pct, and the
  *distance* = pct.
"""
from __future__ import annotations

from typing import Dict

from .genome import Genome


def similarity_standard(g_a: Genome, g_b: Genome,
                        c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
    """Return the standard NEAT distance (lower = more similar)."""
    innovs_a = set(g_a.conns.keys())
    innovs_b = set(g_b.conns.keys())
    shared = innovs_a & innovs_b
    only_a = innovs_a - innovs_b
    only_b = innovs_b - innovs_a

    if not shared and not only_a and not only_b:
        return 0.0

    # disjoint vs excess: find max innov in each, anything beyond the other's max is "excess"
    max_a = max(innovs_a) if innovs_a else -1
    max_b = max(innovs_b) if innovs_b else -1
    max_shared = min(max_a, max_b)
    disjoint = sum(1 for i in (only_a | only_b) if i <= max_shared)
    excess = sum(1 for i in (only_a | only_b) if i > max_shared)

    n = max(len(innovs_a), len(innovs_b))
    n = max(n, 1)
    w_diff = 0.0
    if shared:
        w_diff = sum(abs(g_a.conns[i].weight - g_b.conns[i].weight) for i in shared) / len(shared)

    return (c1 * excess + c2 * disjoint) / n + c3 * w_diff


def similarity_percentage(g_a: Genome, g_b: Genome) -> float:
    """Return the percentage *distance* (lower = more similar)."""
    innovs_a = set(g_a.conns.keys())
    innovs_b = set(g_b.conns.keys())
    all_innovs = innovs_a | innovs_b
    if not all_innovs:
        return 0.0
    total = 0.0
    diff = 0.0
    for i in all_innovs:
        w_a = g_a.conns[i].weight if i in g_a.conns else 0.0
        w_b = g_b.conns[i].weight if i in g_b.conns else 0.0
        total += abs(w_a) + abs(w_b)
        diff += abs(w_a - w_b)
    if total < 1e-12:
        # both genomes have all-zero weights; similarity is determined by topology
        return 0.0 if innovs_a == innovs_b else 1.0
    return diff / total


def similarity(g_a: Genome, g_b: Genome, method: str = "percentage") -> float:
    """Dispatch helper. ``method`` is 'standard' or 'percentage'."""
    if method == "standard":
        return similarity_standard(g_a, g_b)
    if method == "percentage":
        return similarity_percentage(g_a, g_b)
    raise ValueError(f"unknown similarity method {method}")
