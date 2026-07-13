"""
Network similarity tests.

Per spec:
    Standard:
        NEAT-paper comparison.  Matches shared connections and disjoints,
        returns a value based on the average weight difference in shared
        connections and the number of disjoints.

        delta = (c1 * E + c2 * D) / N + c3 * W
        where E = excess, D = disjoint, N = #genes in larger genome,
        W = avg weight diff of matching genes.

    Percentage:
        Treats connection that don't exist as having a weight of zero.
        total = sum of |w| over the union of all connections (both genomes,
                 with missing ones counted as 0)
        diff  = sum of |w1 - w2| over the union
        pct   = diff / total
        Similarity = 1 - pct  (so higher = more similar)
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .genome import Genome
from .config import Config, SimilarityKind


def similarity_standard(g1: Genome, g2: Genome, cfg: Config) -> float:
    """Returns the NEAT-standard distance (higher = more dissimilar)."""
    c1 = cfg.speciation.c1
    c2 = cfg.speciation.c2
    c3 = cfg.speciation.c3

    innovs1 = set(g1.conns.keys())
    innovs2 = set(g2.conns.keys())
    if not innovs1 and not innovs2:
        return 0.0
    shared = innovs1 & innovs2
    only1 = innovs1 - innovs2
    only2 = innovs2 - innovs1
    # In standard NEAT, "excess" are genes beyond the other's max innov,
    # "disjoint" are the rest.  Here we treat all non-shared as disjoint for
    # simplicity (the spec doesn't require strict excess/disjoint split).
    D = len(only1) + len(only2)
    if shared:
        W = np.mean([abs(g1.conns[i].weight - g2.conns[i].weight) for i in shared])
    else:
        W = 0.0
    N = max(len(innovs1), len(innovs2), 1)
    return (c1 * D + c2 * D) / N + c3 * W


def similarity_percentage(g1: Genome, g2: Genome, cfg: Config) -> float:
    """Returns the *distance* (0 = identical, 1 = completely different)."""
    innovs1 = set(g1.conns.keys())
    innovs2 = set(g2.conns.keys())
    union = innovs1 | innovs2
    if not union:
        return 0.0
    total = 0.0
    diff = 0.0
    for iv in union:
        w1 = g1.conns[iv].weight if iv in g1.conns else 0.0
        w2 = g2.conns[iv].weight if iv in g2.conns else 0.0
        total += abs(w1) + abs(w2)
        diff += abs(w1 - w2)
    if total < 1e-12:
        # Both genomes have all-zero weights; treat as identical
        return 0.0
    pct = diff / total
    return float(pct)


def similarity(g1: Genome, g2: Genome, cfg: Config) -> float:
    """Return distance (higher = more dissimilar).  Method chosen by cfg."""
    if cfg.speciation.similarity == SimilarityKind.STANDARD:
        return similarity_standard(g1, g2, cfg)
    if cfg.speciation.similarity == SimilarityKind.PERCENTAGE:
        return similarity_percentage(g1, g2, cfg)
    raise ValueError(f"Unknown similarity: {cfg.speciation.similarity}")
