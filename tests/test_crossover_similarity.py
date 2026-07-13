"""Unit tests for crossover and similarity."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neat.config import (
    Config, CrossoverCfg, CrossoverTopology, CrossoverWeights,
    SimilarityKind, SpeciationKind,
)
from neat.indexing import GlobalIndex
from neat.genome import Genome
from neat.crossover import crossover, _cycle_break_combine
from neat.similarity import similarity, similarity_percentage, similarity_standard


def make_pair():
    cfg = Config(n_inputs=2, n_outputs=1, bias_enabled=True)
    idx = GlobalIndex(2, 1, bias_enabled=True)
    g1 = Genome(cfg, idx); g1.fitness = 1.0
    g2 = Genome(cfg, idx); g2.fitness = 0.5
    return cfg, idx, g1, g2


def test_crossover_fitter_topology():
    cfg, idx, g1, g2 = make_pair()
    g1.add_conn(0, 2, 0.5)
    g1.add_conn(1, 2, 0.5)
    g2.add_conn(0, 2, 0.5)
    cfg.crossover.topology = CrossoverTopology.FITTER
    cfg.crossover.weights = CrossoverWeights.AVERAGE
    rng = np.random.default_rng(0)
    child = crossover(g1, g2, cfg, rng)
    # Child has only g1's topology (fitter) => 2 conns
    assert len(child.conns) == 2
    # Weights should be averaged
    for iv, c in child.conns.items():
        if iv in g1.conns and iv in g2.conns:
            assert abs(c.weight - 0.5*(g1.conns[iv].weight + g2.conns[iv].weight)) < 1e-9
        else:
            assert abs(c.weight - g1.conns[iv].weight) < 1e-9


def test_crossover_more_conns_topology():
    cfg, idx, g1, g2 = make_pair()
    g1.add_conn(0, 2, 0.5)
    g2.add_conn(0, 2, 0.5)
    g2.add_conn(1, 2, 0.5)  # g2 has more
    cfg.crossover.topology = CrossoverTopology.MORE_CONNS
    cfg.crossover.weights = CrossoverWeights.AVERAGE
    rng = np.random.default_rng(0)
    child = crossover(g1, g2, cfg, rng)
    assert len(child.conns) == 2  # uses g2's topology


def test_crossover_combine_topology():
    cfg, idx, g1, g2 = make_pair()
    g1.add_conn(0, 2, 0.5)
    g2.add_conn(1, 2, 0.5)  # different connection
    cfg.crossover.topology = CrossoverTopology.COMBINE
    cfg.crossover.weights = CrossoverWeights.AVERAGE
    rng = np.random.default_rng(0)
    child = crossover(g1, g2, cfg, rng)
    # Child should have both connections
    assert len(child.conns) == 2


def test_crossover_independent_weights():
    cfg, idx, g1, g2 = make_pair()
    g1.add_conn(0, 2, 1.0)
    g2.add_conn(0, 2, -1.0)
    cfg.crossover.topology = CrossoverTopology.COMBINE
    cfg.crossover.weights = CrossoverWeights.INDEPENDENT
    rng = np.random.default_rng(0)
    child = crossover(g1, g2, cfg, rng)
    assert len(child.conns) == 1
    w = list(child.conns.values())[0].weight
    assert w in (1.0, -1.0)


def test_crossover_combine_avoids_cycles():
    """COMBINE must not produce a cyclic child."""
    cfg, idx, g1, g2 = make_pair()
    # g1: 0 -> hidden1 -> 2
    h1 = g1.add_hidden_node().node_id
    g1.add_conn(0, h1, 0.5)
    g1.add_conn(h1, 2, 0.5)
    # g2: 0 -> hidden2 -> 2
    h2 = g2.add_hidden_node().node_id
    g2.add_conn(0, h2, 0.5)
    g2.add_conn(h2, 2, 0.5)
    cfg.crossover.topology = CrossoverTopology.COMBINE
    cfg.crossover.weights = CrossoverWeights.AVERAGE
    rng = np.random.default_rng(0)
    child = crossover(g1, g2, cfg, rng)
    # Forward pass must succeed (no cycle / infinite loop)
    out = child.forward(np.array([1.0, 1.0]))
    assert out.shape == (1,)


def test_similarity_percentage_identical():
    cfg, idx, g1, g2 = make_pair()
    g1.add_conn(0, 2, 0.5)
    g2.add_conn(0, 2, 0.5)
    d = similarity_percentage(g1, g2, cfg)
    assert d < 1e-9, f"identical genomes should have 0 distance, got {d}"


def test_similarity_percentage_disjoint():
    cfg, idx, g1, g2 = make_pair()
    g1.add_conn(0, 2, 1.0)
    g2.add_conn(1, 2, 1.0)  # different innov
    d = similarity_percentage(g1, g2, cfg)
    # total = 1 + 1 = 2, diff = 1 + 1 = 2, pct = 1.0
    assert abs(d - 1.0) < 1e-9, f"got {d}"


def test_similarity_standard_basic():
    cfg, idx, g1, g2 = make_pair()
    cfg.speciation.similarity = SimilarityKind.STANDARD
    g1.add_conn(0, 2, 0.5)
    g2.add_conn(0, 2, 0.5)  # shared
    g2.add_conn(1, 2, 0.5)  # disjoint
    d = similarity_standard(g1, g2, cfg)
    # E=0, D=1, N=2, W=0
    expected = (cfg.speciation.c1 * 1 + cfg.speciation.c2 * 1) / 2 + cfg.speciation.c3 * 0
    assert abs(d - expected) < 1e-9


def test_similarity_zero_genomes():
    cfg, idx, g1, g2 = make_pair()
    d1 = similarity_standard(g1, g2, cfg)
    d2 = similarity_percentage(g1, g2, cfg)
    assert d1 == 0.0
    assert d2 == 0.0


if __name__ == "__main__":
    test_crossover_fitter_topology()
    test_crossover_more_conns_topology()
    test_crossover_combine_topology()
    test_crossover_independent_weights()
    test_crossover_combine_avoids_cycles()
    test_similarity_percentage_identical()
    test_similarity_percentage_disjoint()
    test_similarity_standard_basic()
    test_similarity_zero_genomes()
    print("All crossover/similarity tests passed.")
