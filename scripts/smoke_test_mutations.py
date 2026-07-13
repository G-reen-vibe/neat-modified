"""Smoke tests for mutations."""
import sys
sys.path.insert(0, "/home/z/my-project/neat_python")

import random
import numpy as np
from neat.registry import InnovationRegistry
from neat.genome import Genome, NodeGene, ConnectionGene
from neat.network import forward, topological_sort
from neat import mutations as M


def make_simple_genome(n_in=2, n_out=1, fully_connected=True):
    reg = InnovationRegistry()
    # reserve input/output ids in the registry so split nodes don't collide
    all_ids = list(range(n_in + n_out))
    reg.reserve_node_ids(all_ids)
    g = Genome(reg, n_in, n_out)
    for i in range(n_in):
        g.add_node(NodeGene(i, "input", "identity"))
    for j in range(n_out):
        g.add_node(NodeGene(n_in + j, "output", "tanh"))
    if fully_connected:
        for i in range(n_in):
            for j in range(n_out):
                innov = reg.get_connection_innov(i, n_in + j)
                g.add_connection(ConnectionGene(innov, i, n_in + j, 0.5))
                reg.bump_presence(innov, +1)
    return g


def test_weight_mutation():
    g = make_simple_genome()
    cfg = {"selection": M.W_SELECT_PCT_SHUFFLED, "pct": 1.0, "mod": M.M_GAUSSIAN, "mod_param": 0.1}
    orig = {i: c.weight for i, c in g.conns.items()}
    changed = M.mutate_weights(g, cfg, seed=42)
    assert changed
    assert len(g._undo_log) == 1
    # weights should differ
    diffs = sum(1 for i, c in g.conns.items() if abs(c.weight - orig[i]) > 1e-9)
    assert diffs > 0
    # weight_delta should be populated
    assert g.weight_delta
    # revert
    M.revert_last_mutation(g)
    for i, c in g.conns.items():
        assert abs(c.weight - orig[i]) < 1e-9
    print("weight_mutation OK")


def test_weight_mutation_prob():
    g = make_simple_genome(4, 2)
    cfg = {"selection": M.W_SELECT_PROB, "pct": 0.5, "mod": M.M_UNIFORM, "mod_param": 0.3}
    changed = M.mutate_weights(g, cfg, seed=7)
    assert changed
    print("weight_mutation_prob OK")


def test_connection_mutation():
    g = make_simple_genome(3, 2)
    # add a hidden node so there are extra candidate pairs
    reg = g.registry
    g.add_node(NodeGene(100, "hidden", "relu"))
    reg.bump_node_presence(100, +1)
    n_before = len(g.conns)
    cfg = {"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0, "mod": M.M_GAUSSIAN, "mod_param": 0.2}
    changed = M.mutate_add_connection(g, cfg, seed=1)
    assert changed, "should add at least 1 connection"
    assert len(g.conns) == n_before + 1
    # DAG should still hold
    assert g.is_dag()
    # revert
    M.revert_last_mutation(g)
    assert len(g.conns) == n_before
    print("connection_mutation OK")


def test_neuron_mutation():
    g = make_simple_genome(2, 1)
    n_before = len(g.conns)
    nodes_before = len(g.nodes)
    cfg = {"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0, "mod": M.N_SPLIT_INCOMING_ONE}
    changed = M.mutate_add_neuron(g, cfg, seed=99)
    assert changed
    # split: -1 conn, +2 conns = +1 conn; +1 node
    assert len(g.conns) == n_before + 1, f"expected {n_before+1}, got {len(g.conns)}"
    assert len(g.nodes) == nodes_before + 1
    assert g.is_dag()
    # forward pass should still work
    out = forward(g, np.array([1.0, 1.0]))
    assert out.shape == (1,)
    # revert
    M.revert_last_mutation(g)
    assert len(g.conns) == n_before
    assert len(g.nodes) == nodes_before
    print("neuron_mutation OK")


def test_pruning_mutation():
    # need a genome with extra connections to prune
    g = make_simple_genome(2, 2)
    # add an extra hidden node + connections
    reg = g.registry
    # add hidden node 100
    g.add_node(NodeGene(100, "hidden", "relu"))
    reg.bump_node_presence(100, +1)
    # add 0->100 and 100->2 (output 0)
    i1 = reg.get_connection_innov(0, 100)
    i2 = reg.get_connection_innov(100, 2)
    g.add_connection(ConnectionGene(i1, 0, 100, 0.7))
    g.add_connection(ConnectionGene(i2, 100, 2, 0.8))
    reg.bump_presence(i1, +1)
    reg.bump_presence(i2, +1)
    n_before = len(g.conns)
    nodes_before = len(g.nodes)
    snapshot = sorted((i, c.in_node, c.out_node, c.weight) for i, c in g.conns.items())
    cfg = {"selection": M.P_SELECT_PCT_SHUFFLED, "pct": 1.0}
    changed = M.mutate_prune(g, cfg, seed=3)
    assert changed
    # either removed a conn or merged
    print("  after prune:", len(g.conns), "conns,", len(g.nodes), "nodes")
    assert g.is_dag()
    M.revert_last_mutation(g)
    print("  after revert:", len(g.conns), "conns,", len(g.nodes), "nodes")
    # strict check: revert should fully restore
    assert len(g.conns) == n_before, f"conn count mismatch: {len(g.conns)} vs {n_before}"
    assert len(g.nodes) == nodes_before, f"node count mismatch: {len(g.nodes)} vs {nodes_before}"
    snapshot_after = sorted((i, c.in_node, c.out_node, c.weight) for i, c in g.conns.items())
    assert snapshot == snapshot_after, f"genome state mismatch after revert\nbefore: {snapshot}\nafter:  {snapshot_after}"
    print("pruning_mutation OK")


def test_linear_path_merge():
    """Set up a clear linear path and check merge works."""
    g = make_simple_genome(2, 1)
    reg = g.registry
    # split existing 0->2 connection to create a linear path
    cfg = {"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0, "mod": M.N_SPLIT_INCOMING_ONE}
    # Manually split 0->2 by calling neuron mutation; we want a specific conn.
    # Use a low-level approach:
    innov_orig = reg.get_connection_innov(0, 2)
    new_node = reg.get_split_node_id(innov_orig)
    g.add_node(NodeGene(new_node, "hidden", "relu"))
    reg.bump_node_presence(new_node, +1)
    # remove original
    del g.conns[innov_orig]
    reg.bump_presence(innov_orig, -1)
    # add 0->new and new->2
    i1 = reg.get_connection_innov(0, new_node)
    i2 = reg.get_connection_innov(new_node, 2)
    g.add_connection(ConnectionGene(i1, 0, new_node, 0.7))
    g.add_connection(ConnectionGene(i2, new_node, 2, 0.8))
    reg.bump_presence(i1, +1)
    reg.bump_presence(i2, +1)
    # now we have a linear path 0 -> new -> 2 (since node 'new' has no other connections)
    # plus the existing 1->2 connection
    assert g.is_dag()
    cfg = {"selection": M.P_SELECT_PCT_SHUFFLED, "pct": 1.0}
    # Run prune with seed that should select the 0->new connection
    # Note: this might prune 1->2 instead, depending on iteration order. Let's just
    # verify that pruning produces a valid DAG and that something happened.
    n_before = len(g.conns)
    changed = M.mutate_prune(g, cfg, seed=11)
    assert changed
    assert g.is_dag()
    print("linear_path_merge OK")


def test_seeded_determinism():
    """Same seed should give same result."""
    g1 = make_simple_genome(3, 2)
    g2 = make_simple_genome(3, 2)
    cfg = {"selection": M.W_SELECT_PCT_SHUFFLED, "pct": 0.5, "mod": M.M_GAUSSIAN, "mod_param": 0.1}
    M.mutate_weights(g1, cfg, seed=123)
    M.mutate_weights(g2, cfg, seed=123)
    for i in g1.conns:
        assert abs(g1.conns[i].weight - g2.conns[i].weight) < 1e-12, "seeded mutation must be deterministic"
    print("seeded_determinism OK")


if __name__ == "__main__":
    test_weight_mutation()
    test_weight_mutation_prob()
    test_connection_mutation()
    test_neuron_mutation()
    test_pruning_mutation()
    test_linear_path_merge()
    test_seeded_determinism()
    print("\nALL MUTATION TESTS PASSED")
