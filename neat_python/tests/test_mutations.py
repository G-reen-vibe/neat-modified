"""Tests for mutation operators: weight, connection, neuron, pruning."""
import pytest
import random
import numpy as np

from neat.registry import InnovationRegistry
from neat.genome import Genome, NodeGene, ConnectionGene
from neat.network import forward
from neat import mutations as M


def make_genome(n_in=2, n_out=1, fully_connected=True):
    reg = InnovationRegistry()
    reg.reserve_node_ids(list(range(n_in + n_out)))
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


class TestWeightMutation:
    def test_gaussian_modification(self):
        g = make_genome(3, 2)
        cfg = {"selection": M.W_SELECT_PCT_SHUFFLED, "pct": 1.0,
               "mod": M.M_GAUSSIAN, "mod_param": 0.1}
        orig = {i: c.weight for i, c in g.conns.items()}
        M.mutate_weights(g, cfg, seed=42)
        diffs = sum(1 for i, c in g.conns.items() if abs(c.weight - orig[i]) > 1e-9)
        assert diffs > 0
        assert g.weight_delta  # should be populated

    def test_uniform_modification(self):
        g = make_genome(3, 2)
        cfg = {"selection": M.W_SELECT_PROB, "pct": 0.5,
               "mod": M.M_UNIFORM, "mod_param": 0.3}
        M.mutate_weights(g, cfg, seed=7)
        # all weights should be within +/- 0.3 of 0.5
        for c in g.conns.values():
            assert 0.2 <= c.weight <= 0.8

    def test_bernoulli_modification(self):
        g = make_genome(2, 1)
        cfg = {"selection": M.W_SELECT_PCT_SHUFFLED, "pct": 1.0,
               "mod": M.M_BERNOULLI, "mod_param": 0.1}
        M.mutate_weights(g, cfg, seed=3)
        # all weights should be 0.5 +/- 0.1
        for c in g.conns.values():
            assert c.weight in (0.4, 0.6)

    def test_revert(self):
        g = make_genome(3, 2)
        cfg = {"selection": M.W_SELECT_PCT_SHUFFLED, "pct": 1.0,
               "mod": M.M_GAUSSIAN, "mod_param": 0.1}
        snapshot = {i: c.weight for i, c in g.conns.items()}
        M.mutate_weights(g, cfg, seed=42)
        M.revert_last_mutation(g)
        for i, c in g.conns.items():
            assert abs(c.weight - snapshot[i]) < 1e-12

    def test_seed_determinism(self):
        g1 = make_genome(3, 2)
        g2 = make_genome(3, 2)
        cfg = {"selection": M.W_SELECT_PROB, "pct": 0.5,
               "mod": M.M_GAUSSIAN, "mod_param": 0.1}
        M.mutate_weights(g1, cfg, seed=123)
        M.mutate_weights(g2, cfg, seed=123)
        for i in g1.conns:
            assert abs(g1.conns[i].weight - g2.conns[i].weight) < 1e-12

    def test_at_least_one_selected(self):
        """Even with pct=0, at least one weight must be selected."""
        g = make_genome(2, 1)
        cfg = {"selection": M.W_SELECT_PCT_SHUFFLED, "pct": 0.0,
               "mod": M.M_GAUSSIAN, "mod_param": 0.1}
        M.mutate_weights(g, cfg, seed=42)
        # at least one weight must have changed
        assert any(abs(c.weight - 0.5) > 1e-9 for c in g.conns.values())


class TestConnectionMutation:
    def test_adds_connection(self):
        g = make_genome(2, 2)
        # add hidden node for candidate pairs
        reg = g.registry
        g.add_node(NodeGene(100, "hidden", "relu"))
        reg.bump_node_presence(100, +1)
        n_before = len(g.conns)
        cfg = {"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
               "mod": M.M_GAUSSIAN, "mod_param": 0.2}
        ok = M.mutate_add_connection(g, cfg, seed=1)
        assert ok
        assert len(g.conns) == n_before + 1
        assert g.is_dag()

    def test_dag_preserved(self):
        g = make_genome(3, 2)
        reg = g.registry
        g.add_node(NodeGene(100, "hidden", "relu"))
        reg.bump_node_presence(100, +1)
        for _ in range(5):
            M.mutate_add_connection(g, {"selection": M.C_SELECT_PROB,
                                         "pct": 0.3, "mod": M.M_GAUSSIAN,
                                         "mod_param": 0.2}, seed=random.randint(0, 1000))
            assert g.is_dag(), "DAG must be preserved"

    def test_revert(self):
        g = make_genome(2, 2)
        reg = g.registry
        g.add_node(NodeGene(100, "hidden", "relu"))
        reg.bump_node_presence(100, +1)
        n_before = len(g.conns)
        M.mutate_add_connection(g, {"selection": M.C_SELECT_PCT_SHUFFLED,
                                     "pct": 0.0, "mod": M.M_GAUSSIAN,
                                     "mod_param": 0.2}, seed=1)
        M.revert_last_mutation(g)
        assert len(g.conns) == n_before

    def test_least_common_global(self):
        g = make_genome(3, 2)
        reg = g.registry
        g.add_node(NodeGene(100, "hidden", "relu"))
        reg.bump_node_presence(100, +1)
        # bump presence on some innovs to make them "common"
        candidates = M._safe_candidate_pairs(g)
        if len(candidates) >= 2:
            # bump first candidate's presence
            innov = reg.get_connection_innov(*candidates[0])
            for _ in range(10):
                reg.bump_presence(innov, +1)
            cfg = {"selection": M.C_SELECT_LEAST_COMMON_GLOBAL, "pct": 0.5,
                   "mod": M.M_GAUSSIAN, "mod_param": 0.2}
            M.mutate_add_connection(g, cfg, seed=1)
            assert g.is_dag()


class TestNeuronMutation:
    def test_split_incoming_one(self):
        g = make_genome(2, 1)
        n_conns_before = len(g.conns)
        n_nodes_before = len(g.nodes)
        cfg = {"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
               "mod": M.N_SPLIT_INCOMING_ONE}
        ok = M.mutate_add_neuron(g, cfg, seed=99)
        assert ok
        # split: -1 conn, +2 conns = +1 conn; +1 node
        assert len(g.conns) == n_conns_before + 1
        assert len(g.nodes) == n_nodes_before + 1
        assert g.is_dag()

    def test_split_outgoing_one(self):
        g = make_genome(2, 1)
        n_conns_before = len(g.conns)
        cfg = {"selection": M.C_SELECT_PCT_SHUFFLED, "pct": 0.0,
               "mod": M.N_SPLIT_OUTGOING_ONE}
        ok = M.mutate_add_neuron(g, cfg, seed=99)
        assert ok
        assert len(g.conns) == n_conns_before + 1
        assert g.is_dag()

    def test_revert(self):
        g = make_genome(2, 1)
        snapshot = sorted((i, c.in_node, c.out_node, c.weight) for i, c in g.conns.items())
        nodes_before = len(g.nodes)
        M.mutate_add_neuron(g, {"selection": M.C_SELECT_PCT_SHUFFLED,
                                 "pct": 0.0, "mod": M.N_SPLIT_INCOMING_ONE}, seed=99)
        M.revert_last_mutation(g)
        assert len(g.conns) == len(snapshot)
        assert len(g.nodes) == nodes_before
        snapshot_after = sorted((i, c.in_node, c.out_node, c.weight) for i, c in g.conns.items())
        assert snapshot == snapshot_after

    def test_forward_still_works(self):
        g = make_genome(2, 1)
        M.mutate_add_neuron(g, {"selection": M.C_SELECT_PCT_SHUFFLED,
                                 "pct": 0.0, "mod": M.N_SPLIT_INCOMING_ONE}, seed=1)
        out = forward(g, np.array([1.0, 1.0]))
        assert out.shape == (1,)


class TestPruningMutation:
    def test_prune_nonessential(self):
        g = make_genome(2, 2)
        reg = g.registry
        # add hidden node with multiple connections (so we have nonessential ones)
        g.add_node(NodeGene(100, "hidden", "relu"))
        reg.bump_node_presence(100, +1)
        i1 = reg.get_connection_innov(0, 100)
        i2 = reg.get_connection_innov(1, 100)
        i3 = reg.get_connection_innov(100, 2)
        i4 = reg.get_connection_innov(100, 3)
        for i in [i1, i2, i3, i4]:
            g.add_connection(ConnectionGene(i, *reg._conn_inv_lookup(i) if False else ((0,100) if i==i1 else (1,100) if i==i2 else (100,2) if i==i3 else (100,3)), 0.7))
            reg.bump_presence(i, +1)
        n_before = len(g.conns)
        # try to prune
        M.mutate_prune(g, {"selection": M.P_SELECT_PCT_SHUFFLED, "pct": 0.5}, seed=1)
        assert g.is_dag()

    def test_linear_path_merge(self):
        """Set up a clear linear path and verify merge works."""
        reg = InnovationRegistry()
        reg.reserve_node_ids([0, 1, 2])
        g = Genome(reg, 2, 1)
        g.add_node(NodeGene(0, "input", "identity"))
        g.add_node(NodeGene(1, "input", "identity"))
        g.add_node(NodeGene(2, "output", "identity"))
        # only 0->2; we want a linear path through a hidden node
        i01 = reg.get_connection_innov(0, 2)
        g.add_connection(ConnectionGene(i01, 0, 2, 0.5))
        reg.bump_presence(i01, +1)
        # split into 0 -> hidden -> 2
        M.mutate_add_neuron(g, {"selection": M.C_SELECT_PCT_SHUFFLED,
                                 "pct": 0.0, "mod": M.N_SPLIT_INCOMING_ONE}, seed=1)
        # now we have 0->hid, hid->2, and 1->2 (if 1->2 was added; it wasn't here)
        # Actually we only had 0->2 originally. After split: 0->hid, hid->2.
        # The hidden node has only one in and one out = linear path
        # Pruning should merge this back
        n_before = len(g.conns)
        nodes_before = len(g.nodes)
        # Verify linear path exists
        hids = g.hidden_node_ids()
        assert len(hids) == 1
        hid = hids[0]
        assert len(g.incoming(hid)) == 1
        assert len(g.outgoing(hid)) == 1
        # prune
        M.mutate_prune(g, {"selection": M.P_SELECT_PCT_SHUFFLED, "pct": 1.0}, seed=1)
        # should have merged: -1 conn, -1 node
        assert len(g.conns) == n_before - 1
        assert len(g.nodes) == nodes_before - 1

    def test_revert(self):
        """Revert fully restores the genome."""
        g = make_genome(2, 2)
        reg = g.registry
        g.add_node(NodeGene(100, "hidden", "relu"))
        reg.bump_node_presence(100, +1)
        i1 = reg.get_connection_innov(0, 100)
        i2 = reg.get_connection_innov(100, 2)
        g.add_connection(ConnectionGene(i1, 0, 100, 0.7))
        g.add_connection(ConnectionGene(i2, 100, 2, 0.8))
        reg.bump_presence(i1, +1)
        reg.bump_presence(i2, +1)
        snapshot = sorted((i, c.in_node, c.out_node, c.weight) for i, c in g.conns.items())
        nodes_before = len(g.nodes)
        M.mutate_prune(g, {"selection": M.P_SELECT_PCT_SHUFFLED, "pct": 1.0}, seed=3)
        M.revert_last_mutation(g)
        snapshot_after = sorted((i, c.in_node, c.out_node, c.weight) for i, c in g.conns.items())
        assert snapshot == snapshot_after
        assert len(g.nodes) == nodes_before

    def test_inverse_roulette(self):
        """Inverse roulette should favor pruning high-magnitude weights."""
        g = make_genome(2, 2)
        # set one weight very high
        first_innov = list(g.conns.keys())[0]
        g.conns[first_innov].weight = 10.0
        # run many prunes with inverse roulette and check distribution
        # (statistical test, just verify it runs)
        for s in range(20):
            g_copy = g.clone()
            M.mutate_prune(g_copy, {"selection": M.P_SELECT_PROB_INVERSE_ROULETTE,
                                     "pct": 0.5}, seed=s)
            assert g_copy.is_dag()
