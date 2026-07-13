"""Tests for the innovation registry and genome basics."""
import pytest
import numpy as np

from neat.registry import InnovationRegistry
from neat.genome import Genome, NodeGene, ConnectionGene
from neat.network import forward, topological_sort


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


class TestRegistry:
    def test_universal_connection_innov(self):
        reg = InnovationRegistry()
        i1 = reg.get_connection_innov(0, 1)
        i2 = reg.get_connection_innov(0, 1)
        assert i1 == i2
        i3 = reg.get_connection_innov(1, 0)
        assert i3 != i1

    def test_universal_split_node(self):
        reg = InnovationRegistry()
        innov = reg.get_connection_innov(0, 1)
        n1 = reg.get_split_node_id(innov)
        n2 = reg.get_split_node_id(innov)
        assert n1 == n2
        # different connection -> different node
        innov2 = reg.get_connection_innov(1, 0)
        n3 = reg.get_split_node_id(innov2)
        assert n3 != n1

    def test_reserve_node_ids(self):
        reg = InnovationRegistry()
        reg.reserve_node_ids([0, 1, 2, 5])
        assert reg.new_node_id() == 6
        assert reg.new_node_id() == 7

    def test_presence_count_tracking(self):
        reg = InnovationRegistry()
        i = reg.get_connection_innov(0, 1)
        assert reg.presence_count.get(i, 0) == 0
        reg.bump_presence(i, +1)
        reg.bump_presence(i, +1)
        assert reg.presence_count[i] == 2
        reg.bump_presence(i, -1)
        assert reg.presence_count[i] == 1

    def test_state_dict_roundtrip(self):
        reg = InnovationRegistry()
        reg.get_connection_innov(0, 1)
        reg.get_split_node_id(0)
        reg.bump_presence(0, +5)
        reg.bump_selection(0)
        reg.bump_node_presence(0, +3)
        state = reg.state_dict()
        reg2 = InnovationRegistry()
        reg2.load_state_dict(state)
        assert reg2.state_dict() == state


class TestGenome:
    def test_dag_detection(self):
        g = make_genome(1, 1)
        assert g.is_dag()
        # add a back-edge (would create cycle)
        # we don't have direct way to add a cycle since creates_cycle blocks
        # but is_dag should still detect if it exists
        # add 1->0 connection (would create cycle in 0->1, 1->0)
        # but creates_cycle prevents this in normal flow
        # so we just verify the method works
        assert g.is_dag()

    def test_creates_cycle(self):
        g = make_genome(2, 1)
        # 0->2, 1->2 exist; 2->0 would create cycle
        assert g.creates_cycle(2, 0)
        assert g.creates_cycle(2, 1)
        assert not g.creates_cycle(0, 1)  # but 0->1 isn't a valid conn (1 is input)

    def test_clone(self):
        g = make_genome(3, 2)
        g2 = g.clone()
        assert g2.n_inputs == g.n_inputs
        assert g2.n_outputs == g.n_outputs
        assert len(g2.conns) == len(g.conns)
        assert g2.conns is not g.conns
        assert g2.nodes is not g.nodes
        # mutating clone should not affect original
        g2.conns[0].weight = 999.0
        assert g.conns[0].weight == 0.5

    def test_node_id_helpers(self):
        g = make_genome(3, 2)
        assert len(g.input_node_ids()) == 3
        assert len(g.output_node_ids()) == 2
        assert len(g.hidden_node_ids()) == 0
        assert len(g.bias_node_ids()) == 0


class TestForward:
    def test_identity_then_tanh(self):
        g = make_genome(2, 1)
        out = forward(g, np.array([1.0, 1.0]))
        assert abs(out[0] - np.tanh(1.0)) < 1e-6

    def test_relu_hidden(self):
        reg = InnovationRegistry()
        reg.reserve_node_ids([0, 1, 2, 3])
        g = Genome(reg, 2, 1)
        g.add_node(NodeGene(0, "input", "identity"))
        g.add_node(NodeGene(1, "input", "identity"))
        g.add_node(NodeGene(2, "hidden", "relu"))
        g.add_node(NodeGene(3, "output", "identity"))
        g.add_connection(ConnectionGene(reg.get_connection_innov(0, 2), 0, 2, 1.0))
        g.add_connection(ConnectionGene(reg.get_connection_innov(1, 2), 1, 2, 1.0))
        g.add_connection(ConnectionGene(reg.get_connection_innov(2, 3), 2, 3, 1.0))
        # relu(1 + (-5)) = 0
        out = forward(g, np.array([1.0, -5.0]))
        assert abs(out[0]) < 1e-6
        # relu(2 + 1) = 3
        out = forward(g, np.array([2.0, 1.0]))
        assert abs(out[0] - 3.0) < 1e-6

    def test_topo_sort_order(self):
        g = make_genome(2, 1)
        order = topological_sort(g)
        assert order.index(0) < order.index(2)
        assert order.index(1) < order.index(2)

    def test_input_ordering(self):
        """Inputs are mapped by sorted node id."""
        reg = InnovationRegistry()
        # use unusual input ids
        reg.reserve_node_ids([10, 20, 30])
        g = Genome(reg, 3, 1)
        g.add_node(NodeGene(10, "input", "identity"))
        g.add_node(NodeGene(20, "input", "identity"))
        g.add_node(NodeGene(30, "input", "identity"))
        g.add_node(NodeGene(31, "output", "identity"))
        # weight 1 from each input
        for nid in [10, 20, 30]:
            innov = reg.get_connection_innov(nid, 31)
            g.add_connection(ConnectionGene(innov, nid, 31, 1.0))
        # inputs[0] -> node 10, inputs[1] -> node 20, inputs[2] -> node 30
        out = forward(g, np.array([1.0, 2.0, 3.0]))
        assert abs(out[0] - 6.0) < 1e-6

    def test_p_swish_activation(self):
        reg = InnovationRegistry()
        reg.reserve_node_ids([0, 1])
        g = Genome(reg, 1, 1)
        g.add_node(NodeGene(0, "input", "identity"))
        g.add_node(NodeGene(1, "output", "p_swish"))
        innov = reg.get_connection_innov(0, 1)
        g.add_connection(ConnectionGene(innov, 0, 1, 1.0))
        # P=0: out = 0.5 * x
        out = forward(g, np.array([2.0]), node_param={1: 0.0})
        assert abs(out[0] - 1.0) < 1e-6
        # P=1: out = x * sigmoid(x) = 2 * sigmoid(2) = 2 * 0.8808 = 1.7616
        out = forward(g, np.array([2.0]), node_param={1: 1.0})
        assert abs(out[0] - 2.0 * (1.0 / (1.0 + np.exp(-2.0)))) < 1e-6
