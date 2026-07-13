"""Smoke tests for registry + genome + network."""
import sys
sys.path.insert(0, "/home/z/my-project/neat_python")

import numpy as np
from neat.registry import InnovationRegistry
from neat.genome import Genome, NodeGene, ConnectionGene
from neat.network import forward, topological_sort


def make_simple_genome():
    reg = InnovationRegistry()
    # 2 inputs, 1 output, fully connected
    n_in = 2
    n_out = 1
    g = Genome(reg, n_in, n_out)
    # input nodes 0, 1
    for i in range(n_in):
        g.add_node(NodeGene(i, "input", "identity"))
    # output node 2
    out_id = n_in + n_out - 1 + 100  # any id, but use 2 to keep simple
    out_id = 2
    g.add_node(NodeGene(out_id, "output", "tanh"))
    # connections 0->2, 1->2
    for i in range(n_in):
        innov = reg.get_connection_innov(i, out_id)
        g.add_connection(ConnectionGene(innov, i, out_id, weight=0.5))
    return g


def test_forward_basic():
    g = make_simple_genome()
    out = forward(g, np.array([1.0, 1.0]))
    # tanh(0.5 + 0.5) = tanh(1) ~ 0.7616
    assert abs(out[0] - np.tanh(1.0)) < 1e-6, out
    print("forward_basic OK:", out)


def test_forward_three_layer():
    reg = InnovationRegistry()
    g = Genome(reg, 2, 1)
    g.add_node(NodeGene(0, "input", "identity"))
    g.add_node(NodeGene(1, "input", "identity"))
    g.add_node(NodeGene(2, "hidden", "relu"))
    g.add_node(NodeGene(3, "output", "identity"))
    g.add_connection(ConnectionGene(reg.get_connection_innov(0, 2), 0, 2, 1.0))
    g.add_connection(ConnectionGene(reg.get_connection_innov(1, 2), 1, 2, 1.0))
    g.add_connection(ConnectionGene(reg.get_connection_innov(2, 3), 2, 3, 1.0))
    out = forward(g, np.array([1.0, -5.0]))
    # relu(1 - 5) = 0; 0 -> identity = 0
    assert abs(out[0]) < 1e-6, out
    out = forward(g, np.array([2.0, 1.0]))
    # relu(3) = 3
    assert abs(out[0] - 3.0) < 1e-6, out
    print("forward_three_layer OK:", out)


def test_dag_cycle_detection():
    reg = InnovationRegistry()
    g = Genome(reg, 1, 1)
    g.add_node(NodeGene(0, "input", "identity"))
    g.add_node(NodeGene(1, "hidden", "relu"))
    g.add_node(NodeGene(2, "output", "identity"))
    # 0->1->2 is a DAG
    g.add_connection(ConnectionGene(reg.get_connection_innov(0, 1), 0, 1, 1.0))
    g.add_connection(ConnectionGene(reg.get_connection_innov(1, 2), 1, 2, 1.0))
    assert g.is_dag()
    # check creates_cycle would catch a back-edge
    assert g.creates_cycle(2, 1)
    assert not g.creates_cycle(0, 2)
    print("dag_cycle_detection OK")


def test_topological_sort():
    g = make_simple_genome()
    order = topological_sort(g)
    # 0,1 must come before 2
    assert order.index(0) < order.index(2)
    assert order.index(1) < order.index(2)
    print("topo_sort OK:", order)


def test_universal_innov():
    reg = InnovationRegistry()
    g1 = Genome(reg, 2, 1)
    g2 = Genome(reg, 2, 1)
    i1 = reg.get_connection_innov(0, 1)
    i2 = reg.get_connection_innov(0, 1)
    assert i1 == i2, "innovation must be universal"
    # split node universality
    n1 = reg.get_split_node_id(i1)
    n2 = reg.get_split_node_id(i1)
    assert n1 == n2
    print("universal_innov OK")


if __name__ == "__main__":
    test_forward_basic()
    test_forward_three_layer()
    test_dag_cycle_detection()
    test_topological_sort()
    test_universal_innov()
    print("\nALL SMOKE TESTS PASSED")
