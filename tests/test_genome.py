"""Unit tests for the GlobalIndex."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neat.indexing import GlobalIndex
from neat.config import Config
from neat.genome import Genome


def test_basic_node_ids():
    idx = GlobalIndex(n_inputs=4, n_outputs=2, bias_enabled=True)
    # Layout: 0..3 inputs, 4..5 outputs, 6 bias
    assert idx.bias_id == 6
    h1 = idx.new_hidden_node()
    h2 = idx.new_hidden_node()
    assert h1 == 7 and h2 == 8


def test_universal_conn_innov():
    idx = GlobalIndex(n_inputs=2, n_outputs=1, bias_enabled=True)
    iv1 = idx.get_or_create_conn_innov(0, 2)  # input 0 -> output 0
    iv2 = idx.get_or_create_conn_innov(0, 2)
    assert iv1 == iv2, "same (src,dst) must always map to same innovation"
    iv3 = idx.get_or_create_conn_innov(1, 2)
    assert iv3 != iv1
    iv4 = idx.get_or_create_conn_innov(0, 2)
    assert iv4 == iv1


def test_universal_split_node():
    """Two genomes splitting the same connection must get the same node id."""
    idx = GlobalIndex(n_inputs=2, n_outputs=1, bias_enabled=True)
    # Create connection 0->2 first
    conn_innov = idx.get_or_create_conn_innov(0, 2)
    # Split it twice: must return identical IDs
    n1, in1, out1 = idx.get_or_create_split_node(conn_innov)
    n2, in2, out2 = idx.get_or_create_split_node(conn_innov)
    assert n1 == n2
    assert in1 == in2
    assert out1 == out2
    # The new in-conn is (0 -> n1) and the out-conn is (n1 -> 2)
    assert idx.conn_endpoints(in1) == (0, n1)
    assert idx.conn_endpoints(out1) == (n1, 2)


def test_commonality_tracking():
    idx = GlobalIndex(n_inputs=2, n_outputs=1, bias_enabled=True)
    iv = idx.get_or_create_conn_innov(0, 2)
    assert idx.conn_commonality(iv) == 0
    idx.register_conn_added(iv)
    idx.register_conn_added(iv)
    assert idx.conn_commonality(iv) == 2
    idx.unregister_conn(iv)
    assert idx.conn_commonality(iv) == 1


def test_genome_dag_no_cycles():
    cfg = Config(n_inputs=2, n_outputs=1, bias_enabled=True)
    idx = GlobalIndex(2, 1, bias_enabled=True)
    g = Genome(cfg, idx)
    # Inputs -> output should be fine
    c1 = g.add_conn(0, 2, 0.5)
    c2 = g.add_conn(1, 2, 0.5)
    assert c1 is not None and c2 is not None
    # Adding 2 -> 0 should be forbidden (cycle)
    assert g.add_conn(2, 0, 0.5) is None
    # Add a hidden node and split
    h = g.add_hidden_node()
    hid = h.node_id
    # output -> hidden would create a cycle (hidden -> output already? no, we haven't added it)
    # Actually we need hidden -> output and input -> hidden.  Let's test cycle:
    g.add_conn(hid, 2, 0.3)   # hidden -> output, ok
    g.add_conn(0, hid, 0.3)   # input -> hidden, ok
    # Now output -> hidden would create a cycle (hidden -> output exists)
    assert g.add_conn(2, hid, 0.3) is None


def test_topo_sort_outputs_last():
    cfg = Config(n_inputs=2, n_outputs=1, bias_enabled=True)
    idx = GlobalIndex(2, 1, bias_enabled=True)
    g = Genome(cfg, idx)
    g.add_conn(0, 2, 0.5)
    g.add_conn(1, 2, 0.5)
    order = g.topo_sort()
    # Outputs (node 2) must come after inputs (0,1) and bias (3)
    assert order.index(2) > order.index(0)
    assert order.index(2) > order.index(1)
    assert order.index(2) > order.index(3)


def test_forward_pass_simple():
    """2 inputs, 1 output, no hidden, sigmoid output -> standard NN."""
    import numpy as np
    cfg = Config(n_inputs=2, n_outputs=1, bias_enabled=False)
    cfg.output_activation = "sigmoid"
    idx = GlobalIndex(2, 1, bias_enabled=False)
    g = Genome(cfg, idx)
    g.add_conn(0, 2, 1.0)
    g.add_conn(1, 2, -1.0)
    # sigmoid(0) = 0.5
    out = g.forward(np.array([1.0, 1.0]))
    assert abs(out[0] - 0.5) < 1e-6, f"got {out[0]}"
    # sigmoid(large positive) -> ~1
    out2 = g.forward(np.array([10.0, 0.0]))
    assert out2[0] > 0.99


def test_forward_with_hidden():
    """Hidden node in the middle should actually affect the output."""
    import numpy as np
    cfg = Config(n_inputs=1, n_outputs=1, bias_enabled=False)
    cfg.output_activation = "sigmoid"
    cfg.hidden_activation = "relu"
    idx = GlobalIndex(1, 1, bias_enabled=False)
    g = Genome(cfg, idx)
    h = g.add_hidden_node()
    hid = h.node_id
    g.add_conn(0, hid, 2.0)    # input * 2
    g.add_conn(hid, 1, 1.0)    # hidden -> output
    out = g.forward(np.array([1.0]))
    # relu(2) = 2, sigmoid(2) ~= 0.88
    assert abs(out[0] - 1.0/(1.0+np.exp(-2.0))) < 1e-6, f"got {out[0]}"


if __name__ == "__main__":
    test_basic_node_ids()
    test_universal_conn_innov()
    test_universal_split_node()
    test_commonality_tracking()
    test_genome_dag_no_cycles()
    test_topo_sort_outputs_last()
    test_forward_pass_simple()
    test_forward_with_hidden()
    print("All Genome/Index tests passed.")
