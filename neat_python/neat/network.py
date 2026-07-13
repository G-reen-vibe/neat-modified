"""
Forward-pass engine.

We use the DAG formulation described in the spec: the genome must form a
DAG from inputs to outputs. If the genome is mutated in a way that would
introduce a cycle or disconnect, the mutation is rejected (the mutators
check this before applying). The forward pass uses a topological sort
which gives O(V+E) evaluation.

For small genomes (the common case in CartPole etc.) we provide both a
plain Python implementation and a numpy-batched one. The plain one is
used for individual evaluations; the numpy-batched one is used to push
many genomes through the same input in one shot.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Optional

import numpy as np

from .genome import Genome
from .activations import get_activation


# ---------------------------------------------------------------------- topo
def topological_sort(genome: Genome) -> List[int]:
    """Return node ids in topological order (inputs/bias first).

    Assumes the genome is a DAG (mutators guarantee this). Raises if a
    cycle is detected.
    """
    in_deg: Dict[int, int] = {n: 0 for n in genome.nodes}
    for c in genome.conns.values():
        if c.enabled and c.in_node in in_deg and c.out_node in in_deg:
            in_deg[c.out_node] += 1
    # Start with all zero-degree nodes (inputs, bias, disconnected nodes)
    queue = sorted([n for n, d in in_deg.items() if d == 0])
    order: List[int] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        # collect downstream
        next_zero = []
        for c in genome.conns.values():
            if c.enabled and c.in_node == n:
                in_deg[c.out_node] -= 1
                if in_deg[c.out_node] == 0:
                    next_zero.append(c.out_node)
        # stable ordering
        queue = sorted(queue + next_zero)
    if len(order) != len(genome.nodes):
        raise RuntimeError("Genome is not a DAG; cycle detected")
    return order


# ------------------------------------------------------------ single pass ---
def forward(genome: Genome, inputs: np.ndarray, node_param: Optional[Dict[int, float]] = None) -> np.ndarray:
    """Run a single forward pass.

    Parameters
    ----------
    genome : Genome
    inputs : (n_inputs,) array
    node_param : optional dict node_id -> activation parameter (for p_swish/uaf)

    Returns
    -------
    (n_outputs,) array
    """
    node_param = node_param or {}
    n_in = genome.n_inputs
    if inputs.shape != (n_in,):
        raise ValueError(f"expected inputs shape ({n_in},), got {inputs.shape}")

    # init node values
    values: Dict[int, float] = {}
    for nid, node in genome.nodes.items():
        if node.kind == "bias":
            values[nid] = 1.0
        else:
            values[nid] = 0.0

    # input node ids in sorted order map to input[0..n_in-1]
    input_ids_sorted = sorted(n.node_id for n in genome.nodes.values() if n.kind == "input")
    for i, nid in enumerate(input_ids_sorted):
        values[nid] = float(inputs[i])

    order = topological_sort(genome)
    # Precompute incoming connections per node
    incoming: Dict[int, List] = {}
    for c in genome.conns.values():
        if not c.enabled:
            continue
        incoming.setdefault(c.out_node, []).append(c)

    for nid in order:
        node = genome.nodes[nid]
        if node.kind in ("input", "bias"):
            continue
        s = 0.0
        for c in incoming.get(nid, []):
            s += c.weight * values.get(c.in_node, 0.0)
        act, has_param = get_activation(node.activation)
        x = np.array(s, dtype=np.float64)
        if has_param:
            values[nid] = float(act(x, node_param.get(nid, 1.0)))
        else:
            values[nid] = float(act(x))

    out_ids = sorted(n.node_id for n in genome.nodes.values() if n.kind == "output")
    return np.array([values.get(nid, 0.0) for nid in out_ids], dtype=np.float64)


# ------------------------------------------------------------ action choice
def act_discrete(genome: Genome, inputs: np.ndarray, node_param: Optional[Dict[int, float]] = None) -> int:
    """Pick argmax over the genome's output for a discrete action env."""
    out = forward(genome, inputs, node_param)
    return int(np.argmax(out))


def act_continuous(genome: Genome, inputs: np.ndarray, node_param: Optional[Dict[int, float]] = None) -> np.ndarray:
    """Return the raw output vector (for continuous-control envs)."""
    return forward(genome, inputs, node_param)
