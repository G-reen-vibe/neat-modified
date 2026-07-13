"""
Crossover operations between two genomes.

Topology selection (P):
  * T_FITTER     - use the fitter genome's topology
  * T_MORE_CONNS - use the topology of the genome with more connections
  * T_COMBINE    - attempt to combine both (with cycle breaking for DAG mode)

Weight/neuron selection (P):
  * W_INDEPENDENT - copy each weight independently from one network or the other
  * W_AVERAGE     - average all possible weights
  * W_BY_NEURON   - for a given neuron, copy all outgoing weights from one or the other
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .genome import Genome, NodeGene, ConnectionGene


T_FITTER = 0
T_MORE_CONNS = 1
T_COMBINE = 2

W_INDEPENDENT = 0
W_AVERAGE = 1
W_BY_NEURON = 2


def crossover(
    parent_a: Genome,
    parent_b: Genome,
    topology_method: int = T_FITTER,
    weight_method: int = W_AVERAGE,
    seed: int = 0,
) -> Genome:
    """Produce a child genome from two parents.

    ``parent_a`` is treated as the *primary* parent (the fitter one when
    using T_FITTER). Both parents must share the same registry.
    """
    assert parent_a.registry is parent_b.registry, "parents must share registry"
    reg = parent_a.registry
    rng = random.Random(seed)

    child = Genome(reg, parent_a.n_inputs, parent_a.n_outputs)
    child.generation = max(parent_a.generation, parent_b.generation) + 1

    # Determine which parent is the "base" for topology
    if topology_method == T_FITTER:
        base, other = (parent_a, parent_b) if parent_a.fitness >= parent_b.fitness else (parent_b, parent_a)
    elif topology_method == T_MORE_CONNS:
        base, other = (parent_a, parent_b) if len(parent_a.conns) >= len(parent_b.conns) else (parent_b, parent_a)
    elif topology_method == T_COMBINE:
        base, other = parent_a, parent_b
    else:
        raise ValueError(f"unknown topology method {topology_method}")

    # ---- NODES ----
    # Always include all input/output/bias nodes (from either parent)
    seen_kinds: Dict[int, str] = {}
    for p in (parent_a, parent_b):
        for nid, n in p.nodes.items():
            if nid not in seen_kinds:
                seen_kinds[nid] = n.kind
    # input/output/bias ids
    io_ids = {nid for nid, k in seen_kinds.items() if k in ("input", "output", "bias")}
    for nid in io_ids:
        # find the node in either parent
        n = parent_a.nodes.get(nid) or parent_b.nodes.get(nid)
        child.add_node(NodeGene(nid, n.kind, n.activation))

    # hidden nodes: include from base, plus optionally from other (for COMBINE)
    base_hidden = {nid: n for nid, n in base.nodes.items() if n.kind == "hidden"}
    other_hidden = {nid: n for nid, n in other.nodes.items() if n.kind == "hidden"}

    if topology_method == T_COMBINE:
        # include all hidden nodes from both
        for nid, n in base_hidden.items():
            child.add_node(NodeGene(nid, "hidden", n.activation))
        for nid, n in other_hidden.items():
            if nid not in child.nodes:
                child.add_node(NodeGene(nid, "hidden", n.activation))
    else:
        for nid, n in base_hidden.items():
            child.add_node(NodeGene(nid, "hidden", n.activation))

    # ---- CONNECTIONS ----
    # Innovations in both, only in base, only in other
    base_innovs = set(base.conns.keys())
    other_innovs = set(other.conns.keys())
    shared = base_innovs & other_innovs
    only_base = base_innovs - other_innovs
    only_other = other_innovs - base_innovs

    # Topology: which connections to include
    if topology_method == T_FITTER or topology_method == T_MORE_CONNS:
        # include shared + only_base (matching + disjoint/excess of base)
        candidate_conns = shared | only_base
    else:  # T_COMBINE
        candidate_conns = shared | only_base | only_other

    # Weight selection
    for innov in candidate_conns:
        c_a = base.conns.get(innov)
        c_b = other.conns.get(innov)
        # both endpoints must exist in child
        if c_a is None and c_b is None:
            continue
        # pick a reference for in/out
        ref = c_a if c_a is not None else c_b
        if ref.in_node not in child.nodes or ref.out_node not in child.nodes:
            continue
        # cycle check (for COMBINE; for FITTER the topology came from base so should be fine)
        if child.creates_cycle(ref.in_node, ref.out_node):
            continue
        # determine weight
        if c_a is not None and c_b is not None:
            if weight_method == W_INDEPENDENT:
                w = c_a.weight if rng.random() < 0.5 else c_b.weight
            elif weight_method == W_AVERAGE:
                w = 0.5 * (c_a.weight + c_b.weight)
            elif weight_method == W_BY_NEURON:
                # determined later in a second pass; for now default to average
                w = 0.5 * (c_a.weight + c_b.weight)
            else:
                raise ValueError(f"unknown weight method {weight_method}")
        elif c_a is not None:
            w = c_a.weight
        else:
            w = c_b.weight
        child.add_connection(ConnectionGene(innov, ref.in_node, ref.out_node, w))
        reg.bump_presence(innov, +1)

    # W_BY_NEURON second pass: for each neuron, pick a parent and override
    # the outgoing weights to come from that parent (if they exist there).
    if weight_method == W_BY_NEURON:
        for nid, n in child.nodes.items():
            if n.kind not in ("hidden", "output"):
                continue
            # pick a parent
            src_parent = base if rng.random() < 0.5 else other
            for c in list(child.conns.values()):
                if c.in_node == nid:
                    # try to find a matching connection in src_parent
                    src_c = src_parent.conns.get(c.innov)
                    if src_c is not None:
                        c.weight = src_c.weight

    # Inherit ADAM moments (averaged) - for the optimizer
    if parent_a.m and parent_b.m:
        child.m = {i: 0.5 * (parent_a.m.get(i, 0.0) + parent_b.m.get(i, 0.0))
                   for i in set(parent_a.m) | set(parent_b.m)}
    if parent_a.v and parent_b.v:
        child.v = {i: 0.5 * (parent_a.v.get(i, 0.0) + parent_b.v.get(i, 0.0))
                   for i in set(parent_a.v) | set(parent_b.v)}

    return child
