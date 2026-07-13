"""
Genome representation.

A Genome holds:
  * a dict of node_id -> NodeGene
  * a dict of innovation_number -> ConnectionGene

Genomes are mutated, crossed-over, and evaluated. They are pure data
containers (no forward pass logic here - that lives in ``network.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Iterable

import numpy as np


# ---------------------------------------------------------------------- nodes
@dataclass
class NodeGene:
    node_id: int
    # "input", "output", "hidden", "bias"
    kind: str
    # activation name; inputs/bias use "identity"
    activation: str = "identity"

    def clone(self) -> "NodeGene":
        return NodeGene(self.node_id, self.kind, self.activation)


# ----------------------------------------------------------------- connections
@dataclass
class ConnectionGene:
    innov: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True

    def clone(self) -> "ConnectionGene":
        return ConnectionGene(self.innov, self.in_node, self.out_node, self.weight, self.enabled)


# -------------------------------------------------------------------- genomes
class Genome:
    """A single neural-network genotype."""

    def __init__(
        self,
        registry,
        n_inputs: int,
        n_outputs: int,
        node_ids: Optional[Dict[int, NodeGene]] = None,
        conns: Optional[Dict[int, ConnectionGene]] = None,
    ) -> None:
        self.registry = registry
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.nodes: Dict[int, NodeGene] = dict(node_ids) if node_ids else {}
        self.conns: Dict[int, ConnectionGene] = dict(conns) if conns else {}
        # fitness / bookkeeping
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0
        self.species_id: int = -1
        # for the optimizer: store the last weight-mutation delta per innov
        self.weight_delta: Dict[int, float] = {}
        # ADAM moments keyed by innov
        self.m: Dict[int, float] = {}
        self.v: Dict[int, float] = {}
        # generation this genome was born
        self.generation: int = 0

    # ----------------------------------------------------------- basics ----
    def clone(self) -> "Genome":
        g = Genome(
            self.registry,
            self.n_inputs,
            self.n_outputs,
            {nid: n.clone() for nid, n in self.nodes.items()},
            {i: c.clone() for i, c in self.conns.items()},
        )
        g.fitness = self.fitness
        g.adjusted_fitness = self.adjusted_fitness
        g.species_id = self.species_id
        g.generation = self.generation
        # note: weight_delta / m / v are not copied - children start fresh
        return g

    def add_node(self, node: NodeGene) -> None:
        self.nodes[node.node_id] = node

    def add_connection(self, conn: ConnectionGene) -> None:
        self.conns[conn.innov] = conn

    def has_connection(self, in_node: int, out_node: int) -> bool:
        innov = self.registry.get_connection_innov(in_node, out_node)
        return innov in self.conns

    # ----------------------------------------------------------- helpers ---
    def input_node_ids(self) -> List[int]:
        return sorted(n.node_id for n in self.nodes.values() if n.kind == "input")

    def output_node_ids(self) -> List[int]:
        return sorted(n.node_id for n in self.nodes.values() if n.kind == "output")

    def hidden_node_ids(self) -> List[int]:
        return sorted(n.node_id for n in self.nodes.values() if n.kind == "hidden")

    def bias_node_ids(self) -> List[int]:
        return sorted(n.node_id for n in self.nodes.values() if n.kind == "bias")

    def is_input(self, node_id: int) -> bool:
        n = self.nodes.get(node_id)
        return n is not None and n.kind == "input"

    def is_output(self, node_id: int) -> bool:
        n = self.nodes.get(node_id)
        return n is not None and n.kind == "output"

    def incoming(self, node_id: int) -> List[ConnectionGene]:
        return [c for c in self.conns.values() if c.out_node == node_id]

    def outgoing(self, node_id: int) -> List[ConnectionGene]:
        return [c for c in self.conns.values() if c.in_node == node_id]

    # ------------------------------------------------------ connectivity ---
    def creates_cycle(self, in_node: int, out_node: int) -> bool:
        """Would adding edge in_node -> out_node create a cycle in the DAG?"""
        if in_node == out_node:
            return True
        # BFS from out_node: if we reach in_node, adding edge creates cycle
        stack = [out_node]
        seen: Set[int] = set()
        while stack:
            n = stack.pop()
            if n == in_node:
                return True
            if n in seen:
                continue
            seen.add(n)
            for c in self.conns.values():
                if c.in_node == n:
                    stack.append(c.out_node)
        return False

    def is_dag(self) -> bool:
        """Check that the current (enabled) connection set forms a DAG."""
        # Kahn's algorithm
        in_deg: Dict[int, int] = {n: 0 for n in self.nodes}
        for c in self.conns.values():
            if c.enabled:
                in_deg[c.out_node] = in_deg.get(c.out_node, 0) + 1
        queue = [n for n, d in in_deg.items() if d == 0]
        visited = 0
        while queue:
            n = queue.pop()
            visited += 1
            for c in self.conns.values():
                if c.enabled and c.in_node == n:
                    in_deg[c.out_node] -= 1
                    if in_deg[c.out_node] == 0:
                        queue.append(c.out_node)
        return visited == len(in_deg)

    # ------------------------------------------------------- serialization -
    def to_dict(self) -> Dict:
        return {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "nodes": {nid: {"kind": n.kind, "activation": n.activation}
                      for nid, n in self.nodes.items()},
            "conns": {str(i): {"in": c.in_node, "out": c.out_node,
                                "w": c.weight, "en": c.enabled}
                      for i, c in self.conns.items()},
            "fitness": self.fitness,
            "species_id": self.species_id,
            "generation": self.generation,
        }
