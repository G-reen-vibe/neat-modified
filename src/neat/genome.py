"""
Genome: a single neural network in the population.

Design notes
------------
* We forbid loops and disconnected graphs (per spec's "alternative approach").
  The genome is therefore a DAG from inputs(+bias) to outputs.
* A topological sort is recomputed lazily (after any structural mutation) and
  cached; the forward pass is O(V + E).
* Every connection carries an *innovation number* that is universal across the
  whole population (managed by GlobalIndex).
* Optimizer state (ADAM moments, etc.) is stored per-connection so it can be
  inherited by children and averaged during crossover.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .config import Config, ActivationKind
from .activation import ActivationState, make_activation
from .indexing import GlobalIndex


# ---------------------------------------------------------------------------
# Connection / node records
# ---------------------------------------------------------------------------
@dataclass
class Connection:
    innov: int               # universal innovation number
    src: int                 # node id
    dst: int                 # node id
    weight: float
    enabled: bool = True     # we keep the flag for serialization, but per spec
                              # we *remove* connections rather than disable.
    # Optimizer state (per-connection, inherited/averaged on crossover)
    m: float = 0.0   # 1st moment (Adam/Momentum)
    v: float = 0.0   # 2nd moment (Adam/RMSProp)


@dataclass
class Node:
    node_id: int
    kind: str               # "input" | "output" | "bias" | "hidden"
    activation: ActivationState
    # Optimizer state for the activation params (used by UAF/PSwish)
    act_m: Optional[np.ndarray] = None
    act_v: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------
class Genome:
    """A single individual."""

    def __init__(self, cfg: Config, index: GlobalIndex):
        self.cfg = cfg
        self.index = index

        # nodes keyed by node_id
        self.nodes: Dict[int, Node] = {}
        # connections keyed by innovation number
        self.conns: Dict[int, Connection] = {}

        # bookkeeping
        self.fitness: float = 0.0
        self.parent_species: Optional[int] = None
        self.id: int = -1  # assigned by Population

        # Cached topological order (invalidated on structural change)
        self._topo_order: Optional[List[int]] = None
        self._last_weight_mut_delta: Dict[int, float] = {}  # innov -> delta from last weight mutation
        self._last_act_mut_delta: Dict[int, np.ndarray] = {}  # node_id -> delta vector

        # Build the initial structural nodes (inputs, outputs, bias)
        for i in range(cfg.n_inputs):
            self.nodes[i] = Node(node_id=i, kind="input",
                                 activation=make_activation(ActivationKind.RELU))
        for j in range(cfg.n_outputs):
            nid = cfg.n_inputs + j
            self.nodes[nid] = Node(node_id=nid, kind="output",
                                   activation=make_activation(cfg.output_activation))
        if cfg.bias_enabled and index.bias_id is not None:
            self.nodes[index.bias_id] = Node(node_id=index.bias_id, kind="bias",
                                             activation=make_activation(ActivationKind.RELU))

    # ------------------------------------------------------------------
    # Cloning / copy
    # ------------------------------------------------------------------
    def clone(self) -> "Genome":
        g = Genome(self.cfg, self.index)
        g.nodes = {nid: Node(node_id=n.node_id, kind=n.kind,
                             activation=n.activation.clone())
                   for nid, n in self.nodes.items()}
        g.conns = {iv: Connection(innov=c.innov, src=c.src, dst=c.dst,
                                  weight=c.weight, enabled=c.enabled,
                                  m=c.m, v=c.v)
                   for iv, c in self.conns.items()}
        g.fitness = self.fitness
        g.parent_species = self.parent_species
        g.id = self.id
        # Don't copy cached topo order; will be recomputed
        return g

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------
    @property
    def input_ids(self) -> List[int]:
        return [n.node_id for n in self.nodes.values() if n.kind == "input"]

    @property
    def output_ids(self) -> List[int]:
        return [n.node_id for n in self.nodes.values() if n.kind == "output"]

    @property
    def bias_id(self) -> Optional[int]:
        return self.cfg.n_inputs + self.cfg.n_outputs if self.cfg.bias_enabled else None

    def has_conn(self, src: int, dst: int) -> bool:
        iv = self.index.conn_innov_of(src, dst)
        return iv is not None and iv in self.conns

    def incoming(self, node_id: int) -> List[Connection]:
        return [c for c in self.conns.values() if c.dst == node_id]

    def outgoing(self, node_id: int) -> List[Connection]:
        return [c for c in self.conns.values() if c.src == node_id]

    def is_essential_incoming(self, c: Connection) -> bool:
        """Would removing c leave its dst with no other incoming?"""
        count = sum(1 for cc in self.conns.values() if cc.dst == c.dst)
        return count <= 1

    def is_essential_outgoing(self, c: Connection) -> bool:
        """Would removing c leave its src with no other outgoing?"""
        count = sum(1 for cc in self.conns.values() if cc.src == c.src)
        return count <= 1

    # ------------------------------------------------------------------
    # DAG maintenance
    # ------------------------------------------------------------------
    def _would_create_cycle(self, src: int, dst: int) -> bool:
        """Adding src -> dst creates a cycle iff there's already a path dst -> src."""
        if src == dst:
            return True
        # BFS from dst looking for src
        stack = [dst]
        seen = set()
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            for c in self.conns.values():
                if c.src == cur:
                    stack.append(c.dst)
        return False

    def add_conn(self, src: int, dst: int, weight: float,
                 register: bool = True) -> Optional[Connection]:
        """Add a connection.  Returns None if it would create a cycle."""
        if src not in self.nodes or dst not in self.nodes:
            return None
        if src == dst:
            return None
        if self.has_conn(src, dst):
            return None
        # Don't allow connections *into* inputs or *out of* outputs
        if self.nodes[src].kind == "output":
            return None
        if self.nodes[dst].kind == "input":
            return None
        if self.cfg.forbid_loops and self._would_create_cycle(src, dst):
            return None
        iv = self.index.get_or_create_conn_innov(src, dst)
        c = Connection(innov=iv, src=src, dst=dst, weight=float(weight))
        self.conns[iv] = c
        if register:
            self.index.register_conn_added(iv)
        self._invalidate_cache()
        return c

    def remove_conn(self, innov: int, register: bool = True) -> None:
        c = self.conns.pop(innov, None)
        if c is not None:
            if register:
                self.index.unregister_conn(innov)
            self._invalidate_cache()

    def add_hidden_node(self, node_id: Optional[int] = None,
                        activation_kind: Optional[str] = None) -> Node:
        if node_id is None:
            node_id = self.index.new_hidden_node()
        if node_id not in self.nodes:
            kind = activation_kind or self.cfg.hidden_activation
            n = Node(node_id=node_id, kind="hidden",
                     activation=make_activation(kind))
            self.nodes[node_id] = n
            self.index.register_node_added(node_id)
        return self.nodes[node_id]

    def _invalidate_cache(self) -> None:
        self._topo_order = None

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------
    def topo_sort(self) -> List[int]:
        """Return node ids in topological order (inputs first, outputs last)."""
        if self._topo_order is not None:
            return self._topo_order
        # Build adjacency and in-degree (only over enabled conns)
        in_deg: Dict[int, int] = {nid: 0 for nid in self.nodes}
        adj: Dict[int, List[int]] = {nid: [] for nid in self.nodes}
        for c in self.conns.values():
            if not c.enabled:
                continue
            adj[c.src].append(c.dst)
            in_deg[c.dst] += 1
        # Kahn's
        # Process inputs first to keep ordering stable & deterministic
        order: List[int] = []
        # Use a list as a queue; prioritize inputs, then bias, then hidden, then outputs
        def sort_key(nid: int) -> int:
            k = self.nodes[nid].kind
            return {"input": 0, "bias": 1, "hidden": 2, "output": 3}[k]
        queue = sorted([nid for nid, d in in_deg.items() if d == 0],
                       key=sort_key)
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            next_ready = []
            for nb in adj[cur]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    next_ready.append(nb)
            # Re-sort the queue + new ready nodes for deterministic ordering
            queue = sorted(queue + next_ready, key=sort_key)
        # If there's a cycle (shouldn't happen since we forbid them), append
        # the remaining nodes anyway so the forward pass can still run.
        if len(order) < len(self.nodes):
            for nid in self.nodes:
                if nid not in order:
                    order.append(nid)
        self._topo_order = order
        return order

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute output activations for a single input vector.

        Args:
            inputs: shape (n_inputs,)
        Returns:
            outputs: shape (n_outputs,)
        """
        cfg = self.cfg
        assert inputs.shape[0] == cfg.n_inputs, f"input shape {inputs.shape} != {cfg.n_inputs}"
        order = self.topo_sort()
        # activation buffer
        act = np.zeros(len(self.nodes), dtype=np.float64)
        # map node_id -> index in act
        # Use a dict for simplicity; for performance we could remap to 0..N
        nid_to_idx = {nid: i for i, nid in enumerate(self.nodes)}

        # Seed inputs and bias
        for i, inp_id in enumerate(self.input_ids):
            act[nid_to_idx[inp_id]] = inputs[i]
        if cfg.bias_enabled and self.bias_id is not None:
            act[nid_to_idx[self.bias_id]] = 1.0

        # Precompute incoming connections per node
        in_conns: Dict[int, List[Connection]] = {nid: [] for nid in self.nodes}
        for c in self.conns.values():
            if c.enabled:
                in_conns[c.dst].append(c)

        for nid in order:
            node = self.nodes[nid]
            if node.kind in ("input", "bias"):
                continue
            # Sum incoming
            s = 0.0
            for c in in_conns[nid]:
                s += c.weight * act[nid_to_idx[c.src]]
            act[nid_to_idx[nid]] = node.activation(np.array(s))

        out = np.zeros(cfg.n_outputs, dtype=np.float64)
        for j, out_id in enumerate(self.output_ids):
            out[j] = act[nid_to_idx[out_id]]
        return out

    # ------------------------------------------------------------------
    # Convenience: forward a batch (just loops; genomes are small)
    # ------------------------------------------------------------------
    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """X: (N, n_inputs) -> (N, n_outputs)"""
        out = np.zeros((X.shape[0], self.cfg.n_outputs), dtype=np.float64)
        for i in range(X.shape[0]):
            out[i] = self.forward(X[i])
        return out

    # ------------------------------------------------------------------
    # Bookkeeping for the optimizer (track last weight-mutation deltas)
    # ------------------------------------------------------------------
    def record_weight_delta(self, innov: int, delta: float) -> None:
        self._last_weight_mut_delta[innov] = delta

    def consume_weight_deltas(self) -> Dict[int, float]:
        d = self._last_weight_mut_delta
        self._last_weight_mut_delta = {}
        return d

    # ------------------------------------------------------------------
    # Serialization (for checkpoints)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fitness": self.fitness,
            "parent_species": self.parent_species,
            "nodes": [(nid, n.kind, n.activation.to_dict()) for nid, n in self.nodes.items()],
            "conns": [(c.innov, c.src, c.dst, c.weight, c.enabled, c.m, c.v)
                      for c in self.conns.values()],
        }

    @classmethod
    def from_dict(cls, d: dict, cfg: Config, index: GlobalIndex) -> "Genome":
        g = cls(cfg, index)
        g.id = d["id"]
        g.fitness = d["fitness"]
        g.parent_species = d["parent_species"]
        g.nodes = {}
        for nid, kind, act_d in d["nodes"]:
            g.nodes[nid] = Node(node_id=nid, kind=kind,
                                activation=ActivationState.from_dict(act_d))
        g.conns = {}
        for innov, src, dst, w, en, m, v in d["conns"]:
            g.conns[innov] = Connection(innov=innov, src=src, dst=dst,
                                        weight=w, enabled=en, m=m, v=v)
        return g

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"<Genome id={self.id} nodes={len(self.nodes)} "
                f"conns={len(self.conns)} fit={self.fitness:.3f}>")
