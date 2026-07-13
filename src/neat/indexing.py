"""
Global node / connection indexing.

Per the spec:
    "Every node in the network is assigned a historical ID ... these IDs should
    be universal: the node along the same weight should have the same ID
    irregardless of what order it was mutated in or which species it is in."

We achieve this by keeping a single GlobalIndex shared by the entire population
that records:
    - input/output/bias node IDs (assigned once at construction)
    - every connection (src, dst) ever created -> innovation number
    - every neuron-split ever performed -> (innovation of original conn) -> new node id

Because splits are indexed by the innovation of the connection being split,
two genomes that split the *same* connection will get the *same* new node id and
the *same* two new connection innovation numbers.

We also track usage statistics needed by the spec's selection mechanisms:
    - connection_commonality: how many genomes currently contain each innovation
    - connection_selected_count: how many times each innovation has been
      selected for a *connection* mutation (i.e. created)
    - neuron_split_count: how many times each (existing) innovation has been
      chosen for a neuron-split mutation
    - neuron_commonality: how many genomes currently contain each hidden node
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import threading


# Sentinel for "no innovation yet"
NO_INNOV = -1


class GlobalIndex:
    """Thread-safe global registry of historical IDs.

    A single instance is shared by an entire Population.
    """

    def __init__(self, n_inputs: int, n_outputs: int, bias_enabled: bool = True):
        # --- Fixed structural node IDs -------------------------------------
        # Layout:
        #   [0 .. n_inputs)               -> input nodes
        #   [n_inputs .. n_inputs+n_outputs) -> output nodes
        #   n_inputs + n_outputs           -> bias node (if enabled)
        #   after that                     -> hidden nodes (allocated on demand)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.bias_enabled = bias_enabled
        self.bias_id: Optional[int] = n_inputs + n_outputs if bias_enabled else None
        self._next_node_id = n_inputs + n_outputs + (1 if bias_enabled else 0)

        # Pre-register the input/output node ids as "structural" (always present)
        # Common usage starts at 1 (every genome contains them).
        self._node_commonality: Dict[int, int] = {}
        for i in range(self._next_node_id):
            self._node_commonality[i] = 0  # will be incremented when genomes register

        # --- Connection innovations ---------------------------------------
        # (src, dst) -> innovation number
        self._conn_innov: Dict[Tuple[int, int], int] = {}
        self._next_innov = 0

        # --- Neuron-split registry ----------------------------------------
        # split_innov (innovation of the connection being split) -> new node id
        self._split_to_node: Dict[int, int] = {}
        # split_innov -> (in_innov, out_innov) for the two new connections
        self._split_to_conns: Dict[int, Tuple[int, int]] = {}

        # --- Usage statistics ---------------------------------------------
        # per-innovation: number of genomes currently containing this connection
        self._conn_commonality: Dict[int, int] = {}
        # per-innovation: number of times it has been selected for a connection mutation
        self._conn_selected_count: Dict[int, int] = {}
        # per-innovation: number of times the connection has been selected for splitting
        self._neuron_split_count: Dict[int, int] = {}
        # per-innovation: number of times selected for pruning (stats only)
        self._prune_count: Dict[int, int] = {}

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Node allocation
    # ------------------------------------------------------------------
    def new_hidden_node(self) -> int:
        """Allocate a brand new hidden node id (used when no existing split fits)."""
        with self._lock:
            nid = self._next_node_id
            self._next_node_id += 1
            self._node_commonality.setdefault(nid, 0)
            return nid

    def get_or_create_split_node(self, split_innov: int) -> Tuple[int, int, int]:
        """Look up (or create) the universal IDs for splitting `split_innov`.

        Returns (new_node_id, in_innov, out_innov):
            - new_node_id: the hidden node created in the middle
            - in_innov:   innovation of the (src -> new_node) connection
            - out_innov:  innovation of the (new_node -> dst) connection
        """
        with self._lock:
            if split_innov in self._split_to_node:
                return (self._split_to_node[split_innov],
                        self._split_to_conns[split_innov][0],
                        self._split_to_conns[split_innov][1])
            # Find the (src, dst) of the connection being split
            # (we need to iterate to find it; this is O(n) but only on first split)
            src_dst: Optional[Tuple[int, int]] = None
            for (s, d), iv in self._conn_innov.items():
                if iv == split_innov:
                    src_dst = (s, d)
                    break
            if src_dst is None:
                # Connection not registered (shouldn't happen for valid genomes);
                # fall back to a brand new hidden node id and new innovations.
                new_node = self.new_hidden_node()
                in_innov = self._next_innov
                self._next_innov += 1
                out_innov = self._next_innov
                self._next_innov += 1
                self._split_to_node[split_innov] = new_node
                self._split_to_conns[split_innov] = (in_innov, out_innov)
                return new_node, in_innov, out_innov
            src, dst = src_dst
            new_node = self.new_hidden_node()
            in_innov = self.get_or_create_conn_innov(src, new_node)
            out_innov = self.get_or_create_conn_innov(new_node, dst)
            self._split_to_node[split_innov] = new_node
            self._split_to_conns[split_innov] = (in_innov, out_innov)
            return new_node, in_innov, out_innov

    # ------------------------------------------------------------------
    # Connection innovation allocation
    # ------------------------------------------------------------------
    def get_or_create_conn_innov(self, src: int, dst: int) -> int:
        """Return the innovation number for the connection (src -> dst).

        If `forbid_loops` is in effect, callers must guarantee no cycle is
        introduced *before* calling this; the registry itself does not check.
        """
        with self._lock:
            iv = self._conn_innov.get((src, dst))
            if iv is None:
                iv = self._next_innov
                self._next_innov += 1
                self._conn_innov[(src, dst)] = iv
                self._conn_commonality.setdefault(iv, 0)
                self._conn_selected_count.setdefault(iv, 0)
                self._neuron_split_count.setdefault(iv, 0)
                self._prune_count.setdefault(iv, 0)
            return iv

    def conn_innov_of(self, src: int, dst: int) -> Optional[int]:
        """Return the innovation number for (src, dst) if it exists, else None."""
        with self._lock:
            return self._conn_innov.get((src, dst))

    def conn_endpoints(self, innov: int) -> Optional[Tuple[int, int]]:
        """Reverse lookup: innovation number -> (src, dst)."""
        with self._lock:
            for (s, d), iv in self._conn_innov.items():
                if iv == innov:
                    return (s, d)
            return None

    # ------------------------------------------------------------------
    # Usage statistics (used by spec's "least common" / "least selected"
    # selection mechanisms)
    # ------------------------------------------------------------------
    def register_conn_added(self, innov: int) -> None:
        with self._lock:
            self._conn_commonality[innov] = self._conn_commonality.get(innov, 0) + 1
            self._conn_selected_count[innov] = self._conn_selected_count.get(innov, 0) + 1

    def unregister_conn(self, innov: int) -> None:
        with self._lock:
            c = self._conn_commonality.get(innov, 0)
            if c > 0:
                self._conn_commonality[innov] = c - 1

    def register_node_added(self, node_id: int) -> None:
        with self._lock:
            self._node_commonality[node_id] = self._node_commonality.get(node_id, 0) + 1

    def unregister_node(self, node_id: int) -> None:
        with self._lock:
            c = self._node_commonality.get(node_id, 0)
            if c > 0:
                self._node_commonality[node_id] = c - 1

    def register_split_attempt(self, innov: int) -> None:
        """Mark that `innov` was selected for a neuron-split mutation."""
        with self._lock:
            self._neuron_split_count[innov] = self._neuron_split_count.get(innov, 0) + 1

    def register_prune(self, innov: int) -> None:
        with self._lock:
            self._prune_count[innov] = self._prune_count.get(innov, 0) + 1

    # ------------------------------------------------------------------
    # Read-only views for selection mechanisms
    # ------------------------------------------------------------------
    def conn_commonality(self, innov: int) -> int:
        """How many genomes currently contain this connection."""
        return self._conn_commonality.get(innov, 0)

    def conn_selected_count(self, innov: int) -> int:
        """How many times this connection has been selected for creation."""
        return self._conn_selected_count.get(innov, 0)

    def neuron_commonality(self, node_id: int) -> int:
        """How many genomes currently contain this hidden node."""
        return self._node_commonality.get(node_id, 0)

    def neuron_split_count(self, innov: int) -> int:
        """How many times this connection has been selected for splitting."""
        return self._neuron_split_count.get(innov, 0)

    def all_possible_conns(self) -> List[Tuple[int, int]]:
        """All (src, dst) pairs that have ever been registered."""
        with self._lock:
            return list(self._conn_innov.keys())

    def all_node_ids(self) -> List[int]:
        with self._lock:
            return list(range(self._next_node_id))

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "n_nodes": self._next_node_id,
                "n_conn_innovations": len(self._conn_innov),
                "n_split_nodes": len(self._split_to_node),
            }
