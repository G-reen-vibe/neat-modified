"""
Innovation / node ID registry.

Implements *universal* historical marking: a connection (in_node, out_node)
always gets the same innovation number, and a node created by splitting a
connection always gets the same node id, regardless of which genome or
species performed the mutation first. This is the standard NEAT innovation
marking extended to be globally consistent.
"""
from __future__ import annotations

import threading
from typing import Dict, Tuple, Iterator, List


class InnovationRegistry:
    """Tracks innovation numbers for connections and node ids globally.

    Thread-safe. A registry instance is shared by an entire Population so
    that all genomes see the same historical markings.
    """

    def __init__(self) -> None:
        # connection (in, out) -> innovation number
        self._conn: Dict[Tuple[int, int], int] = {}
        # node created by splitting connection innovation -> node id
        self._node_from_split: Dict[int, int] = {}
        # counters
        self._next_innov: int = 0
        self._next_node: int = 0
        # global selection stats: how many times each connection innovation
        # has been *selected* for connection / neuron mutation, and how many
        # genomes currently contain it.
        self.selection_count: Dict[int, int] = {}
        self.presence_count: Dict[int, int] = {}
        # how many genomes currently contain each hidden node id
        self.node_presence_count: Dict[int, int] = {}
        self._lock = threading.Lock()

    # ---- node id management ---------------------------------------------
    def new_node_id(self) -> int:
        with self._lock:
            n = self._next_node
            self._next_node += 1
            return n

    def reserve_node_range(self, n: int) -> List[int]:
        """Reserve ``n`` fresh node ids atomically."""
        with self._lock:
            ids = list(range(self._next_node, self._next_node + n))
            self._next_node += n
            return ids

    def reserve_node_ids(self, ids: List[int]) -> None:
        """Mark these node ids as used (e.g. inputs/outputs at init)."""
        with self._lock:
            for i in ids:
                if i >= self._next_node:
                    self._next_node = i + 1

    # ---- innovation management ------------------------------------------
    def get_connection_innov(self, in_node: int, out_node: int) -> int:
        """Return the innovation number for (in_node, out_node), creating if needed."""
        key = (in_node, out_node)
        with self._lock:
            if key not in self._conn:
                self._conn[key] = self._next_innov
                self._next_innov += 1
            return self._conn[key]

    def get_split_node_id(self, conn_innov: int) -> int:
        """Return the node id created by splitting connection ``conn_innov``.

        Always returns the same id for the same connection innovation.
        """
        with self._lock:
            if conn_innov not in self._node_from_split:
                self._node_from_split[conn_innov] = self._next_node
                self._next_node += 1
            return self._node_from_split[conn_innov]

    # ---- selection / presence stats -------------------------------------
    def bump_selection(self, conn_innov: int) -> None:
        with self._lock:
            self.selection_count[conn_innov] = self.selection_count.get(conn_innov, 0) + 1

    def bump_presence(self, conn_innov: int, delta: int) -> None:
        with self._lock:
            self.presence_count[conn_innov] = self.presence_count.get(conn_innov, 0) + delta

    def bump_node_presence(self, node_id: int, delta: int) -> None:
        with self._lock:
            self.node_presence_count[node_id] = self.node_presence_count.get(node_id, 0) + delta

    # ---- introspection ---------------------------------------------------
    def all_connection_innovs(self) -> Iterator[Tuple[Tuple[int, int], int]]:
        # snapshot iterator
        with self._lock:
            items = list(self._conn.items())
        return iter(items)

    def state_dict(self) -> Dict:
        with self._lock:
            return {
                "conn": dict(self._conn),
                "node_from_split": dict(self._node_from_split),
                "next_innov": self._next_innov,
                "next_node": self._next_node,
                "selection_count": dict(self.selection_count),
                "presence_count": dict(self.presence_count),
                "node_presence_count": dict(self.node_presence_count),
            }

    def load_state_dict(self, state: Dict) -> None:
        with self._lock:
            self._conn = dict(state["conn"])
            self._node_from_split = dict(state["node_from_split"])
            self._next_innov = state["next_innov"]
            self._next_node = state["next_node"]
            self.selection_count = dict(state.get("selection_count", {}))
            self.presence_count = dict(state.get("presence_count", {}))
            self.node_presence_count = dict(state.get("node_presence_count", {}))
