"""
Crossover operations.

Per spec, three topology methods:
    1. FITTER     - use the fitter genome's topology
    2. MORE_CONNS - use the topology of the genome with more connections
    3. COMBINE    - attempt to combine both (cycle-break to keep DAG)

And three neuron/weight selection methods:
    1. INDEPENDENT - copy each weight from one network or the other
    2. AVERAGE     - average all possible shared weights
    3. BY_NEURON   - for a given neuron, copy all outgoing weights from one
                     network or the other (besides disjoints)

Optimizer state (m, v) on each connection is averaged across the two parents
when both have it, or copied from whichever parent has it.
"""
from __future__ import annotations
from typing import List, Set, Dict, Tuple, Optional
import numpy as np

from .config import Config, CrossoverCfg, CrossoverTopology, CrossoverWeights
from .genome import Genome, Connection, Node
from .indexing import GlobalIndex


def _gather_topology(g: Genome) -> Set[int]:
    """Set of innovation numbers present in g."""
    return set(g.conns.keys())


def _cycle_break_combine(g1: Genome, g2: Genome) -> Set[int]:
    """Combine the topologies of g1 and g2, breaking cycles to keep DAG.

    Strategy: start from g1's conns, then add g2's conns one by one,
    skipping any that would create a cycle in the combined graph.
    """
    # Build a working set of edges (src, dst) from g1
    edges: Set[Tuple[int, int]] = set()
    innovs: Set[int] = set(g1.conns.keys())
    for c in g1.conns.values():
        edges.add((c.src, c.dst))
    # Now try to add g2's conns
    for c in g2.conns.values():
        if c.innov in innovs:
            continue
        if (c.src, c.dst) in edges:
            continue
        # Cycle check: is there a path from c.dst to c.src in `edges`?
        if _has_path(edges, c.dst, c.src):
            continue
        edges.add((c.src, c.dst))
        innovs.add(c.innov)
    return innovs


def _has_path(edges: Set[Tuple[int, int]], src: int, dst: int) -> bool:
    """BFS - is there a path src -> dst through `edges`?"""
    if src == dst:
        return True
    # Build adjacency
    adj: Dict[int, List[int]] = {}
    for (s, d) in edges:
        adj.setdefault(s, []).append(d)
    stack = [src]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur == dst:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj.get(cur, []))
    return False


def crossover(g1: Genome, g2: Genome, cfg: Config, rng: np.random.Generator) -> Genome:
    """Produce a child genome from g1 and g2.

    g1 should be the *fitter* of the two (callers should pass them in that
    order, though the FITTER topology method will pick based on .fitness).
    """
    cc = cfg.crossover
    idx = g1.index
    # Ensure g1 is the fitter one (for FITTER method's determinism)
    if g1.fitness < g2.fitness:
        g1, g2 = g2, g1

    # ---- 1. Topology selection ------------------------------------------
    if cc.topology == CrossoverTopology.FITTER:
        topo_innovs = set(g1.conns.keys())
        primary = g1
    elif cc.topology == CrossoverTopology.MORE_CONNS:
        if len(g2.conns) > len(g1.conns):
            topo_innovs = set(g2.conns.keys())
            primary = g2
        else:
            topo_innovs = set(g1.conns.keys())
            primary = g1
    elif cc.topology == CrossoverTopology.COMBINE:
        topo_innovs = _cycle_break_combine(g1, g2)
        primary = g1
    else:
        raise ValueError(f"Unknown crossover topology: {cc.topology}")

    # ---- 2. Build child -----------------------------------------------
    child = Genome(cfg, idx)
    # Carry over hidden nodes from both parents (so we don't lose nodes
    # that the topology's conns reference).  Inputs/outputs/bias already
    # exist in child.
    for nid, n in {**g1.nodes, **g2.nodes}.items():
        if n.kind == "hidden" and nid not in child.nodes:
            # Use the activation from whichever parent has it (g1 preferred)
            donor = g1 if nid in g1.nodes else g2
            child.nodes[nid] = Node(node_id=nid, kind="hidden",
                                    activation=donor.nodes[nid].activation.clone())
            child.index.register_node_added(nid)

    # ---- 3. Weight selection ------------------------------------------
    for innov in topo_innovs:
        c1 = g1.conns.get(innov)
        c2 = g2.conns.get(innov)
        if c1 is None and c2 is None:
            continue
        # Determine weight + optimizer state
        if cc.weights == CrossoverWeights.AVERAGE:
            if c1 is not None and c2 is not None:
                w = 0.5 * (c1.weight + c2.weight)
                m = 0.5 * (c1.m + c2.m)
                v = 0.5 * (c1.v + c2.v)
            elif c1 is not None:
                w, m, v = c1.weight, c1.m, c1.v
            else:
                w, m, v = c2.weight, c2.m, c2.v
        elif cc.weights == CrossoverWeights.INDEPENDENT:
            # Pick one parent for the entire connection
            if c1 is not None and c2 is not None:
                donor = c1 if rng.random() < 0.5 else c2
            elif c1 is not None:
                donor = c1
            else:
                donor = c2
            w, m, v = donor.weight, donor.m, donor.v
        elif cc.weights == CrossoverWeights.BY_NEURON:
            # Per-neuron: pick a donor for the source node's outgoing weights.
            # For simplicity here, decide per-connection but seed the choice
            # by the src node id so all outgoing conns from the same neuron
            # come from the same parent.
            src = (c1 or c2).src
            # Hash the src node id to a parent (deterministic per src)
            donor = c1 if (hash((src,)) & 1) == 0 else c2
            if donor is None:
                donor = c2 if c1 is None else c1
            w, m, v = donor.weight, donor.m, donor.v
        else:
            raise ValueError(f"Unknown crossover weights: {cc.weights}")

        # Insert the connection (we know it's DAG-safe because both parents
        # are DAGs and the topology method preserved DAG-ness).
        src = (c1 or c2).src
        dst = (c1 or c2).dst
        # If src or dst isn't in child's nodes (e.g. a hidden node we
        # missed), add it
        if src not in child.nodes or dst not in child.nodes:
            continue
        new_c = Connection(innov=innov, src=src, dst=dst, weight=float(w),
                           m=float(m), v=float(v))
        child.conns[innov] = new_c
        child.index.register_conn_added(innov)

    child._invalidate_cache()
    # Inherit parent species from the fitter parent
    child.parent_species = g1.parent_species
    return child


# ---------------------------------------------------------------------------
# Asexual reproduction (just a clone + mutation, applied by Population)
# ---------------------------------------------------------------------------
def asexual(g: Genome, cfg: Config, rng: np.random.Generator) -> Genome:
    """Clone `g` and apply the mutation policy.  Returns the child."""
    from .mutations import apply_mutation_policy
    child = g.clone()
    apply_mutation_policy(child, cfg, rng)
    return child
