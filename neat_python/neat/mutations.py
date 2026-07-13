"""
Mutations: weight, connection, neuron, pruning.

Each mutation type takes a Genome (and its registry), a config dict, and
a seed. All mutations are deterministic given the seed. Each mutation
records an inverse operation in ``genome._undo_log`` so it can be reverted
(``revert_last_mutation``).

Selection mechanisms are specified by index (P) in the spec; we expose
named constants for clarity.

Configuration is a plain dict so it serializes trivially. Each mutation
type has its own sub-config:
  weight_mut    : { selection: 0|1, pct: float, mod: 0|1|2, mod_param: float }
  connection_mut: { selection: 0..5, pct: float, mod: 0|1|2, mod_param: float }
  neuron_mut    : { selection: 0..5, pct: float, mod: 0|1 }
  pruning_mut   : { selection: 0|1|2, pct: float }
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from .genome import Genome, NodeGene, ConnectionGene


# ----------------------------------------------------- selection constants -
# weight mutation
W_SELECT_PCT_SHUFFLED = 0     # select floor(pct * N) weights in shuffled order
W_SELECT_PROB = 1             # select each weight with pct probability

# connection / neuron mutation selection
C_SELECT_PCT_SHUFFLED = 0
C_SELECT_LEAST_COMMON_GLOBAL = 1
C_SELECT_LEAST_SELECTED_GLOBAL = 2
C_SELECT_LEAST_COMMON_SPECIES = 3
C_SELECT_LEAST_SELECTED_GLOBAL_V2 = 4  # alias of #2 per spec
C_SELECT_PROB = 5

# pruning mutation selection
P_SELECT_PCT_SHUFFLED = 0
P_SELECT_PROB = 1
P_SELECT_PROB_INVERSE_ROULETTE = 2

# modification constants (weight / connection)
M_GAUSSIAN = 0
M_UNIFORM = 1
M_BERNOULLI = 2

# neuron modification
N_SPLIT_INCOMING_ONE = 0  # incoming weight = 1, outgoing = original
N_SPLIT_OUTGOING_ONE = 1  # incoming = original, outgoing = 1

# activation choices (for newly-split hidden nodes)
HIDDEN_ACTIVATIONS = ["p_swish", "sigmoid", "tanh", "relu"]


# ----------------------------------------------------- utility functions ---
def _floor_at_least_one(n: int) -> int:
    return max(1, n)


def _sample_shuffled(items: List, pct: float, rng: random.Random) -> List:
    """Take floor(pct * len(items)) items in shuffled order (>=1 if items non-empty)."""
    if not items:
        return []
    n = _floor_at_least_one(int(math.floor(pct * len(items))))
    n = min(n, len(items))
    shuffled = items[:]
    rng.shuffle(shuffled)
    return shuffled[:n]


def _select_prob(items: List, pct: float, rng: random.Random) -> List:
    """Select each item with prob=pct; force at least 1 if items non-empty."""
    if not items:
        return []
    chosen = [x for x in items if rng.random() < pct]
    if not chosen:
        chosen = [rng.choice(items)]
    return chosen


def _modify_value(value: float, mod: int, mod_param: float, rng: random.Random) -> float:
    if mod == M_GAUSSIAN:
        return value + rng.gauss(0.0, mod_param)
    if mod == M_UNIFORM:
        return value + rng.uniform(-mod_param, mod_param)
    if mod == M_BERNOULLI:
        return value + (mod_param if rng.random() < 0.5 else -mod_param)
    raise ValueError(f"unknown mod {mod}")


def _safe_candidate_pairs(genome: Genome) -> List[Tuple[int, int]]:
    """All (in_node, out_node) pairs that could be added as connections.

    Excludes:
      * self loops
      * pairs that would create a cycle (we require DAG)
      * pairs already present in genome
      * pairs that don't make sense (output->*, *->input, *->bias)
    """
    cands: List[Tuple[int, int]] = []
    input_ids = set(genome.input_node_ids())
    bias_ids = set(genome.bias_node_ids())
    output_ids = set(genome.output_node_ids())
    # nodes that can be sources: anything except outputs
    src_kinds = {"input", "bias", "hidden"}
    # nodes that can be targets: anything except inputs and bias
    tgt_kinds = {"hidden", "output"}
    src_nodes = [nid for nid, n in genome.nodes.items() if n.kind in src_kinds]
    tgt_nodes = [nid for nid, n in genome.nodes.items() if n.kind in tgt_kinds]
    existing = {(c.in_node, c.out_node) for c in genome.conns.values()}
    for s in src_nodes:
        for t in tgt_nodes:
            if s == t:
                continue
            if (s, t) in existing:
                continue
            if genome.creates_cycle(s, t):
                continue
            cands.append((s, t))
    return cands


def _splittable_connections(genome: Genome) -> List[ConnectionGene]:
    """Connections that can be split to add a neuron.

    We allow splitting any enabled connection; the resulting neuron
    is hidden. The created node uses the universal split id.
    """
    return [c for c in genome.conns.values() if c.enabled]


# ----------------------------------------------------- WEIGHT MUTATION -----
def mutate_weights(genome: Genome, cfg: Dict, seed: int) -> bool:
    """Mutate existing weights. Returns True if any change was made."""
    rng = random.Random(seed)
    sel = cfg.get("selection", W_SELECT_PROB)
    pct = float(cfg.get("pct", 1.0))
    mod = cfg.get("mod", M_GAUSSIAN)
    mod_param = float(cfg.get("mod_param", 0.05))

    conns = list(genome.conns.values())
    if not conns:
        return False

    if sel == W_SELECT_PCT_SHUFFLED:
        chosen = _sample_shuffled(conns, pct, rng)
    elif sel == W_SELECT_PROB:
        chosen = _select_prob(conns, pct, rng)
    else:
        raise ValueError(f"unknown weight selection {sel}")

    if not chosen:
        return False

    undo = []
    # record the *delta* applied (for the optimizer: genome.weight_delta)
    delta: Dict[int, float] = {}
    for c in chosen:
        old_w = c.weight
        new_w = _modify_value(old_w, mod, mod_param, rng)
        c.weight = new_w
        delta[c.innov] = new_w - old_w
        undo.append((c.innov, old_w))

    genome.weight_delta = delta  # GRPO optimizer needs this
    genome._undo_log.append(("weight", undo))
    return True


# -------------------------------------------------- CONNECTION MUTATION ----
def _rank_candidates(
    genome: Genome,
    pairs: List[Tuple[int, int]],
    sel: int,
    species_members: Optional[List[Genome]] = None,
) -> List[Tuple[int, int]]:
    """Sort candidate pairs according to the selection mechanism.

    Returns the list sorted so that the *least* common/selected come first.
    """
    reg = genome.registry
    if sel in (C_SELECT_LEAST_COMMON_GLOBAL, C_SELECT_LEAST_COMMON_SPECIES):
        if sel == C_SELECT_LEAST_COMMON_SPECIES and species_members:
            # count presence within species
            counts: Dict[int, int] = {}
            for g in species_members:
                for c in g.conns.values():
                    counts[c.innov] = counts.get(c.innov, 0) + 1
            def key(p):
                innov = reg.get_connection_innov(p[0], p[1])
                return counts.get(innov, 0)
            return sorted(pairs, key=key)
        else:
            def key(p):
                innov = reg.get_connection_innov(p[0], p[1])
                return reg.presence_count.get(innov, 0)
            return sorted(pairs, key=key)
    if sel in (C_SELECT_LEAST_SELECTED_GLOBAL, C_SELECT_LEAST_SELECTED_GLOBAL_V2):
        def key(p):
            innov = reg.get_connection_innov(p[0], p[1])
            return reg.selection_count.get(innov, 0)
        return sorted(pairs, key=key)
    # default: shuffle (handled by caller)
    return pairs


def mutate_add_connection(
    genome: Genome,
    cfg: Dict,
    seed: int,
    species_members: Optional[List[Genome]] = None,
    weight_multiplier: float = 1.0,
) -> bool:
    """Add new connections to the genome."""
    rng = random.Random(seed)
    sel = cfg.get("selection", C_SELECT_PCT_SHUFFLED)
    pct = float(cfg.get("pct", 0.0))
    mod = cfg.get("mod", M_GAUSSIAN)
    mod_param = float(cfg.get("mod_param", 0.2))

    pairs = _safe_candidate_pairs(genome)
    if not pairs:
        return False

    if sel == C_SELECT_PCT_SHUFFLED:
        chosen = _sample_shuffled(pairs, pct, rng)
    elif sel == C_SELECT_PROB:
        chosen = _select_prob(pairs, pct, rng)
    elif sel in (C_SELECT_LEAST_COMMON_GLOBAL,
                 C_SELECT_LEAST_SELECTED_GLOBAL,
                 C_SELECT_LEAST_COMMON_SPECIES,
                 C_SELECT_LEAST_SELECTED_GLOBAL_V2):
        ranked = _rank_candidates(genome, pairs, sel, species_members)
        n = _floor_at_least_one(int(math.floor(pct * len(ranked)))) if pct > 0 else 1
        n = min(n, len(ranked))
        chosen = ranked[:n]
        rng.shuffle(chosen)
    else:
        raise ValueError(f"unknown connection selection {sel}")

    if not chosen:
        return False

    undo = []
    for (s, t) in chosen:
        innov = genome.registry.get_connection_innov(s, t)
        # weight init: draw from mod, then apply multiplier
        w0 = _modify_value(0.0, mod, mod_param, rng) * weight_multiplier
        c = ConnectionGene(innov, s, t, w0, enabled=True)
        genome.add_connection(c)
        genome.registry.bump_selection(innov)
        genome.registry.bump_presence(innov, +1)
        undo.append(("add_conn", innov))

    genome._undo_log.append(("connection", undo))
    return True


# ----------------------------------------------------- NEURON MUTATION -----
def mutate_add_neuron(
    genome: Genome,
    cfg: Dict,
    seed: int,
    species_members: Optional[List[Genome]] = None,
) -> bool:
    """Split a connection to add a neuron."""
    rng = random.Random(seed)
    sel = cfg.get("selection", C_SELECT_PCT_SHUFFLED)
    pct = float(cfg.get("pct", 0.0))
    mod = cfg.get("mod", N_SPLIT_INCOMING_ONE)

    conns = _splittable_connections(genome)
    if not conns:
        return False

    # For neuron selection, "common" is based on the *resulting neuron* -
    # i.e. how many genomes contain the split node id.
    reg = genome.registry

    def split_node_id(c: ConnectionGene) -> int:
        return reg.get_split_node_id(c.innov)

    if sel == C_SELECT_PCT_SHUFFLED:
        chosen = _sample_shuffled(conns, pct, rng)
    elif sel == C_SELECT_PROB:
        chosen = _select_prob(conns, pct, rng)
    elif sel == C_SELECT_LEAST_COMMON_GLOBAL:
        # least common split node globally
        def key(c):
            nid = split_node_id(c)
            return reg.node_presence_count.get(nid, 0)
        ranked = sorted(conns, key=key)
        n = _floor_at_least_one(int(math.floor(pct * len(ranked)))) if pct > 0 else 1
        chosen = ranked[:n]
        rng.shuffle(chosen)
    elif sel == C_SELECT_LEAST_COMMON_SPECIES and species_members:
        # least common split node within species
        node_counts: Dict[int, int] = {}
        for g in species_members:
            for nid in g.nodes:
                if g.nodes[nid].kind == "hidden":
                    node_counts[nid] = node_counts.get(nid, 0) + 1
        def key(c):
            return node_counts.get(split_node_id(c), 0)
        ranked = sorted(conns, key=key)
        n = _floor_at_least_one(int(math.floor(pct * len(ranked)))) if pct > 0 else 1
        chosen = ranked[:n]
        rng.shuffle(chosen)
    elif sel in (C_SELECT_LEAST_SELECTED_GLOBAL, C_SELECT_LEAST_SELECTED_GLOBAL_V2):
        # least selected: by the connection's selection_count
        def key(c):
            return reg.selection_count.get(c.innov, 0)
        ranked = sorted(conns, key=key)
        n = _floor_at_least_one(int(math.floor(pct * len(ranked)))) if pct > 0 else 1
        chosen = ranked[:n]
        rng.shuffle(chosen)
    else:
        raise ValueError(f"unknown neuron selection {sel}")

    if not chosen:
        return False

    undo = []
    for c in chosen:
        # split: disable original, add hidden node, add two new connections
        new_node_id = reg.get_split_node_id(c.innov)
        # activation: pick from HIDDEN_ACTIVATIONS deterministically by seed
        act = HIDDEN_ACTIVATIONS[rng.randrange(len(HIDDEN_ACTIVATIONS))]
        # if the hidden node already exists somewhere, prefer that activation;
        # otherwise we register it locally.
        if new_node_id in genome.nodes:
            # already split in this genome (shouldn't happen, but defensive)
            continue
        node = NodeGene(new_node_id, "hidden", act)
        genome.add_node(node)
        reg.bump_node_presence(new_node_id, +1)

        # determine weights per mod
        if mod == N_SPLIT_INCOMING_ONE:
            in_w = 1.0
            out_w = c.weight
        elif mod == N_SPLIT_OUTGOING_ONE:
            in_w = c.weight
            out_w = 1.0
        else:
            raise ValueError(f"unknown neuron mod {mod}")

        # remove original connection (we don't enable/disable, just remove)
        old_w = c.weight
        del genome.conns[c.innov]
        reg.bump_presence(c.innov, -1)

        # add new connections (universal innov)
        in_innov = reg.get_connection_innov(c.in_node, new_node_id)
        out_innov = reg.get_connection_innov(new_node_id, c.out_node)
        genome.add_connection(ConnectionGene(in_innov, c.in_node, new_node_id, in_w))
        genome.add_connection(ConnectionGene(out_innov, new_node_id, c.out_node, out_w))
        reg.bump_presence(in_innov, +1)
        reg.bump_presence(out_innov, +1)
        reg.bump_selection(c.innov)  # the original connection was selected for split

        undo.append(("split", c.innov, new_node_id, in_innov, out_innov, old_w, c.in_node, c.out_node))

    genome._undo_log.append(("neuron", undo))
    return True


# ----------------------------------------------------- PRUNING MUTATION ----
def _is_essential_incoming(genome: Genome, conn: ConnectionGene) -> bool:
    """A conn is essential-incoming if removing it would leave its target
    with no other incoming enabled connections."""
    others = [c for c in genome.conns.values()
              if c.enabled and c.out_node == conn.out_node and c.innov != conn.innov]
    return len(others) == 0


def _is_essential_outgoing(genome: Genome, conn: ConnectionGene) -> bool:
    others = [c for c in genome.conns.values()
              if c.enabled and c.in_node == conn.in_node and c.innov != conn.innov]
    return len(others) == 0


def _find_linear_path_through(genome: Genome, conn: ConnectionGene) -> Optional[Tuple[ConnectionGene, ConnectionGene]]:
    """If conn is the first half of a linear path (mid node has no other
    connections), return (conn, second_conn). Else None.
    """
    mid = conn.out_node
    # mid must have only one incoming (conn) and one outgoing
    in_conns = [c for c in genome.conns.values() if c.enabled and c.out_node == mid]
    out_conns = [c for c in genome.conns.values() if c.enabled and c.in_node == mid]
    if len(in_conns) == 1 and len(out_conns) == 1:
        return (conn, out_conns[0])
    return None


def mutate_prune(genome: Genome, cfg: Dict, seed: int) -> bool:
    """Prune nonessential connections or merge linear paths."""
    rng = random.Random(seed)
    sel = cfg.get("selection", P_SELECT_PCT_SHUFFLED)
    pct = float(cfg.get("pct", 0.0))

    conns = list(genome.conns.values())
    if not conns:
        return False

    # candidates: conns that are nonessential OR part of a linear path
    candidates: List[ConnectionGene] = []
    for c in conns:
        if not c.enabled:
            continue
        if _is_essential_incoming(genome, c) or _is_essential_outgoing(genome, c):
            # check if it's part of a linear path
            lp = _find_linear_path_through(genome, c)
            if lp is None:
                continue
        candidates.append(c)

    if not candidates:
        return False

    if sel == P_SELECT_PCT_SHUFFLED:
        chosen = _sample_shuffled(candidates, pct, rng)
    elif sel == P_SELECT_PROB:
        chosen = _select_prob(candidates, pct, rng)
    elif sel == P_SELECT_PROB_INVERSE_ROULETTE:
        # inverse roulette: higher weight -> lower selection chance
        # selection prob = 1 / (|w| + eps), normalized
        eps = 1e-3
        inv = np.array([1.0 / (abs(c.weight) + eps) for c in candidates])
        inv = inv / inv.sum()
        idx = np.random.default_rng(seed).choice(len(candidates), size=len(candidates),
                                                  replace=False, p=inv)
        n = _floor_at_least_one(int(math.floor(pct * len(candidates)))) if pct > 0 else 1
        n = min(n, len(candidates))
        chosen = [candidates[i] for i in idx[:n]]
    else:
        raise ValueError(f"unknown pruning selection {sel}")

    if not chosen:
        return False

    undo = []
    reg = genome.registry
    removed_innovs: Set[int] = set()
    for c in chosen:
        # connection may have been removed as part of an earlier merge
        if c.innov not in genome.conns:
            continue
        # re-check current state
        if not c.enabled:
            continue
        # check if linear path - merge
        lp = _find_linear_path_through(genome, c)
        if lp is not None and _is_essential_incoming(genome, c):
            # merge: c1 (a->mid) and c2 (mid->b) -> new (a->b) with weight = c1.w * c2.w
            c1, c2 = lp
            # don't merge if there's already a parallel connection (creates ambiguity)
            existing_pairs = {(x.in_node, x.out_node) for x in genome.conns.values()}
            if (c1.in_node, c2.out_node) in existing_pairs:
                continue
            if genome.creates_cycle(c1.in_node, c2.out_node):
                continue
            new_innov = reg.get_connection_innov(c1.in_node, c2.out_node)
            new_w = c1.weight * c2.weight
            # remove c1, c2, mid node, add new conn
            del genome.conns[c1.innov]
            del genome.conns[c2.innov]
            reg.bump_presence(c1.innov, -1)
            reg.bump_presence(c2.innov, -1)
            removed_innovs.add(c1.innov)
            removed_innovs.add(c2.innov)
            mid = c1.out_node
            mid_was_present = mid in genome.nodes
            mid_act = genome.nodes[mid].activation if mid_was_present else "relu"
            if mid_was_present and genome.nodes[mid].kind == "hidden":
                del genome.nodes[mid]
                reg.bump_node_presence(mid, -1)
            genome.add_connection(ConnectionGene(new_innov, c1.in_node, c2.out_node, new_w))
            reg.bump_presence(new_innov, +1)
            undo.append(("merge", c1.innov, c2.innov, new_innov, mid,
                         c1.in_node, c1.out_node, c2.out_node, c1.weight, c2.weight, mid_act))
        else:
            # re-check nonessential in current state
            if _is_essential_incoming(genome, c) or _is_essential_outgoing(genome, c):
                continue
            # simple removal (nonessential)
            del genome.conns[c.innov]
            reg.bump_presence(c.innov, -1)
            removed_innovs.add(c.innov)
            undo.append(("remove_conn", c.innov, c.in_node, c.out_node, c.weight))

    if not undo:
        return False
    genome._undo_log.append(("prune", undo))
    return True


# ----------------------------------------------------- REVERT --------------
def revert_last_mutation(genome: Genome) -> bool:
    """Undo the most recent mutation recorded in the undo log.

    Operations are reverted in LIFO order so that later modifications
    are undone before earlier ones (important when a later merge uses
    the same innovation as an earlier remove).
    """
    if not genome._undo_log:
        return False
    kind, undo = genome._undo_log.pop()
    reg = genome.registry
    if kind == "weight":
        for innov, old_w in undo:
            if innov in genome.conns:
                genome.conns[innov].weight = old_w
        genome.weight_delta = {}
    elif kind == "connection":
        for entry in undo:
            _, innov = entry
            if innov in genome.conns:
                del genome.conns[innov]
                reg.bump_presence(innov, -1)
    elif kind == "neuron":
        for entry in undo:
            _, orig_innov, new_node_id, in_innov, out_innov, old_w, src, dst = entry
            # remove new connections and node, restore original
            if in_innov in genome.conns:
                del genome.conns[in_innov]
                reg.bump_presence(in_innov, -1)
            if out_innov in genome.conns:
                del genome.conns[out_innov]
                reg.bump_presence(out_innov, -1)
            if new_node_id in genome.nodes:
                del genome.nodes[new_node_id]
                reg.bump_node_presence(new_node_id, -1)
            # restore original
            genome.add_connection(ConnectionGene(orig_innov, src, dst, old_w))
            reg.bump_presence(orig_innov, +1)
    elif kind == "prune":
        # LIFO: undo later operations first to avoid innov-key collisions
        for entry in reversed(undo):
            if entry[0] == "remove_conn":
                _, innov, src, dst, w = entry
                genome.add_connection(ConnectionGene(innov, src, dst, w))
                reg.bump_presence(innov, +1)
            elif entry[0] == "merge":
                _, c1_innov, c2_innov, new_innov, mid, src, _, dst, w1, w2, mid_act = entry
                # remove merged
                if new_innov in genome.conns:
                    del genome.conns[new_innov]
                    reg.bump_presence(new_innov, -1)
                # restore c1, c2, mid node
                from .genome import NodeGene as _NG
                genome.add_connection(ConnectionGene(c1_innov, src, mid, w1))
                genome.add_connection(ConnectionGene(c2_innov, mid, dst, w2))
                reg.bump_presence(c1_innov, +1)
                reg.bump_presence(c2_innov, +1)
                if mid not in genome.nodes:
                    genome.add_node(_NG(mid, "hidden", mid_act))
                    reg.bump_node_presence(mid, +1)
    return True


# attach undo log to Genome at runtime (avoid dataclass churn)
def _ensure_undo_log(genome: Genome) -> None:
    if not hasattr(genome, "_undo_log"):
        genome._undo_log = []
    if not hasattr(genome, "_pending_grad"):
        genome._pending_grad = {}


# monkey patch the Genome class to ensure _undo_log always exists
_orig_init = Genome.__init__
def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self._undo_log = []
    self._pending_grad = {}
Genome.__init__ = _patched_init
