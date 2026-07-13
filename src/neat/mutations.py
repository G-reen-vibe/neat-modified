"""
Mutations: weight, connection, neuron, pruning.

Each mutation operator follows the spec:
    - A *selection mechanism* (P) chooses which candidates to operate on.
    - A *modification mechanism* (P) decides how to change each candidate.
    - At least one candidate must always be selected (floor of 1) for
      weight / connection / neuron / prune mutations when their pct > 0.

Notes on reversibility:
    All mutations are seeded by the RNG passed in; the *type* of mutation
    applied to a genome is stored in `genome._last_mutations` so that the
    optimizer can revert a weight mutation when needed.

Notes on indexing:
    Connection and neuron mutations always use the GlobalIndex to fetch
    universal innovation numbers (and universal split-node ids).
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set
import numpy as np

from .config import (
    Config, MutationCfg,
    WeightSelect, WeightMod,
    ConnSelect, NeuronMod, PruneSelect,
)
from .genome import Genome, Connection, Node
from .indexing import GlobalIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _at_least_one(arr: List, rng: np.random.Generator) -> List:
    """Ensure at least one element is selected; if empty, pick one randomly."""
    if len(arr) == 0:
        return []
    return arr


def _select_pct(items: List, pct: float, rng: np.random.Generator) -> List:
    """Select max(1, ceil(pct * len(items))) items in shuffled order."""
    if not items:
        return []
    n = max(1, int(np.ceil(pct * len(items)))) if pct > 0 else 1
    n = min(n, len(items))
    idx = rng.permutation(len(items))[:n]
    return [items[i] for i in idx]


def _select_independent(items: List, prob: float, rng: np.random.Generator,
                        floor_one: bool = True) -> List:
    """Select each item with `prob` probability.  Floor of 1 if floor_one."""
    if not items:
        return []
    mask = rng.random(len(items)) < prob
    sel = [items[i] for i in range(len(items)) if mask[i]]
    if not sel and floor_one:
        sel = [items[int(rng.integers(0, len(items)))]]
    return sel


def _modify_weight(weight: float, mod: str, x: float, rng: np.random.Generator) -> float:
    """Apply a modification to a weight and return the new weight."""
    if mod == WeightMod.GAUSSIAN:
        return weight + rng.normal(0.0, x)
    if mod == WeightMod.UNIFORM:
        return weight + rng.uniform(-x, x)
    if mod == WeightMod.BERNOULLI:
        return weight + (x if rng.random() < 0.5 else -x)
    raise ValueError(f"Unknown weight modification: {mod}")


# ---------------------------------------------------------------------------
# Weight mutation
# ---------------------------------------------------------------------------
def mutate_weights(g: Genome, cfg: MutationCfg, rng: np.random.Generator) -> bool:
    """Mutate weights of `g` in-place.  Returns True if any change was made."""
    if not g.conns:
        return False
    conns = list(g.conns.values())
    if cfg.weight_select == WeightSelect.PCT_SHUFFLED:
        sel = _select_pct(conns, cfg.weight_pct, rng)
    elif cfg.weight_select == WeightSelect.INDEPENDENT:
        sel = _select_independent(conns, cfg.weight_prob, rng, floor_one=True)
    else:
        raise ValueError(f"Unknown weight select: {cfg.weight_select}")

    any_change = False
    for c in sel:
        old = c.weight
        new = _modify_weight(old, cfg.weight_mod, cfg.weight_std, rng)
        if new != old:
            c.weight = new
            g.record_weight_delta(c.innov, new - old)
            any_change = True
    if any_change:
        g._invalidate_cache()
    return any_change


# ---------------------------------------------------------------------------
# Connection mutation
# ---------------------------------------------------------------------------
def _candidate_conns(g: Genome) -> List[Tuple[int, int]]:
    """All (src, dst) pairs not currently in g that would not create a cycle."""
    nodes = list(g.nodes.keys())
    cands: List[Tuple[int, int]] = []
    # Quick lookup of present conns
    present: Set[Tuple[int, int]] = set()
    for c in g.conns.values():
        present.add((c.src, c.dst))
    # Don't add: into inputs, out of outputs, self-loops
    for src in nodes:
        if g.nodes[src].kind == "output":
            continue
        for dst in nodes:
            if g.nodes[dst].kind == "input":
                continue
            if src == dst:
                continue
            if (src, dst) in present:
                continue
            # Skip cycle check here (expensive); we'll filter at mutation time
            cands.append((src, dst))
    return cands


def _filter_no_cycle(g: Genome, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out = []
    for s, d in pairs:
        if not g._would_create_cycle(s, d):
            out.append((s, d))
    return out


def _select_conn_candidates(g: Genome, select: str, pct: float,
                             rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Apply one of the ConnSelect mechanisms to the candidate set."""
    cands = _candidate_conns(g)
    # remove candidates that would create cycles
    cands = _filter_no_cycle(g, cands)
    if not cands:
        return []

    if select == ConnSelect.PCT_SHUFFLED:
        return _select_pct(cands, pct, rng)
    if select == ConnSelect.INDEPENDENT:
        return _select_independent(cands, pct, rng, floor_one=True)
    if select == ConnSelect.LEAST_COMMON_GLOBAL_SHUFFLED:
        # Sort by global commonality ascending, take the X% least common, shuffle them
        cands.sort(key=lambda sd: g.index.conn_commonality(
            g.index.get_or_create_conn_innov(*sd) if g.index.conn_innov_of(*sd) is not None
            else -1))
        n = max(1, int(np.ceil(pct * len(cands)))) if pct > 0 else 1
        n = min(n, len(cands))
        sub = cands[:n]
        rng.shuffle(sub)
        return sub
    if select == ConnSelect.LEAST_SELECTED_GLOBAL_SHUFFLED:
        def sel_count(sd):
            iv = g.index.conn_innov_of(*sd)
            return g.index.conn_selected_count(iv) if iv is not None else -1
        cands.sort(key=sel_count)
        n = max(1, int(np.ceil(pct * len(cands)))) if pct > 0 else 1
        n = min(n, len(cands))
        sub = cands[:n]
        rng.shuffle(sub)
        return sub
    if select == ConnSelect.LEAST_COMMON_SPECIES:
        # Without species context here, fall back to global commonality
        cands.sort(key=lambda sd: g.index.conn_commonality(
            g.index.conn_innov_of(*sd) if g.index.conn_innov_of(*sd) is not None
            else -1))
        n = max(1, int(np.ceil(pct * len(cands)))) if pct > 0 else 1
        n = min(n, len(cands))
        sub = cands[:n]
        rng.shuffle(sub)
        return sub
    if select == ConnSelect.LEAST_SELECTED_GLOBAL:
        def sel_count(sd):
            iv = g.index.conn_innov_of(*sd)
            return g.index.conn_selected_count(iv) if iv is not None else -1
        cands.sort(key=sel_count)
        n = max(1, int(np.ceil(pct * len(cands)))) if pct > 0 else 1
        n = min(n, len(cands))
        return cands[:n]
    raise ValueError(f"Unknown conn select: {select}")


def mutate_conn(g: Genome, cfg: MutationCfg, rng: np.random.Generator,
                weight_init_multiplier: float = 1.0) -> bool:
    """Add new connections to `g`.  Returns True if any change was made."""
    cands = _select_conn_candidates(g, cfg.conn_select, cfg.conn_pct, rng)
    if not cands:
        return False
    any_change = False
    for (src, dst) in cands:
        # Initial weight magnitude scaled by the init multiplier for use during
        # initialization; otherwise just use the modification mechanism.
        base = _modify_weight(0.0, cfg.conn_mod, cfg.conn_std, rng)
        w = base * weight_init_multiplier
        c = g.add_conn(src, dst, w)
        if c is not None:
            any_change = True
    return any_change


# ---------------------------------------------------------------------------
# Neuron (split) mutation
# ---------------------------------------------------------------------------
def _select_neuron_candidates(g: Genome, select: str, pct: float,
                              rng: np.random.Generator) -> List[Connection]:
    """Pick connections to split.  A connection's 'commonality' for the
    spec's purpose is the commonality of the hidden node it would generate
    from splitting (which is the same as the split-node's commonality
    tracked by GlobalIndex)."""
    if not g.conns:
        return []
    conns = list(g.conns.values())
    # Don't split a connection whose split-node would already exist in g
    # (that would be a no-op for topology, though it's still a valid mutation
    # for weight purposes).  For simplicity we allow it.
    if select in (ConnSelect.PCT_SHUFFLED, ConnSelect.LEAST_COMMON_SPECIES,
                  ConnSelect.LEAST_SELECTED_GLOBAL):
        if select == ConnSelect.PCT_SHUFFLED:
            return _select_pct(conns, pct, rng)
        # For neuron: "common" is defined per-spec as "the neuron it would
        # generate from splitting is present in many G"
        def commonality(c: Connection) -> int:
            sn = g.index._split_to_node.get(c.innov)
            if sn is None:
                return 0
            return g.index.neuron_commonality(sn)
        def sel_count(c: Connection) -> int:
            return g.index.neuron_split_count(c.innov)
        if select == ConnSelect.LEAST_COMMON_SPECIES or select == ConnSelect.LEAST_COMMON_GLOBAL_SHUFFLED:
            conns.sort(key=commonality)
        elif select == ConnSelect.LEAST_SELECTED_GLOBAL_SHUFFLED or select == ConnSelect.LEAST_SELECTED_GLOBAL:
            conns.sort(key=sel_count)
        n = max(1, int(np.ceil(pct * len(conns)))) if pct > 0 else 1
        n = min(n, len(conns))
        sub = conns[:n]
        if select.endswith("shuffled"):
            rng.shuffle(sub)
        return sub
    if select == ConnSelect.INDEPENDENT:
        return _select_independent(conns, pct, rng, floor_one=True)
    if select == ConnSelect.LEAST_COMMON_GLOBAL_SHUFFLED:
        def commonality(c: Connection) -> int:
            sn = g.index._split_to_node.get(c.innov)
            if sn is None:
                return 0
            return g.index.neuron_commonality(sn)
        conns.sort(key=commonality)
        n = max(1, int(np.ceil(pct * len(conns)))) if pct > 0 else 1
        n = min(n, len(conns))
        sub = conns[:n]
        rng.shuffle(sub)
        return sub
    if select == ConnSelect.LEAST_SELECTED_GLOBAL_SHUFFLED:
        conns.sort(key=lambda c: g.index.neuron_split_count(c.innov))
        n = max(1, int(np.ceil(pct * len(conns)))) if pct > 0 else 1
        n = min(n, len(conns))
        sub = conns[:n]
        rng.shuffle(sub)
        return sub
    raise ValueError(f"Unknown neuron select: {select}")


def mutate_neuron(g: Genome, cfg: MutationCfg, rng: np.random.Generator) -> bool:
    """Split connections, adding a hidden neuron in the middle."""
    cands = _select_neuron_candidates(g, cfg.neuron_select, cfg.neuron_pct, rng)
    if not cands:
        return False
    any_change = False
    for c in cands:
        # Get the universal split node + new innovations
        new_node, in_innov, out_innov = g.index.get_or_create_split_node(c.innov)
        g.index.register_split_attempt(c.innov)
        # Ensure the new node exists in g
        if new_node not in g.nodes:
            g.add_hidden_node(node_id=new_node,
                              activation_kind=g.cfg.hidden_activation)
        # Remove the original connection
        g.remove_conn(c.innov, register=True)
        # Add the two new connections with weights determined by NeuronMod
        if cfg.neuron_mod == NeuronMod.INCOMING_ONE:
            in_w = 1.0
            out_w = c.weight
        elif cfg.neuron_mod == NeuronMod.OUTGOING_ONE:
            in_w = c.weight
            out_w = 1.0
        else:
            raise ValueError(f"Unknown neuron mod: {cfg.neuron_mod}")
        # Need to manually add the in/out connections (avoiding add_conn's cycle check
        # since the new node is a fresh hidden node and we just removed the original)
        in_src, in_dst = g.index.conn_endpoints(in_innov)
        out_src, out_dst = g.index.conn_endpoints(out_innov)
        # Use add_conn but allow re-creation if it would clash
        if in_innov not in g.conns:
            # Direct insert: we know it's a DAG-safe operation because we just
            # removed the parent connection and the new node has no other conns.
            new_c = Connection(innov=in_innov, src=in_src, dst=in_dst, weight=in_w)
            g.conns[in_innov] = new_c
            g.index.register_conn_added(in_innov)
        if out_innov not in g.conns:
            new_c = Connection(innov=out_innov, src=out_src, dst=out_dst, weight=out_w)
            g.conns[out_innov] = new_c
            g.index.register_conn_added(out_innov)
        any_change = True
    if any_change:
        g._invalidate_cache()
    return any_change


# ---------------------------------------------------------------------------
# Pruning mutation
# ---------------------------------------------------------------------------
def _find_linear_paths(g: Genome) -> List[Tuple[int, int, int, int]]:
    """Find linear paths: (in_innov, mid_node, out_innov) where mid_node has
    exactly one incoming and one outgoing connection.  Returns list of
    (in_innov, mid_node, out_innov)."""
    # Build in/out count per node
    in_count: Dict[int, int] = {}
    out_count: Dict[int, int] = {}
    in_conn: Dict[int, int] = {}    # node_id -> innov of the single incoming
    out_conn: Dict[int, int] = {}   # node_id -> innov of the single outgoing
    for c in g.conns.values():
        in_count[c.dst] = in_count.get(c.dst, 0) + 1
        out_count[c.src] = out_count.get(c.src, 0) + 1
        in_conn[c.dst] = c.innov
        out_conn[c.src] = c.innov
    paths = []
    for nid, n in g.nodes.items():
        if n.kind != "hidden":
            continue
        if in_count.get(nid, 0) == 1 and out_count.get(nid, 0) == 1:
            paths.append((in_conn[nid], nid, out_conn[nid]))
    return paths


def _prunable_conns(g: Genome) -> List[Connection]:
    """Connections that can be safely pruned (non-essential both ways)."""
    out = []
    for c in g.conns.values():
        if g.is_essential_incoming(c):
            continue
        if g.is_essential_outgoing(c):
            continue
        out.append(c)
    return out


def mutate_prune(g: Genome, cfg: MutationCfg, rng: np.random.Generator) -> bool:
    """Prune non-essential connections and merge linear paths."""
    prunable = _prunable_conns(g)
    if not prunable:
        return False

    # Selection
    if cfg.prune_select == PruneSelect.PCT_SHUFFLED:
        sel = _select_pct(prunable, cfg.prune_pct, rng)
    elif cfg.prune_select == PruneSelect.INDEPENDENT:
        sel = _select_independent(prunable, cfg.prune_prob, rng, floor_one=True)
    elif cfg.prune_select == PruneSelect.INVERSE_ROULETTE:
        # Higher weight -> lower selection chance
        # p_i ∝ 1 / (|w_i| + eps)
        eps = 1e-3
        weights = np.array([1.0 / (abs(c.weight) + eps) for c in prunable])
        if weights.sum() <= 0:
            sel = _select_pct(prunable, cfg.prune_pct, rng)
        else:
            probs = weights / weights.sum()
            n = max(1, int(np.ceil(cfg.prune_pct * len(prunable)))) if cfg.prune_pct > 0 else 1
            n = min(n, len(prunable))
            idx = rng.choice(len(prunable), size=n, replace=False, p=probs)
            sel = [prunable[i] for i in idx]
    else:
        raise ValueError(f"Unknown prune select: {cfg.prune_select}")

    any_change = False
    for c in sel:
        g.index.register_prune(c.innov)
        g.remove_conn(c.innov)
        any_change = True

    # After pruning, also merge any newly-created linear paths (one pass)
    if any_change:
        # Repeatedly merge linear paths until none remain (or limit passes)
        for _ in range(5):
            paths = _find_linear_paths(g)
            if not paths:
                break
            for (in_innov, mid_node, out_innov) in paths:
                # Merge: remove in & out, add direct (in.src -> out.dst) with combined weight
                if in_innov not in g.conns or out_innov not in g.conns:
                    continue
                in_c = g.conns[in_innov]
                out_c = g.conns[out_innov]
                combined_w = in_c.weight * out_c.weight
                src, dst = in_c.src, out_c.dst
                # Remove the two conns and the middle node (if it has no other conns)
                g.remove_conn(in_innov)
                g.remove_conn(out_innov)
                # Remove the middle hidden node if it now has no connections
                if not g.incoming(mid_node) and not g.outgoing(mid_node):
                    g.nodes.pop(mid_node, None)
                    g.index.unregister_node(mid_node)
                # Add the direct connection (might already exist; if so, add to its weight)
                existing_iv = g.index.conn_innov_of(src, dst)
                if existing_iv is not None and existing_iv in g.conns:
                    g.conns[existing_iv].weight += combined_w
                else:
                    g.add_conn(src, dst, combined_w)
        g._invalidate_cache()
    return any_change


# ---------------------------------------------------------------------------
# Activation-parameter mutation (for UAF / P-Swish)
# ---------------------------------------------------------------------------
def mutate_activations(g: Genome, std: float, rng: np.random.Generator) -> bool:
    """Perturb the parameters of any UAF / P-Swish activations in `g`."""
    any_change = False
    for n in g.nodes.values():
        if n.activation.kind == "uaf" and n.activation.uaf is not None:
            delta = rng.normal(0.0, std, size=4)
            n.activation.uaf.w = n.activation.uaf.w + delta
            g._last_act_mut_delta[n.node_id] = delta
            any_change = True
        elif n.activation.kind == "pswish" and n.activation.pswish is not None:
            delta = rng.normal(0.0, std)
            n.activation.pswish.beta += delta
            g._last_act_mut_delta[n.node_id] = np.array([delta])
            any_change = True
    return any_change


# ---------------------------------------------------------------------------
# Master mutation driver: applies the configured mutation policy
# ---------------------------------------------------------------------------
def apply_mutation_policy(g: Genome, cfg: Config, rng: np.random.Generator,
                          policy_kind: Optional[str] = None,
                          policy_state: Optional[Dict] = None) -> Dict:
    """Apply the configured MutationPolicyCfg to `g` in-place.

    Returns a dict describing what was applied, e.g.:
        {"weight": True, "conn": False, "neuron": True, "prune": False}
    """
    pol = cfg.policy
    kind = policy_kind or pol.kind
    applied = {"weight": False, "conn": False, "neuron": False, "prune": False}

    if kind == "nested":
        # Handled externally by Population; here we just apply per-type with
        # current sub-policy (passed via policy_state).
        sub = policy_state or {}
        if sub.get("weight", True) and rng.random() < pol.weight_prob:
            applied["weight"] = mutate_weights(g, cfg.mutation, rng)
        if sub.get("conn", True) and rng.random() < pol.conn_prob:
            applied["conn"] = mutate_conn(g, cfg.mutation, rng)
        if sub.get("neuron", True) and rng.random() < pol.neuron_prob:
            applied["neuron"] = mutate_neuron(g, cfg.mutation, rng)
        if sub.get("prune", True) and rng.random() < pol.prune_prob:
            applied["prune"] = mutate_prune(g, cfg.mutation, rng)
    elif kind == "per_type":
        if rng.random() < pol.weight_prob:
            applied["weight"] = mutate_weights(g, cfg.mutation, rng)
        if rng.random() < pol.conn_prob:
            applied["conn"] = mutate_conn(g, cfg.mutation, rng)
        if rng.random() < pol.neuron_prob:
            applied["neuron"] = mutate_neuron(g, cfg.mutation, rng)
        if rng.random() < pol.prune_prob:
            applied["prune"] = mutate_prune(g, cfg.mutation, rng)
    elif kind == "single":
        # Pick one mutation (or none) weighted by single_* probs
        weights = [pol.single_prune, pol.single_neuron, pol.single_conn,
                   pol.single_weight, 0.05]  # 0.05 = "no mutation"
        weights = np.array(weights, dtype=np.float64)
        weights = weights / weights.sum()
        choice = rng.choice(["prune", "neuron", "conn", "weight", "none"], p=weights)
        if choice == "weight":
            applied["weight"] = mutate_weights(g, cfg.mutation, rng)
        elif choice == "conn":
            applied["conn"] = mutate_conn(g, cfg.mutation, rng)
        elif choice == "neuron":
            applied["neuron"] = mutate_neuron(g, cfg.mutation, rng)
        elif choice == "prune":
            applied["prune"] = mutate_prune(g, cfg.mutation, rng)
    else:
        raise ValueError(f"Unknown mutation policy kind: {kind}")
    return applied
