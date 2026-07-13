"""
Genome initialization.

Per spec:
    Each Genome is initialized with some number of weights and some number of
    nodes.  Weights and neurons are added by reusing the connection and neuron
    mutation methods.  A separate multiplier, X, can be specified to multiply
    specifically the magnitude of the output of the connection mutation method.
    To ensure that there are both enough connections to split into neurons,
    and that there are no hanging neurons at the end, the initialization is
    partitioned such that an evenly split number of connection mutations is
    done before each neuron mutation.

Defaults: number of weights = total number of possible weights in an empty
genome with no extra nodes, multiplier of 2, and 0-2 neurons.

For an "empty" genome with `n_inputs` inputs (incl. bias) and `n_outputs`
outputs, the total possible weights = (n_inputs + bias) * n_outputs.
"""
from __future__ import annotations
import numpy as np

from .config import Config
from .genome import Genome
from .indexing import GlobalIndex
from .mutations import mutate_conn, mutate_neuron


def initialize_genome(cfg: Config, index: GlobalIndex,
                      rng: np.random.Generator) -> Genome:
    """Build a fresh genome per the spec's initialization procedure."""
    g = Genome(cfg, index)
    # Total possible weights in an empty genome (inputs+bias -> outputs)
    n_in = cfg.n_inputs + (1 if cfg.bias_enabled else 0)
    n_target_conns = int(round(n_in * cfg.n_outputs * cfg.init.n_conn_multiplier))
    n_neurons_target = int(rng.integers(cfg.init.n_neurons_range[0],
                                         cfg.init.n_neurons_range[1] + 1))

    # Partition: we want to interleave connection mutations with neuron
    # mutations.  If n_neurons_target == 0, just do all conns up front.
    # If n_neurons_target > 0, split n_target_conns into n_neurons_target+1
    # roughly-equal chunks, doing one chunk of conns before each neuron
    # mutation, and the final chunk after.
    if n_neurons_target == 0 or n_target_conns == 0:
        # Just add conns
        for _ in range(max(1, n_target_conns)):
            mutate_conn(g, cfg.mutation, rng,
                        weight_init_multiplier=cfg.init.weight_init_multiplier)
        return g

    chunks = n_neurons_target + 1
    per_chunk = max(1, n_target_conns // chunks)
    for i in range(n_neurons_target):
        for _ in range(per_chunk):
            mutate_conn(g, cfg.mutation, rng,
                        weight_init_multiplier=cfg.init.weight_init_multiplier)
        # Try to add a neuron; if it fails (no conns), skip
        mutate_neuron(g, cfg.mutation, rng)
    # Final chunk
    for _ in range(per_chunk):
        mutate_conn(g, cfg.mutation, rng,
                    weight_init_multiplier=cfg.init.weight_init_multiplier)

    # Sanity: if any hidden node ended up with no connections, remove it
    to_remove = [nid for nid, n in g.nodes.items()
                 if n.kind == "hidden" and not g.incoming(nid) and not g.outgoing(nid)]
    for nid in to_remove:
        g.nodes.pop(nid, None)
        index.unregister_node(nid)
    return g
