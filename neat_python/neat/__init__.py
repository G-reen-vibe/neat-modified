"""NEAT-Modified package."""
from .registry import InnovationRegistry
from .genome import Genome, NodeGene, ConnectionGene
from .network import forward, topological_sort, act_discrete, act_continuous
from . import mutations
from .crossover import crossover, T_FITTER, T_MORE_CONNS, T_COMBINE, W_INDEPENDENT, W_AVERAGE, W_BY_NEURON
from .similarity import similarity, similarity_standard, similarity_percentage
from .speciation import Speciator, Species
from .optimizer import GRPOOptimizer
from .population import Population, MutationPolicy, MP_NESTED, MP_PER_TYPE_PROB, MP_SINGLE_PICK

__all__ = [
    "InnovationRegistry", "Genome", "NodeGene", "ConnectionGene",
    "forward", "topological_sort", "act_discrete", "act_continuous",
    "mutations", "crossover",
    "T_FITTER", "T_MORE_CONNS", "T_COMBINE",
    "W_INDEPENDENT", "W_AVERAGE", "W_BY_NEURON",
    "similarity", "similarity_standard", "similarity_percentage",
    "Speciator", "Species", "GRPOOptimizer",
    "Population", "MutationPolicy", "MP_NESTED", "MP_PER_TYPE_PROB", "MP_SINGLE_PICK",
]
