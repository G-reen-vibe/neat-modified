"""
Modified NEAT implementation per the user's spec.

Modules:
- config:        All tunable parameters
- indexing:      Global node-ID registry (universal historical IDs)
- activation:    UAF, P-Swish, Sigmoid, Tanh, ReLU
- genome:        Genome class + DAG forward pass via topological sort
- mutations:     Weight / connection / neuron / pruning mutators
- crossover:     3 topology methods x 3 weight selection methods
- similarity:    Standard NEAT + Percentage similarity
- speciation:    Single / Standard (adaptive) / Purge
- population:    Generation policy + reproduction
- optimizer:     Optional OpenAI-ES-style GRPO optimizer
- policy:        Mutation policy (nested / per-type / single-pick)

Public API:
    from neat import Config, Genome, GlobalIndex, Population
"""
from .config import Config
from .genome import Genome
from .indexing import GlobalIndex
from .population import Population

__all__ = ["Config", "Genome", "GlobalIndex", "Population"]
__version__ = "0.1.0"
