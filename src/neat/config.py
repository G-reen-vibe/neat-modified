"""
Configuration for the modified NEAT algorithm.

All hyperparameters are gathered here so they can be tweaked, swept, and
serialized.  Field names follow the spec's conventions:
    X / Y / W / Z  -> continuous multipliers
    P              -> discrete choice (string enum)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import copy


# ---------------------------------------------------------------------------
# Enumerations (the "P" choices in the spec)
# ---------------------------------------------------------------------------

class WeightSelect:
    PCT_SHUFFLED    = "pct_shuffled"        # select X% of weights, shuffled
    INDEPENDENT     = "independent"          # each weight with X% prob


class WeightMod:
    GAUSSIAN = "gaussian"   # std = X
    UNIFORM  = "uniform"     # range [-X, X]
    BERNOULLI = "bernoulli"  # +/- X


class ConnSelect:
    PCT_SHUFFLED                = "pct_shuffled"
    LEAST_COMMON_GLOBAL_SHUFFLED = "least_common_global_shuffled"
    LEAST_SELECTED_GLOBAL_SHUFFLED = "least_selected_global_shuffled"
    LEAST_COMMON_SPECIES         = "least_common_species"
    LEAST_SELECTED_GLOBAL        = "least_selected_global"
    INDEPENDENT                  = "independent"


class NeuronMod:
    INCOMING_ONE   = "incoming_one"     # in=1, out=orig
    OUTGOING_ONE   = "outgoing_one"     # in=orig, out=1


class PruneSelect:
    PCT_SHUFFLED    = "pct_shuffled"
    INDEPENDENT     = "independent"
    INVERSE_ROULETTE = "inverse_roulette"   # higher weight -> lower prob


class CrossoverTopology:
    FITTER     = "fitter"
    MORE_CONNS = "more_conns"
    COMBINE    = "combine"


class CrossoverWeights:
    INDEPENDENT = "independent"
    AVERAGE     = "average"
    BY_NEURON   = "by_neuron"


class MutationPolicyKind:
    NESTED    = "nested"      # nested sub-policies with durations
    PER_TYPE  = "per_type"     # check prob for each type independently
    SINGLE    = "single"       # pick one (or none)


class SpeciationKind:
    SINGLE   = "single"
    STANDARD = "standard"
    PURGE    = "purge"


class SimilarityKind:
    STANDARD   = "standard"
    PERCENTAGE = "percentage"


class ActivationKind:
    UAF      = "uaf"
    PSWISH   = "pswish"
    SIGMOID  = "sigmoid"
    TANH     = "tanh"
    RELU     = "relu"


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

@dataclass
class MutationCfg:
    # Weight mutation
    weight_select: str = WeightSelect.PCT_SHUFFLED
    weight_pct: float = 1.0           # X%
    weight_mod: str = WeightMod.GAUSSIAN
    weight_std: float = 0.05          # X for gaussian / range for uniform / magnitude for bernoulli
    weight_prob: float = 1.0          # used when select = INDEPENDENT

    # Connection mutation
    conn_select: str = ConnSelect.PCT_SHUFFLED
    conn_pct: float = 0.0             # 0 means "floor of 1"
    conn_mod: str = WeightMod.GAUSSIAN
    conn_std: float = 0.2

    # Neuron mutation
    neuron_select: str = ConnSelect.PCT_SHUFFLED
    neuron_pct: float = 0.0           # 0 means floor of 1
    neuron_mod: str = NeuronMod.INCOMING_ONE

    # Pruning mutation
    prune_select: str = PruneSelect.PCT_SHUFFLED
    prune_pct: float = 0.0            # 0 means floor of 1
    prune_prob: float = 0.0           # used when INDEPENDENT


@dataclass
class MutationPolicyCfg:
    kind: str = MutationPolicyKind.PER_TYPE
    # per-type probabilities (used when kind = PER_TYPE)
    prune_prob: float = 0.05
    neuron_prob: float = 0.10
    conn_prob: float = 0.25
    weight_prob: float = 1.0
    # for SINGLE: probabilities of picking each (will be normalized, "none" implicit)
    single_prune: float = 0.05
    single_neuron: float = 0.10
    single_conn: float = 0.25
    single_weight: float = 0.60
    # for NESTED: list of (duration_steps, MutationPolicyCfg-as-dict)
    nested_schedule: List[Tuple[int, Dict[str, Any]]] = field(default_factory=list)


@dataclass
class GenerationCfg:
    pop_size: int = 100
    asexual_pct: float = 0.75          # X
    crossover_pct: float = 0.25        # Y
    interspecies: int = 1              # W
    elitism: int = 1                   # Z
    cull_pct: float = 0.5              # fraction of each species that cannot reproduce


@dataclass
class CrossoverCfg:
    topology: str = CrossoverTopology.FITTER
    weights: str = CrossoverWeights.AVERAGE


@dataclass
class SpeciationCfg:
    kind: str = SpeciationKind.PURGE      # initial; switches to STANDARD after gen 0
    initial_kind: str = SpeciationKind.PURGE
    subsequent_kind: str = SpeciationKind.STANDARD
    target_species: int = 10
    min_threshold: float = 0.05
    max_threshold: float = 0.5
    threshold_step: float = 0.025
    similarity: str = SimilarityKind.PERCENTAGE
    # standard NEAT similarity weights (only used if similarity == STANDARD)
    c1: float = 1.0     # disjoint coefficient
    c2: float = 1.0     # excess coefficient
    c3: float = 0.4     # weight-diff coefficient
    purge_keep: int = 10   # for PURGE: keep best X genomes
    purge_extra_mutations: int = 3


@dataclass
class OptimizerCfg:
    enabled: bool = False
    lr: float = 0.1
    weight_decay_l2: float = 0.0
    weight_decay_l1: float = 0.0
    method: str = "adam"      # "adam" | "momentum" | "rmsprop" | "sgd"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    momentum: float = 0.9
    rmsprop_decay: float = 0.99


@dataclass
class InitCfg:
    n_conn_multiplier: float = 1.0   # multiplier on the "fully-connected empty" count
    weight_init_multiplier: float = 2.0   # X: scales conn-mutation weight magnitude
    n_neurons_range: Tuple[int, int] = (0, 2)


@dataclass
class Config:
    # Problem shape
    n_inputs: int = 4
    n_outputs: int = 2
    input_activation: str = ActivationKind.RELU     # applied to input layer's pre-activation = the input itself (acts as identity passthrough actually; we just feed raw inputs)
    output_activation: str = ActivationKind.SIGMOID
    hidden_activation: str = ActivationKind.TANH
    bias_enabled: bool = True
    # Default activation for new hidden neurons (can be overridden per-node)
    # Forbid loops & disconnected graphs -> strict DAG
    forbid_loops: bool = True
    # Reproducibility
    seed: int = 0

    init: InitCfg = field(default_factory=InitCfg)
    mutation: MutationCfg = field(default_factory=MutationCfg)
    policy: MutationPolicyCfg = field(default_factory=MutationPolicyCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    crossover: CrossoverCfg = field(default_factory=CrossoverCfg)
    speciation: SpeciationCfg = field(default_factory=SpeciationCfg)
    optimizer: OptimizerCfg = field(default_factory=OptimizerCfg)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cfg = cls()
        cfg.n_inputs = d.get("n_inputs", cfg.n_inputs)
        cfg.n_outputs = d.get("n_outputs", cfg.n_outputs)
        cfg.input_activation = d.get("input_activation", cfg.input_activation)
        cfg.output_activation = d.get("output_activation", cfg.output_activation)
        cfg.hidden_activation = d.get("hidden_activation", cfg.hidden_activation)
        cfg.bias_enabled = d.get("bias_enabled", cfg.bias_enabled)
        cfg.forbid_loops = d.get("forbid_loops", cfg.forbid_loops)
        cfg.seed = d.get("seed", cfg.seed)
        i = d.get("init", {})
        cfg.init = InitCfg(**i)
        m = d.get("mutation", {})
        cfg.mutation = MutationCfg(**m)
        p = d.get("policy", {})
        # nested_schedule is a list of tuples; json converts to lists, convert back
        if "nested_schedule" in p and p["nested_schedule"]:
            p["nested_schedule"] = [tuple(x) for x in p["nested_schedule"]]
        cfg.policy = MutationPolicyCfg(**p)
        g = d.get("generation", {})
        cfg.generation = GenerationCfg(**g)
        x = d.get("crossover", {})
        cfg.crossover = CrossoverCfg(**x)
        s = d.get("speciation", {})
        cfg.speciation = SpeciationCfg(**s)
        o = d.get("optimizer", {})
        cfg.optimizer = OptimizerCfg(**o)
        return cfg

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def copy(self) -> "Config":
        return copy.deepcopy(self)
