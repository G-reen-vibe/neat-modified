"""Unit tests for initialization, optimizer, and Population."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neat.config import Config, SpeciationKind, OptimizerCfg
from neat.indexing import GlobalIndex
from neat.genome import Genome
from neat.initialization import initialize_genome
from neat.optimizer import Optimizer
from neat.population import Population
from neat.mutations import mutate_weights


def test_initialization_basic():
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    cfg.init.n_conn_multiplier = 1.0
    cfg.init.weight_init_multiplier = 2.0
    cfg.init.n_neurons_range = (0, 2)
    idx = GlobalIndex(4, 2, bias_enabled=True)
    rng = np.random.default_rng(0)
    g = initialize_genome(cfg, idx, rng)
    # At least n_inputs * n_outputs = 8 conns possible
    assert len(g.conns) >= 1
    # Forward pass works
    out = g.forward(np.zeros(4))
    assert out.shape == (2,)


def test_initialization_with_neurons():
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    cfg.init.n_conn_multiplier = 2.0
    cfg.init.n_neurons_range = (2, 2)  # always 2 neurons
    idx = GlobalIndex(4, 2, bias_enabled=True)
    rng = np.random.default_rng(0)
    g = initialize_genome(cfg, idx, rng)
    hidden = [n for n in g.nodes.values() if n.kind == "hidden"]
    assert len(hidden) >= 1, "Should have at least 1 hidden node"
    # No hanging neurons (all hidden should have incoming AND outgoing)
    for n in hidden:
        assert g.incoming(n.node_id), f"Hidden node {n.node_id} has no incoming"
        assert g.outgoing(n.node_id), f"Hidden node {n.node_id} has no outgoing"


def test_population_initialization():
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    cfg.generation.pop_size = 20
    pop = Population(cfg)
    assert len(pop.genomes) == 20
    # All genomes have unique ids
    ids = [g.id for g in pop.genomes]
    assert len(set(ids)) == 20


def test_population_step_runs():
    """End-to-end: evaluate on a dummy task, run a step, check stats."""
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    cfg.generation.pop_size = 20
    cfg.speciation.initial_kind = SpeciationKind.STANDARD
    cfg.speciation.subsequent_kind = SpeciationKind.STANDARD
    cfg.speciation.target_species = 3
    pop = Population(cfg)
    # Dummy eval: number of connections (favors bigger networks)
    def eval_fn(g):
        return float(len(g.conns))
    pop.evaluate(eval_fn)
    stats = pop.step()
    assert stats["pop_size"] == 20
    assert stats["generation"] == 1
    assert "fitness_max" in stats


def test_population_step_with_optimizer():
    """The optimizer step should not crash and should keep weights bounded."""
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    cfg.generation.pop_size = 20
    cfg.speciation.initial_kind = SpeciationKind.SINGLE
    cfg.speciation.subsequent_kind = SpeciationKind.SINGLE
    cfg.optimizer.enabled = True
    cfg.optimizer.lr = 0.05
    cfg.optimizer.method = "adam"
    pop = Population(cfg)
    # Apply a weight mutation manually so deltas exist
    rng = np.random.default_rng(0)
    for g in pop.genomes:
        mutate_weights(g, cfg.mutation, rng)
    # Eval (random fitness proportional to weight magnitude)
    def eval_fn(g):
        return float(sum(abs(c.weight) for c in g.conns.values()))
    pop.evaluate(eval_fn)
    stats = pop.step()
    assert stats["pop_size"] == 20


def test_population_multiple_generations():
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    cfg.generation.pop_size = 30
    cfg.speciation.initial_kind = SpeciationKind.PURGE
    cfg.speciation.subsequent_kind = SpeciationKind.STANDARD
    cfg.speciation.target_species = 5
    cfg.speciation.purge_keep = 5
    pop = Population(cfg)
    def eval_fn(g):
        # Reward = output magnitude for zero input (just to have something)
        out = g.forward(np.zeros(4))
        return float(np.sum(out ** 2))
    for gen in range(5):
        pop.evaluate(eval_fn)
        stats = pop.step()
        print(f"  gen {gen+1}: max={stats['fitness_max']:.3f} species={stats['n_species']}")
    assert pop.generation == 5
    assert len(pop.history) == 5


if __name__ == "__main__":
    test_initialization_basic()
    test_initialization_with_neurons()
    test_population_initialization()
    test_population_step_runs()
    test_population_step_with_optimizer()
    test_population_multiple_generations()
    print("All initialization/optimizer/population tests passed.")
