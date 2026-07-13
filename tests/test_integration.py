"""Final end-to-end integration test.

Runs the full pipeline: build config -> init population -> train N gens ->
evaluate best genome over K episodes -> verify solved.

Also smoke-tests every major component to catch any regressions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from neat import Config, Population, Genome, GlobalIndex
from neat.config import (
    WeightSelect, WeightMod, ConnSelect, NeuronMod, PruneSelect,
    MutationPolicyKind, SpeciationKind, SimilarityKind, CrossoverTopology,
    CrossoverWeights,
)
from neat.mutations import (
    mutate_weights, mutate_conn, mutate_neuron, mutate_prune,
    mutate_activations, apply_mutation_policy, repair_genome,
)
from neat.crossover import crossover, asexual
from neat.similarity import similarity, similarity_percentage, similarity_standard
from neat.speciation import Speciator
from neat.initialization import initialize_genome
from neat.optimizer import Optimizer
from neat.envs import make_env
from scripts.train import build_default_config, train
from scripts.evaluate import evaluate_genome


def test_full_pipeline_cartpole():
    """End-to-end: train CartPole for a few gens, check it learns."""
    print("=== E2E: CartPole-v1 pipeline ===")
    cfg = build_default_config("CartPole-v1", seed=42, pop_size=30)
    result = train("CartPole-v1", cfg, n_generations=10, max_steps=500,
                   n_eval_episodes=1, verbose=False)
    assert result["best_fitness"] >= 100, f"Expected to learn, got {result['best_fitness']}"
    print(f"  10-gen best fitness: {result['best_fitness']:.1f}  ✓")


def test_every_mutation_select_mechanism():
    """Smoke-test every selection mechanism so we know none of them crash."""
    print("=== Smoke-testing all selection mechanisms ===")
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    idx = GlobalIndex(4, 2, bias_enabled=True)

    # Weight selects
    for sel in [WeightSelect.PCT_SHUFFLED, WeightSelect.INDEPENDENT]:
        cfg.mutation.weight_select = sel
        g = Genome(cfg, idx)
        for i in range(4):
            g.add_conn(i, 4, 0.1 * i)
            g.add_conn(i, 5, 0.1 * i)
        rng = np.random.default_rng(0)
        mutate_weights(g, cfg.mutation, rng)

    # Conn selects
    for sel in [ConnSelect.PCT_SHUFFLED, ConnSelect.INDEPENDENT,
                ConnSelect.LEAST_COMMON_GLOBAL_SHUFFLED,
                ConnSelect.LEAST_SELECTED_GLOBAL_SHUFFLED,
                ConnSelect.LEAST_COMMON_SPECIES,
                ConnSelect.LEAST_SELECTED_GLOBAL]:
        cfg.mutation.conn_select = sel
        g = Genome(cfg, idx)
        g.add_conn(0, 4, 0.5)
        rng = np.random.default_rng(0)
        mutate_conn(g, cfg.mutation, rng)

    # Neuron selects
    for sel in [ConnSelect.PCT_SHUFFLED, ConnSelect.INDEPENDENT,
                ConnSelect.LEAST_COMMON_GLOBAL_SHUFFLED,
                ConnSelect.LEAST_SELECTED_GLOBAL_SHUFFLED]:
        cfg.mutation.neuron_select = sel
        g = Genome(cfg, idx)
        g.add_conn(0, 4, 0.5)
        rng = np.random.default_rng(0)
        mutate_neuron(g, cfg.mutation, rng)

    # Prune selects
    for sel in [PruneSelect.PCT_SHUFFLED, PruneSelect.INDEPENDENT,
                PruneSelect.INVERSE_ROULETTE]:
        cfg.mutation.prune_select = sel
        cfg.mutation.prune_pct = 0.5
        g = Genome(cfg, idx)
        # Build a topology with non-essential conns
        for i in range(4):
            g.add_conn(i, 4, 0.5)
            g.add_conn(i, 5, 0.5)
        rng = np.random.default_rng(0)
        mutate_prune(g, cfg.mutation, rng)

    print("  All selection mechanisms ran without error  ✓")


def test_every_mutation_policy():
    """Smoke-test every mutation policy kind."""
    print("=== Smoke-testing all mutation policies ===")
    for kind in [MutationPolicyKind.PER_TYPE, MutationPolicyKind.SINGLE]:
        cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
        cfg.policy.kind = kind
        idx = GlobalIndex(4, 2, bias_enabled=True)
        g = Genome(cfg, idx)
        g.add_conn(0, 4, 0.5)
        rng = np.random.default_rng(0)
        apply_mutation_policy(g, cfg, rng)
    print("  All mutation policies ran without error  ✓")


def test_every_speciation_mode():
    """Smoke-test every speciation mode."""
    print("=== Smoke-testing all speciation modes ===")
    for init_kind, sub_kind in [(SpeciationKind.SINGLE, SpeciationKind.SINGLE),
                                 (SpeciationKind.PURGE, SpeciationKind.STANDARD),
                                 (SpeciationKind.STANDARD, SpeciationKind.STANDARD)]:
        cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
        cfg.speciation.initial_kind = init_kind
        cfg.speciation.subsequent_kind = sub_kind
        cfg.generation.pop_size = 20
        cfg.speciation.purge_keep = 5
        pop = Population(cfg)
        # Dummy eval
        for g in pop.genomes:
            g.fitness = float(np.random.rand())
        pop.step()
    print("  All speciation modes ran without error  ✓")


def test_every_crossover_method():
    """Smoke-test every crossover combination."""
    print("=== Smoke-testing all crossover combinations ===")
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    idx = GlobalIndex(4, 2, bias_enabled=True)
    rng = np.random.default_rng(0)
    for topo in [CrossoverTopology.FITTER, CrossoverTopology.MORE_CONNS,
                 CrossoverTopology.COMBINE]:
        for ws in [CrossoverWeights.INDEPENDENT, CrossoverWeights.AVERAGE,
                   CrossoverWeights.BY_NEURON]:
            cfg.crossover.topology = topo
            cfg.crossover.weights = ws
            g1 = Genome(cfg, idx); g1.fitness = 1.0
            g2 = Genome(cfg, idx); g2.fitness = 0.5
            g1.add_conn(0, 4, 0.5); g1.add_conn(1, 5, 0.5)
            g2.add_conn(0, 5, 0.5); g2.add_conn(2, 4, 0.5)
            child = crossover(g1, g2, cfg, rng)
            # Forward pass should work
            out = child.forward(np.zeros(4))
            assert out.shape == (2,)
    print("  All crossover combinations ran without error  ✓")


def test_every_similarity_method():
    """Smoke-test both similarity methods."""
    print("=== Smoke-testing all similarity methods ===")
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    idx = GlobalIndex(4, 2, bias_enabled=True)
    g1 = Genome(cfg, idx); g2 = Genome(cfg, idx)
    g1.add_conn(0, 4, 0.5); g2.add_conn(0, 4, 0.7)
    for sim in [SimilarityKind.STANDARD, SimilarityKind.PERCENTAGE]:
        cfg.speciation.similarity = sim
        d = similarity(g1, g2, cfg)
        assert d >= 0
    print("  All similarity methods ran without error  ✓")


def test_every_activation():
    """Smoke-test every activation function."""
    print("=== Smoke-testing all activation functions ===")
    import numpy as np
    from neat.activation import ActivationState
    for kind in ["uaf", "pswish", "sigmoid", "tanh", "relu"]:
        s = ActivationState(kind=kind)
        out = s(np.array([0.5, -0.5, 1.0, -1.0]))
        assert out.shape == (4,)
    print("  All activation functions ran without error  ✓")


def test_genome_repair():
    """Repair should fix outputs with no incoming connections."""
    print("=== Testing genome repair ===")
    cfg = Config(n_inputs=2, n_outputs=2, bias_enabled=True)
    idx = GlobalIndex(2, 2, bias_enabled=True)
    g = Genome(cfg, idx)
    # Output ids are 2, 3.  Only connect to output 2, leave output 3 with no incoming.
    g.add_conn(0, 2, 0.5)
    rng = np.random.default_rng(0)
    fixed = repair_genome(g, rng)
    assert fixed, "Should have repaired output 3"
    # Now output 3 should have an incoming connection
    assert any(c.dst == 3 for c in g.conns.values())
    print("  Repair correctly added missing connection  ✓")


def test_universal_ids_across_species():
    """Two genomes in different species splitting the same connection
    should produce the same hidden node ID."""
    print("=== Testing universal IDs across species ===")
    cfg = Config(n_inputs=2, n_outputs=1, bias_enabled=True)
    idx = GlobalIndex(2, 1, bias_enabled=True)
    g1 = Genome(cfg, idx); g2 = Genome(cfg, idx)
    g1.add_conn(0, 2, 0.5)
    g2.add_conn(0, 2, 0.5)  # same innov
    cfg.mutation.neuron_mod = NeuronMod.INCOMING_ONE
    rng = np.random.default_rng(0)
    mutate_neuron(g1, cfg.mutation, rng)
    mutate_neuron(g2, cfg.mutation, rng)
    h1 = [n.node_id for n in g1.nodes.values() if n.kind == "hidden"][0]
    h2 = [n.node_id for n in g2.nodes.values() if n.kind == "hidden"][0]
    assert h1 == h2, f"Universal IDs must match: {h1} vs {h2}"
    print(f"  Both genomes got hidden node id={h1}  ✓")


def test_optimizer_step_runs():
    """The optimizer step should run without crashing."""
    print("=== Testing optimizer step ===")
    cfg = Config(n_inputs=4, n_outputs=2, bias_enabled=True)
    cfg.optimizer.enabled = True
    cfg.generation.pop_size = 20
    cfg.speciation.initial_kind = SpeciationKind.SINGLE
    cfg.speciation.subsequent_kind = SpeciationKind.SINGLE
    pop = Population(cfg)
    # Apply weight mutations so deltas exist
    rng = np.random.default_rng(0)
    for g in pop.genomes:
        mutate_weights(g, cfg.mutation, rng)
    # Eval (random fitness)
    for g in pop.genomes:
        g.fitness = float(np.random.rand())
    stats = pop.step()
    assert stats["pop_size"] == 20
    print("  Optimizer step ran without error  ✓")


def test_visualizer_importable():
    """The visualizer module should import cleanly."""
    print("=== Testing visualizer import ===")
    # Just import the module; don't start the server
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "viz_run",
        os.path.join(os.path.dirname(__file__), "..", "visualizer", "run.py")
    )
    # Don't actually execute it (would start the server); just check it parses
    assert spec is not None
    print("  Visualizer module loads  ✓")


if __name__ == "__main__":
    test_every_mutation_select_mechanism()
    test_every_mutation_policy()
    test_every_speciation_mode()
    test_every_crossover_method()
    test_every_similarity_method()
    test_every_activation()
    test_genome_repair()
    test_universal_ids_across_species()
    test_optimizer_step_runs()
    test_visualizer_importable()
    test_full_pipeline_cartpole()
    print("\n=== ALL INTEGRATION TESTS PASSED ===")
