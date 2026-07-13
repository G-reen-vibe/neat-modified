"""Unit tests for speciation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from neat.config import Config, SpeciationKind, SimilarityKind
from neat.indexing import GlobalIndex
from neat.genome import Genome
from neat.speciation import Speciator
from neat.mutations import mutate_conn


def make_genomes(n=10, n_inputs=2, n_outputs=1):
    cfg = Config(n_inputs=n_inputs, n_outputs=n_outputs, bias_enabled=True)
    idx = GlobalIndex(n_inputs, n_outputs, bias_enabled=True)
    genomes = []
    for i in range(n):
        g = Genome(cfg, idx)
        g.id = i
        g.fitness = float(i)
        # Add some random connections so they're not identical
        g.add_conn(i % n_inputs, n_inputs, 0.1 * i + 0.1)
        if i >= n_inputs:
            g.add_conn((i + 1) % n_inputs, n_inputs, 0.2 * i + 0.1)
        genomes.append(g)
    return cfg, genomes


def test_speciate_single():
    cfg, genomes = make_genomes(5)
    cfg.speciation.initial_kind = SpeciationKind.SINGLE
    cfg.speciation.subsequent_kind = SpeciationKind.SINGLE
    sp = Speciator(cfg)
    sp.speciate(genomes)
    assert len(sp.species) == 1
    assert len(next(iter(sp.species.values())).members) == 5


def test_speciate_purge():
    cfg, genomes = make_genomes(20)
    cfg.speciation.initial_kind = SpeciationKind.PURGE
    cfg.speciation.purge_keep = 5
    cfg.speciation.purge_extra_mutations = 1
    sp = Speciator(cfg)
    sp.speciate(genomes)
    assert len(sp.species) == 5
    # Each species should have exactly 1 member (the original kept genome)
    for s in sp.species.values():
        assert len(s.members) == 1


def test_speciate_standard_creates_species():
    """Standard speciation with a high threshold puts all in one species;
    a low threshold creates many."""
    cfg, genomes = make_genomes(20, n_inputs=4, n_outputs=2)
    cfg.speciation.initial_kind = SpeciationKind.STANDARD
    cfg.speciation.subsequent_kind = SpeciationKind.STANDARD
    cfg.speciation.similarity = SimilarityKind.PERCENTAGE
    # Force a high threshold => everyone in one species
    cfg.speciation.min_threshold = 5.0
    cfg.speciation.max_threshold = 5.0
    sp = Speciator(cfg)
    sp.threshold = 5.0
    sp.speciate(genomes)
    assert len(sp.species) == 1, f"high threshold => 1 species, got {len(sp.species)}"

    # Force a tiny threshold => many species
    cfg.speciation.min_threshold = 0.0
    cfg.speciation.max_threshold = 0.0
    sp2 = Speciator(cfg)
    sp2.threshold = 0.0
    sp2.speciate(genomes)
    assert len(sp2.species) > 1, f"low threshold => many species, got {len(sp2.species)}"


def test_speciate_standard_adapts_threshold():
    cfg, genomes = make_genomes(20, n_inputs=4, n_outputs=2)
    cfg.speciation.initial_kind = SpeciationKind.STANDARD
    cfg.speciation.subsequent_kind = SpeciationKind.STANDARD
    cfg.speciation.similarity = SimilarityKind.PERCENTAGE
    cfg.speciation.target_species = 5
    cfg.speciation.min_threshold = 0.0
    cfg.speciation.max_threshold = 1.0
    cfg.speciation.threshold_step = 0.05
    sp = Speciator(cfg)
    # Run several generations and check threshold moves toward target
    for _ in range(5):
        sp.speciate(genomes)
        sp.advance_generation()
        # After adaptive adjustment, threshold should be within bounds
        assert 0.0 <= sp.threshold <= 1.0


if __name__ == "__main__":
    test_speciate_single()
    test_speciate_purge()
    test_speciate_standard_creates_species()
    test_speciate_standard_adapts_threshold()
    print("All speciation tests passed.")
