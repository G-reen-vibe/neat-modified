"""
GRPO-style optimizer (OpenAI ES inspired) for within-species weight optimization.

For each genome in a species, compute its relative improvement:
  z_i = (reward_i - mean(species)) / std(species)

The partial gradient contributed by genome i is:
  pg_i = weight_delta_i * z_i   (the delta from its last weight mutation)

The applied gradient to genome g is:
  grad_g = sum over i: similarity(g, i) * pg_i
  normalized by total_mass = sum over i: similarity(g, i)

Then weights are updated:
  w_g = w_g + lr * weight_std * grad_g / total_mass  - weight_decay * w_g

Optionally with ADAM/Momentum/RMSProp on the per-weight moments.

The moments m and v are stored per-weight in genome.m / genome.v, and
inherited/averaged during crossover.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .genome import Genome
from .similarity import similarity
from . import mutations as M


class GRPOOptimizer:
    def __init__(
        self,
        enabled: bool = True,
        lr: float = 0.1,
        weight_std: float = 0.05,  # multiplier for gradient (correlates to Fisher info)
        l2: float = 0.0,
        l1: float = 0.0,
        method: str = "sgd",  # "sgd" | "momentum" | "rmsprop" | "adam"
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        similarity_method: str = "percentage",
    ) -> None:
        self.enabled = enabled
        self.lr = lr
        self.weight_std = weight_std
        self.l2 = l2
        self.l1 = l1
        self.method = method
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.similarity_method = similarity_method
        self._t = 0  # timestep for adam bias correction

    def step(self, species_members: List[Genome]) -> None:
        """Apply GRPO update to all genomes in a species.

        Must be called BEFORE culling and BEFORE reproduction, but AFTER
        evaluation. Also reverts the weight mutation (per spec) if a
        weight_delta was recorded.
        """
        if not self.enabled or len(species_members) < 2:
            # still revert weight mutations if any
            for g in species_members:
                if g.weight_delta:
                    M.revert_last_mutation(g)
            return

        rewards = np.array([g.fitness for g in species_members], dtype=np.float64)
        mu = rewards.mean()
        sigma = rewards.std()
        if sigma < 1e-9:
            sigma = 1.0
        z = (rewards - mu) / sigma  # relative improvement

        # partial gradients per genome
        partial_grads: List[Dict[int, float]] = []
        for g in species_members:
            pg: Dict[int, float] = {}
            if g.weight_delta:
                for innov, delta in g.weight_delta.items():
                    pg[innov] = delta * (rewards[list(species_members).index(g)] - mu) / sigma
            # actually use z directly
            idx = species_members.index(g)
            pg = {innov: delta * z[idx] for innov, delta in g.weight_delta.items()} if g.weight_delta else {}
            partial_grads.append(pg)

        # similarity matrix (genome x genome)
        n = len(species_members)
        sim = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = similarity(species_members[i], species_members[j], method=self.similarity_method)
                s = 1.0 - d  # similarity = 1 - distance
                sim[i, j] = s
                sim[j, i] = s

        # apply gradient to each genome
        self._t += 1
        for i, g in enumerate(species_members):
            # weighted sum of partial grads, weighted by similarity to g
            total_mass = 0.0
            applied: Dict[int, float] = {}
            for j, pg in enumerate(partial_grads):
                if not pg:
                    continue
                s = sim[i, j]
                # mass = sum of |sim| over contributors with non-empty pg
                if s == 0:
                    continue
                for innov, v in pg.items():
                    applied[innov] = applied.get(innov, 0.0) + s * v
            # use total_mass = sum of |sim| over contributors with non-empty pg
            mass = sum(abs(sim[i, j]) for j in range(n) if partial_grads[j])
            if mass < 1e-12:
                mass = 1.0
            # apply
            for innov, grad in applied.items():
                if innov not in g.conns:
                    continue
                update = self.lr * self.weight_std * (grad / mass)
                # ADAM/Momentum/RMSProp on the per-weight moment
                if self.method == "adam":
                    m = g.m.get(innov, 0.0)
                    v = g.v.get(innov, 0.0)
                    m = self.beta1 * m + (1 - self.beta1) * update
                    v = self.beta2 * v + (1 - self.beta2) * (update * update)
                    g.m[innov] = m
                    g.v[innov] = v
                    m_hat = m / (1 - self.beta1 ** self._t)
                    v_hat = v / (1 - self.beta2 ** self._t)
                    update = m_hat / (math.sqrt(v_hat) + self.eps)
                elif self.method == "momentum":
                    m = g.m.get(innov, 0.0)
                    m = self.beta1 * m + update
                    g.m[innov] = m
                    update = m
                elif self.method == "rmsprop":
                    v = g.v.get(innov, 0.0)
                    v = self.beta2 * v + (1 - self.beta2) * (update * update)
                    g.v[innov] = v
                    update = update / (math.sqrt(v) + self.eps)
                # SGD: just use update as is
                # We need to apply to PRE-mutation weight. Since we're going to
                # revert the mutation next, store the update and apply after revert.
                if not hasattr(g, '_pending_grad'):
                    g._pending_grad = {}
                g._pending_grad[innov] = g._pending_grad.get(innov, 0.0) + update

        # revert weight mutations (per spec) - now that we've used the delta.
        # After revert, weights are back to pre-mutation values.
        for g in species_members:
            if g.weight_delta:
                M.revert_last_mutation(g)

        # apply pending gradients to PRE-mutation weights (post-revert)
        for g in species_members:
            if not hasattr(g, '_pending_grad') or not g._pending_grad:
                continue
            for innov, update in g._pending_grad.items():
                if innov not in g.conns:
                    continue
                new_w = g.conns[innov].weight + update
                # weight decay
                if self.l2 > 0:
                    new_w -= self.l2 * g.conns[innov].weight
                if self.l1 > 0:
                    new_w -= self.l1 * (1.0 if g.conns[innov].weight > 0 else -1.0)
                g.conns[innov].weight = new_w
            g._pending_grad = {}
