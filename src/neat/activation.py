"""
Activation functions.

Per spec, available activations are:
    - UAF      (Universal Activation Function - a learnable linear combination
                of {tanh, sigmoid, relu, identity} with softmax weights; here
                we keep the mixing weights as a per-node state)
    - P-Swish  (parametric swish: x * sigmoid(beta * x), beta starts at 0
                which makes it the identity)
    - Sigmoid
    - Tanh
    - ReLU

UAF and P-Swish are *parametric*: the parameter is mutated via weight-style
gaussian perturbation in `Genome.mutate_activations`.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable
import numpy as np


# ---------------------------------------------------------------------------
# Pure activations
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def identity(x: np.ndarray) -> np.ndarray:
    return x


def swish(x: np.ndarray, beta: float) -> np.ndarray:
    return x * sigmoid(beta * x)


# ---------------------------------------------------------------------------
# UAF: linear combination of {tanh, sigmoid, relu, id} with learnable weights
# ---------------------------------------------------------------------------
@dataclass
class UAFParams:
    # raw logits for {tanh, sigmoid, relu, identity}
    w: np.ndarray  # shape (4,)

    @classmethod
    def default(cls) -> "UAFParams":
        return cls(w=np.zeros(4, dtype=np.float64))

    def softmax(self) -> np.ndarray:
        w = np.clip(self.w, -30.0, 30.0)
        e = np.exp(w - w.max())
        return e / e.sum()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        s = self.softmax()
        return s[0] * tanh(x) + s[1] * sigmoid(x) + s[2] * relu(x) + s[3] * x


@dataclass
class PSwishParams:
    beta: float = 0.0   # identity at start

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return swish(x, self.beta)


# ---------------------------------------------------------------------------
# Activation state container (per-node)
# ---------------------------------------------------------------------------
@dataclass
class ActivationState:
    kind: str            # one of ActivationKind
    uaf: UAFParams = None
    pswish: PSwishParams = None

    def __post_init__(self):
        if self.kind == "uaf" and self.uaf is None:
            self.uaf = UAFParams.default()
        if self.kind == "pswish" and self.pswish is None:
            self.pswish = PSwishParams()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "sigmoid":
            return sigmoid(x)
        if self.kind == "tanh":
            return tanh(x)
        if self.kind == "relu":
            return relu(x)
        if self.kind == "uaf":
            return self.uaf(x)
        if self.kind == "pswish":
            return self.pswish(x)
        raise ValueError(f"Unknown activation: {self.kind}")

    def clone(self) -> "ActivationState":
        ns = ActivationState(kind=self.kind)
        if self.uaf is not None:
            ns.uaf = UAFParams(w=self.uaf.w.copy())
        if self.pswish is not None:
            ns.pswish = PSwishParams(beta=self.pswish.beta)
        return ns

    def to_dict(self) -> dict:
        d = {"kind": self.kind}
        if self.uaf is not None:
            d["uaf"] = self.uaf.w.tolist()
        if self.pswish is not None:
            d["pswish"] = self.pswish.beta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ActivationState":
        s = cls(kind=d["kind"])
        if "uaf" in d and d["uaf"] is not None:
            s.uaf = UAFParams(w=np.array(d["uaf"], dtype=np.float64))
        if "pswish" in d and d["pswish"] is not None:
            s.pswish = PSwishParams(beta=float(d["pswish"]))
        return s


def make_activation(kind: str) -> ActivationState:
    return ActivationState(kind=kind)
