from typing import Optional

import numpy as np

from benchmarks.benchmark import Benchmark


class Ackley(Benchmark):
    """Benchmark on Ackley Function."""

    _package = "ackley"

    def __init__(self,
                 name: Optional[str] = None,
                 seed: int = 42,
                 a: float = 20.0,
                 b: float = 0.2,
                 c: float = 2 * np.pi,
                 dim: int = 1):
        super(Ackley, self).__init__(name=name, seed=seed)
        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.dim: int = dim

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return super(Ackley, self).evaluate(x=x)

    def _eval(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[1] == self.dim, f"The input vector should have dimension {self.dim}, got {x.shape[1]}"
        term1 = self.a * np.exp(-self.b * np.sqrt(np.sum(x ** 2, axis=-1) / self.dim))
        term2 = np.exp(np.sum(np.cos(self.c * x), axis=-1) / self.dim)
        return term1 + term2 - self.a - np.e
