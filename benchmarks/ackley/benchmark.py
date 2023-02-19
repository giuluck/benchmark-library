from typing import Optional

import numpy as np

from benchmarks.benchmark import Benchmark
from datatypes.parameters import NumericParameter
from datatypes.variables import CustomVariable


class Ackley(Benchmark):
    """
    The Ackley function is a non-convex function used as a performance test problem for optimization algorithms.
    It was proposed by David Ackley in his 1987 PhD dissertation.

    It can be computed as 'f(x) = t1 + t2 - a - e', where:
        - t1 = a * e^[ -b * √(Σ_i x_i ** 2 / d) ]
        - t2 = e^[ Σ_i cos(c * x_i) / d ]
        - d is the dimensionality of the vector x
        - e is the euler number
        - a, b, c are three parameter, whose recommended values are a = 20, b = 0.2 and c = 2π

    The function has its minimum in the vector 0_d.
    """

    _package = "ackley"

    _variables = [
        CustomVariable('x', dtype=np.ndarray, description="the input vector")
    ]

    _parameters = [
        NumericParameter('a', default=20., description="the Ackley function 'a' parameter"),
        NumericParameter('b', default=0.2, description="the Ackley function 'b' parameter"),
        NumericParameter('c', default=2 * np.pi, description="the Ackley function 'c' parameter"),
        NumericParameter('dim', default=1, integer=True, description="the input vector dimension"),
    ]

    def __init__(self,
                 name: Optional[str] = None,
                 seed: int = 42,
                 a: float = 20.0,
                 b: float = 0.2,
                 c: float = 2 * np.pi,
                 dim: int = 1):
        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.dim: int = dim
        super(Ackley, self).__init__(name=name, seed=seed)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return super(Ackley, self).evaluate(x=x)

    def _eval(self, x: np.ndarray) -> np.ndarray:
        assert len(x) == self.dim, f"The input vector should have dimension {self.dim}, got {len(x)}"
        term1 = self.a * np.exp(-self.b * np.sqrt(np.sum(x ** 2) / self.dim))
        term2 = np.exp(np.sum(np.cos(self.c * x)) / self.dim)
        return term1 + term2 - self.a - np.e
