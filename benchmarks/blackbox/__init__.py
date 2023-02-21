from typing import Optional

import numpy as np

from benchmarks import Benchmark
from datatypes.constraints import GenericConstraint
from datatypes.metrics import ReferenceMetric
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

    The function has its minimum in f(0, ..., 0) = 0.
    """

    _structure = lambda: [
        # variables
        CustomVariable('x', dtype=list, description="the input vector"),
        # parameters
        NumericParameter('a', default=20., description="the Ackley function 'a' parameter"),
        NumericParameter('b', default=0.2, description="the Ackley function 'b' parameter"),
        NumericParameter('c', default=2 * np.pi, description="the Ackley function 'c' parameter"),
        NumericParameter('dim', default=1, integer=True, lb=1, description="the input vector dimension"),
        # constraints
        GenericConstraint(
            name='input_dim',
            satisfied_fn=lambda x, dim: len(x) == dim,
            description=f"the input vector should have the correct input dimension 'dim'"
        ),
        # metrics
        ReferenceMetric(
            name='gap',
            metric='mae',
            reference=0.0,
            description='absolute gap from the optimum'
        )
    ]

    @staticmethod
    def _query(x, a, b, c, dim):
        x = np.array(x)
        term1 = a * np.exp(-b * np.sqrt(np.sum(x ** 2) / dim))
        term2 = np.exp(np.sum(np.cos(c * x)) / dim)
        return term1 + term2 - a - np.e

    def __init__(self,
                 name: Optional[str] = None,
                 seed: int = 42,
                 a: float = 20.,
                 b: float = 0.2,
                 c: float = 2 * np.pi,
                 dim: int = 1):
        super(Ackley, self).__init__(name=name, seed=seed, a=a, b=b, c=c, dim=dim)

    def query(self, x: list) -> np.ndarray:
        return super(Ackley, self).query(x=x)


class Rosenbrock(Benchmark):
    """
    The Rosenbrock function (a.k.a. Rosenbrock's valley or Rosenbrock's banana function) is a non-convex function.
    It was introduced by Howard H. Rosenbrock in 1960, and it is used as a performance test for optimization algorithms.
    The global minimum is inside a long, narrow, parabolic shaped flat valley.
    Hence, to find the valley is trivial, while to converge to the global minimum is difficult.

    It can be computed as 'f(x) = t1 + t2', where:
        - t1 = Σ_{i=0}^{d-1} [ b * (x_{i+1} - x_i^2)^2 ]
        - t2 = Σ_{i=0}^{d-1} [ (1 - x_i)^2 ]
        - d is the dimensionality of the vector x
        - b is a parameter whose recommended values is b = 100

    The function has its minimum in f(1, ..., 1) = 0.
    """

    _structure = lambda: [
        # variables
        CustomVariable('x', dtype=list, description="the input vector"),
        # parameters
        NumericParameter('b', default=100., description="the Rosenbrock function 'b' parameter"),
        NumericParameter('dim', default=2, integer=True, lb=2, description="the input vector dimension"),
        # constraints
        GenericConstraint(
            name='input_dim',
            satisfied_fn=lambda x, dim: len(x) == dim,
            description=f"the input vector should have the correct input dimension 'dim'"
        ),
        # metrics
        ReferenceMetric(
            name='gap',
            metric='mae',
            reference=0.0,
            description='absolute gap from the optimum'
        )
    ]

    @staticmethod
    def _query(x, b, dim):
        x = np.array(x)
        term1 = b * (x[1:] - x[:-1] ** 2) ** 2
        term2 = (1 - x[:-1]) ** 2
        return np.sum(term1 + term2)

    def __init__(self,
                 name: Optional[str] = None,
                 seed: int = 42,
                 b: float = 100.,
                 dim: int = 1):
        super(Rosenbrock, self).__init__(name=name, seed=seed, b=b, dim=dim)

    def query(self, x: list) -> np.ndarray:
        return super(Rosenbrock, self).query(x=x)
