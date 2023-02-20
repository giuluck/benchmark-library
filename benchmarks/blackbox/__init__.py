import numpy as np

from benchmarks.benchmark import Benchmark
from datatypes.constraints import CustomConstraint
from datatypes.parameters import NumericParameter
from datatypes.variables import CustomVariable
from utils.decorators import benchmark


@benchmark
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
        CustomVariable('x', dtype=list, description="the input vector")
    ]

    _parameters = [
        NumericParameter('a', default=20., description="the Ackley function 'a' parameter"),
        NumericParameter('b', default=0.2, description="the Ackley function 'b' parameter"),
        NumericParameter('c', default=2 * np.pi, description="the Ackley function 'c' parameter"),
        NumericParameter('dim', default=1, integer=True, description="the input vector dimension")
    ]

    _constraints = [
        CustomConstraint(
            name='input_dim',
            satisfied_fn=lambda x, dim: len(x) == dim,
            description=f"the input vector should have the correct input dimension 'dim'"
        )
    ]

    @staticmethod
    def _query(x: list, a: float, b: float, c: float, dim: int) -> np.ndarray:
        x = np.array(x)
        term1 = a * np.exp(-b * np.sqrt(np.sum(x ** 2) / dim))
        term2 = np.exp(np.sum(np.cos(c * x)) / dim)
        return term1 + term2 - a - np.e

    def __init__(self, name, seed, a, b, c, dim):
        super(Ackley, self).__init__(name=name, seed=seed, a=a, b=b, c=c, dim=dim)

    def query(self, x) -> np.ndarray:
        return super(Ackley, self).query(x=x)
