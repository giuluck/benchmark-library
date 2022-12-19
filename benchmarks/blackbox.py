from typing import Tuple, Optional

import numpy as np
import pandas as pd
from skopt import Space
from skopt.sampler import Lhs

from benchmarks.benchmark import Benchmark
from parameters import PositiveParameter, DiscreteParameter, CustomParameter
from utils import num


def ackley_function(x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2 * np.pi) -> np.ndarray:
    """Computes the Ackley function.

    :param x:
        A bi-dimensional numpy array (NxD), where the first dimension is the number of samples to be evaluated and the
        second dimension if the dimensionality of each sample.

    :param a:
        The first parameter of the Ackley function.

    :param b:
        The second parameter of the Ackley function.

    :param c:
        The third parameter of the Ackley function.

    :return:
        The evaluation of the Ackley function in the given N points.
    """
    d = x.shape[1]
    term1 = a * np.exp(-b * np.sqrt(np.sum(x ** 2, axis=-1) / d))
    term2 = np.exp(np.sum(np.cos(c * x), axis=-1) / d)
    return term1 + term2 - a - np.e


class BlackBoxOptimization(Benchmark):
    """Benchmark on Black-box Optimization."""

    ALIAS = "Black-box Optimization"

    DESCRIPTION = """Standard benchmark for black-box function optimization."""

    PARAMETERS = [
        CustomParameter(name="func", description="the black-box function f(x) -> y", default=ackley_function),
        PositiveParameter(name="samples", description="number of initial samples", default=10, strict=True),
        DiscreteParameter(
            name="mode",
            description="method used to compute the initial samples",
            default="lhs",
            categories=["uniform", "lhs", "max_min"]
        ),
        CustomParameter(
            name="ranges",
            description="the range of each feature along each dimension",
            default=[(-5.0, 10.0)],
            in_domain=lambda value: np.all([
                isinstance(v, tuple) and len(v) == 2
                and isinstance(v[0], num)
                and isinstance(v[1], num)
                for v in value
            ]),
            domain='a list of tuples <lb, ub>'
        )
    ]

    def generate(self) -> pd.DataFrame:
        # unpack data
        func = self.config['func']
        samples = self.config['samples']
        mode = self.config['mode']
        ranges = self.config['ranges']
        # sample points
        space = Space(ranges)
        match mode:
            case 'uniform':
                x = space.rvs(n_samples=samples, random_state=self.seed)
            case 'lhs':
                lhs = Lhs(lhs_type="classic", criterion=None)
                x = lhs.generate(space.dimensions, n_samples=samples, random_state=self.seed)
            case 'max_min':
                lhs = Lhs(criterion="maximin", iterations=100)
                x = lhs.generate(space.dimensions, n_samples=samples, random_state=self.seed)
            case _:
                raise RuntimeError("Something went wrong during sampling")
        # build data
        x = np.array(x)
        data = pd.DataFrame(x, columns=[f'x{i + 1}' for i, _ in enumerate(space.dimensions)])
        data['y'] = func(x)
        return data

    def plot(self, figsize: Optional[Tuple[int, int]] = (16, 9)):
        pass
