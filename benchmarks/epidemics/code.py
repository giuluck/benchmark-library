from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import odeint

from benchmarks.benchmark import Benchmark


class EpidemicControl(Benchmark):
    """Benchmark on Epidemic Control Scenarios."""

    _package = 'epidemics'

    def __init__(self, name: Optional[str] = None, seed: int = 42, gamma: float = 1. / 14, horizon: int = 300):
        super(EpidemicControl, self).__init__(name=name, seed=seed)
        self.gamma: float = gamma
        self.horizon: int = horizon

    def evaluate(self, s0: float, i0: float, r0: float, beta: float) -> pd.DataFrame:
        return super(EpidemicControl, self).evaluate(s0=s0, i0=i0, r0=r0, beta=beta)

    def _eval(self, s0: float, i0: float, r0: float, beta: float) -> pd.DataFrame:
        # define sir dynamics
        def sir(s, i, r):
            n = sum([s, i, r])
            ds = -beta * s * i / n
            di = beta * s * i / n - self.gamma * i
            dr = self.gamma * i
            return np.array([ds, di, dr])

        # solve ode
        data = odeint(
            func=lambda y, t: sir(*y),
            y0=np.array([s0, i0, r0]),
            t=np.arange(self.horizon)
        )
        # build dataframe
        data = pd.DataFrame(data=data, index=np.arange(self.horizon), columns=["S", "I", "R"])
        data.index.rename("time", inplace=True)
        return data
