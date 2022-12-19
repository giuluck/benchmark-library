from typing import Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from benchmarks.benchmark import Benchmark
from parameters import PositiveParameter, IntervalParameter


class EpidemicControl(Benchmark):
    """Benchmark on Epidemic Control Scenarios."""

    ALIAS = "Epidemic Control"

    # TODO: add a final paragraph about the task purposes and metrics.
    DESCRIPTION = """
        Let us assume that we are at the early stages of an epidemic and we want to do our best to control while we
        wait for a cure/vaccine. We need to decide which actions to take in order to minimize the total number of
        infected, but we are subjected to a variety of constraints (e.g. socio-economical impact).
        
        For our epidemics we will rely on a SIR model, which is a type of compartmental model where the population
        is divided into three groups (compartments), i.e. Susceptibles (S), Infected (I), and Recovered (R). The
        classical SIR model is dynamic, indeed the size of the three groups evolves over time according to the
        following system of Ordinary Differential Equations:
            dS = -𝛽 * S * I / N
            dI = 𝛽 * S * I / N - 𝛾 * I
            dR = 𝛾 * I
        with N being the total population size, and 𝛽 and 𝛾 the infection and recovery rate, respectively.
    """.strip("\n").replace("        ", "")

    PARAMETERS = [
        IntervalParameter(name="s0", description="percentage of initial susceptibles", default=0.99, lb=0.0, ub=1.0),
        IntervalParameter(name="i0", description="percentage of initial infected", default=0.01, lb=0.0, ub=1.0),
        IntervalParameter(name="r0", description="percentage of initial recovered", default=0.00, lb=0.0, ub=1.0),
        PositiveParameter(name="beta", description="infection rate", default=0.1, strict=True),
        PositiveParameter(name="gamma", description="recovery rate", default=1.0 / 14, strict=True),
        PositiveParameter(name="horizon", description="time horizon of the simulation", default=300, strict=True)
    ]

    def generate(self) -> pd.DataFrame:
        # unpack configuration
        s0 = self.config['s0']
        i0 = self.config['i0']
        r0 = self.config['r0']
        beta = self.config['beta']
        gamma = self.config['gamma']
        horizon = self.config['horizon']

        # define sir dynamics
        def sir(s, i, r):
            n = sum([s, i, r])
            ds = -beta * s * i / n
            di = beta * s * i / n - gamma * i
            dr = gamma * i
            return np.array([ds, di, dr])

        # solve ode
        data = odeint(
            func=lambda y, t: sir(*y),
            y0=np.array([s0, i0, r0]),
            t=np.arange(horizon)
        )
        # build dataframe
        data = pd.DataFrame(data=data, index=np.arange(horizon), columns=["S", "I", "R"])
        data.index.rename("time", inplace=True)
        return data

    def plot(self, figsize: Optional[Tuple[int, int]] = (16, 9)):
        plt.figure(figsize=figsize)
        data = self.data.melt(var_name="Compartment", value_name="%", ignore_index=False).reset_index()
        sns.lineplot(data, x="time", y="%", hue="Compartment", hue_order=["S", "I", "R"])
        plt.tight_layout()
        plt.show()
