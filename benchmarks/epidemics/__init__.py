import numpy as np
import pandas as pd
from scipy.integrate import odeint

from benchmarks.benchmark import Benchmark
from datatypes.constraints import CustomConstraint
from datatypes.parameters import PositiveParameter, NumericParameter
from datatypes.variables import PositiveVariable
from utils.decorators import benchmark


@benchmark
class EpidemicSIR(Benchmark):
    """
    Let us assume that we are at the early stages of an epidemic.
    We want to do our best to control while we wait for a cure/vaccine.
    Hence, we need to decide which actions to take in order to minimize the total number of infected.
    But we are subjected to a variety of constraints (e.g. socio-economical impact).

    For our epidemics we will rely on a SIR model, which is the simplest type of compartmental model.
    The population is divided into three groups (compartments), i.e. Susceptibles (S), Infected (I), and Recovered (R).
    The classical SIR model is dynamic, indeed the size of the three groups evolves over time.
    The evolution of the model is defined by the following system of Ordinary Differential Equations (ODEs):
        dS = -𝛽 * S * I / N
        dI = 𝛽 * S * I / N - 𝛾 * I
        dR = 𝛾 * I
    with N being the total population size, and 𝛽 and 𝛾 the infection and recovery rate, respectively.
    """

    _package = 'epidemics'

    _variables = [
        PositiveVariable('s0', description='the percentage of initial susceptibles'),
        PositiveVariable('i0', description='the percentage of initial infected'),
        PositiveVariable('r0', description='the percentage of initial recovered'),
        PositiveVariable('beta', strict=True, description='the infection rate')
    ]

    _parameters = [
        PositiveParameter('gamma', default=1. / 14, strict=True, description='the recovery rate'),
        NumericParameter('horizon', default=300, lb=1, integer=True, description='the time horizon of the simulation')
    ]

    _constraints = [
        CustomConstraint(
            name='percentage',
            satisfied_fn=lambda s0, i0, r0: s0 + i0 + r0 == 1,
            description='the percentages of initial susceptibles, infected, and recovered must sum up to one'
        )
    ]

    @staticmethod
    def _query(s0: float, i0: float, r0: float, beta: float, gamma: float, horizon: int) -> pd.DataFrame:
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

    def __init__(self, name, seed, gamma, horizon):
        super(EpidemicSIR, self).__init__(name=name, seed=seed, gamma=gamma, horizon=horizon)

    def query(self, s0, i0, r0, beta) -> pd.DataFrame:
        return super(EpidemicSIR, self).query(s0=s0, i0=i0, r0=r0, beta=beta)
