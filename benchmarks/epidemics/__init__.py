import numpy as np
import pandas as pd
from descriptors import classproperty
from scipy.integrate import odeint

from benchmarks import Benchmark
from datatypes.constraints import CustomConstraint
from datatypes.parameters import NumericParameter
from datatypes.variables import PositiveVariable
from utils.decorators import benchmark


@benchmark
class SIR(Benchmark):
    """
    Models an epidemics leveraging the SIR model, which is the simplest type of compartmental model.
    The population is divided into three groups (compartments), i.e. Susceptibles (S), Infected (I), and Recovered (R).
    The classical SIR model is dynamic, indeed the size of the three groups evolves over time.
    The evolution of the model is defined by the following system of Ordinary Differential Equations (ODEs):
        dS = -𝛽 * S * I / N
        dI = 𝛽 * S * I / N - 𝛾 * I
        dR = 𝛾 * I
    with N being the total population size, and 𝛽 and 𝛾 the infection and recovery rate, respectively.
    """

    @classproperty
    def _structure(self):
        return [
            # variables
            PositiveVariable('s0', description='the percentage of initial susceptibles'),
            PositiveVariable('i0', description='the percentage of initial infected'),
            PositiveVariable('r0', description='the percentage of initial recovered'),
            PositiveVariable('beta', strict=True, description='the infection rate'),
            PositiveVariable('gamma', strict=True, description='the recovery rate'),
            # parameter
            NumericParameter('horizon', default=300, lb=1, integer=True, description='the timespan of the simulation'),
            # constraint
            CustomConstraint(
                name='percentage',
                satisfied_fn=lambda s0, i0, r0: s0 + i0 + r0 == 1,
                description='the percentages of susceptibles, infected, and recovered must sum up to one'
            )
        ]

    @staticmethod
    def _query(s0, i0, r0, beta, gamma, horizon):
        # define sir dynamics (normalization is not necessary since we assume the population to sum up to 1)
        def sir(s, i, _):
            t1 = beta * s * i
            t2 = gamma * i
            return np.array([-t1, t1 - t2, t2])

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

    def __init__(self, name, seed, horizon):
        super(SIR, self).__init__(name=name, seed=seed, horizon=horizon)

    def query(self, s0, i0, r0, beta, gamma) -> pd.DataFrame:
        return super(SIR, self).query(s0=s0, i0=i0, r0=r0, beta=beta, gamma=gamma)


@benchmark
class SEIR(Benchmark):
    """
    Models an epidemic leveraging the SEIR model.
    The population is divided into three groups (compartments), i.e. Susceptibles (S), Infected (I), and Recovered (R).
    The classical SIR model is dynamic, indeed the size of the three groups evolves over time.
    The evolution of the model is defined by the following system of Ordinary Differential Equations (ODEs):
        dS = -𝛽 * S * I / N
        dI = 𝛽 * S * I / N - 𝛾 * I
        dR = 𝛾 * I
    with N being the total population size, and 𝛽 and 𝛾 the infection and recovery rate, respectively.
    """

    @classproperty
    def _structure(self):
        return [
            # variables
            PositiveVariable('s0', description='the percentage of initial susceptibles'),
            PositiveVariable('e0', description='the percentage of initial exposed'),
            PositiveVariable('i0', description='the percentage of initial infected'),
            PositiveVariable('r0', description='the percentage of initial recovered'),
            PositiveVariable('beta', strict=True, description='the infection rate'),
            PositiveVariable('gamma', strict=True, description='the recovery rate'),
            PositiveVariable('latency', strict=True, description='the exposition latency period'),
            # parameters
            NumericParameter('horizon', default=300, lb=1, integer=True, description='the timespan of the simulation'),
            # constraints
            CustomConstraint(
                name='percentage',
                satisfied_fn=lambda s0, e0, i0, r0: s0 + e0 + i0 + r0 == 1,
                description='the percentages of susceptibles, exposed, infected, and recovered must sum up to one'
            )
        ]

    @staticmethod
    def _query(s0, e0, i0, r0, beta, gamma, latency, horizon):
        # define seir dynamics (normalization is not necessary since we assume the population to sum up to 1)
        def seir(s, e, i, _):
            t1 = beta * s * i
            t2 = latency * e
            t3 = gamma * i
            return np.array([-t1, t1 - t2, t2 - t3, t3])

        # solve ode
        data = odeint(
            func=lambda y, t: seir(*y),
            y0=np.array([s0, e0, i0, r0]),
            t=np.arange(horizon)
        )
        # build dataframe
        data = pd.DataFrame(data=data, index=np.arange(horizon), columns=["S", "E", "I", "R"])
        data.index.rename("time", inplace=True)
        return data

    def __init__(self, name, seed, horizon):
        super(SEIR, self).__init__(name=name, seed=seed, horizon=horizon)

    def query(self, s0, e0, i0, r0, beta, gamma, latency) -> pd.DataFrame:
        return super(SEIR, self).query(s0=s0, e0=e0, i0=i0, r0=r0, beta=beta, gamma=gamma, latency=latency)
