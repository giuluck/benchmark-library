import importlib.resources
from typing import Optional, Literal

import joblib
import numpy as np
import pandas as pd
from powerplantsim import Plant
from powerplantsim.plant import RecourseAction
from powerplantsim.plant.execution import SimulationOutput
from powerplantsim.utils.typing import Plan
from sklearn.gaussian_process import GaussianProcessRegressor

from benchmarks.unit_commitment import plants, metrics
from model import Benchmark, Structure, querymethod


class UnitCommitment(Benchmark):
    _MAX_INT: int = 2 ** 32 - 1

    @staticmethod
    def build(structure: Structure):
        # VARIABLES
        structure.add_categorical_parameter(
            name='scenario',
            default='simple',
            categories=['simple', 'medium', 'hard'],
            description='the type of simulated scenario'
        )
        structure.add_numeric_parameter(
            name='day',
            default=1,
            lb=1,
            ub=365,
            integer=True,
            description='the day of the year that is taken into account'
        )
        structure.add_positive_parameter(
            name='prediction_error',
            default=1.0,
            description='the amount of error introduced by wrong predictions'
        )
        structure.add_positive_parameter(
            name='uncertainty_error',
            default=1.0,
            description='the amount of error introduced by stochastic behaviour of the problem'
        )
        structure.add_custom_variable(
            name='plan',
            dtype=dict,
            description='a dictionary <machine | edge: states | flows> that maps each machine (indexed by its name) to '
                        'either a fixed state or an iterable of such which should match the time horizon of the '
                        'simulation, and each edge (indexed by the tuple of source and destination) to either a fixed '
                        'flow or an iterable of such which should match the time horizon of the simulation.'
        )
        structure.add_custom_variable(
            name='action',
            dtype=RecourseAction,
            description='a RecourseAction object that defines the recourse action to apply to the predicted plan to '
                        'generate the actual plan, after the true costs and demands are disclosed.'
        )
        # METRICS
        structure.add_value_metric(
            name='demands_satisfied',
            value='output',
            function=lambda output: float(output.status['satisfied'].all()),
            description='whether all the demands were correctly satisfied (1.0) or not (0.0)'
        )
        structure.add_value_metric(
            name='max_gap',
            value='output',
            function=lambda output: metrics.demands_gap(output=output).max().max(),
            description='the maximal gap between the demands and the supplied commodities'
        )
        structure.add_value_metric(
            name='average_gap',
            value='output',
            function=lambda output: metrics.demands_gap(output=output).mean().mean(),
            description='the average gap between the demands and the supplied commodities'
        )
        structure.add_value_metric(
            name='cost',
            value='output',
            function=lambda output: metrics.costs(output=output).sum().sum(),
            description='the overall cost of the production'
        )

    def _variance(self, index: pd.Series, gp: GaussianProcessRegressor) -> np.ndarray:
        x = pd.DataFrame({
            'holiday': index.map(lambda dt: dt.dayofweek in [5, 6]),
            'day': index.map(lambda dt: dt.dayofyear),
            'hour': index.map(lambda dt: dt.hour)
        })
        return gp.sample_y(x, random_state=self._rng.integers(UnitCommitment._MAX_INT)).squeeze()

    def _plant(self, variance: bool) -> Plant:
        # retrieve data and gaussian process regressor
        with importlib.resources.path('benchmarks.unit_commitment', 'data.csv') as file:
            df = pd.read_csv(file).iloc[(self.day - 1) * 24: self.day * 24]
            df['datetime'] = pd.to_datetime(df['datetime'])
        with importlib.resources.path('benchmarks.unit_commitment', 'gp.joblib') as file:
            gp = joblib.load(file)
        # handle plant
        if self.scenario == 'simple':
            keys = ['heat_demand', 'electricity_selling', 'gas_purchase']
            plant = plants.simple
        elif self.scenario == 'medium':
            keys = ['heat_demand', 'electricity_selling', 'gas_purchase']
            plant = plants.medium
        elif self.scenario == 'hard':
            keys = ['heat_demand', 'cooling_demand', 'electricity_selling', 'electricity_purchase', 'gas_purchase']
            plant = plants.hard
        else:
            raise AssertionError(f"Unknown scenario '{self.scenario}'")
        # build predictions and variance models
        kwargs = dict(horizon=df['datetime'])
        for key in keys:
            # no variance model for gas purchase, otherwise add both prediction and uncertainty error
            if variance and key != 'gas_purchase':
                std = df[f'{key}_std']
                # compute prediction error and add it to the original predictions, then clip to zero
                prediction_errors = self.prediction_error * self._variance(index=df['datetime'], gp=gp) * std
                predictions = np.clip(df[key] + prediction_errors, a_min=0, a_max=float('inf'))
                kwargs[key] = predictions
                # compute uncertainty error, use them to compute the clipped values and return variance via subtraction
                uncertainty_errors = self.uncertainty_error * self._variance(index=df['datetime'], gp=gp) * std
                values = np.clip(predictions + uncertainty_errors, a_min=0, a_max=float('inf'))
                kwargs[f'{key}_variance'] = values - predictions
            else:
                kwargs[key] = df[key]
        # return plant
        return plant(**kwargs)

    @querymethod
    def query(self, plan: Plan, action: RecourseAction) -> SimulationOutput:
        """Runs a simulation up to the time horizon using the given plan.

        :param plan:
            The energetic plan of the power plant defined as vectors of edge flows and machine states. It is a
            dictionary <machine | edge: states | flows> that maps each machine (indexed by its name) to either a fixed
            state or an iterable of such which should match the time horizon of the simulation, and each edge (indexed
            by the tuple of source and destination) to either a fixed flow or an iterable of such which should match
            the time horizon of the simulation.

        :param action:
            A RecourseAction object that defines the recourse action to apply to the predicted plan to generate the
            actual plan, after the true costs and demands are disclosed.

        :return:
            An object containing all the history of the simulation, i.e., the list of flows, states, true prices, true
            demands, and some additional information about the simulation.
        """
        return self._plant(variance=True).run(plan=plan, action=action, progress=True)

    # noinspection PyShadowingNames
    def __init__(self,
                 name: Optional[str] = None,
                 seed: int = 42,
                 scenario: Literal['simple', 'medium', 'hard'] = 'simple',
                 day: int = 1,
                 prediction_error: float = 1.0,
                 uncertainty_error: float = 1.0):
        super(UnitCommitment, self).__init__(
            name=name,
            seed=seed,
            scenario=scenario,
            day=day,
            prediction_error=prediction_error,
            uncertainty_error=uncertainty_error
        )

    @property
    def scenario(self) -> Literal['simple', 'medium', 'hard']:
        return self.configuration['scenario']

    @property
    def day(self) -> int:
        return self.configuration['day']

    @property
    def prediction_error(self) -> float:
        return self.configuration['prediction_error']

    @property
    def uncertainty_error(self) -> float:
        return self.configuration['uncertainty_error']

    @property
    def plant(self) -> Plant:
        return self._plant(variance=False)
