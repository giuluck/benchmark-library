from typing import Optional, Literal

import pandas as pd
from ppsim.plant import RecourseAction
from ppsim.plant.execution import SimulationOutput
from ppsim.utils.typing import Plan

from benchmarks.unit_commitment import plants, metrics
from model import Benchmark, Structure, querymethod


class UnitCommitment(Benchmark):

    @staticmethod
    def build(structure: Structure):
        # VARIABLES
        structure.add_categorical_parameter(
            name='scenario',
            default='simple',
            categories=['simple', 'medium', 'hard'],
            description='the type of simulated scenario'
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
            name='average_recourse',
            value='output',
            function=lambda output: output.status['recourse'].astype(int).mean(),
            description='the average number of timestamps where a recourse action was invoked'
        )
        structure.add_value_metric(
            name='cost',
            value='output',
            function=lambda output: metrics.costs(output=output).sum().sum(),
            description='the overall cost of the production'
        )

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
        # TODO: load correct data and process them
        df = pd.DataFrame(columns=[
            'heat demand',
            'cooling demand',
            'electricity selling',
            'electricity purchase',
            'gas purchase'
        ])
        # build plant
        if self.scenario == 'simple':
            plant = plants.simple(df=df)
        elif self.scenario == 'medium':
            plant = plants.medium(df=df)
        elif self.scenario == 'hard':
            plant = plants.hard(df=df)
        else:
            raise AssertionError(f"Unknown scenario '{self.scenario}'")
        # run simulation
        return plant.run(plan=plan, action=action, progress=True)


    # noinspection PyShadowingNames
    def __init__(self,
                 name: Optional[str] = None,
                 seed: int = 42,
                 scenario: Literal['simple', 'medium', 'hard'] = 'simple',
                 prediction_error: float = 1.0,
                 uncertainty_error: float = 1.0):
        super(UnitCommitment, self).__init__(
            name=name,
            seed=seed,
            scenario=scenario,
            prediction_error=prediction_error,
            uncertainty_error=uncertainty_error
        )

    @property
    def scenario(self) -> Literal['simple', 'medium', 'hard']:
        return self.configuration['scenario']

    @property
    def prediction_error(self) -> float:
        return self.configuration['prediction_error']

    @property
    def uncertainty_error(self) -> float:
        return self.configuration['uncertainty_error']
