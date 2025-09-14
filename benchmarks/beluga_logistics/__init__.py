from typing import Optional

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'beluga_challenge_toolkit'))

from model import Benchmark, Structure, querymethod
from beluga_lib.beluga_problem import BelugaProblemDecoder, BelugaProblem
from evaluation.planner_api import DeterministicPlannerAPI, BelugaPlan
from evaluation.evaluators import DeterministicEvaluator, SingleSimulationOutcome


class PlanWrapper(DeterministicPlannerAPI):

    def __init__(self, json_plan):
        self.json_plan = json_plan

    def setup(self):
        pass

    def build_plan(self, prb : BelugaProblem):
        # Return the plan
        plan = BelugaPlan.from_json_obj(self.json_plan, prb)
        return plan


class BelugaLogisticsDeterministic(Benchmark):
    """
    Models a planning problem that involves handling parts transported by Aribus Beluga aircrafts
    This benchmark is a wrapper around the planning competition run during the TUPLES project
    """

    @staticmethod
    def build(structure: Structure):
        # variables
        structure.add_custom_variable('problem', dtype = object,
                                      description='problem description, in json format')
        structure.add_custom_variable('plan', dtype = object,
                                      description='plan description, in json format')
        # parameter
        structure.add_positive_parameter(
            name='max_steps',
            default=100,
            strict=True, integer=True, description='the maximum number of steps to be simulated')

        structure.add_positive_parameter(
            name='time_limit',
            default=60,
            strict=True, description='time limit for the evaluation (in seconds)')

        structure.add_positive_parameter(
            name='alpha',
            default=0.7,
            strict=True, description='alpha parameter in the main score function')

        structure.add_positive_parameter(
            name='beta',
            default=0.0004,
            strict=True, description='beta parameter in the main score function')

        # metrics
        structure.add_value_metric(
            name='score',
            function=lambda out: out.score['value'],
            description='total score for the instance'
        )

        structure.add_value_metric(
            name='score_A',
            function=lambda out: out.score['A'],
            description='A component of the total score for the instance'
        )

        structure.add_value_metric(
            name='score_B',
            function=lambda out: out.score['B'],
            description='B component of the total score for the instance'
        )

        structure.add_value_metric(
            name='score_C',
            function=lambda out: out.score['C'],
            description='C component of the total score for the instance'
        )

    @querymethod
    def query(self, problem : dict, plan : dict) -> SingleSimulationOutcome:
        # Convert the problem json into the internal format
        inst = json.loads(json.dumps(problem), cls=BelugaProblemDecoder)

        # Build a plan object
        planner = PlanWrapper(plan)

        # Build an evaluator
        evaluator = DeterministicEvaluator(prb=inst,
                              problem_name='test_problem.json',
                              problem_folder=None,
                              max_steps=self.max_steps, # TODO change this
                              time_limit=self.time_limit,
                              planner=planner,
                              alpha=self.alpha,
                              beta=self.beta)

        # Setup the evaluator
        evaluator.setup()

        # Start the evaluation
        outcome = evaluator.evaluate()

        return outcome


    def __init__(self, name: Optional[str] = None,
                 max_steps : int = 100,
                 time_limit : float = 60,
                 alpha : float = 0.7,
                 beta : float = 0.0004):
        super(BelugaLogisticsDeterministic, self).__init__(name=name, seed=None,
                                                           max_steps=max_steps,
                                                           time_limit=time_limit,
                                                           alpha=alpha,
                                                           beta=beta)

    @property
    def max_steps(self) -> int:
        return self._configuration['max_steps']

    @property
    def time_limit(self) -> float:
        return self._configuration['time_limit']

    @property
    def alpha(self) -> float:
        return self._configuration['alpha']

    @property
    def beta(self) -> float:
        return self._configuration['beta']

