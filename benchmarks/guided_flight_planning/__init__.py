from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hybrid_rl'))

from model import Benchmark, Structure, querymethod
import pandas as pd
from pandas import DataFrame
import numpy as np

from parameters import *
from flight_environments import FPD_custom, calculate_network, domain_factory
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import AircraftState
from unify import FlightEnv
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import \
    PerformanceModelEnum
from skdecide.hub.solver.astar import Astar
from custom_utils import suppress_stdout_stderr

# from custom_utils import generate_weather_date, custom_obs_rollout, get_model_name
# from flight_environments import FPD_custom, calculate_boundaries
# from skdecide.hub.solver.astar import Astar

class GuidedFlightPlanning(Benchmark):
    """
    Models a flight planning problem guided by a coarse trajectory.
    The guiding trajectory is meant to speed up the computation of the actual trajectory
    """

    @staticmethod
    def build(structure: Structure):
        # variables
        structure.add_custom_variable('origin', dtype = object,
                                      description='origin, as a (latitude, longitude, altitude) collection')
        structure.add_custom_variable('destination', dtype = object,
                                      description='destination, as a (latitude, longitude, altitude) collection')
        structure.add_custom_variable('lat', object,
                                      description='sequence of latitude values for the guiding trajectory')
        structure.add_custom_variable('lon', object,
                                      description='sequence of longitude values for the guiding trajectory')
        structure.add_custom_variable('alt', object,
                                      description='sequence of altitude values for the guiding trajectory')
        structure.add_custom_variable('mass', object,
                                      description='sequence of aircraft mass values for the guiding trajectory')
        # parameter
        structure.add_positive_parameter(
            name='steps',
            default=FWD_POINTS,
            strict=True, integer=True, description='the number of steps in the final trajectory')

        structure.add_positive_parameter(
            name='width',
            default=LAT_POINTS,
            strict=True, integer=True, description='the width of the corridor around the guiding trajectory')

        structure.add_positive_parameter(
            name='height',
            default=VERTICAL_POINTS,
            strict=True, integer=True, description='the height of the corridor considered by the solver')

        structure.add_categorical_parameter(
            name='metric',
            default='fuel',
            categories=('fuel', 'time'),
            dtype=str,
            description='the metric that the planner should target')

        # metrics
        structure.add_value_metric(
            name='total_fuel',
            function=lambda out: out['fuel'].sum(),
            description='total amount of consumed fuel'
        )

    @querymethod
    def query(self, origin : str, destination : str,
              lat: list[float],
              lon: list[float],
              alt: list[float],
              mass: list[float],
              ) -> DataFrame:
        # Assemble the guiding path
        guiding_path = pd.DataFrame({'lat': lat, 'lon': lon, 'alt': alt, 'mass': mass})

        # Define the initial aircraft state
        perf_model_name = PerformanceModelEnum.POLL_SCHUMANN
        base_env = FlightEnv(start=np.array([origin[0], origin[1], 0]), perf_model_name=perf_model_name,
                             target=np.array([destination[0], destination[1], 0]), steps=4, randomize=False)
        aircraft_state = base_env.initial_state

        # Build a scikit-decide domain
        solver_factory = domain_factory(
            origin, destination, avg_path=guiding_path,
            acState=aircraft_state, use_hybrid=True, node_id=0, fuel_loaded=None)

        solver_domain = solver_factory()

        # Solve the planning problem
        solver = Astar(
            domain_factory=solver_factory,
            heuristic=lambda d, s: d.heuristic(s, heuristic_name=self._configuration['metric']),
            parallel=False
        )
        with suppress_stdout_stderr():
            # _, rl_obs = custom_obs_rollout(solver_domain, solver=solver)
            solver_domain.custom_rollout(solver=solver, make_img=False)
        solver.solve()

        # Extract the trajectory
        trajectory = solver_domain.observation.trajectory[[
            'lat', 'lon', 'alt', 'fuel', 'mass']]
        fuel = sum(trajectory['fuel'])
        flight_time = solver_domain.observation.time - solver_domain.starting_time
        solver_domain.close()

        return trajectory

    def __init__(self, name: Optional[str] = None,
                 steps: int = FWD_POINTS,
                 width : int = LAT_POINTS,
                 height : int = VERTICAL_POINTS,
                 metric : str = 'fuel'):
        super(GuidedFlightPlanning, self).__init__(name=name, seed=None,
                                                   steps=steps,
                                                   width=width,
                                                   height=height,
                                                   metric=metric)

    @property
    def steps(self) -> int:
        return self._configuration['steps']

    @property
    def width(self) -> int:
        return self._configuration['width']

    @property
    def height(self) -> int:
        return self._configuration['height']

    @property
    def metric(self) -> int:
        return self._configuration['metric']
