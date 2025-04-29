from typing import Optional

from model import Benchmark, Structure, querymethod
import pandas as pd
from pandas import DataFrame

# Imports from Alberto's code, to be streamlined
from .parameters import *
from .custom_utils import generate_weather_date, custom_obs_rollout, get_model_name
from .flight_environments import FPD_custom, calculate_boundaries
from skdecide.hub.solver.astar import Astar

def domain_factory(ori: tuple, des: tuple, avg_path: pd.DataFrame, lat_points: int = LAT_POINTS,
                   fwd_points: int = FWD_POINTS, vertical_points: int = VERTICAL_POINTS):
    """
    Domain factory for the A* solver to use with RL enhanced solver
    :param ori: Origin coordinates
    :param des: Destination coordinates
    :param avg_path: Original path from the RL model
    :param lat_points: Number of lateral points
    :param fwd_points: Number of forward points
    :param vertical_points: Number of vertical points

    :return: Domain factory lambda function
    """

    weather_date = generate_weather_date()

    def RL_factory():
        fpd = FPD_custom(
            ori,
            des,
            "A320",
            weather_date=weather_date,
            heuristic_name="fuel",
            perf_model_name="PS",
            objective="fuel",
            fuel_loop=False,
            graph_width="normal",
            nb_lateral_points=lat_points,
            nb_forward_points=fwd_points,
            nb_vertical_points=vertical_points,
            noisy=False
        )

        fpdn = fpd.get_network()  # 41, 11, 5
        lower_bound , upper_bound = calculate_boundaries(avg_path, fpdn)  # Inclusive boundaries
        new_net = [row[lower_bound:upper_bound + 1] for row in fpdn]
        fpd.network = new_net
        fpd.nb_lateral_points = upper_bound - lower_bound + 1

        fpd.start.pos = (0, fpd.nb_lateral_points // 2, 0)
        return fpd

    return RL_factory


class GuidedFlightPlanning(Benchmark):
    """
    Models a flight planning problem guided by a coarse trajectory.
    The guiding trajectory is meant to speed up the computation of the actual trajectory
    """

    @staticmethod
    def build(structure: Structure):
        # variables
        structure.add_custom_variable('origin', dtype = str,
                                      description='origin airport code')
        structure.add_custom_variable('destination', dtype = str,
                                      description='destination airport code')
        structure.add_custom_variable('lat', object,
                                      description='sequence of latitude values for the guiding trajectory')
        structure.add_custom_variable('lon', object,
                                      description='sequence of longitude values for the guiding trajectory')
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

        # metrics
        structure.add_value_metric(
            name='total_fuel',
            function=lambda out: out['fuel'].sum(),
            description='total amount of consumed fuel'
        )

    @querymethod
    def query(self, origin : str, destination : str, lat: list[float], lon: list[float]) -> DataFrame:
        # Assemble the guiding path
        guiding_pah = pd.DataFrame({'lat': lat, 'lon': lon})

        # Build a scikit-decide domain
        solver_factory = domain_factory(origin,
                                        destination, guiding_pah)
        solver_domain = solver_factory()

        # Solve the planning problem
        with Astar(
                heuristic=lambda d, s: d.heuristic(s),
                domain_factory=solver_factory,
                parallel=False,
        ) as solver:
            solver.solve()
            _, rl_obs = custom_obs_rollout(solver_domain, solver=solver)
            solver.close()

        # Extract the final trajectory
        trajectory = rl_obs.trajectory[["lat", "lon", "alt", "fuel", "mass"]]
        solver_domain.close()
        return trajectory

    def __init__(self, name: Optional[str] = None,
                 steps: int = FWD_POINTS,
                 width : int = LAT_POINTS,
                 height : int = VERTICAL_POINTS):
        super(GuidedFlightPlanning, self).__init__(name=name, seed=None,
                                                   steps=steps, width=width, height=height)

    @property
    def steps(self) -> int:
        return self._configuration['steps']

    @property
    def width(self) -> int:
        return self._configuration['width']

    @property
    def height(self) -> int:
        return self._configuration['height']


# class SEIR(Benchmark):
#     """
#     Models an epidemic leveraging the SEIR model.
#     The population is divided into three groups (compartments), i.e. Susceptibles (S), Infected (I), and Recovered (R).
#     The classical SIR model is dynamic, indeed the size of the three groups evolves over time.
#     The evolution of the model is defined by the following system of Ordinary Differential Equations (ODEs):
#         dS = -𝛽 * S * I / N
#         dI = 𝛽 * S * I / N - 𝛾 * I
#         dR = 𝛾 * I
#     with N being the total population size, and 𝛽 and 𝛾 the infection and recovery rate, respectively.
#     """
#
#     @staticmethod
#     def build(structure: Structure):
#         # variables
#         structure.add_positive_variable('s0', description='the percentage of initial susceptibles')
#         structure.add_positive_variable('e0', description='the percentage of initial exposed')
#         structure.add_positive_variable('i0', description='the percentage of initial infected')
#         structure.add_positive_variable('r0', description='the percentage of initial recovered')
#         structure.add_positive_variable('beta', strict=True, description='the infection rate')
#         structure.add_positive_variable('gamma', strict=True, description='the recovery rate')
#         structure.add_positive_variable('latency', strict=True, description='the exposition latency period'),
#         # parameter
#         structure.add_positive_parameter(
#             name='horizon',
#             default=300,
#             strict=True, integer=True, description='the timespan of the simulation')
#         # constraint
#         structure.add_generic_constraint(
#             name='percentage',
#             check=lambda s0, e0, i0, r0: s0 + e0 + i0 + r0 == 1,
#             description='the percentages of susceptibles, exposed, infected, and recovered must sum up to one'
#         )
#         # metrics
#         structure.add_value_metric(
#             name='susceptibles',
#             function=lambda out: out['S'].iloc[-1],
#             description='number of final susceptibles'
#         )
#         structure.add_value_metric(
#             name='exposed',
#             function=lambda out: out['E'].iloc[-1],
#             description='number of final Exposed'
#         )
#         structure.add_value_metric(
#             name='infected',
#             function=lambda out: out['I'].iloc[-1],
#             description='number of final infected'
#         )
#         structure.add_value_metric(
#             name='recovered',
#             function=lambda out: out['R'].iloc[-1],
#             description='number of final recovered'
#         )
#
#     @querymethod
#     def query(self, s0: float, e0: float, i0: float, r0: float, beta: float, gamma: float, latency: float) -> DataFrame:
#         # define seir dynamics (normalization is not necessary since we assume the population to sum up to 1)
#         def seir(s, e, i, _):
#             t1 = beta * s * i
#             t2 = latency * e
#             t3 = gamma * i
#             return np.array([-t1, t1 - t2, t2 - t3, t3])
#
#         # solve ode
#         data = odeint(
#             func=lambda y, t: seir(*y),
#             y0=np.array([s0, e0, i0, r0]),
#             t=np.arange(self.horizon)
#         )
#         # build dataframe
#         data = DataFrame(data=data, index=np.arange(self.horizon), columns=["S", "E", "I", "R"])
#         data.index.rename("time", inplace=True)
#         return data
#
#     def __init__(self, name: Optional[str] = None, horizon: int = 300):
#         super(SEIR, self).__init__(name=name, seed=None, horizon=horizon)
#
#     @property
#     def horizon(self) -> int:
#         return self._configuration['horizon']
