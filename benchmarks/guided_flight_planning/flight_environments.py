import math
from typing import Tuple, Callable

import numpy as np
import pandas as pd
from openap.prop import aircraft
from pygeodesy.ellipsoidalVincenty import LatLon
from skdecide import Value, load_registered_solver
from openap.extra.aero import bearing as aero_bearing
from openap.extra.aero import ft, kts, mach2tas

from .get_ensemble_weather import get_wind_values
from .parameters import *
from skdecide.hub.domain.flight_planning import FlightPlanningDomain
from .custom_utils import flying as custom_flying, generate_weather_date, fuel_optimisation_PS, get_wind_interpolator


def compute_gspeed(tas: float, tc: float, ws: float, wd: float):
    swc = ws / tas * math.sin(wd - tc)
    if abs(swc) >= 1.0:
        gs = tas
    else:
        gs = tas * math.sqrt(1 - swc * swc) - ws * math.cos(wd - tc)

    if gs < 0:
        gs = tas
    return gs

class FPD_custom(FlightPlanningDomain):
    """
    Custom Flight Planning Domain for the RL comparison

    Defaults the flying function to Noise=False to make A* feasible #TODO: Check if this is avoidable
    """
    def __init__(
            self,origin, destination, actype, weather_date=None, wind_interpolator=None, objective="fuel",
            heuristic_name="fuel", perf_model_name=PERF_MODEL, constraints=None,
            nb_forward_points=FWD_POINTS, nb_lateral_points=LAT_POINTS, nb_vertical_points=None,
            take_off_weight=None, fuel_loop=False, fuel_loaded=None,
            fuel_loop_solver_cls=None, fuel_loop_solver_kwargs=None, fuel_loop_tol=1e-3,
            climbing_slope=None, descending_slope=None, graph_width=None,
            res_img_dir=None, starting_time=3_600.0 * 8.0,
            noisy=False, noise_amount=0
    ):
        if constraints is None:
            constraints = {}
        constraints["fuel"] = fuel_loaded if fuel_loaded else 24000

        if fuel_loop:
            if fuel_loop_solver_cls is None:
                LazyAstar = load_registered_solver("LazyAstar")
                fuel_loop_solver_cls = LazyAstar
                fuel_loop_solver_kwargs = dict(heuristic=lambda d, s: d.heuristic(s))
            elif fuel_loop_solver_kwargs is None:
                fuel_loop_solver_kwargs = {}
            fuel_loaded = fuel_optimisation_PS(
                origin=origin,
                destination=destination,
                actype=actype,
                constraints=constraints,
                weather_date=weather_date,
                solver_cls=fuel_loop_solver_cls,
                solver_kwargs=fuel_loop_solver_kwargs,
                fuel_tol=fuel_loop_tol,
            )
            fuel_loaded = 1.1 * fuel_loaded  # Adding 10% fuel reserve
            fuel_loop = False

        super().__init__(origin, destination, actype, weather_date, wind_interpolator, objective, heuristic_name,
                         perf_model_name, constraints, nb_forward_points, nb_lateral_points, nb_vertical_points,
                         take_off_weight, fuel_loaded, fuel_loop, fuel_loop_solver_cls, fuel_loop_solver_kwargs,
                         fuel_loop_tol, climbing_slope, descending_slope, graph_width, res_img_dir, starting_time)
        _, _, _, self.ds = get_wind_values(0, 0, 0, noisy=False, noise_amount=0) \
            if WIND_DS[0] is None else (0, 0, 0, WIND_DS[0])
        self._memory = None
        self.ac = aircraft(actype)
        self.noisy = noisy
        self.noise_amount = noise_amount

    def flying(self, from_: pd.DataFrame, to_: Tuple[float, float, int]) -> pd.DataFrame:
        df, _, _ = custom_flying(from_=from_, to_=to_, noisy=self.noisy, noise_amount=self.noise_amount, env=self,
                                 wind_interpolator=self.weather_interpolator, ds=self.ds, ac=self.ac)
        return df

    def get_trajectory(self):
        return self._memory.trajectory

    def heuristic(self, s, heuristic_name: str = None):
        """
        Custom heuristic, re-implemented to add dynamic noise and eventually RL
        :param s: D.T_state, Current state
        :param heuristic_name: str, Name of the heuristic to use
        :return: Value[D.T_value], Heuristic cost
        """

        if heuristic_name is None:
            heuristic_name = self.heuristic_name
        pos = s.trajectory.iloc[-1]
        lat_to, lon_to, alt_to = self.lat2, self.lon2, self.alt2
        lat_start, lon_start, alt_start = self.lat1, self.lon1, self.alt1

        # Compute distance in meters
        distance_to_goal = LatLon.distanceTo(
            LatLon(pos["lat"], pos["lon"], height=pos["alt"] * ft),  # alt ft -> meters
            LatLon(lat_to, lon_to, height=alt_to * ft),  # alt ft -> meters
        )
        distance_to_start = LatLon.distanceTo(
            LatLon(pos["lat"], pos["lon"], height=pos["alt"] * ft),  # alt ft -> meters
            LatLon(lat_start, lon_start, height=alt_start * ft),  # alt ft -> meters
        )

        if heuristic_name == "distance":
            cost = distance_to_goal
        elif heuristic_name == "fuel":
            bearing_degrees = aero_bearing(pos["lat"], pos["lon"], lat_to, lon_to)

            # weather computations & A/C speed modification
            we, wn = 0, 0
            temp = 273.15
            if self.weather_interpolator:
                wind_ms = self.weather_interpolator.interpol_wind_classic(
                    lat=pos["lat"], longi=pos["lon"], alt=pos["alt"], t=pos["ts"]
                )
                we, wn = wind_ms[2][0], wind_ms[2][1]
                temp = self.weather_interpolator.interpol_field(
                    [pos["ts"], pos["alt"], pos["lat"], pos["lon"]], field="T"
                )
                wspd = math.sqrt(wn * wn + we * we)
            else:
                wspd, wd, tt, _ = get_wind_values(pos["lat"], pos["lon"], pos["alt"], noisy=False, ds=self.ds, noise_amount=0)

            if math.isnan(temp):
                print("NaN values in temp")

            tas = mach2tas(pos["mach"], pos["alt"] * ft)  # alt ft -> meters
            gs = compute_gspeed(tas=tas, tc=math.radians(bearing_degrees), ws=wspd,
                                wd=3 * math.pi / 2 - math.atan2(wn, we))
            values_current = {
                "mass": pos["mass"],
                "alt": pos["alt"],
                "speed": tas / kts,
                "temp": temp,
            }
            dt = distance_to_goal / gs
            if distance_to_goal == 0:
                return Value(cost=0)

            if self.perf_model_name == "PS":
                cost = self.perf_model.compute_fuel_consumption(
                    values_current,
                    dt,
                    math.degrees((alt_to - pos["alt"]) * ft / distance_to_goal)
                    # approximation for small angles: tan(alpha) ~ alpha
                )
            else:
                cost = self.perf_model.compute_fuel_consumption(values_current,delta_time=dt)

        elif heuristic_name in ["time", "lazy_fuel", "lazy_time"]:
            raise Exception(f"Heuristic {heuristic_name} is not yet implemented")
        else:
            cost = 0

        return Value(cost=cost)

    def close(self):
        del self

class Trajectory_Container:
    def __init__(self):
        self.get_traj = None

    def modify_container(self, getter: Callable):
        self.get_traj = getter

    def get_trajectory(self):
        return self.get_traj()


def calculate_boundaries(avg_path: pd.DataFrame, fpdn: list) -> Tuple[int, int]:
    """
    :param avg_path: Original RL path
    :param fpdn: Environment network
    :return: lower_bound, upper_bound
    """
    lower_bound = np.inf
    upper_bound = -np.inf
    substep = math.ceil(FWD_POINTS / 3.0)
    for point in range(FWD_POINTS):
        path_p = avg_path.iloc[math.floor(point / substep)]
        closest_point = np.argmin([np.linalg.norm(np.array([p[0].lat, p[0].lon]) -
                                   np.array([path_p["lat"], path_p["lon"]])) for p in fpdn[point]])
        if lower_bound > closest_point:
            lower_bound = closest_point
        if upper_bound < closest_point:
            upper_bound = closest_point

    if upper_bound - lower_bound < LAT_POINTS * MIN_CUT:
        # print("Expanding boundaries")
        lower_bound = max(0, lower_bound - LAT_POINTS // 3)
        upper_bound = min(LAT_POINTS - 1, upper_bound + LAT_POINTS // 3)
    while upper_bound - lower_bound > LAT_POINTS * MAX_CUT:
        # print("Reducing boundaries")
        upper_bound -= 1
        lower_bound += 1
    return lower_bound, upper_bound


def domain_factory(ori: tuple, des: tuple, fuel_loaded=None, noisy=False) -> Callable[[], FPD_custom]:
    """
    Domain factory for the A* solver to use with solver only
    :param ori: Origin coordinates
    :param des: Destination coordinates
    :param fuel_loaded: Fuel loaded, added to avoid multiple fuel loops
    :param noisy: bool, Noise flag
    :return: Domain factory lambda function
    """

    weather_date = generate_weather_date()

    def create_domain():
        domain = FPD_custom(
            ori,
            des,
            "A320",
            weather_date=weather_date,
            heuristic_name="fuel",
            perf_model_name=PERF_MODEL,
            objective="fuel",
            fuel_loop=False,
            fuel_loaded=fuel_loaded,
            graph_width="normal",
            nb_lateral_points=LAT_POINTS,
            nb_forward_points=FWD_POINTS,
            noisy=noisy
        )
        # cont.modify_container(domain.get_trajectory)
        return domain

    return create_domain

def RL_V_domain_factory(ori: tuple, des: tuple, avg_path: pd.DataFrame, fuel_loaded=None, noisy=False)\
        -> Callable[[], FPD_custom]:
    """
    Domain factory for the A* solver to use with RL enhanced solver
    :param ori: Origin coordinates
    :param des: Destination coordinates
    :param avg_path: Original path from the RL model
    :param fuel_loaded: Fuel loaded, added to avoid multiple fuel loops
    :param noisy: bool, Noise flag
    :return: Domain factory lambda function
    """
    lat_points = LAT_POINTS
    fwd_points = FWD_POINTS

    simple_cut = False

    weather_date = generate_weather_date()

    def RL_factory():
        fpd = FPD_custom(
            ori,
            des,
            "A320",
            weather_date=weather_date,
            heuristic_name="fuel",
            perf_model_name=PERF_MODEL,
            objective="fuel",
            fuel_loop=False,
            fuel_loaded=fuel_loaded,
            graph_width="normal",
            nb_lateral_points=lat_points,
            nb_forward_points=fwd_points,
            noisy=noisy
        )

        fpdn = fpd.get_network()  # 41, 11
        if simple_cut:
            # Deprecated
            avg_lat = np.mean(avg_path["lat"])
            avg_lon = np.mean(avg_path["lon"])
            pos_points = fpdn[fwd_points // 2][lat_points // 2 + 1]
            neg_points = fpdn[fwd_points // 2][lat_points // 2 - 1]
            if np.linalg.norm(np.array([pos_points[0].lat, pos_points[0].lon]) - np.array([avg_lon, avg_lat])) < np.linalg.norm(np.array([neg_points[0].lat, neg_points[0].lon]) - np.array([avg_lon, avg_lat])):
                fpd.network = [row[:(lat_points // 2)] for row in fpdn]
            else:
                fpd.network = [row[(lat_points // 2 + 1):] for row in fpdn]
                fpd.nb_lateral_points = lat_points // 2
                fpd.start.pos = (0, fpd.nb_lateral_points // 2, 0)
        else:
            lower_bound , upper_bound = calculate_boundaries(avg_path, fpdn) # Inclusive boundaries
            new_net = [row[lower_bound:upper_bound + 1] for row in fpdn]
            # print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
            fpd.network = new_net
            fpd.nb_lateral_points = upper_bound - lower_bound + 1

            fpd.start.pos = (0, fpd.nb_lateral_points // 2, 0)

        # cont.modify_container(fpd.get_trajectory)
        return fpd

    return RL_factory
