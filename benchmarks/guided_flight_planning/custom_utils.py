import math
import os
import random
from tokenize import String
from typing import Tuple, Any
import pandas as pd
from skdecide.hub.domain.flight_planning.aircraft_performance.base import AircraftPerformanceModel, PollSchumannModel
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.GenericInterpolator import (
    GenericWindInterpolator,
)
from math import radians, sqrt
import numpy as np
from openap.prop import aircraft
from openap.extra.aero import bearing as aero_bearing
from openap.extra.aero import distance, ft, kts, latlon, mach2tas
from pandas import DataFrame
from skdecide.hub.domain.flight_planning.domain import compute_gspeed, simple_fuel_loop, FlightPlanningDomain
from datetime import date
from skdecide.hub.domain.flight_planning.domain import WeatherDate
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.get_weather_noaa import get_weather_matrix
from skdecide.hub.solver.astar import Astar

from .parameters import *
from .get_ensemble_weather import get_wind_values


def get_wind_interpolator(weather_date: WeatherDate) -> GenericWindInterpolator:
    w_dict = weather_date.to_dict()

    if NPZ_PATH is None:
        mat = get_weather_matrix(
            year=w_dict["year"],
            month=w_dict["month"],
            day=w_dict["day"],
            forecast=w_dict["forecast"],
            delete_npz_from_local=False,
            delete_grib_from_local=False,
        )
        weather_interpolator = GenericWindInterpolator(file_npz=mat)
    else:
        npz_file = np.load(NPZ_PATH)
        weather_interpolator = GenericWindInterpolator(file_npz=npz_file)

    return weather_interpolator


def generate_weather_date():
    # today = date.today()
    # month = today.month
    # year = today.year
    # day = 1
    # TODO fixed for debugging
    month = 4
    year = 2025
    day = 21
    return WeatherDate(year=year, month=month, day=day)


def get_model_name(model_name="unify_test") -> String:
    if TRAIN_ITERATIONS > 9999:
        model_name += "_long"
    if HIDDEN_SIZE != 64:
        model_name += "_" + str(HIDDEN_SIZE)
    if not SIMULATE_FLYING:
        model_name += "_no_fuel"
    if INCLUDE_ATMOS:
        if INCLUDE_T:
            model_name += "_atmosT"
        else:
            model_name += "_atmos"

    if not os.path.isfile(os.path.join(MODEL_PATH, model_name)) and LOAD_MODEL:
        model_name.replace("_long", "")
    return model_name


def flying(from_: pd.DataFrame, to_: Tuple[float, float, int], noisy=False, noise_amount=0,
           dynamic_noise=False, env=None, wind_interpolator=None, ds=WIND_DS[0], ac=None):
    """Compute the trajectory of a flying object from a given point to a given point

    # Parameters
        from_ (pd.DataFrame): the trajectory of the object so far
        to_ (Tuple[float, float]): the destination of the object

    # Returns
        pd.DataFrame: the final trajectory of the object
    """
    lat_to, lon_to, alt_to = to_[0], to_[1], to_[2]

    pos = from_.to_dict("records")[-1]
    alt = to_[2]
    dist_ = distance(pos["lat"], pos["lon"], to_[0], to_[1], alt)
    data = []
    epsilon = 100
    dt = 600
    dist = dist_
    loop = 0
    wspd, wd, wd = 0, 0, 0
    temp = 273.15
    if ac is None:
        ac = aircraft('A320')
    mach = ac["cruise"]["mach"]

    while dist > epsilon:
        bearing_degrees = aero_bearing(pos["lat"], pos["lon"], lat_to, lon_to)

        if noisy or wind_interpolator is None:
            if env and ds is None:
                ds = env.ds
            wspd, wd, tt, _ = get_wind_values(pos["lat"], pos["lon"], pos["alt"], noisy=noisy, ds=ds, noise_amount=noise_amount)
            if INCLUDE_T:
                temp = tt
            tas = mach2tas(mach, alt * ft)  # alt ft -> meters
        else:
            if env:
                wind_interpolator = env.weather_interpolator
            time = pos["ts"] % (3_600 * 24)
            wind_ms = wind_interpolator.interpol_wind_classic(
                lat=pos["lat"], longi=pos["lon"], alt=alt, t=time
            )
            we, wn = wind_ms[2][0], wind_ms[2][1]
            wd = np.arctan2(we, wn)

            wspd = sqrt(wn * wn + we * we)
            tas = mach2tas(mach, alt * ft)

        gs = compute_gspeed(
            tas=tas,
            true_course=radians(bearing_degrees),
            wind_speed=wspd,
            wind_direction=wd
        )

        if gs * dt > dist:
            # Last step. make sure we go to destination.
            dt = dist / gs
            ll = lat_to, lon_to
        else:
            ll = latlon(
                pos["lat"],
                pos["lon"],
                d=gs * dt,
                brg=bearing_degrees,
                h=alt_to * ft,
            )

        values_current = {
            "mass": pos["mass"],
            "alt": pos["alt"],
            "speed": tas / kts,
            "temp": temp,
        }

        if env:
            if env.perf_model_name == "PS":
                pos["fuel"] = env.perf_model.compute_fuel_consumption(
                    values_current,
                    dt,
                    math.degrees((alt_to - pos["alt"]) * ft / (gs * dt)),
                )
            else:
                pos["fuel"] = env.perf_model.compute_fuel_consumption(
                    values_current,
                    dt,
                )
        else:
            perf_model = PollSchumannModel("A320")
            pos["fuel"] = perf_model.compute_fuel_consumption(
                values_current,
                dt,
                math.degrees((alt_to - pos["alt"]) * ft / (gs * dt)),
            )

        mass = pos["mass"] - pos["fuel"]

        new_row = {
            "ts": pos["ts"] + dt,
            "lat": ll[0],
            "lon": ll[1],
            "mass": mass,
            "mach": mach,
            "fuel": pos["fuel"],
            "alt": alt,  # to be modified
        }

        # New distance to the next 'checkpoint'
        dist = distance(
            new_row["lat"], new_row["lon"], to_[0], to_[1], new_row["alt"]
        )

        if dist < dist_:
            data.append(new_row)
            dist_ = dist
            pos = data[-1]
        else:
            dt = int(dt / 10)
            print("going in the wrong part.")
            assert dt > 0

        loop += 1

    return pd.DataFrame(data), wspd, wd


def custom_obs_rollout(env: FlightPlanningDomain, solver: Astar, max_steps: int = 100) -> Tuple[bool, pd.DataFrame]:
    """
    Custom rollout function for the A* solver (needed for retrieving the trajectory)
    :param env: Current envrionment
    :param solver: A* solver
    :param max_steps: Function stops after goal is reached or after max_steps steps
    :return: tuple containing whether the goal was reached and the final observation
    """
    observation = env.reset()
    solver.reset()
    # loop until max_steps or goal is reached
    for i_step in range(1, max_steps + 1):
        # print(f"Rollout n. {i_step} with {observation}:")

        # choose action according to solver
        action = solver.sample_action(observation)

        # print(action)
        # get corresponding action
        outcome = env.step(action)
        observation = outcome.observation

        # final state reached?
        if env.is_terminal(observation):
            break

    is_goal_reached = env.is_goal(observation)

    return is_goal_reached, observation


def fuel_optimisation_PS(origin, destination, actype="A320",constraints=None,
                weather_date=WeatherDate(year=2024, month=6, day=1), max_steps: int = 100,
                solver_cls=None, solver_kwargs=None, fuel_tol=1e-3) -> float:
    """
    Optimize the fuel consumption of a flight from origin to destination.
    :param origin: ICAO code of the departure airport of th flight plan e.g LFPG for Paris-CDG, or a tuple (lat,lon)
    :param destination: ICAO code of the arrival airport of th flight plan e.g LFBO for Toulouse-Blagnac airport, or a tuple (lat,lon)
    :param actype: Aircarft type describe in openap datas (https://github.com/junzis/openap/tree/master/openap/data/aircraft)
    :param constraints: Constraints that will be defined for the flight plan
    :param weather_date: Date of the flight plan
    :param max_steps: Maximum number of steps in the fuel loop
    :param solver_cls: Solver class used in the fuel loop.
    :param solver_kwargs: Kwargs to initialize the solver used in the fuel loop.
    :param fuel_tol: tolerance on fuel used to stop the optimization
    :return: quantity of fuel to be loaded in the plane for the flight
    """

    if constraints is None:
        constraints = {"fuel": 24000}

    small_diff = False
    step = 0
    new_fuel = constraints["fuel"]
    while not small_diff:
        domain_factory = lambda: FlightPlanningDomain(
            origin=origin,
            destination=destination,
            actype=actype,
            constraints=constraints,
            weather_date=weather_date,
            objective="distance",
            heuristic_name="distance",
            nb_forward_points=FWD_POINTS,
            nb_lateral_points=LAT_POINTS,
            fuel_loaded=new_fuel,
            starting_time=0.0,
            perf_model_name=PERF_MODEL,
        )
        solver_kwargs = dict(solver_kwargs)
        solver_kwargs["domain_factory"] = domain_factory
        solver_factory = lambda: solver_cls(**solver_kwargs)
        fuel_prec = new_fuel
        new_fuel = simple_fuel_loop(
            solver_factory=solver_factory,
            domain_factory=domain_factory,
            max_steps=max_steps,
        )
        step += 1
        small_diff = (fuel_prec - new_fuel) <= fuel_tol

    return new_fuel


def get_angle_to_target(north_point):
    # Calculate the angle between the original vector and the North vector
    original_angle = np.arctan2(0, 1)
    north_angle = np.arctan2(north_point[1], north_point[0])
    angle_difference = north_angle - original_angle

    return angle_difference


def rotate_vector_to_north(vector, angle_difference):
    # Rotate the original vector to align with the new North direction
    rotation_matrix = np.array([[np.cos(angle_difference), -np.sin(angle_difference)],
                                [np.sin(angle_difference), np.cos(angle_difference)]])
    rotated_vector = np.dot(rotation_matrix, vector)

    return rotated_vector


def is_point_A_closer(pointA, pointB, pointC):
    distA = np.linalg.norm(pointA - pointC)
    distB = np.linalg.norm(pointB - pointC)

    if distA < distB:
        return True
    else:
        return False


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
