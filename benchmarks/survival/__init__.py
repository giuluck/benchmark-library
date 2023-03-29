from typing import Optional, Callable, List, Tuple

import numpy as np
from pandas import DataFrame
from scipy.integrate import odeint

from model import Benchmark, Structure, querymethod

# TODO :- consider time horizon greater than the actual span of interest;
# you have an a posteriori for censored


class Cox(Benchmark):
    @staticmethod
    def build(structure: Structure):
        # variables
        structure.add_custom_variable(
            "h0", dtype=Callable, description="baseline hazard function"
        )
        structure.add_custom_variable("x", dtype=np.ndarray, description="covariates")
        structure.add_custom_variable(
            "beta", dtype=np.ndarray, description="coefficients of covariates"
        )
        # parameters
        structure.add_positive_parameter(
            name="horizon",
            default=1827,  # 365 * 5 + 2 where 2 anni bisestili
            strict=True,
            integer=True,
            description="the timespan of the simulation",
        )
        # constraint
        # TODO
        # metrics
        # TODO

    @querymethod
    def query(self, h0: Callable, x: np.ndarray, beta: np.ndarray) -> DataFrame:
        """Simulate patients survival according to a Cox-like hazard."""
        # cox model
        H = []
        for t in range(self.horizon):
            H.append(h0(t) * np.exp(np.sum(x * beta, axis=1)))
        H = np.array(H)

        # build dataframe
        n_pat_ = x.shape[0]
        id_pat = np.arange(0, n_pat_, 1).reshape(-1, 1)
        x = np.concatenate([id_pat, x], axis=1)
        data = np.array(
            [
                np.concatenate(
                    [
                        np.arange(0, self.horizon, 1).reshape(-1, 1),
                        np.repeat(x[i].reshape(1, -1), self.horizon, axis=0),
                        H[:, i].reshape(-1, 1),
                    ],
                    axis=1,
                )
                for i in range(n_pat_)
            ]
        )
        data = np.concatenate(data, axis=0)
        cov_name = [f"x_{i}" for i in range(x.shape[1] - 1)]
        data = DataFrame(data, columns=["horizon", "id_pat"] + cov_name + ["h_t"])

        return data

    def __init__(self, name: Optional[str] = None, horizon: int = 1827):
        super(Cox, self).__init__(name=name, seed=None, horizon=horizon)

    @property
    def horizon(self) -> int:
        return self._configuration["horizon"]


# constant prob of dying/censoring
DEATH_PROB = 0.1
CENSORING_PROB = 0.3


def draw_patient_flag(df_pat: DataFrame) -> Tuple[int, str]:
    """Random sample deceased/censored flag for single patient based on constant censoring/death
    probability over time.

    Args:
        df_pat (DataFrame): dataframe resulting from query method
    Returns:
        Tuple[int, str]: event time, event status
    """
    from random import random

    # print("\nRandom chances:")
    for i in df_pat.horizon.iloc[1:]:
        chance = random()
        # print(i, chance)
        if chance > 1 - CENSORING_PROB:
            return i, "censored"
        elif chance < DEATH_PROB:
            return i, "dead"
    return i, "censored"


def explode_flag(h: int, status: str, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Take event time and status and explode corresponding columns up to fixed horizon.

    Args:
        h (int): time when the patient experienced the event
        status (str): event status: censored or dead
        horizon (int): fixed simulation horizon

    Returns:
        Tuple[np.ndarray, np.ndarray]: flag_censored, flag_deceased
    """
    if status == "censored":
        flag_censored = np.array([0] * (h) + [1] * (horizon - h + 1))
        flag_deceased = np.zeros_like(flag_censored)
    else:
        flag_deceased = np.array([0] * (h) + [1] * (horizon - h + 1))
        flag_censored = np.zeros_like(flag_deceased)
    return flag_censored, flag_deceased


def simulate_patient_flag(df: DataFrame) -> None:
    """Take simulation dataframe and add two columns (in place) with censoring/deceased statuses.

    Args:
        df (DataFrame): dataframe resulting from query method
    """
    flag_censored_table = []
    flag_deceased_table = []
    for id_pat in df.id_pat.unique():
        h, status = draw_patient_flag(df[df.id_pat == id_pat])
        flag_censored, flag_deceased = explode_flag(h, status, horizon=df.horizon.max())
        flag_censored_table.extend(flag_censored)
        flag_deceased_table.extend(flag_deceased)
    df["censored"] = flag_censored_table
    df["deceased"] = flag_deceased_table


if __name__ == "__main__":
    # instantiate simulator
    instance = Cox(horizon=10)

    # set config
    N_PAT = 3
    N_COV = 1


    def baseline_hazard(t: int) -> float:
        return np.sqrt(t * 0.2)
    
    covars = np.random.rand(N_PAT, N_COV)
    covars_coeff = np.random.rand(N_COV)

    # simulate
    simulation_df = instance.query(h0=baseline_hazard, x=covars, beta=covars_coeff)
    print(f"{simulation_df.shape=}")
    simulation_df.head(5)
