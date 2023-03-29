import numpy as np
from benchmarks.survival import Cox, simulate_patient_flag
import matplotlib.pyplot as plt
from typing import Tuple


if __name__ == "__main__":
    instance = Cox(horizon=10)

    n_pat = 3
    n_cov = 1
    h0 = lambda t: np.sqrt(t * 0.2)
    x = np.random.rand(n_pat, n_cov)
    beta = np.random.rand(n_cov)

    # simulate
    print("Simulating data for:")
    print(f"{n_pat=}\t{n_cov=}\t{instance.horizon=}")
    df = instance.query(h0=h0, x=x, beta=beta)
    print(f"{df.shape=}")
    print("\nDataframe head:\n\n", df.head(5))

    from random import seed

    seed(42)
    df["horizon"] = df.horizon.astype("int")
    simulate_patient_flag(df)

    print("\n\nSimulated flags:\n\n", df)
