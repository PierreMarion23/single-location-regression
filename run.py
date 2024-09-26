import os
import pickle

from jax import numpy as jnp
from jax import random
import pandas as pd

from dynamics import (
    generate_batch,
    initialize_on_manifold,
    initialize_on_sphere,
    risk_fn,
    sgd_step,
    pgd_step,
    R,
)
from utils import lambd_value, zeta_integ


def run(config, key):
    os.makedirs("pickles", exist_ok=True)

    d = config["d"]
    L = config["L"]
    gamma = config["gamma"]
    lambd0 = config["lambd0"]
    alpha = config["alpha"]
    n_steps = config["n_steps"]
    log_every = config["log_every"]
    n_repeats = config["n_repeats"]
    lambd_schedule = config["lambd_schedule"]
    flag_initialize_sphere = config["flag_initialize_sphere"]
    flag_sgd = config["flag_sgd"]
    if flag_sgd:
        batch_size = config["batch_size"]
        eps = config["eps"]
        test_batch_size = config["test_batch_size"]
    else:
        batch_size, eps, test_batch_size = None, None, None
    filename = config["filename"]

    subkeys = random.split(key, n_repeats)

    # define update function
    if flag_sgd:
        step_fn = sgd_step
    else:
        step_fn = pgd_step

    # precompute zeta values
    if not flag_sgd:
        print("Precomputing zeta values...")
        zeta_values = {}
        zeta_values[0] = zeta_integ(0, lambd0)
        for step in range(n_steps):
            if (step + 1) % log_every == 0:
                if lambd_schedule:
                    print("Value {}/{}".format(step + 1, n_steps))
                    lambd = lambd_value(lambd0, step, lambd_schedule)
                    zeta_values[step + 1] = zeta_integ(0, lambd)
                else:
                    zeta_values[step + 1] = zeta_values[0]

    logs = {
        "Run": [],
        "Step": [],
        "Kappa": [],
        "Nu": [],
        "Distance to manifold": [],
        "Excess risk": [],
    }

    print("Beginning simulation")
    for repeat in range(n_repeats):
        print("Repeat {}/{}".format(repeat + 1, n_repeats))
        if flag_sgd:
            subsubkeys = random.split(subkeys[repeat], n_steps + 2)
            key_init = subsubkeys[0]
        else:
            key_init = subkeys[repeat]
        if flag_initialize_sphere:
            k, v, k_star, v_star = initialize_on_sphere(d, key_init)
        else:
            k, v, k_star, v_star = initialize_on_manifold(d, key_init)
        logs["Run"].append(repeat)
        logs["Step"].append(0)
        logs["Kappa"].append(float(k @ k_star))
        logs["Nu"].append(float(v @ v_star))
        logs["Distance to manifold"].append(
            float(jnp.sqrt((v @ k_star) ** 2 + (k @ v_star) ** 2 + (k @ v) ** 2))
        )
        if flag_sgd:
            test_X, test_y = generate_batch(
                k_star, v_star, test_batch_size, d, L, eps, gamma, subsubkeys[1]
            )
            lambd = lambd_value(lambd0, 0, lambd_schedule)
            test_risk = risk_fn(test_X, test_y, (k, v), lambd)
            logs["Excess risk"].append(float(test_risk - eps**2))
        else:
            risk_value = R(k, v, k_star, v_star, lambd0, gamma, d, L)
            risk_value += (L - 1) * zeta_values[0]
            logs["Excess risk"].append(float(risk_value))

        for step in range(n_steps):
            key_sgd_step = subsubkeys[step + 2] if flag_sgd else None
            lambd = lambd_value(lambd0, step, lambd_schedule)
            (k, v), risk_value, kappa, nu, theta, eta, rho = step_fn(
                (k, v),
                k_star,
                v_star,
                d,
                L,
                gamma,
                lambd,
                alpha,
                batch_size,
                eps,
                key_sgd_step,
            )
            if (step + 1) % log_every == 0:
                logs["Run"].append(repeat)
                logs["Step"].append(step + 1)
                logs["Kappa"].append(float(kappa))
                logs["Nu"].append(float(nu))
                logs["Distance to manifold"].append(
                    float(jnp.sqrt((theta) ** 2 + (eta) ** 2 + (rho) ** 2))
                )
                if flag_sgd:
                    test_risk = risk_fn(test_X, test_y, (k, v), lambd)
                    logs["Excess risk"].append(float(test_risk - eps**2))
                else:
                    risk_value += (L - 1) * zeta_values[step + 1]
                    logs["Excess risk"].append(float(risk_value))

    df = pd.DataFrame(logs)

    with open("pickles/result_gd_exp_{}.pickle".format(filename), "wb") as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    return df
