import pickle

from jax import random

from config import (
    config_gd_manifold,
    config_gd_sphere_no_schedule_large_lambda,
    config_gd_sphere_no_schedule_small_lambda,
    config_gd_sphere_schedule,
    config_sgd_sphere_schedule,
)
from plot import plot
from run import run

if __name__ == "__main__":
    for config in [
        config_gd_manifold,
        config_gd_sphere_no_schedule_large_lambda,
        config_gd_sphere_no_schedule_small_lambda,
        config_gd_sphere_schedule,
        config_sgd_sphere_schedule,
    ]:
        filename = config["filename"]
        print("Starting experiment {}".format(filename))
        load_exp = False  # if True, looks for pickled results and re-makes the plots.

        key = random.key(0)
        if not load_exp:
            df = run(config, key)
        else:
            with open("pickles/result_gd_exp_{}.pickle".format(filename), "rb") as f:
                df = pickle.load(f)

        print("Plotting results...")
        plot(config, df)
