from jax import numpy as jnp

# initialize on manifold, no lambda schedule
config_gd_manifold = {
    "d": 400,
    "L": 10,
    "gamma": float(jnp.sqrt(0.5)),
    "lambd0": 0.1,
    "alpha": 4e-3,
    "n_steps": 20_000,
    "log_every": 200,
    "n_repeats": 30,
    "lambd_schedule": False,
    "flag_initialize_sphere": False,
    "flag_sgd": False,
    "filename": "init_manifold",
    "plot_run_idx_1": 0,
    "plot_run_idx_2": 1,
}
#

# initialize on sphere, no lambda schedule, large lambda
config_gd_sphere_no_schedule_large_lambda = {
    "d": 400,
    "L": 10,
    "gamma": float(jnp.sqrt(0.5)),
    "lambd0": 1.0,
    "alpha": 1e-3,
    "n_steps": 120_000,
    "log_every": 300,
    "n_repeats": 30,
    "lambd_schedule": False,
    "flag_initialize_sphere": True,
    "flag_sgd": False,
    "filename": "init_sphere_no_schedule_large_lambda",
    "plot_run_idx_1": 0,
    "plot_run_idx_2": 2,
}


# initialize on sphere, no lambda schedule, small lambda
config_gd_sphere_no_schedule_small_lambda = {
    "d": 400,
    "L": 10,
    "gamma": float(jnp.sqrt(0.5)),
    "lambd0": 0.1,
    "alpha": 4e-3,
    "n_steps": 20_000,
    "log_every": 200,
    "n_repeats": 30,
    "lambd_schedule": False,
    "flag_initialize_sphere": True,
    "flag_sgd": False,
    "filename": "init_sphere_no_schedule_small_lambda",
    "plot_run_idx_1": 5,
    "plot_run_idx_2": 3,
    "plot_run_idx_3": 12,
}


# initialize on sphere, lambda schedule
config_gd_sphere_schedule = {
    "d": 400,
    "L": 10,
    "gamma": float(jnp.sqrt(0.5)),
    "lambd0": 1.0,
    "alpha": 4e-3,
    "n_steps": 120_000,
    "log_every": 300,
    "n_repeats": 30,
    "lambd_schedule": True,
    "flag_initialize_sphere": True,
    "flag_sgd": False,
    "filename": "init_sphere_schedule",
    "plot_run_idx_1": 0,
    "plot_run_idx_2": 2,
}

# initialize on sphere, lambda schedule, sgd
config_sgd_sphere_schedule = {
    "batch_size": 5,
    "test_batch_size": 10**3,
    "d": 80,
    "L": 10,
    "eps": 0.1,
    "gamma": float(jnp.sqrt(0.5)),
    "lambd0": 2.0,
    "alpha": 1e-3,
    "n_steps": 200_000,
    "log_every": 500,
    "n_repeats": 30,
    "lambd_schedule": True,
    "flag_initialize_sphere": True,
    "flag_sgd": True,
    "filename": "sgd_init_sphere_schedule",
    "plot_run_idx_1": 0,
    "plot_run_idx_2": 1,
}
