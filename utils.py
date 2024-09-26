import jax
from jax import numpy as jnp
from jax.scipy import special as scs
import quadax


def erf_prime(x):
    """First derivative of erf."""
    return 2 * jnp.exp(-(x**2)) / jnp.sqrt(jnp.pi)


def erf_second(x):
    """Second derivative of erf."""
    return -4 * x * jnp.exp(-(x**2)) / jnp.sqrt(jnp.pi)


def der_zeta(t, gamma):
    """Derivative of zeta."""
    return (
        2
        / jnp.sqrt(1 + 2 * gamma**2)
        * scs.erf(t / jnp.sqrt((1 + 4 * gamma**2) * (1 + 2 * gamma**2)))
        * erf_prime(t / jnp.sqrt(1 + 2 * gamma**2))
    )


@jax.custom_jvp
def zeta_integ(t, gamma):
    """zeta is defined as the integral of its derivative."""
    integ_value, _ = quadax.quadgk(der_zeta, interval=[-jnp.inf, t], args=(gamma,))
    return 1.0 + integ_value


# This tells jax to use the definition of the derivative of zeta rather than try to autodiff it.
@zeta_integ.defjvp
def der_zeta_jvp(primals, tangents):
    t, gamma = primals
    t_dot, _ = tangents
    return zeta_integ(gamma, t), der_zeta(t, gamma) * t_dot


def lambd_value(lambd0, step, schedule, increment=1e-4):
    """Definition of the schedule for lambda."""
    if schedule:
        return lambd0 / (1 + increment * step)
    return lambd0
