from functools import partial

import jax
from jax import numpy as jnp
from jax import random
from jax.scipy import special as scs

from utils import erf_prime, erf_second, zeta_integ


def init_star(d, key):
    """Initialization of k* and v* on the sphere."""
    subkeys = random.split(key, 2)
    k_star = random.normal(subkeys[0], shape=(d,))
    k_star = k_star / jnp.linalg.norm(k_star)
    v_star = random.normal(subkeys[1], shape=(d,))
    v_star = v_star - k_star @ v_star * k_star
    v_star = v_star / jnp.linalg.norm(v_star)
    return k_star, v_star


def initialize_on_manifold(d, key):
    """Initialization of k and v on the manifold M."""
    subkeys = random.split(key, 3)
    k_star, v_star = init_star(d, subkeys[0])
    k = random.normal(subkeys[1], shape=(d,))
    k = k - v_star @ k * v_star
    k = k / jnp.linalg.norm(k)
    v = random.normal(subkeys[2], shape=(d,))
    # first orthogonalize v and k_star
    v = v - k_star @ v * k_star
    # then, to orthogonalize v and k, project v on the orthogonal of the part of k that is orthogonal to k_star
    orth_k = k - k_star @ k * k_star
    orth_k = orth_k / jnp.linalg.norm(orth_k)
    v = v - orth_k @ v * orth_k
    v = v / jnp.linalg.norm(v)
    return k, v, k_star, v_star


def initialize_on_sphere(d, key):
    """Initialization of k and v on the sphere."""
    subkeys = random.split(key, 3)
    k_star, v_star = init_star(d, subkeys[0])
    k = random.normal(subkeys[1], shape=(d,))
    k = k / jnp.linalg.norm(k)
    v = random.normal(subkeys[2], shape=(d,))
    v = v / jnp.linalg.norm(v)
    return k, v, k_star, v_star


@partial(jax.jit, static_argnames=["d", "L"])
def R(k, v, k_star, v_star, lambd, gamma, d, L):
    """Closed-form formula for the value of the risk."""
    rho = k @ v
    theta = v @ k_star
    kappa = k @ k_star
    eta = v_star @ k
    nu = v @ v_star

    arg_erf = lambd * jnp.sqrt(d / (2 * (1 + 2 * lambd**2 * gamma**2))) * kappa
    C = 2 * lambd / jnp.sqrt(jnp.pi * (1 + 2 * lambd**2))
    zeta_value = zeta_integ(lambd * jnp.sqrt(d / 2) * kappa, lambd * gamma)

    return (
        gamma**2
        - 2
        * (
            gamma**2 * nu * scs.erf(arg_erf)
            + lambd
            * gamma**2
            * jnp.sqrt(d / 2)
            * eta
            * theta
            / jnp.sqrt(1 + 2 * lambd**2 * gamma**2)
            * erf_prime(arg_erf)
            + lambd**2
            * gamma**4
            * eta
            * rho
            / (1 + 2 * lambd**2 * gamma**2)
            * erf_second(arg_erf)
        )
        + (d / 2 * theta**2 + gamma**2) * zeta_value
        + 4
        * lambd
        * gamma**2
        * jnp.sqrt(d / 2)
        * (
            theta * rho
            - lambd**2 * gamma**2 * rho**2 * kappa / (1 + 2 * lambd**2 * gamma**2)
        )
        / jnp.sqrt(1 + 2 * lambd**2 * gamma**2)
        * scs.erf(arg_erf / jnp.sqrt(1 + 4 * lambd**2 * gamma**2))
        * erf_prime(arg_erf)
        + 4
        / jnp.sqrt(jnp.pi)
        * lambd**2
        * gamma**4
        * rho**2
        / ((1 + 2 * lambd**2 * gamma**2) * jnp.sqrt(1 + 4 * lambd**2 * gamma**2))
        * erf_prime(
            -lambd * jnp.sqrt(d) * kappa / jnp.sqrt(1 + 4 * lambd**2 * gamma**2)
        )
        + (L - 1)
        * (
            8
            / jnp.pi
            * lambd**2
            / jnp.sqrt(1 + 4 * lambd**2)
            / (1 + 2 * lambd**2)
            * rho**2
        )
        + 4 * lambd**2 * (L - 1) * (L - 2) / (jnp.pi * (1 + 2 * lambd**2)) * rho**2
        - 2
        * (L - 1)
        * C
        * rho
        * (
            -jnp.sqrt(d / 2) * theta * scs.erf(arg_erf)
            - lambd
            * gamma**2
            * rho
            / jnp.sqrt(1 + 2 * lambd**2 * gamma**2)
            * erf_prime(arg_erf)
        )
    )


grad_R_k = jax.grad(R, 0)
grad_R_v = jax.grad(R, 1)


# Some arguments of this function are only here to match the signature of the sgd_step
# function.
@partial(jax.jit, static_argnames=["d", "L"])
def pgd_step(params, k_star, v_star, d, L, gamma, lambd, alpha, batch_size, eps, key):
    """One step of PGD."""
    k, v = params
    risk_value = R(k, v, k_star, v_star, lambd, gamma, d, L)
    grad_k_value = grad_R_k(k, v, k_star, v_star, lambd, gamma, d, L)
    grad_v_value = grad_R_v(k, v, k_star, v_star, lambd, gamma, d, L)
    new_k = k - alpha * (grad_k_value - k @ grad_k_value * k)
    new_v = v - alpha * (grad_v_value - v @ grad_v_value * v)
    new_k = new_k / jnp.linalg.norm(new_k)
    new_v = new_v / jnp.linalg.norm(new_v)
    new_params = (new_k, new_v)
    kappa = new_k @ k_star
    nu = new_v @ v_star
    theta = new_v @ k_star
    eta = new_k @ v_star
    rho = new_k @ new_v
    return (
        new_params,
        risk_value,
        kappa,
        nu,
        theta,
        eta,
        rho,
    )


# X is a Lxd matrix
@jax.jit
def T(X, k, v, lambd):
    """Our simplified attention-based predictor."""
    return scs.erf(lambd * X @ k).T @ (X @ v)


batched_T = jax.vmap(T, in_axes=[0, None, None, None], out_axes=0)


def risk_fn(X, y, params, lambd):
    """Loss with mean-squared error for our problem."""
    k, v = params
    return jnp.mean((batched_T(X, k, v, lambd) - y) ** 2)


grad_fn = jax.value_and_grad(risk_fn, argnums=2)


@partial(jax.jit, static_argnames=["batch_size", "d", "L"])
def generate_batch(k_star, v_star, batch_size, d, L, eps, gamma, key):
    """Generates one batch of data of the single-location regression distribution."""
    subkeys = random.split(key, 3)
    X_1 = jax.random.multivariate_normal(
        subkeys[0],
        jnp.sqrt(d / 2) * k_star,
        gamma**2 * jnp.eye(d),
        shape=(batch_size, 1),
    )
    other_X_i = jax.random.normal(subkeys[1], shape=(batch_size, L - 1, d))
    xi = eps * jax.random.normal(subkeys[2], shape=(batch_size,))
    y = X_1[:, 0, :] @ v_star + xi
    return jnp.concatenate([X_1, other_X_i], axis=1), y


@partial(jax.jit, static_argnames=["batch_size", "d", "L"])
def sgd_step(params, k_star, v_star, d, L, gamma, lambd, alpha, batch_size, eps, key):
    """One step of SGD."""
    X, y = generate_batch(k_star, v_star, batch_size, d, L, eps, gamma, key)
    risk, grad = grad_fn(X, y, params, lambd)
    k, v = params
    grad_k_value, grad_v_value = grad
    new_k = k - alpha * (grad_k_value - k @ grad_k_value * k)
    new_v = v - alpha * (grad_v_value - v @ grad_v_value * v)
    k, v = new_k / jnp.linalg.norm(new_k), new_v / jnp.linalg.norm(new_v)
    kappa = k @ k_star
    nu = v @ v_star
    theta = v @ k_star
    eta = k @ v_star
    rho = k @ v
    return (k, v), risk, kappa, nu, theta, eta, rho
