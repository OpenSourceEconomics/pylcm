from dataclasses import dataclass
from typing import Literal

import jax
from jax import numpy as jnp
from jax.scipy.stats.norm import cdf

from lcm.typing import Float1D, FloatND, ParamsDict


@dataclass(frozen=True, kw_only=True)
class Shock:
    distribution_type: Literal["uniform", "normal", "tauchen", "rouwenhorst"]
    n_points: int
    shock_params: ParamsDict

    def get_gridpoints(self) -> Float1D:
        return SHOCK_GRIDPOINT_FUNCTIONS[self.distribution_type](
            self.n_points, **self.shock_params
        )

    def get_transition_probs(self) -> FloatND:
        return SHOCK_TRANSITION_PROBABILITY_FUNCTIONS[self.distribution_type](
            self.n_points, **self.shock_params
        )

    def draw_shock(self, key: FloatND, prev_value: Float1D) -> Float1D:
        return SHOCK_CALCULATION_FUNCTIONS[self.distribution_type](
            params=self.shock_params, key=key, prev_value=prev_value
        )


def _discretized_uniform_distribution_gridpoints(
    n_points: int, start: float = 0, stop: float = 1
) -> FloatND:
    """Calculate the gridpoints for a discretized uniform distribution.

    Args:
        n_points: Number of discretization points
        start: Min value of the distribution.
        stop: Max value of the distribution.

    Returns:
        Values at discretization points and transition matrix.

    """
    return jnp.linspace(start=start, stop=stop, num=n_points)


def _discretized_uniform_distribution_probs(n_points: int) -> FloatND:
    """Calculate the transition probabilities for a discretized uniform distribution.

    Args:
        n_points: Number of discretization points
        start: Min value of the distribution.
        stop: Max value of the distribution.

    Returns:
        Values at discretization points and transition matrix.

    """
    return jnp.full((n_points, n_points), fill_value=1 / n_points)


def _discretized_normal_distribution_gridpoints(
    n_points: int, mu_eps: float = 0.0, sigma_eps: float = 1.0, n_std: int = 3
) -> FloatND:
    """Calculate the gridpoints for a discretized uniform distribution.

    Args:
        n_points: Number of discretization points
        mu_eps: Mean of the distribution.
        sigma_eps: Std. Dev. of the distribution.
        n_std: Distance from mean to the values of the lowest and highest
            discretized value.

    Returns:
        Values at discretization points and transition matrix.

    """
    x_min = mu_eps - n_std * sigma_eps
    x_max = mu_eps + n_std * sigma_eps
    return jnp.linspace(start=x_min, stop=x_max, num=n_points)


def _discretized_normal_distribution_probs(
    n_points: int, mu_eps: float = 0.0, sigma_eps: float = 1.0, n_std: int = 3
) -> FloatND:
    """Calculate the transition probabilities for a discretized normal distribution.

    Args:
        n_points: Number of discretization points
        mu_eps: Mean of the distribution.
        sigma_eps: Std. Dev. of the distribution.
        n_std: Distance from mean to the values of the lowest and highest
            discretized value.

    Returns:
        Values at discretization points and transition matrix.

    """
    x_min = mu_eps - n_std * sigma_eps
    x_max = mu_eps + n_std * sigma_eps
    x, stepsize = jnp.linspace(start=x_min, stop=x_max, num=n_points, retstep=True)
    P = jnp.zeros(n_points)
    P = P.at[1:].set(jnp.diff(cdf(x + 0.5 * stepsize, loc=mu_eps, scale=sigma_eps)))
    P = P.at[0].set(cdf(x_min + 0.5 * stepsize, loc=mu_eps, scale=sigma_eps))
    P = P.at[-1].set(1 - cdf(x_max - 0.5 * stepsize, loc=mu_eps, scale=sigma_eps))

    return jnp.full((n_points, n_points), fill_value=P)


def _tauchen_gridpoints(
    n_points: int,
    rho: float,
    sigma_eps: float = 1.0,
    mu_eps: float = 0.0,
    n_std: int = 2,
) -> FloatND:
    r"""Calculate the gridpoints for an AR1 process with the 'Tauchen'-method.

    X_t = \rho*X_(t-1) + \\eps_t, \\eps_t= N(0, sigma_eps)

    Args:
        n_points: Number of discretization points
        rho: See AR1-process equation.
        mu_eps: See AR1-process equation.
        sigma_eps: See AR1-process equation.
        n_std: Distance from mean to the values of the lowest and highest
            discretized value.

    Returns:
        Values at discretization points and transition matrix.

    """
    # standard deviation of demeaned y_t
    std_y = jnp.sqrt(sigma_eps**2 / (1 - rho**2))

    # top of discrete state space for demeaned y_t
    x_max = n_std * std_y

    # bottom of discrete state space for demeaned y_t
    x_min = -x_max

    # discretized state space for demeaned y_t
    x = jnp.linspace(x_min, x_max, n_points)

    # shifts the state values by the long run mean of y_t
    mu_eps = mu_eps / (1 - rho)

    return x + mu_eps


def _tauchen_probs(
    n_points: int,
    rho: float,
    sigma_eps: float = 1.0,
    mu_eps: float = 0.0,
    n_std: int = 2,
) -> FloatND:
    r"""Calculate the transition probs for an AR1 process with the 'Tauchen'-method.

    X_t = \rho*X_(t-1) + \\eps_t, \\eps_t= N(0, sigma_eps)

    Args:
        n_points: Number of discretization points
        rho: See AR1-process equation.
        mu_eps: See AR1-process equation.
        sigma_eps: See AR1-process equation.
        n_std: Distance from mean to the values of the lowest and highest
            discretized value.

    Returns:
        Values at discretization points and transition matrix.

    """
    # standard deviation of demeaned y_t
    std_y = jnp.sqrt(sigma_eps**2 / (1 - rho**2))

    # top of discrete state space for demeaned y_t
    x_max = n_std * std_y

    # bottom of discrete state space for demeaned y_t
    x_min = -x_max

    # discretized state space for demeaned y_t
    x = jnp.linspace(x_min, x_max, n_points)

    step = (x_max - x_min) / (n_points - 1)
    half_step = 0.5 * step

    # approximate Markov transition matrix for
    # demeaned y_t
    P = jnp.empty((n_points, n_points))
    for i in range(n_points):
        P = P.at[i, 0].set(cdf((x[0] - rho * x[i] + half_step) / sigma_eps))
        P = P.at[i, -1].set(
            1 - cdf((x[n_points - 1] - rho * x[i] - half_step) / sigma_eps)
        )
        for j in range(1, n_points - 1):
            z = x[j] - rho * x[i]
            P = P.at[i, j].set(
                cdf((z + half_step) / sigma_eps) - cdf((z - half_step) / sigma_eps)
            )
    # shifts the state values by the long run mean of y_t
    mu_eps = mu_eps / (1 - rho)

    return P


def _rouwenhorst_gridpoints(
    n_points: int, rho: float, sigma_eps: float = 1.0, mu_eps: float = 0.0
) -> FloatND:
    r"""Calculate the gridpoints for the AR1 process with the 'Rouwenhorst'-method.

    X_t = \rho*X_(t-1) + \\eps_t, \\eps_t= N(mu_eps, sigma_eps)

    The distance between the outermost points and the mean is always two times
    the standard deviation.

    Args:
        n_points: Number of discretization points
        rho: See AR1-process equation.
        mu_eps: See AR1-process equation.
        sigma_eps: See AR1-process equation.

    Returns:
        Values at discretization points.

    """
    nu = jnp.sqrt((n_points - 1) / (1 - rho**2)) * sigma_eps

    return jnp.linspace(mu_eps / (1.0 - rho) - nu, mu_eps / (1.0 - rho) + nu, n_points)


def _rouwenhorst_probs(n_points: int, rho: float) -> FloatND:
    r"""Calculate the gridpoints for the AR1 process with the 'Rouwenhorst'-method.

    X_t = \rho*X_(t-1) + \\eps_t, \\eps_t= N(mu_eps, sigma_eps)

    The distance between the outermost points and the mean is always two times
    the standard deviation.

    Args:
        n_points: Number of discretization points
        rho: See AR1-process equation.
        mu_eps: See AR1-process equation.
        sigma_eps: See AR1-process equation.

    Returns:
        Transition probability matrix.

    """
    q = (rho + 1) / 2
    P = jnp.zeros((n_points, n_points))

    P = P.at[0, 0].set(q)
    P = P.at[0, 1].set(1 - q)
    P = P.at[1, 0].set(1 - q)
    P = P.at[1, 1].set(q)

    for i in range(2, n_points):
        P11 = jnp.zeros((i + 1, i + 1))
        P12 = jnp.zeros((i + 1, i + 1))
        P21 = jnp.zeros((i + 1, i + 1))
        P22 = jnp.zeros((i + 1, i + 1))

        P11 = P11.at[0:i, 0:i].set(P[0:i, 0:i])
        P12 = P12.at[0:i, 1 : i + 1].set(P[0:i, 0:i])
        P21 = P21.at[1 : i + 1, 0:i].set(P[0:i, 0:i])
        P22 = P22.at[1 : i + 1, 1 : i + 1].set(P[0:i, 0:i])

        P = P.at[0 : i + 1, 0 : i + 1].set(
            q * P11 + (1 - q) * P12 + (1 - q) * P21 + q * P22
        )
        P = P.at[1:i, :].set(P[1:i, :] / 2)

    return P


SHOCK_TRANSITION_PROBABILITY_FUNCTIONS = {
    "uniform": _discretized_uniform_distribution_probs,
    "normal": _discretized_normal_distribution_probs,
    "tauchen": _tauchen_probs,
    "rouwenhorst": _rouwenhorst_probs,
}

SHOCK_GRIDPOINT_FUNCTIONS = {
    "uniform": _discretized_uniform_distribution_gridpoints,
    "normal": _discretized_normal_distribution_gridpoints,
    "tauchen": _tauchen_gridpoints,
    "rouwenhorst": _rouwenhorst_gridpoints,
}


def uniform(params: ParamsDict, key: FloatND, prev_value: Float1D) -> Float1D:  # noqa: ARG001
    return jax.random.uniform(key=key)


def normal(params: ParamsDict, key: FloatND, prev_value: Float1D) -> Float1D:  # noqa: ARG001
    return jax.random.normal(key=key)


def ar1_tauchen(params: ParamsDict, key: FloatND, prev_value: Float1D) -> Float1D:
    return prev_value * params["rho"] + jax.random.normal(key=key)


def ar1_rouwenhorst(params: ParamsDict, key: FloatND, prev_value: Float1D) -> Float1D:
    return prev_value * params["rho"] + jax.random.normal(key=key)


SHOCK_CALCULATION_FUNCTIONS = {
    "uniform": uniform,
    "normal": normal,
    "tauchen": ar1_tauchen,
    "rouwenhorst": ar1_rouwenhorst,
}
