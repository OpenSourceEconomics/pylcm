from copy import deepcopy

import jax
from jax import numpy as jnp
from jax.scipy.stats.norm import cdf

from lcm.interfaces import InternalRegime
from lcm.typing import Float1D, FloatND, ParamsDict


def discretized_uniform_distribution(
    n_points: int, start: float, stop: float
) -> tuple[Float1D, FloatND]:
    """Discretize the specified uniform distribution.

    Args:
        n_points: Number of discretization points
        start: Min value of the distribution.
        stop: Max value of the distribution.

    Returns:
        Values at discretization points and transition matrix.

    """
    return jnp.linspace(start=start, stop=stop, num=n_points), jnp.full(
        (n_points, n_points), fill_value=1 / n_points
    )


def discretized_normal_distribution(
    n_points: int, mu_eps: float, sigma_eps: float, n_std: int = 3
) -> tuple[Float1D, FloatND]:
    """Discretize the specified normal distribution.

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

    return x, jnp.full((n_points, n_points), fill_value=P)


def tauchen(
    n_points: int, rho: float, sigma_eps: float, mu_eps: float = 0.0, n_std: int = 2
) -> tuple[Float1D, FloatND]:
    r"""Discretize the specified AR1 process with the 'Tauchen'-method.

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
    P = _fill_tauchen(x, n_points, rho, sigma_eps, half_step)

    # shifts the state values by the long run mean of y_t
    mu_eps = mu_eps / (1 - rho)

    return x + mu_eps, P


def _fill_tauchen(
    x: Float1D, n: int, rho: float, sigma: float, half_step: Float1D
) -> FloatND:
    P = jnp.empty((n, n))
    for i in range(n):
        P = P.at[i, 0].set(cdf((x[0] - rho * x[i] + half_step) / sigma))
        P = P.at[i, -1].set(1 - cdf((x[n - 1] - rho * x[i] - half_step) / sigma))
        for j in range(1, n - 1):
            z = x[j] - rho * x[i]
            P = P.at[i, j].set(
                cdf((z + half_step) / sigma) - cdf((z - half_step) / sigma)
            )
    return P


def rouwenhorst(
    rho: float, sigma_eps: float, n_points: int, mu_eps: float = 0.0
) -> tuple[Float1D, FloatND]:
    r"""Discretize the specified AR1 process with the 'Rouwenhorst'-method.

    X_t = \rho*X_(t-1) + \\eps_t, \\eps_t= N(mu_eps, sigma_eps)

    The distance between the outermost points and the mean is always two times
    the standard deviation.

    Args:
        n_points: Number of discretization points
        rho: See AR1-process equation.
        mu_eps: See AR1-process equation.
        sigma_eps: See AR1-process equation.

    Returns:
        Values at discretization points and transition matrix.

    """
    q = (rho + 1) / 2
    nu = jnp.sqrt((n_points - 1) / (1 - rho**2)) * sigma_eps
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

    return jnp.linspace(
        mu_eps / (1.0 - rho) - nu, mu_eps / (1.0 - rho) + nu, n_points
    ), P.T


SHOCK_DISCRETIZATION_FUNCTIONS = {
    "uniform": discretized_uniform_distribution,
    "normal": discretized_normal_distribution,
    "tauchen": tauchen,
    "rouwenhorst": rouwenhorst,
}


def uniform(params: ParamsDict, key: Float1D, prev_value: Float1D) -> Float1D:  # noqa: ARG001
    return jax.random.uniform(minval=params["start"], maxval=params["stop"], key=key)


def normal(params: ParamsDict, key: Float1D, prev_value: Float1D) -> Float1D:  # noqa: ARG001
    return jax.random.normal(key=key) * params["sigma_eps"] + params["mu_eps"]


def ar1_tauchen(params: ParamsDict, key: Float1D, prev_value: Float1D) -> Float1D:
    return prev_value * params["rho"] + jax.random.normal(key=key) * params["sigma_eps"]


def ar1_rouwenhorst(params: ParamsDict, key: Float1D, prev_value: Float1D) -> Float1D:
    return prev_value * params["rho"] + jax.random.normal(key=key) * params["sigma_eps"]


SHOCK_CALCULATION_FUNCTIONS = {
    "uniform": uniform,
    "normal": normal,
    "tauchen": ar1_tauchen,
    "rouwenhorst": ar1_rouwenhorst,
}


def pre_compute_shock_probabilities(
    internal_regimes: dict[str, InternalRegime], params: ParamsDict
) -> ParamsDict:
    """Pre-compute the discretized probabilities for shocks.

    The parameters for the transition functions will be augmented with the
    pre-calculated probability distributions of the given shock.

    Args:
        internal_regimes: The internal regimes containing the shocks.
        params: The parameters that need augmentation as given by the user.

    Returns:
        A ParamsDict where every transition function that uses a pre-implemented shock
        is augmented with a new entry containing the shocks discretized probability
        distribution.

    """
    new_params = deepcopy(params)
    for regime_name, regime in internal_regimes.items():
        transition_info = regime.transition_info
        need_precompute = transition_info.index[
            ~transition_info["type"].isin(["custom", "none"])
        ].tolist()

        for trans_name in need_precompute:
            n_points = regime.gridspecs[trans_name.removeprefix("next_")].n_points
            new_params[regime_name][trans_name]["pre_computed"] = (
                SHOCK_DISCRETIZATION_FUNCTIONS[transition_info.loc[trans_name, "type"]](
                    **(params[regime_name][trans_name] | {"n_points": n_points})
                )[1]
            )
    return new_params


def fill_shock_grids(
    internal_regimes: dict[str, InternalRegime], params: ParamsDict
) -> dict[str, InternalRegime]:
    """Fill the shock grids.

    As the values for the shock grids depend on the parameters that the user supplies,
    they have to be calculated at the start of the solution or simulation respectively.

    Args:
        internal_regimes: The internal regimes whose grids need to be replaced.
        params: The parameters as given by the user.

    Returns:
        The original internal regimes, but with the filled shock grids.

    """
    new_internal_regimes = deepcopy(internal_regimes)
    for regime_name, regime in new_internal_regimes.items():
        transition_info = regime.transition_info
        need_precompute = transition_info.index[
            ~transition_info["type"].isin(["custom", "none"])
        ].tolist()
        for trans_name in need_precompute:
            n_points = regime.gridspecs[trans_name.removeprefix("next_")].n_points
            param_copy = params[regime_name][trans_name].copy()
            param_copy.pop("pre_computed")
            new_values = SHOCK_DISCRETIZATION_FUNCTIONS[
                transition_info.loc[trans_name, "type"]
            ](**(param_copy | {"n_points": n_points}))[0]

            if trans_name.removeprefix("next_") in regime.state_action_spaces.states:
                new_internal_regimes[
                    regime_name
                ].state_action_spaces = regime.state_action_spaces.replace(
                    states=regime.state_action_spaces.states
                    | {trans_name.removeprefix("next_"): new_values}
                )
    return new_internal_regimes
