from jax import Array
from jax import numpy as jnp
from jax.scipy.stats.norm import cdf

from lcm.exceptions import ShockInitializationError
from lcm.typing import Float1D, FloatND


class UniformShock:
    values: Array
    transition_probs: Array

    def init(self, n: int, x_min: float, x_max: float) -> None:
        self.values, self.transition_probs = discretized_uniform_distribution(
            n, x_min, x_max
        )


class IIDShock:
    values: Array
    transition_probs: Array

    def init(self, n: int, mu_eps: float, sigma_eps: float, n_std: int = 3) -> None:
        self.values, self.transition_probs = discretized_normal_distribution(
            n, mu_eps, sigma_eps, n_std
        )


class AR1Shock:
    values: Array
    transition_probs: Array

    def init(
        self,
        method: str,
        n: int,
        rho: float,
        sigma_eps: float,
        mu_eps: float = 0.0,
        n_std: int = 1,
    ) -> None:
        if method == "tauchen":
            values, transition_probs = tauchen(
                n=n, rho=rho, sigma_eps=sigma_eps, mu_eps=mu_eps, n_std=n_std
            )
            self.values = values
            self.transition_probs = transition_probs
        elif method == "rouwenhorst":
            values, transition_probs = rouwenhorst(
                n=n, rho=rho, sigma_eps=sigma_eps, mu_eps=mu_eps, n_std=n_std
            )
            self.values = values
            self.transition_probs = transition_probs
        else:
            raise ShockInitializationError(
                "The requested discretization method does not exist.",
            )


def discretized_uniform_distribution(
    n: int, x_min: float, x_max: float
) -> tuple[Float1D, FloatND]:
    """Discretize the specified uniform distribution.

    Args:
        n: Number of discretization points
        x_min: Min value of the distribution.
        x_max: Max value of the distribution.

    Returns:
        Values at discretization points and transition matrix.

    """
    return jnp.linspace(start=x_min, stop=x_max, num=n), jnp.full(
        (n, n), fill_value=1 / n
    )


def discretized_normal_distribution(
    n: int, mu_eps: float, sigma_eps: float, n_std: int = 3
) -> tuple[Float1D, FloatND]:
    """Discretize the specified normal distribution.

    Args:
        n: Number of discretization points
        mu_eps: Mean of the distribution.
        sigma_eps: Std. Dev. of the distribution.
        n_std: Distance from mean to the values of the lowest and highest
            discretized value.

    Returns:
        Values at discretization points and transition matrix.

    """
    x_min = mu_eps - n_std * sigma_eps
    x_max = mu_eps + n_std * sigma_eps
    x, stepsize = jnp.linspace(start=x_min, stop=x_max, num=n, retstep=True)
    P = jnp.zeros(n)
    P = P.at[1:].set(jnp.diff(cdf(x + 0.5 * stepsize, loc=mu_eps, scale=sigma_eps)))
    P = P.at[0].set(cdf(x_min + 0.5 * stepsize, loc=mu_eps, scale=sigma_eps))
    P = P.at[-1].set(1 - cdf(x_max - 0.5 * stepsize, loc=mu_eps, scale=sigma_eps))

    return x, jnp.full((n, n), fill_value=P)


def tauchen(
    n: int, rho: float, sigma_eps: float, mu_eps: float = 0.0, n_std: int = 2
) -> tuple[Float1D, FloatND]:
    r"""Discretize the specified AR1 process with the 'Tauchen'-method.

    X_t = \rho*X_(t-1) + \\eps_t, \\eps_t= N(mu_eps, sigma_eps)

    Args:
        n: Number of discretization points
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
    x = jnp.linspace(x_min, x_max, n)

    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step

    # approximate Markov transition matrix for
    # demeaned y_t
    P = _fill_tauchen(x, n, rho, sigma_eps, half_step)

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
    rho: float, sigma_eps: float, n: int, mu_eps: float = 0.0, n_std: int = 2
) -> tuple[Float1D, FloatND]:
    r"""Discretize the specified AR1 process with the 'Rouwenhorst'-method.

    X_t = \rho*X_(t-1) + \\eps_t, \\eps_t= N(mu_eps, sigma_eps)

    Args:
        n: Number of discretization points
        rho: See AR1-process equation.
        mu_eps: See AR1-process equation.
        sigma_eps: See AR1-process equation.
        n_std: Distance from mean to the values of the lowest and highest
            discretized value.

    Returns:
        Values at discretization points and transition matrix.

    """
    q = (rho + 1) / 2
    nu = jnp.sqrt((n - 1) / (1 - rho**2)) * sigma_eps
    P = jnp.zeros((n, n))

    P = P.at[0, 0].set(q)
    P = P.at[0, 1].set(1 - q)
    P = P.at[1, 0].set(1 - q)
    P = P.at[1, 1].set(q)

    for i in range(2, n):
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

    return jnp.linspace(mu_eps / (1.0 - rho) - nu, mu_eps / (1.0 - rho) + nu, n), P.T
