"""Spec for regime-level EV1 taste shocks under the brute-force solver.

Taste shocks are a model property declared on the `Regime` (`taste_shocks=
ExtremeValueTasteShocks()`), with the scale a runtime param under the pseudo-
function name `taste_shocks`. The solve replaces the hard max over discrete-action
axes with the smoothed expected maximum `scale * logsumexp(Qc / scale)` after the
masked max over continuous actions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm import logsum
from _lcm.regime_building.max_Q_over_a import draw_taste_shock_noise
from lcm.exceptions import InvalidParamsError, ModelInitializationError
from lcm.taste_shocks import (
    ExtremeValueTasteShocks,
)
from tests.test_models import taste_shocks_toy


def test_logsum_matches_logsumexp_identity():
    """`logsum_and_softmax` equals `scale * log(sum(exp(values / scale)))`."""
    values = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    scale = 1.0

    got_logsum, _ = logsum.logsum_and_softmax(values=values, scale=scale, axes=(0,))

    expected = np.log(np.sum(np.exp(np.asarray(values)), axis=0))
    aaae(got_logsum, expected, decimal=12)


def test_logsum_small_scale_approaches_hard_max():
    """With a small scale, the smoothed max is numerically the hard max."""
    values = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    got_logsum, _ = logsum.logsum_and_softmax(values=values, scale=0.1, axes=(0,))

    aaae(got_logsum, jnp.array([4.0, 5.0]), decimal=5)


def test_logsum_scale_zero_is_exactly_hard_max_and_one_hot():
    """`scale = 0.0` returns the hard max and one-hot probabilities, no NaN."""
    values = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    got_logsum, got_probs = logsum.logsum_and_softmax(
        values=values, scale=0.0, axes=(0,)
    )

    aaae(got_logsum, jnp.array([4.0, 5.0]), decimal=15)
    aaae(got_probs, jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]), decimal=15)


def test_logsum_multiple_axes():
    """Aggregation over all discrete axes collapses them jointly."""
    values = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    scale = 0.5

    got_logsum, _ = logsum.logsum_and_softmax(values=values, scale=scale, axes=(0, 1))

    expected = scale * np.log(np.sum(np.exp(np.asarray(values) / scale)))
    aaae(got_logsum, expected, decimal=10)


def test_logsum_masked_rows_get_zero_probability():
    """`-inf` (infeasible) entries contribute zero probability and no NaN."""
    values = jnp.array([[1.0], [-jnp.inf]])

    got_logsum, got_probs = logsum.logsum_and_softmax(
        values=values, scale=0.7, axes=(0,)
    )

    aaae(got_logsum, jnp.array([1.0]), decimal=12)
    aaae(got_probs, jnp.array([[1.0], [0.0]]), decimal=12)


def test_logsum_all_masked_is_neg_inf_without_nan():
    """A fully infeasible slice yields `-inf` value and finite (zero) probs."""
    values = jnp.array([[-jnp.inf], [-jnp.inf]])

    got_logsum, got_probs = logsum.logsum_and_softmax(
        values=values, scale=0.7, axes=(0,)
    )

    assert bool(jnp.isneginf(got_logsum).all())
    assert not bool(jnp.isnan(got_probs).any())


def _reference_alive_V(*, scale: float, discount_factor: float) -> np.ndarray:
    """Recompute the smoothed period-0 value function with numpy on the same grids."""
    wealth = np.linspace(1.0, 10.0, 6)
    terminal_wealth = np.linspace(0.0, 12.0, 25)
    consumption = np.linspace(0.5, 5.0, 8)
    v_done = np.log(terminal_wealth + 1.0)

    qc = np.empty((wealth.size, 2))
    for d in (0, 1):
        next_w = wealth[:, None] - consumption[None, :] + taste_shocks_toy.WAGE * d
        continuation = np.interp(next_w, terminal_wealth, v_done)
        q = (
            np.log(consumption)[None, :]
            - taste_shocks_toy.KAPPA * d
            + discount_factor * continuation
        )
        feasible = consumption[None, :] <= wealth[:, None]
        qc[:, d] = np.max(np.where(feasible, q, -np.inf), axis=1)

    shifted = qc - qc.max(axis=1, keepdims=True)
    return qc.max(axis=1) + scale * np.log(np.sum(np.exp(shifted / scale), axis=1))


def test_brute_force_solve_matches_reference_logsum_value_function():
    """Solved V in the decision period equals the numpy logsum reference exactly.

    Both compute the same grid-restricted `Qc`, so agreement is up to float
    precision, not up to grid error.
    """
    scale = 0.2
    discount_factor = 0.95
    model = taste_shocks_toy.get_model()
    params = taste_shocks_toy.get_params(scale=scale, discount_factor=discount_factor)

    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    got = np.asarray(period_to_regime_to_V_arr[0]["alive"])
    expected = _reference_alive_V(scale=scale, discount_factor=discount_factor)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-8)


def test_smoothed_value_weakly_exceeds_hard_max_value():
    """The EV1 expected maximum dominates the hard max pointwise."""
    model = taste_shocks_toy.get_model()
    smoothed = model.solve(
        params=taste_shocks_toy.get_params(scale=0.5), log_level="debug"
    )
    hard = model.solve(params=taste_shocks_toy.get_params(scale=0.0), log_level="debug")

    assert bool(jnp.all(smoothed[0]["alive"] >= hard[0]["alive"] - 1e-12))


def test_params_template_contains_taste_shock_scale():
    """The params template exposes the scale under the `taste_shocks` pseudo-entry."""
    model = taste_shocks_toy.get_model()

    template = model.get_params_template()

    assert "scale" in template["alive"]["taste_shocks"]


def test_taste_shock_noise_is_mean_zero():
    """The simulation taste-shock draw has mean zero.

    The solve uses `scale * logsumexp(Qc / scale)`, which equals the expected
    maximum only for mean-zero EV1 shocks. A raw Gumbel(0, 1) draw has mean
    `EULER_GAMMA`, so the draw is centered by it; the sample mean of a large
    draw is therefore zero up to Monte Carlo error.
    """
    scale = jnp.array(2.0)
    noise = draw_taste_shock_noise(key=jax.random.key(0), shape=(200_000,), scale=scale)
    np.testing.assert_allclose(float(jnp.mean(noise)), 0.0, atol=0.03)


def test_expected_max_with_taste_shock_noise_matches_logsum():
    """The expected realized maximum equals the smoothed (logsum) solve value.

    With the centered draw, `E[max_d (v_d + noise_d)]` reproduces
    `scale * logsumexp(v / scale)` — the value the solve assigns — so solved
    value and simulated realized value agree up to Monte Carlo error.
    """
    values = jnp.array([0.0, 1.0, 0.5])
    scale = jnp.array(0.7)
    n_draws = 400_000

    noise = draw_taste_shock_noise(
        key=jax.random.key(1), shape=(n_draws, values.size), scale=scale
    )
    simulated_expected_max = float(jnp.mean(jnp.max(values + noise, axis=1)))

    solved_value, _ = logsum.logsum_and_softmax(values=values, scale=scale, axes=(0,))
    np.testing.assert_allclose(simulated_expected_max, float(solved_value), atol=0.02)


def test_negative_taste_shock_scale_raises():
    """A negative taste-shock scale is rejected with a clear error.

    `scale = 0` (the hard maximum) stays valid; only a negative scale, which
    would multiply the Gumbel draw by a negative number in simulation, is an
    error.
    """
    model = taste_shocks_toy.get_model()
    params = taste_shocks_toy.get_params(scale=-0.1)
    with pytest.raises(InvalidParamsError, match="scale"):
        model.solve(params=params, log_level="debug")


def test_taste_shocks_without_discrete_action_raises():
    """Declaring taste shocks on a regime without discrete actions is an error."""
    no_discrete_action = taste_shocks_toy.alive.replace(
        actions={"consumption": taste_shocks_toy.CONSUMPTION_GRID},
        taste_shocks=ExtremeValueTasteShocks(),
    )
    with pytest.raises(ModelInitializationError, match="discrete action"):
        taste_shocks_toy.Model(
            regimes={"alive": no_discrete_action, "done": taste_shocks_toy.done},
            ages=taste_shocks_toy.AgeGrid(start=40, stop=41, step="Y"),
            regime_id_class=taste_shocks_toy.ToyRegimeId,
        )
