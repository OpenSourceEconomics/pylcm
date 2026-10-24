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
from lcm import EULER_GAMMA, AgeGrid, DiscreteGrid, Model, Regime, categorical
from lcm.exceptions import InvalidParamsError, ModelInitializationError
from lcm.solvers import GridSearch
from lcm.taste_shocks import (
    ExtremeValueTasteShocks,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserParamsNode,
)
from tests.conftest import DECIMAL_PRECISION
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


def test_logsum_small_scale_does_not_overflow():
    """A small positive scale keeps the smoothed value at the finite hard max.

    `values / scale` overflows to `inf` in float32 when the scale is small
    enough (here a normal float32 scale), but the smoothed maximum is bounded
    by the hard max, so the result must stay finite with finite probabilities.
    """
    values = jnp.array([40.0, 50.0], dtype=jnp.float32)

    got_logsum, got_probs = logsum.logsum_and_softmax(
        values=values, scale=jnp.float32(1e-37), axes=(0,)
    )

    assert bool(jnp.isfinite(got_logsum).all())
    aaae(got_logsum, jnp.float32(50.0), decimal=3)
    assert not bool(jnp.isnan(got_probs).any())
    aaae(got_probs, jnp.array([0.0, 1.0], dtype=jnp.float32), decimal=6)


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


@pytest.mark.slow
def test_brute_force_solve_matches_reference_logsum_value_function():
    """Solved V in the decision period equals the numpy logsum reference exactly.

    Both compute the same grid-restricted `Qc`, so agreement is up to float
    precision, not up to grid error.
    """
    scale = 0.2
    discount_factor = 0.95
    model = taste_shocks_toy.get_model()
    params = taste_shocks_toy.get_params(scale=scale, discount_factor=discount_factor)

    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug").values

    got = np.asarray(period_to_regime_to_V_arr[0]["alive"])
    expected = _reference_alive_V(scale=scale, discount_factor=discount_factor)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.slow
def test_smoothed_value_weakly_exceeds_hard_max_value():
    """The EV1 expected maximum dominates the hard max pointwise.

    The smoothed value increases in the scale and tends to the hard max as the
    scale vanishes, so a vanishing-scale solve serves as the hard-max baseline.
    """
    model = taste_shocks_toy.get_model()
    smoothed = model.solve(
        params=taste_shocks_toy.get_params(scale=0.5), log_level="debug"
    ).values
    hard = model.solve(
        params=taste_shocks_toy.get_params(scale=1e-8), log_level="debug"
    ).values

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


@pytest.mark.slow
@pytest.mark.parametrize("scale", [0.0, -0.1])
def test_nonpositive_taste_shock_scale_raises(scale):
    """A taste-shock scale that is not strictly positive is rejected.

    Declaring taste shocks means opting into smoothing, so the scale must be
    positive. The hard maximum is the no-taste-shocks model, not `scale = 0`.
    """
    model = taste_shocks_toy.get_model()
    params = taste_shocks_toy.get_params(scale=scale)
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
            edges={"alive": {"done": 40}},
            regimes={"alive": no_discrete_action, "done": taste_shocks_toy.done},
            ages=taste_shocks_toy.AgeGrid(start=40, inclusive_stop=41, step="Y"),
            regime_id_class=taste_shocks_toy.ToyRegimeId,
            initial_nodes={40: "alive"},
        )


def _utility_alive_location_zero(
    *,
    consumption: ContinuousAction,
    work: DiscreteAction,
    kappa: float,
    taste_shock_scale: float,
) -> FloatND:
    return (
        taste_shocks_toy.utility_alive(consumption=consumption, work=work, kappa=kappa)
        + taste_shock_scale * EULER_GAMMA
    )


def _get_location_zero_model() -> Model:
    alive = taste_shocks_toy.alive.replace(
        functions={"utility": _utility_alive_location_zero}
    )
    return Model(
        edges={"alive": {"done": 40}},
        regimes={"alive": alive, "done": taste_shocks_toy.done},
        ages=AgeGrid(start=40, inclusive_stop=41, step="Y"),
        regime_id_class=taste_shocks_toy.ToyRegimeId,
        initial_nodes={40: "alive"},
    )


@pytest.mark.slow
def test_location_zero_utility_shift_moves_value_by_constant_and_keeps_choices():
    """Adding `scale * EULER_GAMMA` to utility gives location-0 EV1 shocks.

    Under the default linear Koopmans aggregator, the constant enters every
    discrete action's `Q` with unit slope. In this toy every action leads to
    the same shock-free terminal regime, so each `Q` moves by the same
    constant: the decision-period value rises by exactly `scale * EULER_GAMMA`
    and the same seed draws the same discrete choices.
    """
    scale = 0.2
    discount_factor = 0.95
    wealth = np.linspace(1.0, 10.0, 6)
    n_subjects = wealth.size
    initial_conditions = {
        "age": jnp.full(n_subjects, 40.0),
        "wealth": jnp.asarray(wealth),
        "regime_id": jnp.full(
            n_subjects, taste_shocks_toy.ToyRegimeId.alive, dtype=jnp.int32
        ),
    }
    mean_zero_params = taste_shocks_toy.get_params(
        scale=scale, discount_factor=discount_factor
    )
    location_zero_params = {
        "discount_factor": discount_factor,
        "alive": {
            "utility": {"kappa": taste_shocks_toy.KAPPA, "taste_shock_scale": scale},
            "next_wealth": {"wage": taste_shocks_toy.WAGE},
            "taste_shocks": {"scale": scale},
        },
    }
    mean_zero_model = taste_shocks_toy.get_model()
    location_zero_model = _get_location_zero_model()

    mean_zero_V = mean_zero_model.solve(
        params=mean_zero_params, log_level="debug"
    ).values
    location_zero_V = location_zero_model.solve(
        params=location_zero_params, log_level="debug"
    ).values

    aaae(
        location_zero_V[0]["alive"],
        mean_zero_V[0]["alive"] + scale * EULER_GAMMA,
        decimal=DECIMAL_PRECISION,
    )

    mean_zero_choices = mean_zero_model.simulate(
        params=mean_zero_params,
        initial_conditions=initial_conditions,
        log_level="debug",
        seed=5471,
    ).to_dataframe(use_categorical_codes=True)
    location_zero_choices = location_zero_model.simulate(
        params=location_zero_params,
        initial_conditions=initial_conditions,
        log_level="debug",
        seed=5471,
    ).to_dataframe(use_categorical_codes=True)

    first_period = "period == 0"
    np.testing.assert_array_equal(
        location_zero_choices.query(first_period)["work"].to_numpy(),
        mean_zero_choices.query(first_period)["work"].to_numpy(),
    )


@categorical(ordered=True)
class LocationZeroDependencyAction:
    zero: ScalarInt
    one: ScalarInt


@categorical(ordered=False)
class LocationZeroDependencyRegime:
    alive: ScalarInt
    done: ScalarInt


def _dependency_base_utility(*, decision: DiscreteAction, entry_cost: float) -> FloatND:
    return decision - entry_cost


def _dependency_mean_zero_utility(*, base_utility: FloatND) -> FloatND:
    return base_utility


def _dependency_location_zero_utility(
    *, base_utility: FloatND, taste_shock_scale: float
) -> FloatND:
    return base_utility + taste_shock_scale * EULER_GAMMA


def _dependency_feasible_from_utility(*, utility: FloatND) -> BoolND:
    return utility >= 0


def _dependency_feasible_from_base_utility(*, base_utility: FloatND) -> BoolND:
    return base_utility >= 0


def _dependency_terminal_utility() -> FloatND:
    return jnp.asarray(0.0)


def _get_utility_dependency_model(
    *, shifted: bool, preserve_constraint_input: bool
) -> Model:
    utility = (
        _dependency_location_zero_utility if shifted else _dependency_mean_zero_utility
    )
    feasibility = (
        _dependency_feasible_from_base_utility
        if preserve_constraint_input
        else _dependency_feasible_from_utility
    )
    alive = Regime(
        actions={"decision": DiscreteGrid(category_class=LocationZeroDependencyAction)},
        functions={"base_utility": _dependency_base_utility, "utility": utility},
        constraints={"nonnegative_felicity": feasibility},
        taste_shocks=ExtremeValueTasteShocks(),
        solver=GridSearch(),
    )
    return Model(
        edges={"alive": {"done": 40}},
        regimes={
            "alive": alive,
            "done": Regime(functions={"utility": _dependency_terminal_utility}),
        },
        ages=AgeGrid(start=40, inclusive_stop=41, step="Y"),
        regime_id_class=LocationZeroDependencyRegime,
        initial_nodes={40: "alive"},
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("entry_cost", "scale", "adds_feasible_action"),
    [
        pytest.param(0.25, 0.2, False, id="shift-below-feasibility-threshold"),
        pytest.param(0.25, 1.0, True, id="shift-crosses-feasibility-threshold"),
        pytest.param(0.0, 0.2, False, id="zero-utility-action-is-feasible"),
        pytest.param(1.0, 0.2, False, id="only-zero-utility-action-is-feasible"),
    ],
)
def test_location_zero_shift_preserves_unshifted_constraint_inputs(
    *, entry_cost: float, scale: float, adds_feasible_action: bool
) -> None:
    """A location shift changes shocks while preserving the original feasible set.

    Utility is a shared DAG output. A constraint reading it must keep reading
    unshifted felicity when the public recipe adds the Gumbel mean. A naive
    rewrite can admit a formerly infeasible action, even with LinearAggregator.
    The two boundary cases also keep the original non-strict constraint literal.
    """
    mean_zero_params: dict[str, UserParamsNode] = {
        "discount_factor": 0.95,
        "alive": {
            "base_utility": {"entry_cost": entry_cost},
            "taste_shocks": {"scale": scale},
        },
    }
    location_zero_params: dict[str, UserParamsNode] = {
        "discount_factor": 0.95,
        "alive": {
            "base_utility": {"entry_cost": entry_cost},
            "utility": {"taste_shock_scale": scale},
            "taste_shocks": {"scale": scale},
        },
    }
    mean_zero_model = _get_utility_dependency_model(
        shifted=False, preserve_constraint_input=False
    )
    naive_model = _get_utility_dependency_model(
        shifted=True, preserve_constraint_input=False
    )
    location_zero_model = _get_utility_dependency_model(
        shifted=True, preserve_constraint_input=True
    )

    mean_zero_value = mean_zero_model.solve(
        params=mean_zero_params, log_level="debug"
    ).values[0]["alive"]
    naive_value = naive_model.solve(
        params=location_zero_params, log_level="debug"
    ).values[0]["alive"]
    location_zero_value = location_zero_model.solve(
        params=location_zero_params, log_level="debug"
    ).values[0]["alive"]

    # Enumerate the two deterministic actions independently of the model DAG.
    base_values = np.array([0.0, 1.0]) - entry_cost
    shift = scale * np.euler_gamma
    original_mask = base_values >= 0.0
    naive_mask = base_values + shift >= 0.0
    expected_mean_zero = scale * np.logaddexp.reduce(base_values[original_mask] / scale)
    expected_location_zero = expected_mean_zero + shift
    expected_naive = shift + scale * np.logaddexp.reduce(
        base_values[naive_mask] / scale
    )

    aaae(mean_zero_value, expected_mean_zero, decimal=DECIMAL_PRECISION)
    aaae(location_zero_value, expected_location_zero, decimal=DECIMAL_PRECISION)
    aaae(naive_value, expected_naive, decimal=DECIMAL_PRECISION)
    if adds_feasible_action:
        assert float(naive_value) > float(location_zero_value)
    else:
        aaae(naive_value, location_zero_value, decimal=DECIMAL_PRECISION)
