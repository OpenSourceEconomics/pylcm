"""Policy lookup in a regime with EV1 taste shocks.

The lookup publishes, per queried state row and discrete-action combination, the
conditional value `Qc` (the max of `Q` over the feasible continuous actions), the
choice probability `softmax(Qc / scale)` over the feasible combinations, and the
continuous actions attaining `Qc`. The value is the smoothed maximum
`scale * logsumexp(Qc / scale)` the solve stores, and no single action is
returned, since the discrete choice is random.

The hand model has one decision period and a terminal bequest regime:

- wealth `w ∈ {1, 2}`, work `d ∈ {0, 1}`, consumption `c ∈ {0.5, 1.0, 1.5}`
  feasible iff `c <= w`;
- `u = log(c) - 0.3 d`, `w' = w - c + d`, terminal `V(w') = log(w' + 1)` on the
  nodes `0, 0.5, ..., 3`, so every feasible `w'` is a node;
- discount factor `0.9`, taste-shock scale `0.5`.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.exceptions import InvalidSimulationInputError
from lcm.regime import Regime as UserRegime
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    FunctionName,
    ScalarInt,
    UserParamsNode,
)
from tests.conftest import DECIMAL_PRECISION
from tests.simulation.test_simulate_taste_shocks import _reference_work_probability
from tests.test_models import dcegm_paper_twin, taste_shocks_toy

SCALE = 0.5
DISCOUNT_FACTOR = 0.9
WEALTH = np.array([1.0, 2.0])
CONSUMPTION = np.array([0.5, 1.0, 1.5])

# Hand-computed at w = 1 and w = 2 (rows), work off and on (columns).
EXPECTED_QC = np.array([[0.0, 0.32383246], [0.77038371, 0.93012677]])
EXPECTED_CONSUMPTION = np.array([[1.0, 1.0], [1.5, 1.5]])
EXPECTED_PROBABILITIES = np.array([[0.34351593, 0.65648407], [0.42080099, 0.57919901]])
EXPECTED_VALUE = np.array([0.53426089, 1.20318134])
# Hand values carry eight decimals.
HAND_DECIMALS = min(DECIMAL_PRECISION, 7)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    done: ScalarInt


@categorical(ordered=True)
class Work:
    off: ScalarInt
    on: ScalarInt


def utility_alive(*, consumption: ContinuousAction, work: DiscreteAction) -> FloatND:
    return jnp.log(consumption) - 0.3 * work


def utility_done(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 1.0)


def next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction, work: DiscreteAction
) -> ContinuousState:
    return wealth - consumption + 1.0 * work


def budget_constraint(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def no_work_when_rich(*, work: DiscreteAction, wealth: ContinuousState) -> BoolND:
    return (work == Work.off) | (wealth < 2.0)


def get_model(*, forbid_work_when_rich: bool = False) -> Model:
    constraints: dict[FunctionName, Callable[..., BoolND]] = {
        "budget_constraint": budget_constraint
    }
    if forbid_work_when_rich:
        constraints["no_work_when_rich"] = no_work_when_rich
    alive = UserRegime(
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
        state_transitions={"wealth": next_wealth},
        actions={
            "work": DiscreteGrid(category_class=Work),
            "consumption": LinSpacedGrid(start=0.5, stop=1.5, n_points=3),
        },
        constraints=constraints,
        functions={"utility": utility_alive},
        taste_shocks=ExtremeValueTasteShocks(),
    )
    done = UserRegime(
        states={"wealth": LinSpacedGrid(start=0.0, stop=3.0, n_points=7)},
        functions={"utility": utility_done},
    )
    return Model(
        edges={"alive": {"done": 40}},
        regimes={"alive": alive, "done": done},
        ages=AgeGrid(start=40, inclusive_stop=41, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={40: "alive"},
    )


def get_params(*, scale: float = SCALE) -> dict[str, UserParamsNode]:
    return {
        "discount_factor": DISCOUNT_FACTOR,
        "alive": {"taste_shocks": {"scale": scale}},
    }


def lookup(*, model, params, solution=None, **kwargs):
    """Look up `alive` in period 0 at both wealth nodes."""
    if solution is None:
        solution = model.solve(params=params, log_level="off")
    return model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="alive",
        states={"wealth": jnp.asarray(WEALTH)},
        **kwargs,
    )


@pytest.fixture(scope="module")
def solved():
    model = get_model()
    params = get_params()
    solution = model.solve(params=params, log_level="off")
    return model, params, solution


@pytest.fixture(scope="module")
def looked_up(solved):
    model, params, solution = solved
    return lookup(model=model, params=params, solution=solution)


def test_lookup_returns_no_single_action(looked_up):
    assert looked_up.actions is None


def test_lookup_outputs_have_one_axis_per_discrete_action(looked_up):
    assert (
        looked_up.Qc.shape,
        looked_up.choice_probabilities.shape,
        {name: a.shape for name, a in looked_up.conditional_actions.items()},
    ) == ((2, 2), (2, 2), {"consumption": (2, 2)})


def test_lookup_Qc_is_the_conditional_max_over_consumption(looked_up):
    np.testing.assert_array_almost_equal(
        np.asarray(looked_up.Qc), EXPECTED_QC, decimal=HAND_DECIMALS
    )


def test_lookup_conditional_consumption_attains_Qc(looked_up):
    np.testing.assert_array_equal(
        np.asarray(looked_up.conditional_actions["consumption"]),
        EXPECTED_CONSUMPTION,
    )


def test_lookup_choice_probabilities_are_the_softmax_of_Qc(looked_up):
    np.testing.assert_array_almost_equal(
        np.asarray(looked_up.choice_probabilities),
        EXPECTED_PROBABILITIES,
        decimal=HAND_DECIMALS,
    )


def test_lookup_choice_probabilities_sum_to_one(looked_up):
    np.testing.assert_array_almost_equal(
        np.asarray(looked_up.choice_probabilities).sum(axis=1),
        np.ones(2),
        decimal=DECIMAL_PRECISION,
    )


def test_lookup_value_is_the_logsum_of_Qc(looked_up):
    np.testing.assert_array_almost_equal(
        np.asarray(looked_up.value), EXPECTED_VALUE, decimal=HAND_DECIMALS
    )


def test_lookup_value_equals_the_solved_value_at_grid_rows(*, solved, looked_up):
    _, _, solution = solved
    np.testing.assert_array_almost_equal(
        np.asarray(looked_up.value),
        np.asarray(solution.values[0]["alive"]),
        decimal=DECIMAL_PRECISION,
    )


def test_lookup_infeasible_combination_has_probability_exactly_zero():
    model = get_model(forbid_work_when_rich=True)
    got = lookup(model=model, params=get_params())
    np.testing.assert_array_equal(
        np.asarray(got.choice_probabilities)[1], np.array([1.0, 0.0])
    )


def test_lookup_infeasible_combination_leaves_value_at_the_feasible_Qc():
    model = get_model(forbid_work_when_rich=True)
    got = lookup(model=model, params=get_params())
    assert (float(got.Qc[1, 1]), float(got.value[1])) == (
        float("-inf"),
        float(got.Qc[1, 0]),
    )


def test_lookup_small_scale_puts_all_mass_on_the_best_combination():
    model = get_model()
    got = lookup(model=model, params=get_params(scale=1e-3))
    np.testing.assert_array_almost_equal(
        np.asarray(got.choice_probabilities),
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        decimal=DECIMAL_PRECISION,
    )


def test_lookup_restricted_work_grid_shrinks_the_discrete_axis(solved):
    model, params, solution = solved
    got = lookup(
        model=model,
        params=params,
        solution=solution,
        action_grids={"work": jnp.asarray([Work.on])},
    )
    assert (got.Qc.shape, np.asarray(got.choice_probabilities).tolist()) == (
        (2, 1),
        [[1.0], [1.0]],
    )


def test_lookup_with_action_values_has_Qc_as_the_masked_max_of_Q(solved):
    model, params, solution = solved
    got = lookup(
        model=model, params=params, solution=solution, return_action_values=True
    )
    masked = np.where(np.asarray(got.F), np.asarray(got.Q), -np.inf)
    np.testing.assert_array_equal(np.asarray(got.Qc), masked.max(axis=2))


def test_lookup_without_action_values_returns_no_Q(looked_up):
    assert (looked_up.Q, looked_up.F, looked_up.U, looked_up.CE) == (
        None,
        None,
        None,
        None,
    )


@pytest.mark.slow
def test_lookup_work_probability_matches_the_toy_reference():
    scale = 0.2
    discount_factor = 0.95
    model = taste_shocks_toy.get_model()
    params = taste_shocks_toy.get_params(scale=scale, discount_factor=discount_factor)
    solution = model.solve(params=params, log_level="off")
    wealth = model.state_grid(params=params, regime_name="alive", state_name="wealth")
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="alive",
        states={"wealth": jnp.asarray(wealth)},
    )
    expected = [
        _reference_work_probability(
            wealth=float(w), scale=scale, discount_factor=discount_factor
        )
        for w in np.asarray(wealth)
    ]
    np.testing.assert_array_almost_equal(
        np.asarray(got.choice_probabilities)[:, 1],
        expected,
        decimal=DECIMAL_PRECISION - 1,
    )


def test_lookup_refuses_a_taste_shock_regime_not_solved_by_grid_search():
    model = dcegm_paper_twin.get_model("dcegm")
    params = dcegm_paper_twin.get_params()
    solution = model.solve(params=params, log_level="off")
    with pytest.raises(InvalidSimulationInputError, match="GridSearch"):
        model.lookup_policy(
            params=params,
            solution=solution,
            period=0,
            regime_name="working_life",
            states={
                name: jnp.asarray(
                    model.state_grid(
                        params=params, regime_name="working_life", state_name=name
                    )[:1]
                )
                for name in model.state_names(regime_name="working_life")
            },
        )
