"""Policy lookup in a regime with EV1 taste shocks.

The lookup publishes, per queried state row and discrete-action combination, the
conditional value `Qc` (the max of `Q` over the feasible continuous actions), the
choice probability `softmax(Qc / scale)` over the feasible combinations, and the
continuous actions attaining `Qc`. The value is the smoothed maximum
`scale * logsumexp(Qc / scale)` of the simulation decision's conditional values,
using the solved continuation, and no single action is returned, since the
discrete choice is random. The hand model has the same utility in both phases, so
its grid-row lookup also equals the stored solve value.

The hand model has one decision period and a terminal bequest regime:

- wealth `w ∈ {1, 2}`, work `d ∈ {0, 1}`, consumption `c ∈ {0.5, 1.0, 1.5}`
  feasible iff `c <= w`;
- `u = log(c) - 0.3 d`, `w' = w - c + d`, terminal `V(w') = log(w' + 1)` on the
  nodes `0, 0.5, ..., 3`, so every feasible `w'` is a node;
- discount factor `0.9`, taste-shock scale `0.5`.
"""

import math
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    categorical,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.phased import Phased
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


def utility_alive_with_work_bonus(
    *, consumption: ContinuousAction, work: DiscreteAction
) -> FloatND:
    return utility_alive(consumption=consumption, work=work) + math.log(3.0) * work


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


def high_consumption(*, consumption: ContinuousAction) -> BoolND:
    return consumption >= 1.5


def work_costs_half_a_unit(
    *, consumption: ContinuousAction, work: DiscreteAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth - 0.5 * work


def get_model(
    *,
    extra_constraints: dict[FunctionName, Callable[..., BoolND]] | None = None,
    execution_config: ExecutionConfig | None = None,
    utility: Callable[..., FloatND] | Phased = utility_alive,
) -> Model:
    constraints: dict[FunctionName, Callable[..., BoolND]] = {
        "budget_constraint": budget_constraint,
        **(extra_constraints or {}),
    }
    alive = UserRegime(
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
        state_transitions={"wealth": next_wealth},
        actions={
            "work": DiscreteGrid(category_class=Work),
            "consumption": LinSpacedGrid(start=0.5, stop=1.5, n_points=3),
        },
        constraints=constraints,
        functions={"utility": utility},
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
        execution_config=execution_config or ExecutionConfig(),
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


@pytest.fixture(scope="module")
def no_work_when_rich_lookup():
    model = get_model(extra_constraints={"no_work_when_rich": no_work_when_rich})
    return lookup(model=model, params=get_params())


def test_lookup_infeasible_combination_has_probability_exactly_zero(
    no_work_when_rich_lookup,
):
    np.testing.assert_array_equal(
        np.asarray(no_work_when_rich_lookup.choice_probabilities)[1],
        np.array([1.0, 0.0]),
    )


def test_lookup_infeasible_combination_leaves_value_at_the_feasible_Qc(
    no_work_when_rich_lookup,
):
    got = no_work_when_rich_lookup
    assert (float(got.Qc[1, 1]), float(got.value[1])) == (
        float("-inf"),
        float(got.Qc[1, 0]),
    )


def test_lookup_row_without_feasible_combination_has_no_mass_and_no_value():
    model = get_model(extra_constraints={"high_consumption": high_consumption})
    got = lookup(model=model, params=get_params())
    assert (
        float(got.value[0]),
        np.asarray(got.choice_probabilities)[0].tolist(),
        np.asarray(got.conditional_actions["consumption"])[0].tolist(),
    ) == (float("-inf"), [0.0, 0.0], [0.5, 0.5])


def test_lookup_under_a_device_budget_decodes_the_same_conditional_actions():
    model = get_model(execution_config=ExecutionConfig(device_memory_bytes=2**30))
    got = lookup(model=model, params=get_params())
    np.testing.assert_array_equal(
        np.asarray(got.conditional_actions["consumption"]), EXPECTED_CONSUMPTION
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


def test_lookup_keeps_no_intermediates_unless_requested(solved):
    """The choice distribution is always published; `Q`, `F`, `U`, `CE` on request."""
    model, params, solution = solved
    got = lookup(model=model, params=params, solution=solution)
    assert (got.Q, got.F, got.U, got.CE) == (None, None, None, None)
    assert got.Qc is not None
    assert got.choice_probabilities is not None


def test_lookup_keeps_only_the_requested_intermediates(solved):
    model, params, solution = solved
    full = lookup(
        model=model,
        params=params,
        solution=solution,
        keep_intermediates=("Q", "F", "U", "CE"),
    )
    got = lookup(
        model=model, params=params, solution=solution, keep_intermediates=("CE",)
    )
    assert (got.Q, got.F, got.U) == (None, None, None)
    np.testing.assert_array_equal(np.asarray(got.CE), np.asarray(full.CE))
    np.testing.assert_array_equal(
        np.asarray(got.choice_probabilities), np.asarray(full.choice_probabilities)
    )
    np.testing.assert_array_equal(np.asarray(got.value), np.asarray(full.value))


def test_lookup_with_action_values_has_Qc_as_the_masked_max_of_Q(solved):
    model, params, solution = solved
    got = lookup(
        model=model,
        params=params,
        solution=solution,
        keep_intermediates=("Q", "F", "U", "CE"),
    )
    masked = np.where(np.asarray(got.F), np.asarray(got.Q), -np.inf)
    np.testing.assert_array_equal(np.asarray(got.Qc), masked.max(axis=2))


def test_lookup_flow_utility_is_the_utility_of_each_action(solved):
    model, params, solution = solved
    got = lookup(
        model=model,
        params=params,
        solution=solution,
        keep_intermediates=("Q", "F", "U", "CE"),
    )
    expected = np.log(CONSUMPTION)[None, :] - 0.3 * np.array([0.0, 1.0])[:, None]
    np.testing.assert_array_almost_equal(
        np.asarray(got.U),
        np.broadcast_to(expected, (2, 2, 3)),
        decimal=DECIMAL_PRECISION,
    )


def test_lookup_continuation_is_the_terminal_value_at_feasible_actions(solved):
    model, params, solution = solved
    got = lookup(
        model=model,
        params=params,
        solution=solution,
        keep_intermediates=("Q", "F", "U", "CE"),
    )
    next_wealth_values = (
        WEALTH[:, None, None]
        - CONSUMPTION[None, None, :]
        + np.array([0.0, 1.0])[None, :, None]
    )
    feasible = np.asarray(got.F)
    np.testing.assert_array_almost_equal(
        np.asarray(got.CE)[feasible],
        np.log(next_wealth_values + 1.0)[feasible],
        decimal=DECIMAL_PRECISION,
    )


def test_lookup_conditional_consumption_differs_across_discrete_choices():
    """Working costs half a unit of wealth, so at w = 1 it caps consumption at 0.5."""
    model = get_model(
        extra_constraints={"work_costs_half_a_unit": work_costs_half_a_unit}
    )
    got = lookup(model=model, params=get_params())
    np.testing.assert_array_equal(
        np.asarray(got.conditional_actions["consumption"]),
        np.array([[1.0, 0.5], [1.5, 1.5]]),
    )


@pytest.fixture(scope="module")
def phased_solved():
    """Working pays `log(3)` more in the simulate phase than in the solve phase."""
    model = get_model(
        utility=Phased(solve=utility_alive, simulate=utility_alive_with_work_bonus)
    )
    params = get_params()
    solution = model.solve(params=params, log_level="off")
    return solution, lookup(model=model, params=params, solution=solution)


def test_lookup_with_phased_utility_uses_the_simulated_utility(phased_solved):
    _, got = phased_solved
    np.testing.assert_array_almost_equal(
        np.asarray(got.Qc),
        EXPECTED_QC + np.array([0.0, math.log(3.0)]),
        decimal=HAND_DECIMALS,
    )


def test_lookup_with_phased_utility_smooths_the_simulated_Qc(phased_solved):
    _, got = phased_solved
    simulated_Qc = EXPECTED_QC + np.array([0.0, math.log(3.0)])
    expected = SCALE * np.log(np.exp(simulated_Qc / SCALE).sum(axis=1))
    np.testing.assert_array_almost_equal(
        np.asarray(got.value), expected, decimal=HAND_DECIMALS
    )


def test_solve_with_phased_utility_stores_the_solve_phase_value(phased_solved):
    solution, _ = phased_solved
    np.testing.assert_array_almost_equal(
        np.asarray(solution.values[0]["alive"]), EXPECTED_VALUE, decimal=HAND_DECIMALS
    )


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


@pytest.mark.slow
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
