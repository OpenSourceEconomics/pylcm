"""A joint transition prices one shared support node in backward induction."""

from collections.abc import Callable, Mapping

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.transition_plans import (
    TargetTransitionPlan,
    TransitionLotteryInfo,
    TransitionOutputInfo,
)
from lcm import (
    AgeGrid,
    DiscreteGrid,
    EdgeLeg,
    GatedEdge,
    IrregSpacedGrid,
    JointTransition,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    SamePeriodRef,
    categorical,
)
from lcm.exceptions import (
    InvalidStateTransitionProbabilitiesError,
    RegimeInitializationError,
)
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _certain_target() -> FloatND:
    return jnp.asarray(1.0)


def _joint_probabilities() -> FloatND:
    return jnp.asarray([0.5, 0.5])


def _too_many_joint_probabilities() -> FloatND:
    return jnp.asarray([0.2, 0.3, 0.5])


def _negative_joint_probabilities() -> FloatND:
    return jnp.asarray([1.1, -0.1])


def _nonunit_joint_probabilities() -> FloatND:
    return jnp.asarray([0.2, 0.3])


def _nonfinite_joint_probabilities() -> FloatND:
    return jnp.asarray([jnp.nan, 0.5])


def _wrong_sized_support() -> Mapping[str, FloatND]:
    return {
        "wealth": jnp.asarray([0.0, 0.5, 1.0]),
        "income": jnp.asarray([0.0, 0.5, 1.0]),
    }


_SUPPORT = {
    "wealth": jnp.asarray([0.0, 1.0]),
    "income": jnp.asarray([0.0, 1.0]),
}


def _next_wealth(match: Mapping[str, FloatND]) -> FloatND:
    return match["wealth"]


def _next_income(match: Mapping[str, FloatND]) -> FloatND:
    return match["income"]


def _build_model(
    *,
    enable_jit: bool,
    probabilities: Callable[[], FloatND] = _joint_probabilities,
    support: object = _SUPPORT,
) -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_certain_target)},
                active=lambda age: age < 21,
                functions={"utility": lambda: jnp.asarray(0.0)},
                joint_transitions={
                    "target": {
                        "match": JointTransition(
                            support_size=2,
                            support=support,
                            probabilities=probabilities,
                            outputs={
                                "wealth": _next_wealth,
                                "income": _next_income,
                            },
                        )
                    }
                },
            ),
            "target": Regime(
                transition=None,
                states={
                    "wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
                    "income": LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
                },
                functions={"utility": lambda wealth, income: wealth + 2 * income},
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )


def _params() -> UserParams:
    return {
        "source": {
            "target": {
                "next_regime": {},
                "match": {"support": {}, "probabilities": {}},
                "next_wealth": {},
                "next_income": {},
            },
            "koopmans_aggregator": {"discount_factor": 1.0},
        },
        "target": {"utility": {}},
    }


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_joint_transition_prices_shared_nodes_without_a_cartesian_product(
    *, enable_jit: bool
) -> None:
    """The source continuation averages the two declared pairs, not four marginals."""
    model = _build_model(enable_jit=enable_jit)

    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["source"]), 1.5, decimal=DECIMAL_PRECISION
    )


def test_joint_transition_rejects_probability_vector_with_wrong_length() -> None:
    """A joint kernel probability axis must match its declared support size."""
    model = _build_model(enable_jit=False, probabilities=_too_many_joint_probabilities)

    with pytest.raises(
        InvalidStateTransitionProbabilitiesError,
        match=r"match.*length 3.*support_size is 2",
    ):
        model.solve(params=_params(), log_level="debug")


def test_callable_joint_support_matches_declared_size_after_params_are_bound() -> None:
    """A callable support keeps its declared leading node dimension at runtime."""
    model = _build_model(enable_jit=False, support=_wrong_sized_support)

    with pytest.raises(
        RegimeInitializationError,
        match=r"match\.support.*leading axis.*support_size=2",
    ):
        model.solve(params=_params(), log_level="debug")


@pytest.mark.parametrize(
    "probabilities",
    [
        _negative_joint_probabilities,
        _nonunit_joint_probabilities,
        _nonfinite_joint_probabilities,
    ],
    ids=["out-of-range", "nonunit", "nonfinite"],
)
def test_joint_transition_rejects_invalid_probability_values(
    probabilities: Callable[[], FloatND],
) -> None:
    """Joint probabilities must be finite, bounded, and sum to one row-wise."""
    model = _build_model(enable_jit=False, probabilities=probabilities)

    with pytest.raises(
        InvalidStateTransitionProbabilitiesError,
        match=(
            "contains nonfinite or out-of-range values, or rows that do not sum to one"
        ),
    ):
        model.solve(params=_params(), log_level="debug")


def test_simulation_shares_one_realization_and_hides_the_joint_node() -> None:
    """All outputs use one sampled node, which never appears in the result schema."""
    model = _build_model(enable_jit=False)
    n_subjects = 200

    result = model.simulate(
        params=_params(),
        initial_conditions={
            "age": jnp.full(n_subjects, 20.0),
            "regime_id": jnp.full(n_subjects, RegimeId.source),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    frame = result.to_dataframe(terminal_rows="all")
    target = frame.query('regime_name == "target"')

    np.testing.assert_array_equal(target["wealth"], target["income"])
    assert set(target["wealth"]) == {0.0, 1.0}
    assert "match" not in frame.columns
    assert "match" not in result.state_names
    assert "match" not in result.available_targets


def test_joint_transition_lowers_to_one_lottery_and_two_outputs() -> None:
    """The canonical target plan separates one lottery from genuine outputs."""
    model = _build_model(enable_jit=False)

    plan = model._regimes["source"].solution.transition_plans["target"]

    assert isinstance(plan, TargetTransitionPlan)
    assert list(plan.lotteries) == ["match"]
    assert isinstance(plan.lotteries["match"], TransitionLotteryInfo)
    assert list(plan.outputs) == ["wealth", "income"]
    assert all(
        isinstance(output, TransitionOutputInfo) for output in plan.outputs.values()
    )
    assert all(
        output.lottery_dependencies == frozenset({"match"})
        for output in plan.outputs.values()
    )
    assert plan.output_order == ("wealth", "income")


@categorical(ordered=False)
class BDYRegimeId:
    single: ScalarInt
    single_terminal: ScalarInt
    couple: ScalarInt


@categorical(ordered=False)
class BDYEps:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class BDYChoice:
    only: ScalarInt


_BDY_WEALTH_POINTS = tuple(float(value) for value in range(6))
_BDY_MATCH_SUPPORT = {
    "partner_wealth": jnp.asarray([0.25, 0.75]),
    "ybar_p": jnp.asarray([-1.0, 1.0]),
    "eps_p": jnp.asarray([BDYEps.high, BDYEps.low]),
}
_BDY_EXPECTED = np.asarray([2.0, 5.5, 11.0, 18.5, 28.0])


def _bdy_certain_couple() -> FloatND:
    return jnp.asarray(1.0)


def _bdy_match_support_with_size(
    support_size: int,
) -> dict[str, FloatND | IntND]:
    indices = jnp.arange(support_size) % 2
    return {name: values[indices] for name, values in _BDY_MATCH_SUPPORT.items()}


def _bdy_probabilities_with_size(support_size: int) -> Callable[[], FloatND]:
    def probabilities() -> FloatND:
        return jnp.full((support_size,), 1.0 / support_size)

    return probabilities


def _bdy_next_wealth(
    wealth: ContinuousState, partner_match: Mapping[str, FloatND | IntND]
) -> FloatND:
    return wealth + partner_match["partner_wealth"]


def _bdy_next_ybar_p(
    partner_match: Mapping[str, FloatND | IntND],
) -> FloatND:
    return partner_match["ybar_p"]


def _bdy_next_eps_p(
    partner_match: Mapping[str, FloatND | IntND],
) -> IntND:
    return partner_match["eps_p"].astype(jnp.int32)


def _bdy_couple_utility_f(
    wealth: ContinuousState,
    ybar_p: ContinuousState,
    eps_p: DiscreteState,
    household_choice: DiscreteAction,
) -> FloatND:
    return (
        wealth**2
        + 1.5 * wealth
        + 0.75
        + ybar_p
        + (2 * eps_p - 1)
        + 0.0 * household_choice
    )


def _bdy_couple_utility_m(
    wealth: ContinuousState,
    ybar_p: ContinuousState,
    eps_p: DiscreteState,
    household_choice: DiscreteAction,
) -> FloatND:
    return 2 * _bdy_couple_utility_f(wealth, ybar_p, eps_p, household_choice)


def _bdy_single_utility(wealth: ContinuousState) -> FloatND:
    return 0.0 * wealth


def _bdy_fallback_utility(wealth: ContinuousState) -> FloatND:
    return -100.0 + 0.0 * wealth


def _bdy_identity_wealth(wealth: ContinuousState) -> FloatND:
    return wealth


def _bdy_gate_always_open(V_target_f: FloatND) -> jnp.ndarray:
    return jnp.ones_like(V_target_f, dtype=bool)


def _bdy_model(*, enable_jit: bool, support_size: int = 2) -> Model:
    return Model(
        regimes={
            "single": Regime(
                transition={"couple": MarkovTransition(_bdy_certain_couple)},
                active=lambda age: age < 1,
                states={"wealth": IrregSpacedGrid(points=_BDY_WEALTH_POINTS[:-1])},
                functions={"utility": _bdy_single_utility},
                joint_transitions={
                    "couple": {
                        "partner_match": JointTransition(
                            support_size=support_size,
                            support=_bdy_match_support_with_size(support_size),
                            probabilities=_bdy_probabilities_with_size(support_size),
                            outputs={
                                "wealth": _bdy_next_wealth,
                                "ybar_p": _bdy_next_ybar_p,
                                "eps_p": _bdy_next_eps_p,
                            },
                        )
                    }
                },
                gated_edges={
                    "couple": GatedEdge(
                        gate=_bdy_gate_always_open,
                        legs={
                            "f": EdgeLeg(
                                target_stakeholder="f",
                                fallback=SamePeriodRef(
                                    regime="single_terminal",
                                    projection={"wealth": _bdy_identity_wealth},
                                ),
                            )
                        },
                    )
                },
            ),
            "single_terminal": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"wealth": IrregSpacedGrid(points=_BDY_WEALTH_POINTS[:-1])},
                functions={"utility": _bdy_fallback_utility},
            ),
            "couple": Regime(
                transition=None,
                active=lambda age: age >= 1,
                stakeholders=("f", "m"),
                actions={"household_choice": DiscreteGrid(BDYChoice)},
                states={
                    "wealth": IrregSpacedGrid(points=_BDY_WEALTH_POINTS),
                    "ybar_p": IrregSpacedGrid(points=(-1.0, 1.0)),
                    "eps_p": DiscreteGrid(BDYEps),
                },
                functions={
                    "utility_f": _bdy_couple_utility_f,
                    "utility_m": _bdy_couple_utility_m,
                },
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=BDYRegimeId,
        enable_jit=enable_jit,
    )


def _bdy_params() -> UserParams:
    return {
        "single": {
            "couple": {
                "next_regime": {},
                "partner_match": {"support": {}, "probabilities": {}},
                "next_wealth": {},
                "next_ybar_p": {},
                "next_eps_p": {},
            },
            "koopmans_aggregator": {"discount_factor": 0.95},
        },
        "single_terminal": {"utility": {}},
        "couple": {"utility_f": {}, "utility_m": {}},
    }


def _bdy_scalar_oracle() -> np.ndarray:
    target_grid = np.asarray(_BDY_WEALTH_POINTS)
    probabilities = np.asarray([0.5, 0.5])
    expected = []
    for wealth in _BDY_WEALTH_POINTS[:-1]:
        continuation = 0.0
        for node, probability in enumerate(probabilities):
            pooled = wealth + float(_BDY_MATCH_SUPPORT["partner_wealth"][node])
            lower = int(np.floor(pooled))
            fraction = pooled - target_grid[lower]
            ybar_p = float(_BDY_MATCH_SUPPORT["ybar_p"][node])
            eps_p = int(_BDY_MATCH_SUPPORT["eps_p"][node])

            landed_wealth = target_grid[lower : lower + 2]
            values = (
                landed_wealth**2 + 1.5 * landed_wealth + 0.75 + ybar_p + (2 * eps_p - 1)
            )
            interpolated = (1 - fraction) * values[0] + fraction * values[1]
            continuation += probability * interpolated
        expected.append(continuation)
    return np.asarray(expected)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_bdy_joint_match_matches_scalar_oracle_without_a_stored_match_axis(
    *, enable_jit: bool
) -> None:
    """BDY matching folds one mixed-coordinate draw before collective V storage."""
    model = _bdy_model(enable_jit=enable_jit)

    solution = model.solve(params=_bdy_params(), log_level="debug")
    oracle = _bdy_scalar_oracle()

    np.testing.assert_array_almost_equal(
        oracle, _BDY_EXPECTED, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["single"]),
        0.95 * _BDY_EXPECTED,
        decimal=DECIMAL_PRECISION,
    )
    assert np.asarray(solution[1]["couple"]).shape == (2, 6, 2, 2)
    assert "partner_match" not in model._regimes["couple"].solution.state_names


def test_bdy_target_storage_is_independent_of_match_support_size() -> None:
    """Transient BDY match nodes never become stored value-function axes."""
    storage_signatures = []
    for support_size in (5, 45, 225):
        model = _bdy_model(enable_jit=False, support_size=support_size)
        solution = model.solve(params=_bdy_params(), log_level="debug")
        target = solution[1]["couple"]
        total_entries = sum(
            np.asarray(array).size
            for regimes in solution.values()
            for array in regimes.values()
        )
        storage_signatures.append(
            (target.ndim, target.shape, type(target.sharding), total_entries)
        )

    assert (
        storage_signatures
        == [(4, (2, 6, 2, 2), type(solution[1]["couple"].sharding), 58)] * 3
    )


def test_bdy_simulation_samples_one_shared_match_and_agrees_with_solve() -> None:
    """BDY simulation samples one row whose outputs reproduce the folded value."""
    model = _bdy_model(enable_jit=False)
    solution = model.solve(params=_bdy_params(), log_level="debug")
    n_subjects = 10_000

    result = model.simulate(
        params=_bdy_params(),
        initial_conditions={
            "age": jnp.zeros(n_subjects),
            "wealth": jnp.full(n_subjects, 2.0),
            "regime_id": jnp.full(n_subjects, BDYRegimeId.single),
        },
        period_to_regime_to_V_arr=solution,
        log_level="debug",
    )
    frame = result.to_dataframe(use_labels=False, terminal_rows="all")
    couple = frame.query('regime_name == "couple"')

    realized_pairs = set(zip(couple["ybar_p"], couple["eps_p"], strict=True))
    assert realized_pairs == {(-1.0, 1), (1.0, 0)}
    expected_wealth = np.where(couple["ybar_p"].to_numpy() == -1.0, 2.25, 2.75)
    np.testing.assert_array_equal(couple["wealth"], expected_wealth)

    folded_node_values = np.where(expected_wealth == 2.25, 9.375, 12.625)
    solved_continuation = float(np.asarray(solution[0]["single"])[2] / 0.95)
    assert abs(folded_node_values.mean() - solved_continuation) < 0.08
    assert "partner_match" not in frame.columns
    assert "partner_match" not in result.state_names
    assert "partner_match" not in result.available_targets
