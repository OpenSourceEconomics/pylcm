"""A gated edge is gated where the source lands, not on the target's nodes.

A gate is a predicate, and a predicate does not commute with interpolation.
Baking it into a surface on the target's grid and interpolating that surface at
the realized landing point reports, in every cell whose corners fall on opposite
sides of the gate, a blend of the open and the closed branch — a number neither
branch pays, and one that can rank a source action above the one it should have
taken. Reading each operand at the landing point and gating it there is the
order forward simulation routes in, so the value the source maximizes and the
branch a subject is sent down agree.

The model below is exactly hand-computable, so every expected number is an
integer or a half-integer rather than a tolerance-bounded approximation.
"""

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    ParetoObjective,
    ParticipationCheck,
    ParticipationRead,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidValueFunctionError,
    ModelInitializationError,
    OffGridStateError,
)
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    OffGrid,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION

# Nodes 0, 1, 2. The gate opens strictly above 1.0, so the cell [1, 2] straddles
# it and is where gating on the nodes and gating at the landing point differ.
_X = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)

# 1.5 lands inside the straddled cell; 2.0 lands exactly on a node.
_SAVING = IrregSpacedGrid(points=np.array([1.5, 2.0]))

_ON_GRID_SAVING = IrregSpacedGrid(points=np.array([1.0, 2.0]))


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _certain_target(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _utility_source(*, x: ContinuousState, saving: ContinuousAction) -> FloatND:
    return -2.0 * saving + 0.0 * x


def _next_x(saving: ContinuousAction) -> ContinuousState:
    return saving


def _utility_target(x: ContinuousState) -> FloatND:
    return 10.0 + x


def _utility_fallback(x: ContinuousState) -> FloatND:
    return 0.0 * x


def _identity_x(x: ContinuousState) -> ContinuousState:
    return x


def _outside_hull_x(x: ContinuousState) -> ContinuousState:
    return x + 3.0


def _gate(x: ContinuousState) -> BoolND:
    return x > 1.0


def _params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "target": {},
        "fallback": {},
    }


def _make_model(
    *,
    saving_grid=_SAVING,
    off_grid: Literal["pointwise", "reject"] = "pointwise",
    fallback_off_grid: OffGrid = "extrapolate",
    project_outside_hull: bool = False,
) -> Model:
    return Model(
        regimes={
            "source": Regime(
                states={"x": _X},
                state_transitions={"x": _next_x},
                actions={"saving": saving_grid},
                functions={"utility": _utility_source},
            ),
            "target": Regime(
                states={"x": _X},
                functions={"utility": _utility_target},
            ),
            "fallback": Regime(
                states={"x": _X.replace(off_grid=fallback_off_grid)},
                functions={"utility": _utility_fallback},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "source"},
        edges={
            "source": Transition(
                targets={"target": 0, "fallback": 0},
                law=ByAge(
                    cases={0: {"target": StochasticTransition(func=_certain_target)}}
                ),
                gates={
                    "target": Gate(
                        predicate=_gate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={
                                        "x": _outside_hull_x
                                        if project_outside_hull
                                        else _identity_x
                                    },
                                )
                            )
                        },
                        off_grid=off_grid,
                    )
                },
            )
        },
    )


def test_source_value_is_a_value_one_branch_actually_pays() -> None:
    """The source publishes `8.5`, the best gated value at a realized landing.

    Terminal values are `V_target(x) = 10 + x` and `V_fallback(x) = 0`. Saving
    `1.5` lands inside the straddled cell, where the gate is open, so the branch
    pays `V_target(1.5) = 11.5` against a flow cost of `-3`, i.e. `8.5`. Saving
    `2.0` pays `V_target(2) = 12` against `-4`, i.e. `8.0`.

    Gating on the nodes first turns the continuation into `[0, 0, 12]`, whose
    interpolant at `1.5` is `6` — halfway between the closed branch at `x = 1`
    and the open branch at `x = 2`, an amount no branch delivers — and would
    publish `8.0`.
    """
    solution = _make_model().solve(params=_params(), log_level="debug").values

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["source"]),
        np.full(3, 8.5),
        decimal=DECIMAL_PRECISION,
    )


@categorical(ordered=False)
class FallbackCategory:
    low: ScalarInt
    high: ScalarInt


def _project_category(category: DiscreteState) -> DiscreteState:
    return jnp.asarray(category, dtype=jnp.int32)


def _project_zero_shock(x: ContinuousState) -> FloatND:
    return 0.0 * x


def _utility_categorical_fallback(
    *, category: DiscreteState, shock: FloatND
) -> FloatND:
    return 12.0 * category + shock


def _make_categorical_process_fallback_model(
    category_projection: Callable[[DiscreteState], DiscreteState],
) -> Model:
    categories = DiscreteGrid(category_class=FallbackCategory)
    return Model(
        regimes={
            "source": Regime(
                states={"category": categories, "x": _X},
                state_transitions={"category": _project_category, "x": _next_x},
                actions={"saving": IrregSpacedGrid(points=np.array([1.0, 1.5]))},
                functions={"utility": _utility_source},
            ),
            "target": Regime(
                states={"category": categories, "x": _X},
                functions={"utility": _utility_target},
            ),
            "fallback": Regime(
                states={
                    "category": categories,
                    "shock": NormalIIDProcess(
                        n_points=3,
                        gauss_hermite=False,
                        mu=0.0,
                        sigma=1.0,
                        n_std=1.0,
                    ),
                },
                functions={"utility": _utility_categorical_fallback},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "source"},
        edges={
            "source": Transition(
                targets={"target": 0, "fallback": 0},
                law=ByAge(
                    cases={0: {"target": StochasticTransition(func=_certain_target)}}
                ),
                gates={
                    "target": Gate(
                        predicate=_gate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={
                                        "category": category_projection,
                                        "shock": _project_zero_shock,
                                    },
                                )
                            )
                        },
                        off_grid="pointwise",
                    )
                },
            )
        },
    )


@pytest.mark.coverage(backends=("cpu", "gpu-small"), precisions="both")
def test_solve_reads_integer_categories_in_a_process_fallback() -> None:
    """Categorical fallback values change which gated action is optimal.

    Saving 1 closes the gate and pays 12 * category - 2 at shock zero.
    Saving 1.5 opens it and pays 11.5 - 3 = 8.5. Thus category zero
    chooses the open branch (8.5), while category one chooses the closed
    branch (10). These values do not depend on the source's x coordinate.
    """
    model = _make_categorical_process_fallback_model(
        category_projection=_project_category
    )
    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution.values[0]["source"]),
        np.array([[8.5, 8.5, 8.5], [10.0, 10.0, 10.0]]),
        decimal=DECIMAL_PRECISION,
    )


def _project_unknown_category(category: DiscreteState) -> DiscreteState:
    return category + jnp.int32(2)


@pytest.mark.coverage(backends=("cpu", "gpu-small"), precisions="both")
def test_solve_rejects_unknown_categories_in_a_process_fallback() -> None:
    """Codes outside the categorical domain cannot publish a finite value."""
    model = _make_categorical_process_fallback_model(
        category_projection=_project_unknown_category
    )

    with pytest.raises(InvalidValueFunctionError, match="NaN"):
        model.solve(params=_params(), log_level="debug")


def test_the_action_taken_is_the_one_the_gated_value_ranks_first() -> None:
    """The simulated subject saves `1.5`, not `2.0`.

    The two candidates are separated by a half unit of value, so this is a
    genuine reversal of the argmax rather than a rounding difference, and it is
    asserted as the discrete decision it is.
    """
    model = _make_model()

    result = model.simulate(
        params=_params(),
        initial_conditions={
            "age": jnp.array([0.0]),
            "x": jnp.array([0.0]),
            "regime_id": jnp.array([RegimeId.source]),
        },
        log_level="debug",
    )
    first_period = result.to_dataframe().query("period == 0")

    assert first_period["saving"].to_numpy().tolist() == [1.5]


@pytest.mark.parametrize(
    ("fallback_off_grid", "expected"),
    [("clip_lookup", 3.5), ("clip_state", 2.0)],
)
def test_projected_fallback_preserves_or_clips_its_economic_handoff(
    *,
    fallback_off_grid: OffGrid,
    expected: float,
) -> None:
    """A closed gate writes the fallback projection under the fallback's rule."""
    model = _make_model(
        saving_grid=IrregSpacedGrid(points=np.array([0.5, 0.6])),
        fallback_off_grid=fallback_off_grid,
        project_outside_hull=True,
    )
    result = model.simulate(
        params=_params(),
        initial_conditions={
            "age": jnp.array([0.0]),
            "x": jnp.array([0.0]),
            "regime_id": jnp.array([RegimeId.source]),
        },
        log_level="off",
    )
    frame = result.to_dataframe().query("period == 1")
    np.testing.assert_array_equal(frame["regime_name"], ["fallback"])
    np.testing.assert_array_equal(frame["x"], [expected])


@pytest.mark.parametrize("runtime_checks", [False, True])
def test_projected_realized_fallback_raise_is_mandatory(
    *, runtime_checks: bool
) -> None:
    """A reached fallback projection fails the solve even with diagnostics off."""
    model = _make_model(
        saving_grid=IrregSpacedGrid(points=np.array([0.5, 0.6])),
        fallback_off_grid="raise",
        project_outside_hull=True,
    )
    with pytest.raises(OffGridStateError, match=r"regime='source'.*state='x'"):
        model.solve(params=_params(), log_level="off", runtime_checks=runtime_checks)


def test_landings_that_all_fall_on_nodes_are_unaffected() -> None:
    """With every landing point on a node, the gate is read exactly as declared.

    Saving `1.0` lands on the closed side and pays the fallback `0` against a
    flow cost of `-2`; saving `2.0` lands on the open side and pays `12` against
    `-4`. The best is `8.0`. Without this control the number above could come
    from gating differently rather than from gating in a different place.
    """
    model = _make_model(saving_grid=_ON_GRID_SAVING)

    solution = model.solve(params=_params(), log_level="debug").values

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["source"]),
        np.full(3, 8.0),
        decimal=DECIMAL_PRECISION,
    )


def test_off_grid_reject_refuses_a_continuous_target() -> None:
    """`off_grid="reject"` names the target's continuous state and stops.

    The setting promises the gate is never asked about a point the target's grid
    does not hold. A continuous target state breaks that promise, and building
    the model anyway would publish an interpolated gate under a declaration that
    says none is possible.
    """
    with pytest.raises(ModelInitializationError) as excinfo:
        _make_model(off_grid="reject")

    message = str(excinfo.value)
    assert "off_grid" in message
    assert "'x'" in message


def test_off_grid_reject_accepts_a_target_reached_exactly() -> None:
    """A target carrying only a discrete state satisfies the promise and builds.

    Every landing point is a node there, so the gate is read exactly where it
    was declared. Without this control the rejection above could come from
    declaring `off_grid="reject"` at all rather than from the target's grid.
    """
    model = _make_discrete_target_model(off_grid="reject")

    solution = model.solve(params=_params(), log_level="debug").values

    # `sick` is closed and pays the fallback 0; `healthy` is open and pays 7.
    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["source"]),
        np.array([0.0, 7.0]),
        decimal=DECIMAL_PRECISION,
    )


@categorical(ordered=False)
class Health:
    sick: ScalarInt
    healthy: ScalarInt


def _utility_zero() -> FloatND:
    return jnp.asarray(0.0)


def _fallback_utility(health: DiscreteState) -> FloatND:
    """Zero either way, but read through `health` so the state is live."""
    return jnp.where(health == Health.healthy, 0.0, 0.0)


def _keep_health(health: DiscreteState) -> DiscreteState:
    return health


def _utility_healthy_target(health: DiscreteState) -> FloatND:
    return jnp.where(health == Health.healthy, 7.0, 3.0)


def _healthy_gate(health: DiscreteState) -> BoolND:
    return health == Health.healthy


def _make_discrete_target_model(*, off_grid: Literal["pointwise", "reject"]) -> Model:
    health = DiscreteGrid(category_class=Health)
    return Model(
        regimes={
            "source": Regime(
                states={"health": health},
                state_transitions={"health": _keep_health},
                functions={"utility": _utility_zero},
            ),
            "target": Regime(
                states={"health": health},
                functions={"utility": _utility_healthy_target},
            ),
            "fallback": Regime(
                states={"health": health},
                functions={"utility": _fallback_utility},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "source"},
        edges={
            "source": Transition(
                targets={"target": 0, "fallback": 0},
                law=ByAge(
                    cases={0: {"target": StochasticTransition(func=_certain_target)}}
                ),
                gates={
                    "target": Gate(
                        predicate=_healthy_gate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"health": _keep_health},
                                )
                            )
                        },
                        off_grid=off_grid,
                    )
                },
            )
        },
    )


# The canonical off-grid witness, in exact halves and fifths: a target whose
# value crosses its gate reference between the two nodes, so the gate is open at
# one node and closed at the other and every intermediate point is a cell the
# nodewise surface cannot describe.
_WITNESS_BETA = 0.95
_WITNESS_LANDING = 0.6
_WITNESS_X = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)


@categorical(ordered=True)
class Risk:
    safe: ScalarInt
    risky: ScalarInt


def _witness_utility(*, risk: DiscreteState, x: ContinuousState) -> FloatND:
    """The safe action pays one unit now, whatever the source's own state."""
    return jnp.where(risk == Risk.safe, 1.0, 0.0) + 0.0 * x


def _witness_next_x(risk: DiscreteState) -> ContinuousState:
    """The safe action lands on the closed node, the risky one inside the cell."""
    return jnp.where(risk == Risk.safe, 0.0, _WITNESS_LANDING)


def _witness_target(x: ContinuousState) -> FloatND:
    """`V_target = (1, 3)` on the two nodes."""
    return 1.0 + 2.0 * x


def _witness_reference(x: ContinuousState) -> FloatND:
    """`V_reference = (2, 2.5)` on the two nodes, so the two cross inside."""
    return 2.0 + 0.5 * x


def _witness_fallback(x: ContinuousState) -> FloatND:
    return 0.0 * x


def _witness_gate(*, V_target: FloatND, V_reference: FloatND) -> BoolND:
    return V_target > V_reference


@categorical(ordered=False)
class WitnessRegimeId:
    source: ScalarInt
    target: ScalarInt
    reference: ScalarInt
    fallback: ScalarInt


def _witness_params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "source": {"koopmans_aggregator": {"discount_factor": _WITNESS_BETA}},
        "target": {},
        "reference": {},
        "fallback": {},
    }


def _witness_model() -> Model:
    return Model(
        regimes={
            "source": Regime(
                states={"x": _WITNESS_X},
                state_transitions={"x": _witness_next_x},
                actions={"risk": DiscreteGrid(category_class=Risk)},
                functions={"utility": _witness_utility},
            ),
            "target": Regime(
                states={"x": _WITNESS_X},
                functions={"utility": _witness_target},
            ),
            "reference": Regime(
                states={"x": _WITNESS_X},
                functions={"utility": _witness_reference},
            ),
            "fallback": Regime(
                states={"x": _WITNESS_X},
                functions={"utility": _witness_fallback},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=WitnessRegimeId,
        initial_nodes={0: "source"},
        edges={
            "source": Transition(
                targets={"target": 0, "fallback": 0},
                law=ByAge(
                    cases={0: {"target": StochasticTransition(func=_certain_target)}}
                ),
                participation={
                    "target": ParticipationCheck(
                        predicate=_witness_gate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"x": _identity_x},
                                )
                            )
                        },
                        reads={
                            "V_target": ParticipationRead(
                                kind="IntegratedFull",
                                regime="target",
                                projection={"x": _identity_x},
                            ),
                            "V_reference": ParticipationRead(
                                kind="IntegratedFull",
                                regime="reference",
                                projection={"x": _identity_x},
                            ),
                        },
                    )
                },
            )
        },
    )


def test_a_gate_that_flips_inside_a_cell_does_not_reward_the_risky_action() -> None:
    """The source takes the safe action worth `1.0`, not the risky one worth `0`.

    At the landing point `x = 0.6` the operands read `V_target = 2.2` and
    `V_reference = 2.3`, so the gate is closed and the branch pays the fallback
    `0`. The nodewise surface `where(V_target > V_reference, V_target, 0)` is
    `(0, 3)` instead, whose interpolant at `0.6` is `1.8` — neither branch's
    value — and discounting it gives the risky action `0.95 * 1.8 = 1.71`,
    above the safe action's `1.0`. The reversal is a whole unit of value wide,
    so it is the argmax that is asserted.
    """
    model = _witness_model()

    solution = model.solve(params=_witness_params(), log_level="debug").values

    # The nodewise surrogate, computed from the published terminal values: it
    # ranks the risky action above the safe one, so the fixture discriminates
    # between the two orders rather than agreeing under both.
    terminal = solution[1]
    nodewise = np.where(
        np.asarray(terminal["target"]) > np.asarray(terminal["reference"]),
        np.asarray(terminal["target"]),
        np.asarray(terminal["fallback"]),
    )
    surrogate = _WITNESS_BETA * np.interp(_WITNESS_LANDING, [0.0, 1.0], nodewise)
    np.testing.assert_almost_equal(surrogate, 1.71, decimal=DECIMAL_PRECISION)

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["source"]),
        np.full(2, 1.0),
        decimal=DECIMAL_PRECISION,
    )


def _crossing_target(
    *, x: ContinuousState, risk: DiscreteState, period: ScalarInt, age: FloatND
) -> FloatND:
    return jnp.where(risk == Risk.safe, x, 1.0 - x) + (period - 1) + (age - 1)


def _crossing_source(risk: DiscreteState) -> FloatND:
    return jnp.where(risk == Risk.safe, 0.0, 0.25)


def _crossing_landing(risk: DiscreteState) -> ContinuousState:
    return jnp.where(risk == Risk.safe, 0.0, 0.5)


def _crossing_floor(V_target: FloatND) -> BoolND:
    return V_target > 0.75


def _negative_fallback(x: ContinuousState) -> FloatND:
    return -2.0 + 0.0 * x


def _nonfinite_risky_landing(risk: DiscreteState) -> ContinuousState:
    return jnp.where(risk == Risk.safe, 0.0, jnp.nan)


def _safe_risk_only(risk: DiscreteState) -> BoolND:
    return risk == Risk.safe


def _risky_risk_only(risk: DiscreteState) -> BoolND:
    return risk == Risk.risky


def _participation_floor_model(
    *, nonfinite_risky: bool = False, safe_only: bool = False
) -> Model:
    """One actual edge and date, with optional impossible source-action landings."""
    actions = DiscreteGrid(category_class=Risk)
    return Model(
        regimes={
            "source": Regime(
                actions={"risk": actions},
                functions={"utility": _crossing_source},
                state_transitions={
                    "x": _nonfinite_risky_landing
                    if nonfinite_risky
                    else _crossing_landing
                },
                constraints={"safe_only": _safe_risk_only} if safe_only else {},
            ),
            "target": Regime(
                states={"x": _WITNESS_X},
                actions={"risk": actions},
                functions={"utility": _crossing_target},
            ),
            "fallback": Regime(
                states={"x": _WITNESS_X},
                functions={"utility": _negative_fallback},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "source"},
        edges={
            "source": Transition(
                law=ByAge(cases={0: "target"}),
                participation={
                    "target": ParticipationCheck(
                        predicate=_crossing_floor,
                        reads={
                            "V_target": ParticipationRead(
                                kind="Participating",
                                regime="target",
                                projection={"x": _identity_x},
                            )
                        },
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={"x": _identity_x}
                                )
                            )
                        },
                    )
                },
            )
        },
    )


def test_participation_policy_floor_prices_the_actual_landing_action_problem() -> None:
    """A target's interior action crossing cannot manufacture a passing floor.

    Direct enumeration gives target action values (0, 1) at the safe landing
    and (0.5, 0.5) at the risky landing. The strict floor 0.75 therefore opens
    only for safe. With discount 0.5 the source candidates are 0.5 and -0.75;
    interpolating target maxima instead gives 0.5 and 0.75 and reverses choice.
    The rejected destination still beats fallback, so maximization cannot
    substitute for the declared Boolean rule.
    Both time terms vanish at landing; borrowing the source's period or age
    changes the floor decision and therefore the selected source value.
    """
    solution = _participation_floor_model().solve(
        params={"discount_factor": 0.5}, log_level="off"
    )
    np.testing.assert_array_equal(np.asarray(solution.values[0]["source"]), 0.5)


def test_participation_integrity_ignores_a_genuinely_infeasible_source_action() -> None:
    """An impossible source action cannot poison the valid action's participation."""
    solution = _participation_floor_model(nonfinite_risky=True, safe_only=True).solve(
        params={"discount_factor": 0.5}, log_level="off"
    )
    np.testing.assert_array_equal(np.asarray(solution.values[0]["source"]), 0.5)


def test_feasible_invalid_participation_remains_mandatory_at_zero_discount() -> None:
    """Discounting cannot erase a feasible source action's unresolved reader."""
    with pytest.raises(ExecutionPlanningError, match="unresolved numerical status"):
        _participation_floor_model(nonfinite_risky=True).solve(
            params={"discount_factor": 0.0}, log_level="off", runtime_checks=False
        )


@categorical(ordered=False)
class _FutureParticipationRegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt
    future: ScalarInt


def _future_crossing_target(
    *, x: ContinuousState, risk: DiscreteState, period: ScalarInt, age: FloatND
) -> FloatND:
    return _crossing_target(x=x, risk=risk, period=period, age=age) - 0.5


def _future_participation_value(
    *, x: ContinuousState, period: ScalarInt, age: FloatND
) -> FloatND:
    return 1.0 + 0.0 * x + (period - 2) + (age - 2)


def _sloped_negative_fallback(x: ContinuousState) -> FloatND:
    return -2.0 + x


def _settlement_x(x: ContinuousState) -> ContinuousState:
    return x + 0.25


def _future_participation_floor_model(
    *,
    force_risky: bool = False,
    phased_settlement: bool = False,
    allow_initial_target: bool = False,
) -> Model:
    """The participation check's actual Q requires a genuinely later stored value."""
    actions = DiscreteGrid(category_class=Risk)
    return Model(
        regimes={
            "source": Regime(
                actions={"risk": actions},
                functions={"utility": _crossing_source},
                state_transitions={"x": _crossing_landing},
                constraints={"risky_only": _risky_risk_only} if force_risky else {},
            ),
            "target": Regime(
                states={"x": _WITNESS_X},
                actions={"risk": actions},
                functions={"utility": _future_crossing_target},
                state_transitions={"x": _identity_x},
            ),
            "fallback": Regime(
                states={"x": _WITNESS_X},
                functions={
                    "utility": _sloped_negative_fallback
                    if phased_settlement
                    else _negative_fallback
                },
            ),
            "future": Regime(
                states={"x": _WITNESS_X},
                functions={"utility": _future_participation_value},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_FutureParticipationRegimeId,
        initial_nodes={0: "source", 1: "target"}
        if allow_initial_target
        else {0: "source"},
        edges={
            "source": Transition(
                law=ByAge(cases={0: "target"}),
                participation={
                    "target": ParticipationCheck(
                        predicate=_crossing_floor,
                        reads={
                            "V_target": ParticipationRead(
                                kind="Participating",
                                regime="target",
                                projection={"x": _identity_x},
                            )
                        },
                        routes={
                            "only": StakeholderRoute(
                                fallback=Phased(
                                    solve=ProjectedRegimeValue(
                                        regime="fallback",
                                        projection={"x": _identity_x},
                                    ),
                                    simulate=ProjectedRegimeValue(
                                        regime="fallback",
                                        projection={"x": _settlement_x},
                                    ),
                                )
                                if phased_settlement
                                else ProjectedRegimeValue(
                                    regime="fallback", projection={"x": _identity_x}
                                )
                            )
                        },
                    )
                },
            ),
            "target": Transition(law=ByAge(cases={1: "future"})),
        },
    )


def test_nonterminal_participation_prices_its_actual_future_continuation() -> None:
    """Future value at date two restores the crossing at the true participation date."""
    solution = _future_participation_floor_model().solve(
        params={"discount_factor": 0.5}, log_level="off"
    )
    np.testing.assert_array_equal(np.asarray(solution.values[2]["future"]), 1.0)
    np.testing.assert_array_equal(np.asarray(solution.values[1]["target"]), 1.0)
    np.testing.assert_array_equal(np.asarray(solution.values[0]["source"]), 0.5)


def test_actual_participation_publisher_borrows_its_declared_future_native_input(
    *, monkeypatch
) -> None:
    """The source really reads the original later buffer through its dated alias."""
    from _lcm.execution.value_transfer import ValueArtifactKind  # noqa: PLC0415
    from _lcm.participation_continuation import (  # noqa: PLC0415
        PARTICIPATION_CONTINUATION,
        ParticipationContinuation,
    )
    from _lcm.participation_publication import (  # noqa: PLC0415
        _BorrowedParticipationKernel,
    )
    from _lcm.policy_queries import _PolicyQueryKernel  # noqa: PLC0415

    published = []
    consumed = []
    publish = _BorrowedParticipationKernel.__call__
    consume = _PolicyQueryKernel.__call__

    def observe_publish(self, **arguments):
        output = publish(self, **arguments)
        payload = output.continuations[PARTICIPATION_CONTINUATION]
        assert isinstance(payload, ParticipationContinuation)
        original = arguments["next_regime_to_V_arr"]["future"]
        assert payload.period == 1
        assert payload.regime == "target"
        assert all(leaf is original for leaf in payload.leaves().values())
        published.append((payload, original))
        return output

    def observe_consume(self, **arguments):
        if arguments["period"] == 0:
            payload, original = published[0]
            source_payload = arguments["next_regime_to_continuation"]["target"]
            assert isinstance(source_payload, ParticipationContinuation)
            assert source_payload.schema == payload.schema
            assert all(leaf is original for leaf in source_payload.leaves().values())
            assert not original.is_deleted()
            consumed.append(original)
        return consume(self, **arguments)

    monkeypatch.setattr(_BorrowedParticipationKernel, "__call__", observe_publish)
    monkeypatch.setattr(_PolicyQueryKernel, "__call__", observe_consume)
    model = _future_participation_floor_model()
    spec = model._regimes["target"].solution.continuation_spec
    assert spec is not None
    aliases = spec.stored_value_aliases()
    assert aliases
    for alias, origin in aliases.items():
        assert (alias.kind, alias.period, alias.regime) == (
            ValueArtifactKind.CONTINUATION_LEAF,
            1,
            "target",
        )
        assert (origin.kind, origin.period, origin.regime) == (
            ValueArtifactKind.REGIME_VALUE,
            2,
            "future",
        )
    source = model._regimes["source"].solution.period_kernels[0]
    assert isinstance(source, _PolicyQueryKernel)
    readers = {
        read.target
        for program in source.core_programs().values()
        for read in program.requirements.value_reads
    }
    assert set(aliases) <= readers
    solution = model.solve(params={"discount_factor": 0.5}, log_level="off")
    assert consumed
    assert consumed[0] is published[0][1]
    np.testing.assert_array_equal(np.asarray(solution.values[0]["source"]), 0.5)


@pytest.mark.parametrize("corruption", ["missing", "wrong_date"])
def test_participation_source_refuses_a_missing_or_wrong_date_payload(
    *, monkeypatch, corruption: str
) -> None:
    """A failed actual source reader cannot commit or free its borrowed origin."""
    from _lcm.participation_continuation import (  # noqa: PLC0415
        ParticipationContinuation,
    )
    from _lcm.policy_queries import _PolicyQueryKernel  # noqa: PLC0415

    invoke = _PolicyQueryKernel.__call__
    borrowed = []

    def corrupt_source(self, **arguments):
        if arguments["period"] == 0:
            payload = arguments["next_regime_to_continuation"]["target"]
            assert isinstance(payload, ParticipationContinuation)
            borrowed.extend(payload.leaves().values())
            arguments["next_regime_to_continuation"] = (
                {}
                if corruption == "missing"
                else {
                    "target": ParticipationContinuation(
                        regime=payload.regime,
                        period=payload.period + 1,
                        schema=payload.schema,
                        operands=payload.operands,
                    )
                }
            )
        return invoke(self, **arguments)

    monkeypatch.setattr(_PolicyQueryKernel, "__call__", corrupt_source)
    with pytest.raises(ExecutionPlanningError, match="producer-date operand payload"):
        _future_participation_floor_model().solve(
            params={"discount_factor": 0.5}, log_level="off"
        )
    assert borrowed
    assert all(not leaf.is_deleted() for leaf in borrowed)


def _joint_crossing_f(*, x: ContinuousState, risk: DiscreteState) -> FloatND:
    return jnp.where(risk == Risk.safe, 2 * x, 2 - 2 * x)


def _joint_crossing_m(*, x: ContinuousState, risk: DiscreteState) -> FloatND:
    return jnp.where(risk == Risk.safe, 2 - x, 1 + x)


def _joint_participation_floor(*, V_target_f: FloatND, V_target_m: FloatND) -> BoolND:
    return (V_target_f > 1.5) & (V_target_m > 0.75)


def test_public_collective_participation_prices_one_actual_target_policy_tuple() -> (
    None
):
    """The source receives (1,.5), rather than independently maximized (1,1)."""
    actions = DiscreteGrid(category_class=Risk)
    objective = ParetoObjective(weights={"f": 0.75, "m": 0.25})
    model = Model(
        regimes={
            "source": Regime(
                actions={"risk": actions},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _crossing_source, "m": _crossing_source},
                        objective=objective,
                    )
                },
                state_transitions={"x": _crossing_landing},
            ),
            "target": Regime(
                states={"x": _WITNESS_X},
                actions={"risk": actions},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _joint_crossing_f, "m": _joint_crossing_m},
                        objective=objective,
                    )
                },
            ),
            "fallback": Regime(
                states={"x": _WITNESS_X},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _negative_fallback, "m": _negative_fallback},
                        objective=objective,
                    )
                },
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "source"},
        edges={
            "source": Transition(
                law=ByAge(cases={0: "target"}),
                participation={
                    "target": ParticipationCheck(
                        predicate=_joint_participation_floor,
                        reads={
                            f"V_target_{member}": ParticipationRead(
                                kind="Participating",
                                regime="target",
                                stakeholder=member,
                                projection={"x": _identity_x},
                            )
                            for member in ("f", "m")
                        },
                        routes={
                            member: StakeholderRoute(
                                target_stakeholder=member,
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    stakeholder=member,
                                    projection={"x": _identity_x},
                                ),
                            )
                            for member in ("f", "m")
                        },
                    )
                },
            )
        },
    )
    solution = model.solve(params={"discount_factor": 0.5}, log_level="off")
    np.testing.assert_array_equal(np.asarray(solution.values[0]["source"]), [1.0, 0.5])


def test_participation_source_replay_uses_the_same_actual_future_price() -> None:
    """Simulation selects the safe source action selected by the solved price."""
    model = _future_participation_floor_model()
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.asarray([0.0]),
            "regime_id": jnp.asarray([_FutureParticipationRegimeId.source]),
            "x": jnp.asarray([0.0]),
        },
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame.query("period == 0")["risk"], [int(Risk.safe)])
    participation = frame.query("period == 1")
    np.testing.assert_array_equal(participation["regime_name"], ["target"])
    np.testing.assert_array_equal(participation["x"], [0.0])


def test_actual_participation_applies_incoming_phased_settlement_once() -> None:
    """Solve prices its own law while replay settles the rejected actual landing."""
    model = _future_participation_floor_model(force_risky=True, phased_settlement=True)
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    np.testing.assert_array_equal(np.asarray(solution.values[0]["source"]), -0.5)
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.asarray([0.0]),
            "regime_id": jnp.asarray([_FutureParticipationRegimeId.source]),
            "x": jnp.asarray([0.0]),
        },
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    participation = frame.query("period == 1")
    np.testing.assert_array_equal(participation["regime_name"], ["fallback"])
    np.testing.assert_array_equal(participation["x"], [0.75])
    assert frame.query("period == 2").empty


def test_initial_target_does_not_borrow_a_participation_checks_incoming_route() -> None:
    """An initial target occupant has no incoming edge's fallback entitlement."""
    model = _future_participation_floor_model(
        force_risky=True, phased_settlement=True, allow_initial_target=True
    )
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.asarray([1.0]),
            "regime_id": jnp.asarray([_FutureParticipationRegimeId.target]),
            "x": jnp.asarray([0.5]),
        },
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    original = frame.query("period == 1")
    np.testing.assert_array_equal(original["regime_name"], ["target"])
    np.testing.assert_array_equal(original["x"], [0.5])
    np.testing.assert_array_equal(frame.query("period == 2")["regime_name"], ["future"])


@categorical(ordered=False)
class _SharedParticipationRegimeId:
    single: ScalarInt
    pair: ScalarInt
    target: ScalarInt
    fallback: ScalarInt
    done: ScalarInt


def _shared_participation_landing() -> ContinuousState:
    return jnp.asarray(0.5)


def _shared_participation_target_f(risk: DiscreteState) -> FloatND:
    return jnp.where(risk == Risk.safe, 2.0, 1.0)


def _shared_participation_target_m(risk: DiscreteState) -> FloatND:
    return jnp.where(risk == Risk.safe, 1.0, 2.0)


def _shared_participation_fallback_f(x: ContinuousState) -> FloatND:
    return 10.0 + x


def _shared_participation_fallback_m(x: ContinuousState) -> FloatND:
    return 20.0 + x


def _shared_participation_done_m(x: ContinuousState) -> FloatND:
    return 2.0 + 0.0 * x


def _shared_participation_eighth(x: ContinuousState) -> ContinuousState:
    return x + 0.125


def _shared_participation_pair_floor(V_target_f: FloatND) -> BoolND:
    return V_target_f > 3.0


def test_distinct_incoming_sources_keep_their_own_tuple_routes() -> None:
    """Shared target/date keeps scalar and household incoming programs distinct."""
    objective = ParetoObjective(weights={"f": 0.75, "m": 0.25})
    model = Model(
        regimes={
            "single": Regime(
                functions={"utility": _utility_zero},
                state_transitions={"x": _shared_participation_landing},
            ),
            "pair": Regime(
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _utility_zero, "m": _utility_zero},
                        objective=objective,
                    )
                },
                state_transitions={"x": _shared_participation_landing},
            ),
            "target": Regime(
                states={"x": _WITNESS_X},
                actions={"risk": DiscreteGrid(category_class=Risk)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            "f": _shared_participation_target_f,
                            "m": _shared_participation_target_m,
                        },
                        objective=objective,
                    )
                },
                state_transitions={"x": _settlement_x},
            ),
            "fallback": Regime(
                states={"x": _WITNESS_X},
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            "f": _shared_participation_fallback_f,
                            "m": _shared_participation_fallback_m,
                        },
                        objective=objective,
                    )
                },
                state_transitions={"x": _shared_participation_eighth},
            ),
            "done": Regime(
                states={"x": _WITNESS_X},
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            "f": _future_participation_value,
                            "m": _shared_participation_done_m,
                        },
                        objective=objective,
                    )
                },
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_SharedParticipationRegimeId,
        initial_nodes={0: ("single", "pair")},
        edges={
            "single": Transition(
                law=ByAge(cases={0: "target"}),
                participation={
                    "target": ParticipationCheck(
                        predicate=_crossing_floor,
                        reads={
                            "V_target": ParticipationRead(
                                kind="Participating",
                                regime="target",
                                stakeholder="m",
                                projection={"x": _identity_x},
                            )
                        },
                        routes={
                            "only": StakeholderRoute(
                                target_stakeholder="m",
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    stakeholder="f",
                                    projection={"x": _identity_x},
                                ),
                            )
                        },
                    )
                },
            ),
            "pair": Transition(
                law=ByAge(cases={0: "target"}),
                participation={
                    "target": ParticipationCheck(
                        predicate=_shared_participation_pair_floor,
                        reads={
                            "V_target_f": ParticipationRead(
                                kind="Participating",
                                regime="target",
                                stakeholder="f",
                                projection={"x": _identity_x},
                            )
                        },
                        routes={
                            "f": StakeholderRoute(
                                target_stakeholder="f",
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    stakeholder="m",
                                    projection={"x": _settlement_x},
                                ),
                            ),
                            "m": StakeholderRoute(
                                target_stakeholder="m",
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    stakeholder="f",
                                    projection={"x": _shared_participation_eighth},
                                ),
                            ),
                        },
                    )
                },
            ),
            "target": Transition(law=ByAge(cases={1: "done"})),
            "fallback": Transition(law=ByAge(cases={1: "done"})),
        },
    )
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    np.testing.assert_array_equal(np.asarray(solution.values[0]["single"]), 1.0)
    np.testing.assert_array_equal(
        np.asarray(solution.values[0]["pair"]), [10.875, 5.5625]
    )
    role = model.stakeholder_names_to_ids
    frame = (
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "age": jnp.asarray([0.0, 0.0, 0.0]),
                "regime_id": jnp.asarray(
                    [
                        _SharedParticipationRegimeId.single,
                        _SharedParticipationRegimeId.pair,
                        _SharedParticipationRegimeId.pair,
                    ]
                ),
                "own_stakeholder": jnp.asarray([-1, role["f"], role["m"]]),
                "x": jnp.asarray([0.0, 0.0, 0.0]),
            },
            log_level="off",
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values(["period", "subject_id"])
    )
    participation = frame.query("period == 1")
    np.testing.assert_array_equal(participation["subject_id"], [0, 1, 2])
    np.testing.assert_array_equal(
        participation["regime_name"], ["target", "fallback", "fallback"]
    )
    np.testing.assert_array_equal(
        participation["own_stakeholder"], [role["m"], role["m"], role["f"]]
    )
    np.testing.assert_array_equal(participation["x"], [0.5, 0.75, 0.625])
    np.testing.assert_array_equal(participation["value_f"], [2.5, 11.25, 11.125])
    np.testing.assert_array_equal(participation["value_m"], [2.0, 21.75, 21.625])
    np.testing.assert_array_equal(
        participation.query("regime_name == 'target'")["risk"], [int(Risk.safe)]
    )
    final = frame.query("period == 2")
    np.testing.assert_array_equal(final["subject_id"], [0, 1, 2])
    np.testing.assert_array_equal(final["regime_name"], ["done", "done", "done"])
    np.testing.assert_array_equal(
        final["own_stakeholder"], [role["m"], role["m"], role["f"]]
    )
    np.testing.assert_array_equal(final["x"], [0.75, 0.875, 0.75])
    np.testing.assert_array_equal(final["value_f"], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(final["value_m"], [2.0, 2.0, 2.0])


@categorical(ordered=False)
class _ExtraOriginRegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt
    future: ScalarInt
    fallback_future: ScalarInt


def _extra_origin_future_value(
    *, x: ContinuousState, payoff: FloatND, period: ScalarInt, age: FloatND
) -> FloatND:
    return payoff + 0.0 * x + (period - 2) + (age - 2)


def _extra_origin_participation_model() -> Model:
    """The rejected branch reads a later value outside target Full's inputs."""
    return Model(
        regimes={
            "source": Regime(
                actions={"risk": DiscreteGrid(category_class=Risk)},
                functions={"utility": _crossing_source},
                state_transitions={"x": _crossing_landing},
            ),
            "target": Regime(
                states={"x": _WITNESS_X},
                actions={"risk": DiscreteGrid(category_class=Risk)},
                functions={"utility": _future_crossing_target},
                state_transitions={"x": _identity_x},
            ),
            "fallback": Regime(
                states={"x": _WITNESS_X},
                functions={"utility": _negative_fallback},
                state_transitions={"x": _identity_x},
            ),
            "future": Regime(
                states={"x": _WITNESS_X},
                functions={"utility": _future_participation_value},
            ),
            "fallback_future": Regime(
                states={"x": _WITNESS_X},
                functions={"utility": _extra_origin_future_value},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_ExtraOriginRegimeId,
        initial_nodes={0: "source"},
        edges={
            "source": Transition(
                law=ByAge(cases={0: "target"}),
                participation={
                    "target": ParticipationCheck(
                        predicate=_crossing_floor,
                        reads={
                            "V_target": ParticipationRead(
                                kind="Participating",
                                regime="target",
                                projection={"x": _identity_x},
                            )
                        },
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"x": _identity_x},
                                )
                            )
                        },
                    )
                },
            ),
            "target": Transition(law=ByAge(cases={1: "future"})),
            "fallback": Transition(law=ByAge(cases={1: "fallback_future"})),
        },
    )


def test_participation_borrows_a_distinct_fallback_future_original(
    *, monkeypatch
) -> None:
    """Extra native custody preserves the original that reverses the source rank."""
    from _lcm.execution.value_transfer import ValueArtifactKind  # noqa: PLC0415
    from _lcm.participation_continuation import (  # noqa: PLC0415
        PARTICIPATION_CONTINUATION,
        ParticipationContinuation,
    )
    from _lcm.participation_publication import (  # noqa: PLC0415
        _BorrowedParticipationKernel,
    )
    from _lcm.policy_queries import _PolicyQueryKernel  # noqa: PLC0415

    published = []
    consumed = []
    publish = _BorrowedParticipationKernel.__call__
    consume = _PolicyQueryKernel.__call__

    def observe_publish(self, **arguments):
        output = publish(self, **arguments)
        if self.template.regime == "target" and arguments["period"] == 1:
            payload = output.continuations[PARTICIPATION_CONTINUATION]
            assert isinstance(payload, ParticipationContinuation)
            original = arguments["next_regime_to_V_arr"]["fallback_future"]
            names = [
                name
                for name, origin in self.origins.items()
                if origin.kind is ValueArtifactKind.REGIME_VALUE
                and origin.regime == "fallback_future"
                and origin.period == 2
            ]
            assert names
            for name in names:
                assert payload.operands[name] is original
                origin = self.origins[name]
                reads = [
                    read
                    for program in self.core_programs().values()
                    for read in program.requirements.value_reads
                    if read.target == origin
                ]
                assert reads
                assert all(
                    read.source.source_regime == "target"
                    and read.source.source_period == 1
                    and read.source.argument is not None
                    and read.source.path
                    for read in reads
                )
            published.append((payload, original, tuple(names)))
        return output

    def observe_consume(self, **arguments):
        if self.root_context.regime_name == "source" and arguments["period"] == 0:
            published_payload, original, names = published[0]
            payload = arguments["next_regime_to_continuation"]["target"]
            assert isinstance(payload, ParticipationContinuation)
            assert payload.schema == published_payload.schema
            assert payload.period == 1
            assert all(payload.operands[name] is original for name in names)
            assert not original.is_deleted()
            consumed.append(original)
        return consume(self, **arguments)

    monkeypatch.setattr(_BorrowedParticipationKernel, "__call__", observe_publish)
    monkeypatch.setattr(_PolicyQueryKernel, "__call__", observe_consume)
    model = _extra_origin_participation_model()
    params = {"discount_factor": 0.5, "fallback_future": {"payoff": 6.0}}
    solution = model.solve(params=params, log_level="off")
    np.testing.assert_array_equal(
        np.stack((solution.values[2]["future"], solution.values[2]["fallback_future"])),
        [[1.0, 1.0], [6.0, 6.0]],
    )
    np.testing.assert_array_equal(np.asarray(solution.values[0]["source"]), 0.75)
    assert consumed
    assert consumed[0] is published[0][1]
    # Public ValueStore reads are independent copies; live custody above uses
    # the original native array throughout publication and source consumption.
    np.testing.assert_array_equal(
        np.asarray(published[0][1]),
        np.asarray(solution.values[2]["fallback_future"]),
    )
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.asarray([0.0]),
            "regime_id": jnp.asarray([_ExtraOriginRegimeId.source]),
            "x": jnp.asarray([0.0]),
        },
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame.query("period == 0")["risk"], [int(Risk.risky)])
    participation = frame.query("period == 1")
    np.testing.assert_array_equal(participation["regime_name"], ["fallback"])
    np.testing.assert_array_equal(participation["x"], [0.5])
    np.testing.assert_array_equal(participation["value"], [1.0])
    np.testing.assert_array_equal(
        frame.query("period == 2")["regime_name"], ["fallback_future"]
    )


@pytest.mark.parametrize("extra_origin", [False, True])
def test_participation_publisher_reuses_stable_native_callable_identities(
    *, extra_origin: bool
) -> None:
    """Authority, preparation and dispatch share the same native callables."""
    from types import MappingProxyType  # noqa: PLC0415

    from _lcm.participation_publication import (  # noqa: PLC0415
        _BorrowedParticipationKernel,
    )

    model = (
        _extra_origin_participation_model()
        if extra_origin
        else _future_participation_floor_model()
    )
    kernel = model._regimes["target"].solution.period_kernels[1]
    assert isinstance(kernel, _BorrowedParticipationKernel)
    graph = kernel.core_programs()
    assert graph is kernel.core_programs()
    for name, program in graph.items():
        repeated = kernel.core_programs()[name]
        assert repeated is program
        assert repeated.function is program.function
        assert type(program.function).__eq__ is object.__eq__
        assert type(program.function).__hash__ is object.__hash__
        assert repeated.requirements.value_reads is program.requirements.value_reads
        if program.checked_variant is not None:
            assert repeated.checked_variant is program.checked_variant
            assert type(program.checked_variant.function).__eq__ is object.__eq__
            assert type(program.checked_variant.function).__hash__ is object.__hash__
        if not extra_origin:
            assert program is kernel.base.core_programs()[name]
    assert set(kernel.origins.values()) <= {
        read.target
        for program in graph.values()
        for read in program.requirements.value_reads
    }
    rebound = kernel.with_fixed_params(fixed_flat_params=MappingProxyType({}))
    assert rebound is not kernel
    assert rebound.core_programs() is rebound.core_programs()
    assert rebound.core_programs() is not graph
    assert tuple(
        (read.target, read.source)
        for program in rebound.core_programs().values()
        for read in program.requirements.value_reads
    ) == tuple(
        (read.target, read.source)
        for program in graph.values()
        for read in program.requirements.value_reads
    )
