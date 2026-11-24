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

import logging
from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    DiscreteGrid,
    Gate,
    IrregSpacedGrid,
    LinSpacedGrid,
    MemberRoute,
    Model,
    NormalIIDProcess,
    ProjectedRegimeValue,
    Regime,
    Transition,
    categorical,
)
from lcm.exceptions import (
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
    saving_grid: IrregSpacedGrid = _SAVING,
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
                            "only": MemberRoute(
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
                            "only": MemberRoute(
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
def test_solve_rejects_unknown_categories_in_a_process_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Codes outside the categorical domain cannot publish a finite value.

    The stored values do not hold the gated continuation the regime read, so the
    NaN report comes without the U/F/E/Q breakdown rather than from a failed
    attempt to rebuild it.
    """
    model = _make_categorical_process_fallback_model(
        category_projection=_project_unknown_category
    )

    with (
        caplog.at_level(logging.WARNING, logger="lcm"),
        pytest.raises(InvalidValueFunctionError, match="NaN") as exc_info,
    ):
        model.solve(params=_params(), log_level="debug")
    assert exc_info.value.diagnostics is None
    assert "Diagnostic enrichment failed" not in caplog.text


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
                            "only": MemberRoute(
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
