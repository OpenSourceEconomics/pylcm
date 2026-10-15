"""Composite states at the edges of their domain.

- Member laws that read no other discrete input, and edges with no feasible
  successor, keep their tables' ranks.
- A constraint the enumeration cannot evaluate never lets a tuple off its members'
  grids into the code set, and a feasible off-grid law still raises.
- Initial conditions are encoded only for subjects whose regime carries the members,
  after the usual integer checks.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    CompositeState,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    LinSpacedGrid,
    Model,
    PeriodRange,
    Periods,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidInitialConditionsError, InvalidStateTransitionError
from lcm.regime import ConstraintEntry, StateTransitionEntry
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    FunctionName,
    ScalarInt,
    StateName,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION, assert_agrees_to_ulp

_PARAMS = {"discount_factor": 0.9}
_MISSING = -(2**31)


@categorical(ordered=True)
class _Bit:
    zero: ScalarInt
    one: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    live: ScalarInt
    dead: ScalarInt


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _identity_utility(*, a: DiscreteState, b: DiscreteState) -> FloatND:
    return 1.0 + a + b


def _utility(*, a: DiscreteState, b: DiscreteState, choice: DiscreteAction) -> FloatND:
    return 0.0 + a + b + choice


def _utility_reading_x(
    *, a: DiscreteState, b: DiscreteState, choice: DiscreteAction, x: ContinuousState
) -> FloatND:
    return 0.0 + a + b + choice + 0.0 * x


def _no_action(*, choice: DiscreteAction) -> BoolND:
    return choice < 0


def _next_b(*, choice: DiscreteAction) -> DiscreteState:
    return choice


def _step_up(*, a: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return a + choice


def _cap_reading_x(
    *, a: DiscreteState, choice: DiscreteAction, x: ContinuousState
) -> BoolND:
    return (a + choice <= 1) & (x >= 0)


def _model(
    *,
    states: Mapping[StateName, DiscreteGrid | LinSpacedGrid],
    actions: Mapping[StateName, DiscreteGrid],
    transitions: Mapping[StateName, StateTransitionEntry],
    utility: UserFunction,
    constraints: Mapping[FunctionName, ConstraintEntry],
    n_decisions: int = 2,
    starts: tuple[str, ...] = ("live",),
    composite: bool,
) -> Model:
    """A live regime for `n_decisions` periods, then a terminal one."""
    return Model(
        n_periods=n_decisions + 1,
        regimes={
            "live": Regime(
                states=states,
                actions=actions,
                state_transitions=transitions,
                functions={"utility": utility},
                constraints=constraints,
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        edges={
            "live": {
                "live": PeriodRange(start=0, exclusive_stop=n_decisions - 1),
                "dead": Periods(values=(n_decisions - 1,)),
            }
        },
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: starts}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
        composite_states=(
            {
                "history": CompositeState(
                    members=("a", "b"), start_codes={"a": (0,), "b": (0,)}
                )
            }
            if composite
            else {}
        ),
    )


def _fixed_laws_without_actions(*, composite: bool) -> Model:
    return _model(
        states={"a": DiscreteGrid(_Bit), "b": DiscreteGrid(_Bit)},
        actions={},
        transitions={"a": fixed_transition("a"), "b": fixed_transition("b")},
        utility=_identity_utility,
        constraints={},
        composite=composite,
    )


def test_fixed_laws_without_other_inputs_solve_to_the_closed_form() -> None:
    """`1 + 0.9 * 1` at the start: one empty assignment of other inputs, not zero."""
    values = _fixed_laws_without_actions(composite=True).solve(
        params=_PARAMS, log_level="off"
    )
    np.testing.assert_almost_equal(
        float(np.asarray(values.values[0]["live"]).ravel()[0]),
        1.9,
        decimal=DECIMAL_PRECISION,
    )


def _infeasible(*, composite: bool) -> Model:
    return _model(
        states={"a": DiscreteGrid(_Bit), "b": DiscreteGrid(_Bit)},
        actions={"choice": DiscreteGrid(_Bit)},
        transitions={"a": fixed_transition("a"), "b": fixed_transition("b")},
        utility=_utility,
        constraints={"no_action": _no_action},
        composite=composite,
    )


def _start(regime_ids: tuple[int, ...]) -> dict[str, jnp.ndarray]:
    ids = np.asarray(regime_ids, dtype=np.int32)
    codes = np.where(ids == _RegimeId.live, 0, _MISSING).astype(np.int32)
    return {
        "regime_id": jnp.asarray(ids),
        "period": jnp.zeros(len(ids), dtype=jnp.int32),
        "a": jnp.asarray(codes),
        "b": jnp.asarray(codes),
    }


def test_edge_without_feasible_successor_builds_and_starts_are_infeasible() -> None:
    """An empty successor set is a table without rows; the start stays infeasible."""
    mask = _infeasible(composite=True).initial_conditions_feasibility(
        params=_PARAMS, initial_conditions=_start((0,))
    )
    np.testing.assert_array_equal(np.asarray(mask), [False])


def _skipped_cap(
    *, composite: bool, cap: Mapping[FunctionName, ConstraintEntry]
) -> Model:
    return _model(
        states={
            "a": DiscreteGrid(_Bit),
            "b": DiscreteGrid(_Bit),
            "x": LinSpacedGrid(start=0, stop=1, n_points=2),
        },
        actions={"choice": DiscreteGrid(_Bit)},
        transitions={
            "a": _step_up,
            "b": fixed_transition("b"),
            "x": fixed_transition("x"),
        },
        utility=_utility_reading_x,
        constraints=cap,
        n_decisions=3,
        composite=composite,
    )


def test_unevaluable_constraint_keeps_the_code_set_on_the_member_grids() -> None:
    """`a + choice` leaves the grid only where the skipped cap forbids it."""
    lowering = _skipped_cap(
        composite=True, cap={"cap": _cap_reading_x}
    )._composite_states["history"]
    assert lowering.reachable.union == ((0, 0), (1, 0))


def test_unevaluable_constraint_leaves_values_equal_to_the_dense_model() -> None:
    """Composite V at the start equals dense V, `1 + 0.9 + 0.81`."""
    dense = _skipped_cap(composite=False, cap={"cap": _cap_reading_x})
    composite = _skipped_cap(composite=True, cap={"cap": _cap_reading_x})
    dense_v = np.asarray(dense.solve(params=_PARAMS, log_level="off").values[0]["live"])
    composite_v = np.asarray(
        composite.solve(params=_PARAMS, log_level="off").values[0]["live"]
    )
    order = list(composite._regimes["live"].solution.state_names)
    start = tuple(0 for _ in order)
    dense_order = list(dense._regimes["live"].solution.state_names)
    assert_agrees_to_ulp(
        got=composite_v[start],
        expected=dense_v[tuple(0 for _ in dense_order)],
        n_ulp=8,
    )


def test_feasible_off_grid_member_law_still_raises() -> None:
    """Without the cap, `a + choice` leaves the grid at a feasible cell."""
    with pytest.raises(InvalidStateTransitionError):
        _skipped_cap(composite=True, cap={}).solve(
            params=_PARAMS, log_level="off", runtime_checks=True
        )


def _cohort(*, composite: bool) -> Model:
    return _model(
        states={"a": DiscreteGrid(_Bit), "b": DiscreteGrid(_Bit)},
        actions={"choice": DiscreteGrid(_Bit)},
        transitions={"a": fixed_transition("a"), "b": _next_b},
        utility=_utility,
        constraints={},
        starts=("dead", "live"),
        composite=composite,
    )


@pytest.mark.parametrize("regime_ids", [(0, 1), (1, 0), (1, 1), (1, 0, 1)])
def test_subjects_starting_without_the_members_are_not_encoded(
    regime_ids: tuple[int, ...],
) -> None:
    """Subjects starting where no member exists are feasible, as in the dense model."""
    masks = [
        np.asarray(
            _cohort(composite=composite).initial_conditions_feasibility(
                params=_PARAMS, initial_conditions=_start(regime_ids)
            )
        )
        for composite in (False, True)
    ]
    np.testing.assert_array_equal(masks[1], masks[0])


@pytest.mark.parametrize("code", [0.5, -0.5, 1.5])
def test_fractional_member_code_is_refused_before_encoding(code: float) -> None:
    """A fractional member code is refused, never truncated to a valid one."""
    conditions = _start((0,)) | {"a": jnp.asarray([code])}
    with pytest.raises(ValueError, match="integer values required"):
        _cohort(composite=True).initial_conditions_feasibility(
            params=_PARAMS, initial_conditions=conditions
        )


def test_labelled_frame_without_member_on_other_subjects_starts_like_dense() -> None:
    """A labelled frame leaves members empty where the start regime lacks them."""
    frame = pd.DataFrame(
        {
            "regime_name": ["live", "dead"],
            "period": [0, 0],
            "a": ["zero", None],
            "b": ["zero", None],
        }
    )
    masks = [
        np.asarray(
            _cohort(composite=composite).initial_conditions_feasibility(
                params=_PARAMS, initial_conditions=frame
            )
        )
        for composite in (False, True)
    ]
    np.testing.assert_array_equal(masks[1], masks[0])


def _only(*, regime_ids: tuple[int, ...], column: str) -> dict[str, jnp.ndarray]:
    """Start conditions that supply one member column and omit the other."""
    conditions = _start(regime_ids)
    conditions.pop("b" if column == "a" else "a")
    return conditions


@pytest.mark.parametrize("column", ["a", "b"])
def test_one_member_column_for_subjects_without_members_is_accepted(
    column: str,
) -> None:
    """Without a carrying subject, a single optional member column is fine."""
    masks = [
        np.asarray(
            _cohort(composite=composite).initial_conditions_feasibility(
                params=_PARAMS,
                initial_conditions=_only(regime_ids=(1, 1), column=column),
            )
        )
        for composite in (False, True)
    ]
    np.testing.assert_array_equal(masks[1], masks[0])


def test_carrying_subject_without_every_member_column_is_refused() -> None:
    """A subject starting where the members exist needs every member column."""
    with pytest.raises(InvalidInitialConditionsError, match="'b'"):
        _cohort(composite=True).initial_conditions_feasibility(
            params=_PARAMS, initial_conditions=_only(regime_ids=(0, 1), column="a")
        )


@pytest.mark.parametrize("value", [0.5, 2.0**31, float("nan")])
def test_optional_member_column_is_still_checked_as_integers(value: float) -> None:
    """An optional member column keeps the integrality and int32 range checks."""
    conditions = _only(regime_ids=(1,), column="a") | {"a": jnp.asarray([value])}
    with pytest.raises(ValueError, match=r"^a: "):
        _cohort(composite=True).initial_conditions_feasibility(
            params=_PARAMS, initial_conditions=conditions
        )


def test_optional_member_column_of_the_wrong_length_is_refused() -> None:
    """An optional member column must have one entry per subject."""
    conditions = _only(regime_ids=(1,), column="a") | {
        "a": jnp.zeros(2, dtype=jnp.int32)
    }
    with pytest.raises(InvalidInitialConditionsError, match="'a'"):
        _cohort(composite=True).initial_conditions_feasibility(
            params=_PARAMS, initial_conditions=conditions
        )
