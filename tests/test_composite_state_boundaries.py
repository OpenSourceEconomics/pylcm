"""Composite states at the edges of their domain.

- Member laws that read no other discrete input, and edges with no feasible
  successor, keep their tables' ranks.
- A constraint the enumeration cannot evaluate never lets a tuple off its members'
  grids into the code set, and a feasible off-grid law still raises.
- Initial conditions are encoded only for subjects whose regime carries the members,
  after the usual integer checks.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jaxtyping import Array, Int

from _lcm.grids.discrete import MISSING_CAT_CODE
from _lcm.regime_building.composite_states import _read_flat
from lcm import (
    AgeGrid,
    ByPeriod,
    CompositeState,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    LinSpacedGrid,
    Model,
    PeriodRange,
    Periods,
    Phased,
    Regime,
    StochasticTransition,
    Transition,
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


type _Int64 = Int[Array, "..."]


def _jump_off_the_grid(*, a: DiscreteState, choice: DiscreteAction) -> _Int64:
    return jnp.where(choice == 1, jnp.int64(2**32 + 1), a.astype(jnp.int64))


def test_wide_off_grid_successor_never_wraps_into_the_code_set() -> None:
    """`2**32 + 1` is off the grid in int64; narrowed to int32 it would read `1`."""
    if not jax.config.jax_enable_x64:
        pytest.skip("An int64 code needs x64.")
    lowering = _model(
        states={"a": DiscreteGrid(_Bit), "b": DiscreteGrid(_Bit)},
        actions={"choice": DiscreteGrid(_Bit)},
        transitions={"a": _jump_off_the_grid, "b": fixed_transition("b")},
        utility=_utility,
        constraints={},
        composite=True,
    )._composite_states["history"]
    assert lowering.reachable.union == ((0, 0),)


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


def _next_a_before_half(
    *, a: DiscreteState, choice: DiscreteAction, age: FloatND
) -> DiscreteState:
    return jnp.where((choice == 1) & (age < 0.5), jnp.int32(1), a)


def _half_a(*, a: DiscreteState) -> FloatND:
    return a + 0.5


def _next_a_from_half(*, half_a: FloatND, choice: DiscreteAction) -> DiscreteState:
    return jnp.where(choice == 1, jnp.int32(1), jnp.floor(half_a).astype(jnp.int32))


def _quarterly(
    *,
    a_law: StateTransitionEntry,
    functions: Mapping[FunctionName, UserFunction],
    composite: bool,
) -> Model:
    """Two half-year decisions on a sub-annual clock, then a terminal period."""
    return Model(
        regimes={
            "live": Regime(
                states={"a": DiscreteGrid(_Bit), "b": DiscreteGrid(_Bit)},
                actions={"choice": DiscreteGrid(_Bit)},
                state_transitions={"a": a_law, "b": fixed_transition("b")},
                functions={"utility": _utility} | dict(functions),
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        edges={"live": {"live": 0, "dead": 0.5}},
        ages=AgeGrid(start=0, inclusive_stop=1, step="2Q"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "live"},
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


def _start_values(model: Model) -> FloatND:
    """V at the start tuple `(0, 0)` in the first period."""
    v = np.asarray(model.solve(params=_PARAMS, log_level="off").values[0]["live"])
    return v.ravel()[0]


@pytest.mark.parametrize(
    ("a_law", "functions"),
    [(fixed_transition("a"), {}), (_next_a_before_half, {})],
    ids=["no-law-reads-age", "member-law-reads-age"],
)
def test_sub_annual_clock_builds_and_matches_dense(
    *, a_law: StateTransitionEntry, functions: Mapping[FunctionName, UserFunction]
) -> None:
    """A quarterly `AgeGrid` hands numeric ages to the enumeration."""
    values = [
        _start_values(_quarterly(a_law=a_law, functions=functions, composite=c))
        for c in (False, True)
    ]
    assert_agrees_to_ulp(got=values[1], expected=values[0], n_ulp=8)


def test_member_law_reading_a_float_helper_matches_dense() -> None:
    """A member law may read a floating helper of a member through the encoder."""
    models = [
        _quarterly(a_law=_next_a_from_half, functions={"half_a": _half_a}, composite=c)
        for c in (False, True)
    ]
    values = [_start_values(model) for model in models]
    assert_agrees_to_ulp(got=values[1], expected=values[0], n_ulp=8)


def _wide_a(*, a: DiscreteState) -> _Int64:
    return a.astype(jnp.int64)


def _next_a_from_wide(*, wide_a: _Int64, choice: DiscreteAction) -> DiscreteState:
    return jnp.where(choice == 1, jnp.int32(1), wide_a.astype(jnp.int32))


def _next_a_as_int64(*, a: DiscreteState, choice: DiscreteAction) -> _Int64:
    return jnp.where(choice == 1, jnp.int64(1), a.astype(jnp.int64))


@pytest.mark.parametrize(
    ("a_law", "functions"),
    [(_next_a_from_wide, {"wide_a": _wide_a}), (_next_a_as_int64, {})],
    ids=["reads-int64-helper", "returns-int64-code"],
)
def test_member_law_with_int64_codes_matches_dense(
    *, a_law: StateTransitionEntry, functions: Mapping[FunctionName, UserFunction]
) -> None:
    """A member law may read or return signed integer codes of any width."""
    if not jax.config.jax_enable_x64:
        pytest.skip("An int64 code needs x64.")
    models = [
        _quarterly(a_law=a_law, functions=functions, composite=c) for c in (False, True)
    ]
    values = [_start_values(model) for model in models]
    assert_agrees_to_ulp(got=values[1], expected=values[0], n_ulp=8)


def test_member_law_reading_a_float_helper_simulates_like_dense() -> None:
    """The simulated choices with a floating helper equal those of the dense model."""
    conditions = {
        "regime_id": jnp.full(3, _RegimeId.live, dtype=jnp.int32),
        "age": jnp.zeros(3),
        "a": jnp.zeros(3, dtype=jnp.int32),
        "b": jnp.zeros(3, dtype=jnp.int32),
    }
    panels = [
        _quarterly(a_law=_next_a_from_half, functions={"half_a": _half_a}, composite=c)
        .simulate(
            params=_PARAMS, initial_conditions=conditions, log_level="off", seed=5
        )
        .to_dataframe(use_labels=False)[
            ["subject_id", "period", "regime_name", "choice"]
        ]
        .reset_index(drop=True)
        for c in (False, True)
    ]
    pd.testing.assert_frame_equal(panels[1], panels[0])


@categorical(ordered=True)
class _Trit:
    zero: ScalarInt
    one: ScalarInt
    two: ScalarInt


@categorical(ordered=False)
class _SideRegimeId:
    live: ScalarInt
    side: ScalarInt
    dead: ScalarInt


def _step_a(*, a: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return jnp.minimum(a + choice, 2)


def _jump_a(*, a: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return jnp.minimum(a + 2 * choice, 2)


def _side_utility(*, a: DiscreteState, b: DiscreteState) -> FloatND:
    return 1.0 + a + b


def _branching(*, side_in: str) -> Model:
    """`live` steps `a` toward itself and jumps it toward `side`.

    `side_in` names the one phase whose graph reaches `side`; the other phase keeps
    `live` for a second period instead. Both declare `live`'s edge out of period 1,
    which the solve phase needs wherever the simulation can stand.
    """
    reaches_side = {
        "live": {"side": Periods(values=(0,)), "dead": Periods(values=(1,))}
    }
    stays_live = {"live": {"live": Periods(values=(0,)), "dead": Periods(values=(1,))}}
    shared = {"side": {"dead": Periods(values=(1,))}}
    side_phase = reaches_side | shared
    live_phase = stays_live | shared
    return Model(
        n_periods=3,
        regimes={
            "live": Regime(
                states={"a": DiscreteGrid(_Trit), "b": DiscreteGrid(_Bit)},
                actions={"choice": DiscreteGrid(_Bit)},
                state_transitions={
                    "a": {"live": _step_a, "side": _jump_a},
                    "b": fixed_transition("b"),
                },
                functions={"utility": _utility},
            ),
            "side": Regime(
                states={"a": DiscreteGrid(_Trit), "b": DiscreteGrid(_Bit)},
                state_transitions={
                    "a": fixed_transition("a"),
                    "b": fixed_transition("b"),
                },
                functions={"utility": _side_utility},
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        edges=Phased(
            solve=side_phase if side_in == "solve" else live_phase,
            simulate=side_phase if side_in == "simulate" else live_phase,
        ),
        regime_id_class=_SideRegimeId,
        initial_nodes=InitialNodes(by_period={0: "live"}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
        composite_states={
            "history": CompositeState(
                members=("a", "b"), start_codes={"a": (0,), "b": (0,)}
            )
        },
    )


@pytest.mark.parametrize("side_in", ["simulate", "solve"])
def test_each_phase_and_each_target_law_reaches_its_own_tuples(side_in: str) -> None:
    """`live` gets `a + choice` and `side` gets `a + 2 * choice`, whichever phase."""
    by_node = _branching(side_in=side_in)._composite_states["history"].reachable.by_node
    assert {node: set(tuples) for node, tuples in by_node.items()} == {
        (0, "live"): {(0, 0)},
        (1, "live"): {(0, 0), (1, 0)},
        (1, "side"): {(0, 0), (2, 0)},
    }


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _two_targets(*, composite: bool) -> Model:
    """`live` moves to `live` or `side` with probability one half each.

    Toward `live` the member law steps `a`; toward `side` it jumps `a` by two.
    """
    return Model(
        n_periods=3,
        regimes={
            "live": Regime(
                states={"a": DiscreteGrid(_Trit), "b": DiscreteGrid(_Bit)},
                actions={"choice": DiscreteGrid(_Bit)},
                state_transitions={
                    "a": {"live": _step_a, "side": _jump_a},
                    "b": fixed_transition("b"),
                },
                functions={"utility": _utility},
            ),
            "side": Regime(
                states={"a": DiscreteGrid(_Trit), "b": DiscreteGrid(_Bit)},
                state_transitions={
                    "a": fixed_transition("a"),
                    "b": fixed_transition("b"),
                },
                functions={"utility": _side_utility},
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        edges={
            "live": Transition(
                targets={
                    "live": Periods(values=(0,)),
                    "side": Periods(values=(0,)),
                    "dead": Periods(values=(1,)),
                },
                law=ByPeriod(
                    cases={
                        0: {
                            "live": StochasticTransition(func=_half),
                            "side": StochasticTransition(func=_half),
                        },
                        1: "dead",
                    }
                ),
            ),
            "side": {"dead": Periods(values=(1,))},
        },
        regime_id_class=_SideRegimeId,
        initial_nodes=InitialNodes(by_period={0: "live"}),
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


def test_two_targets_of_one_phase_keep_their_own_member_laws() -> None:
    """From one source and period, each target receives its own law's tuples."""
    by_node = (
        _two_targets(composite=True)._composite_states["history"].reachable.by_node
    )
    assert {node: set(tuples) for node, tuples in by_node.items()} == {
        (0, "live"): {(0, 0)},
        (1, "live"): {(0, 0), (1, 0)},
        (1, "side"): {(0, 0), (2, 0)},
    }


def test_two_targets_of_one_phase_value_like_dense() -> None:
    """The start value weighs each target's own successor, as the dense model does."""
    values = [
        np.asarray(
            _two_targets(composite=c)
            .solve(params=_PARAMS, log_level="off")
            .values[0]["live"]
        ).ravel()[0]
        for c in (False, True)
    ]
    assert_agrees_to_ulp(got=values[1], expected=values[0], n_ulp=8)


@pytest.mark.parametrize("code", [2**32, 2**32 + 1, -(2**32)])
def test_wide_code_outside_the_table_never_wraps_into_it(code: int) -> None:
    """An int64 code is checked against the table before it is narrowed to int32."""
    if not jax.config.jax_enable_x64:
        pytest.skip("An int64 code needs x64.")
    table = jnp.arange(4, dtype=jnp.int32)
    got = _read_flat(table=table, shape=(2, 2), index=(jnp.int64(code), jnp.int64(0)))
    assert int(got) == MISSING_CAT_CODE
