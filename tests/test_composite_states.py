"""A composite state solves and simulates the model on reachable member tuples only.

Declaring deterministic discrete member states as one composite state replaces their
axes by a single generated state whose codes index the tuples reachable from the
declared start codes. User functions keep reading the members. A generated
constraint makes every tuple infeasible at the periods it cannot be reached in.

The model reproduces the dense model's values at every reachable cell and its
simulated panel exactly.
"""

import dataclasses
import itertools
from collections.abc import Mapping
from types import MappingProxyType

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.grids.discrete import MISSING_CAT_CODE
from lcm import (
    AgeGrid,
    CompositeState,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    StochasticTransition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidSimulationInputError,
    ModelInitializationError,
)
from lcm.regime import ConstraintEntry, StateTransitionEntry
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    FunctionName,
    RegimeName,
    ScalarInt,
    StateName,
)
from tests.conftest import assert_agrees_to_ulp

_CAP = 2
_MEMBERS = ("count_a", "count_b", "count_c")
_N_PERIODS = 6
_PARAMS = {"discount_factor": 0.95}


@categorical(ordered=True)
class _Count:
    zero: ScalarInt
    one: ScalarInt
    two: ScalarInt


@categorical(ordered=False)
class _Choice:
    a: ScalarInt
    b: ScalarInt
    c: ScalarInt
    rest: ScalarInt


@categorical(ordered=False)
class _Offer:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _next_count_a(*, count_a: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return jnp.where(choice == _Choice.a, jnp.minimum(count_a + 1, _CAP), count_a)


def _next_count_b(
    *, count_b: DiscreteState, choice: DiscreteAction, cap: ScalarInt
) -> DiscreteState:
    return jnp.where(choice == _Choice.b, jnp.minimum(count_b + 1, cap), count_b)


def _next_count_c(*, count_c: DiscreteState, choice: DiscreteAction) -> DiscreteState:
    return jnp.where(choice == _Choice.c, jnp.minimum(count_c + 1, _CAP), count_c)


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption + 1.0


def _offer_probs(count_a: DiscreteState) -> FloatND:
    p = 0.2 + 0.3 * count_a
    return jnp.stack([1 - p, p])


def _c_needs_a(*, choice: DiscreteAction, count_a: DiscreteState) -> BoolND:
    return (choice != _Choice.c) | (count_a > 0)


def _b_needs_offer(*, choice: DiscreteAction, offer: DiscreteState) -> BoolND:
    return (choice != _Choice.b) | (offer == _Offer.yes)


def _budget(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _utility(
    *,
    choice: DiscreteAction,
    count_a: DiscreteState,
    count_b: DiscreteState,
    count_c: DiscreteState,
    consumption: ContinuousAction,
) -> FloatND:
    skill = 0.3 * count_a + 0.2 * count_b + 0.5 * count_c * count_a
    return jnp.log(consumption) + skill - 0.05 * (choice != _Choice.rest)


def _retired_utility(
    *, count_a: DiscreteState, count_b: DiscreteState, count_c: DiscreteState
) -> FloatND:
    return 0.4 * count_a + 0.3 * count_b + 0.2 * count_c


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _model(
    *,
    composite_states: Mapping[StateName, CompositeState] | None = None,
    count_a_law: StateTransitionEntry = _next_count_a,
    retired_count_c: DiscreteGrid | None = None,
    extra_constraints: Mapping[FunctionName, ConstraintEntry] | None = None,
) -> Model:
    """Counters move in `alive`, freeze in `retired`, and `dead` is terminal."""
    alive = Regime(
        states={name: DiscreteGrid(_Count) for name in _MEMBERS}
        | {
            "offer": DiscreteGrid(_Offer),
            "wealth": LinSpacedGrid(start=1, stop=4, n_points=4),
        },
        state_transitions={
            "count_a": count_a_law,
            "count_b": _next_count_b,
            "count_c": _next_count_c,
            "offer": StochasticTransition(func=_offer_probs),
            "wealth": _next_wealth,
        },
        actions={
            "choice": DiscreteGrid(_Choice),
            "consumption": LinSpacedGrid(start=0.5, stop=3, n_points=4),
        },
        functions={"utility": _utility},
        constraints={
            "c_needs_a": _c_needs_a,
            "b_needs_offer": _b_needs_offer,
            "budget": _budget,
        }
        | dict(extra_constraints or {}),
    )
    retired = Regime(
        states={name: DiscreteGrid(_Count) for name in _MEMBERS}
        | ({} if retired_count_c is None else {"count_c": retired_count_c}),
        state_transitions={name: fixed_transition(name) for name in _MEMBERS},
        functions={"utility": _retired_utility},
    )
    return Model(
        regimes={
            "alive": alive,
            "retired": retired,
            "dead": Regime(functions={"utility": _zero_utility}),
        },
        edges={
            "alive": {"alive": range(_N_PERIODS - 3), "retired": _N_PERIODS - 3},
            "retired": {"dead": _N_PERIODS - 2},
        },
        ages=AgeGrid(start=0, inclusive_stop=_N_PERIODS - 1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        fixed_params={"alive": {"next_count_b": {"cap": _CAP}}},
        execution_config=ExecutionConfig(device_memory_bytes=None),
        composite_states=composite_states or {},
    )


def _composite() -> Model:
    return _model(composite_states=_history())


def _member_first(
    *,
    model: Model,
    regime: RegimeName,
    values: np.ndarray,
    names: tuple[StateName, ...],
) -> np.ndarray:
    """Move the named state axes of a value array to the front."""
    order = list(model._regimes[regime].solution.state_names)
    return np.moveaxis(
        values, [order.index(name) for name in names], list(range(len(names)))
    )


def test_values_agree_with_dense_model_at_every_reachable_cell() -> None:
    """Composite V equals dense V wherever a member tuple is reached."""
    dense, composite = _model(), _composite()
    lowering = composite._composite_states["history"]
    dense_values = dense.solve(params=_PARAMS, log_level="off").values
    composite_values = composite.solve(params=_PARAMS, log_level="off").values
    gathered_dense, gathered_composite = [], []
    for (period, regime), tuples in lowering.reachable.by_node.items():
        full = _member_first(
            model=dense,
            regime=regime,
            values=np.asarray(dense_values[period][regime]),
            names=_MEMBERS,
        )
        reduced = _member_first(
            model=composite,
            regime=regime,
            values=np.asarray(composite_values[period][regime]),
            names=("history",),
        )
        for members in sorted(tuples):
            gathered_dense.append(np.ravel(full[members]))
            gathered_composite.append(np.ravel(reduced[lowering.encode(members)]))
    assert_agrees_to_ulp(
        got=np.concatenate(gathered_composite),
        expected=np.concatenate(gathered_dense),
        n_ulp=4,
    )


def _initial_conditions() -> dict[str, jnp.ndarray]:
    n = 40
    rng = np.random.default_rng(seed=3)
    return {
        "regime_id": jnp.full(n, _RegimeId.alive, dtype=jnp.int32),
        "age": jnp.zeros(n),
        "wealth": jnp.asarray(rng.uniform(1, 4, n)),
        "offer": jnp.asarray(rng.integers(0, 2, n), dtype=jnp.int32),
    } | {name: jnp.zeros(n, dtype=jnp.int32) for name in _MEMBERS}


def _panel(*, model: Model, composite: bool) -> pd.DataFrame:
    frame = model.simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        log_level="off",
        seed=11,
    ).to_dataframe(
        use_categorical_codes=True,
        additional_targets=list(_MEMBERS) if composite else None,
    )
    columns = ["subject_id", "period", "regime_name", "choice", "consumption"]
    out = frame[[*columns, "wealth", "offer", *_MEMBERS]].copy()
    out[list(_MEMBERS)] = out[list(_MEMBERS)].astype(float)
    return out.reset_index(drop=True)


def test_simulated_panel_equals_dense_panel() -> None:
    """Choices, states and decoded members match the dense model row for row."""
    pd.testing.assert_frame_equal(
        _panel(model=_composite(), composite=True),
        _panel(model=_model(), composite=False),
        check_dtype=False,
    )


def test_value_array_has_one_entry_per_code_and_other_state_cell() -> None:
    """The composite axis replaces the member product in the stored value array."""
    composite = _composite()
    values = composite.solve(params=_PARAMS, log_level="off").values
    n_codes = len(composite._composite_states["history"].table)
    assert np.asarray(values[0]["alive"]).shape == (n_codes, 2, 4)


def test_code_unreachable_at_its_period_is_infeasible() -> None:
    """At period 0 only the start tuple is reachable, so every other code reads -inf."""
    composite = _composite()
    lowering = composite._composite_states["history"]
    start = lowering.encode((0, 0, 0))
    v0 = np.asarray(composite.solve(params=_PARAMS, log_level="off").values[0]["alive"])
    other = np.delete(v0, start, axis=0)
    assert (np.isneginf(other).all(), np.isfinite(v0[start]).all()) == (True, True)


def test_initial_member_tuple_outside_the_code_set_is_refused() -> None:
    """A start the declaration never reaches has no composite code."""
    conditions = _initial_conditions() | {
        "count_c": jnp.full(40, _CAP, dtype=jnp.int32)
    }
    with pytest.raises(InvalidInitialConditionsError, match="count_c"):
        _composite().simulate(
            params=_PARAMS, initial_conditions=conditions, log_level="off", seed=1
        )


def test_labelled_initial_frame_names_members_like_the_dense_model() -> None:
    """A DataFrame of member labels starts subjects at the same composite codes."""
    conditions = _initial_conditions()
    frame = pd.DataFrame(
        {
            "regime_name": "alive",
            "age": np.asarray(conditions["age"]),
            "wealth": np.asarray(conditions["wealth"]),
            "offer": np.where(np.asarray(conditions["offer"]) == 1, "yes", "no"),
        }
        | dict.fromkeys(_MEMBERS, "zero")
    )

    def run(model: Model) -> pd.DataFrame:
        return model.simulate(
            params=_PARAMS, initial_conditions=frame, log_level="off", seed=11
        ).to_dataframe(use_categorical_codes=True)[["subject_id", "period", "choice"]]

    pd.testing.assert_frame_equal(run(_composite()), run(_model()))


def test_unknown_member_label_in_initial_frame_is_named() -> None:
    """A label that is no category of its member is reported as that label."""
    frame = pd.DataFrame(
        {
            "regime_name": ["alive"],
            "age": [0.0],
            "wealth": [2.0],
            "offer": ["no"],
            "count_a": ["zero"],
            "count_b": ["zero"],
            "count_c": ["bogus"],
        }
    )
    with pytest.raises(InvalidInitialConditionsError, match="bogus"):
        _composite().simulate(
            params=_PARAMS, initial_conditions=frame, log_level="off", seed=1
        )


def test_composite_name_taken_by_an_action_is_refused() -> None:
    """The generated state may not reuse the name of a declared action."""
    with pytest.raises(ModelInitializationError, match="choice"):
        _model(
            composite_states={
                "choice": CompositeState(
                    members=_MEMBERS, start_codes=dict.fromkeys(_MEMBERS, (0,))
                )
            }
        )


def _history() -> dict[StateName, CompositeState]:
    return {
        "history": CompositeState(
            members=_MEMBERS, start_codes=dict.fromkeys(_MEMBERS, (0,))
        )
    }


def _keep_count_a(*, count_a: DiscreteState) -> DiscreteState:
    return count_a


def test_phase_specific_member_law_is_refused() -> None:
    """A member whose law differs between solve and simulate names its law."""
    with pytest.raises(ModelInitializationError, match="count_a"):
        _model(
            composite_states=_history(),
            count_a_law=Phased(solve=_next_count_a, simulate=_keep_count_a),
        )


@categorical(ordered=True)
class _WideCount:
    zero: ScalarInt
    one: ScalarInt
    two: ScalarInt
    three: ScalarInt


def test_member_held_on_different_grids_is_refused() -> None:
    """Every carrying regime must hold a member on one grid."""
    with pytest.raises(ModelInitializationError, match="count_c"):
        _model(composite_states=_history(), retired_count_c=DiscreteGrid(_WideCount))


def _always(*, choice: DiscreteAction) -> BoolND:
    return choice >= 0


def test_constraint_named_like_the_generated_one_is_refused() -> None:
    """The generated reachability constraint may not replace a declared one."""
    with pytest.raises(ModelInitializationError, match="history_is_reachable"):
        _model(
            composite_states=_history(),
            extra_constraints={"history_is_reachable": _always},
        )


def test_round_tripped_model_simulates_the_same_panel() -> None:
    """A pickled and restored model keeps its composite lowering."""
    composite = _composite()
    restored = cloudpickle.loads(cloudpickle.dumps(composite))
    pd.testing.assert_frame_equal(
        _panel(model=restored, composite=True), _panel(model=composite, composite=True)
    )


@categorical(ordered=True)
class _Bit:
    zero: ScalarInt
    one: ScalarInt


@categorical(ordered=False)
class _LiveId:
    live: ScalarInt
    dead: ScalarInt


def _bit_utility(
    *, a: DiscreteState, b: DiscreteState, choice: DiscreteAction
) -> FloatND:
    return 0.0 + a + b + choice


def _next_b_from_choice(*, choice: DiscreteAction) -> DiscreteState:
    return choice


def _next_a_from_numeric_choice(*, choice: DiscreteAction) -> DiscreteState:
    return jnp.where(choice == 20, jnp.int32(1), jnp.int32(0))


def _two_bit_model(
    *,
    a_grid: DiscreteGrid,
    choice_grid: DiscreteGrid,
    transitions: Mapping[StateName, StateTransitionEntry],
) -> Model:
    """Two binary members, one binary choice, two decisions and a terminal period."""
    return Model(
        regimes={
            "live": Regime(
                states={"a": a_grid, "b": DiscreteGrid(_Bit)},
                actions={"choice": choice_grid},
                state_transitions=transitions,
                functions={"utility": _bit_utility},
            ),
            "dead": Regime(functions={"utility": _zero_utility}),
        },
        edges={"live": {"live": 0, "dead": 1}},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_LiveId,
        initial_nodes={0: "live"},
        execution_config=ExecutionConfig(device_memory_bytes=None),
        composite_states={
            "history": CompositeState(
                members=("a", "b"), start_codes={"a": (0,), "b": (0,)}
            )
        },
    )


def test_member_on_a_value_grid_is_refused() -> None:
    """Members are read as codes, so a member declared by its values is refused."""
    with pytest.raises(ModelInitializationError, match=r"'a'.*values"):
        _two_bit_model(
            a_grid=DiscreteGrid(values=np.array([10, 20])),
            choice_grid=DiscreteGrid(_Bit),
            transitions={"a": fixed_transition("a"), "b": _next_b_from_choice},
        )


def test_member_law_reading_a_value_grid_is_refused() -> None:
    """A member law reading a variable declared by its values names that input."""
    with pytest.raises(ModelInitializationError, match=r"next_a.*'choice'"):
        _two_bit_model(
            a_grid=DiscreteGrid(_Bit),
            choice_grid=DiscreteGrid(values=np.array([10, 20])),
            transitions={"a": _next_a_from_numeric_choice, "b": fixed_transition("b")},
        )


def _independent_reachable_sets() -> dict[tuple[int, str], set[tuple[int, ...]]]:
    """Every counter tuple some admissible choice path reaches, per node.

    `alive` decides at periods `0 .. _N_PERIODS - 3`, `retired` holds the tuples of
    the period after. Counter `c` may rise only once `a` is positive; the offer is
    enumerated over both codes, so `b` may always rise.
    """
    reached: dict[tuple[int, str], set[tuple[int, ...]]] = {}
    for period in range(_N_PERIODS - 1):
        regime = "alive" if period <= _N_PERIODS - 3 else "retired"
        for path in itertools.product(range(4), repeat=period):
            counts = [0, 0, 0]
            for choice in path:
                if choice == int(_Choice.c) and counts[0] == 0:
                    break
                if choice < len(counts):
                    counts[choice] = min(counts[choice] + 1, _CAP)
            else:
                reached.setdefault((period, regime), set()).add(tuple(counts))
    return reached


def test_value_is_finite_exactly_at_codes_reachable_at_each_period() -> None:
    """At every node the finite cells of V are the independently reachable codes."""
    composite = _composite()
    lowering = composite._composite_states["history"]
    values = composite.solve(params=_PARAMS, log_level="off").values
    finite: dict[tuple[int, str], set[tuple[int, ...]]] = {}
    for period, regimes in values.items():
        for regime in ("alive", "retired"):
            if regime not in regimes:
                continue
            v = _member_first(
                model=composite,
                regime=regime,
                values=np.asarray(regimes[regime]),
                names=("history",),
            )
            codes = np.flatnonzero(np.isfinite(v.reshape(len(v), -1)).any(axis=1))
            if codes.size:
                finite[(period, regime)] = {
                    tuple(int(code) for code in lowering.table[c]) for c in codes
                }
    assert finite == _independent_reachable_sets()


def test_missing_code_marks_in_grid_and_off_grid_tuples_outside_the_set() -> None:
    """`(0, 0, 2)` is on the grids but unreachable; `(3, 0, 0)` is off the grid."""
    lowering = _composite()._composite_states["history"]
    codes = lowering.encode_rows(np.asarray([[0, 0, 2], [3, 0, 0]]))
    assert codes.tolist() == [int(MISSING_CAT_CODE)] * 2


def _two_start_codes() -> Model:
    """`count_a` may start at zero or one."""
    return _model(
        composite_states={
            "history": CompositeState(
                members=_MEMBERS,
                start_codes={"count_a": (0, 1), "count_b": (0,), "count_c": (0,)},
            )
        }
    )


def test_labelled_start_with_nonzero_member_label_decodes_to_its_code() -> None:
    """A subject starting at `count_a="one"` reads member code 1 in period 0."""
    frame = pd.DataFrame(
        {
            "regime_name": ["alive", "alive"],
            "age": [0.0, 0.0],
            "wealth": [2.0, 3.0],
            "offer": ["no", "yes"],
            "count_a": ["one", "zero"],
            "count_b": ["zero", "zero"],
            "count_c": ["zero", "zero"],
        }
    )
    panel = (
        _two_start_codes()
        .simulate(params=_PARAMS, initial_conditions=frame, log_level="off", seed=3)
        .to_dataframe(use_categorical_codes=True, additional_targets=list(_MEMBERS))
    )
    first = panel[panel["period"] == 0].sort_values("subject_id")
    assert first[list(_MEMBERS)].to_numpy().tolist() == [[1, 0, 0], [0, 0, 0]]


@pytest.mark.parametrize("round_trip", [False, True])
def test_initial_state_grids_expand_composite_members(*, round_trip: bool) -> None:
    """Initial-condition introspection exposes member grids, never generated codes."""
    model = _composite()
    if not round_trip:
        for declaration in model._composite_states.values():
            assert declaration.member_grids is not None
            for regime_name in ("alive", "retired"):
                originals = declaration.member_grids[regime_name]
                exposed = model.initial_state_grids(regime_name=regime_name)
                for member in _MEMBERS:
                    assert exposed[member] is originals[member]
    if round_trip:
        model = cloudpickle.loads(cloudpickle.dumps(model))
    grids = model.initial_state_grids(regime_name="alive")
    assert set(grids) == {*_MEMBERS, "offer", "wealth"}
    assert set(model.initial_state_names(regime_name="alive")) == set(grids)
    for member in _MEMBERS:
        grid = grids[member]
        assert isinstance(grid, DiscreteGrid)
        assert grid.categories == ("zero", "one", "two")
        assert grid.codes == (0, 1, 2)
    assert grids["wealth"] is model.user_regimes["alive"].states["wealth"]
    assert set(model.initial_state_grids(regime_name="retired")) == set(_MEMBERS)
    assert not model.initial_state_grids(regime_name="dead")
    with pytest.raises(TypeError):
        grids["invented"] = grids["wealth"]  # ty: ignore[invalid-assignment]
    with pytest.raises(InvalidSimulationInputError, match="Unknown regime"):
        model.initial_state_grids(regime_name="absent")


def test_composite_archive_without_member_grids_keeps_simulation() -> None:
    """Legacy archives remain simulatable but refuse unavailable grid inspection."""
    model = _composite()
    model._composite_states = MappingProxyType(
        {
            name: dataclasses.replace(lowering, member_grids=None)
            for name, lowering in model._composite_states.items()
        }
    )
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    assert set(restored.initial_state_names(regime_name="alive")) == {
        *_MEMBERS,
        "offer",
        "wealth",
    }
    with pytest.raises(InvalidSimulationInputError, match="rebuild the model"):
        restored.initial_state_grids(regime_name="alive")
    pd.testing.assert_frame_equal(
        _panel(model=restored, composite=True),
        _panel(model=_composite(), composite=True),
    )
