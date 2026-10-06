"""Simulating subjects grouped by an invariant type leaves every panel byte unchanged.

`ExecutionConfig(invariant_block_widths={"pref_type": 1})` also groups the
simulated population by `pref_type` whenever the forward phase provably never
changes it. Each group is simulated in its own chunks, reading every
continuation that carries `pref_type` through that type's block. Subjects keep
the random keys of their original rows, and the result restores the original
subject order, so the panel equals the ungrouped panel of the same solution bit
for bit. A model whose simulate phase does not preserve the type keeps the
ungrouped route.
"""

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import tests.conftest as test_config
from _lcm.simulation import value_reads
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.random import generate_simulation_keys
from lcm import (
    AgeGrid,
    AgeRange,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    StochasticTransition,
    categorical,
    fixed_transition,
    load_solution,
)
from lcm.result import SimulationResult
from lcm.solver_api import SolutionResult
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)

_LAST_AGE = 4
_N_TYPES = 3
_BUDGET = 2**30


@categorical(ordered=False)
class _PrefType:
    patient: ScalarInt
    average: ScalarInt
    impatient: ScalarInt


@categorical(ordered=False)
class _Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    work: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _OutsideRegimeId:
    work: ScalarInt
    outside: ScalarInt
    dead: ScalarInt


def _work_utility(
    *,
    consumption: ContinuousAction,
    pref_type: DiscreteState,
    health: DiscreteState,
    weight: FloatND,
) -> FloatND:
    return weight[pref_type] * jnp.log(consumption) + 0.2 * health


def _outside_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _typed_bequest(
    *, wealth: ContinuousState, pref_type: DiscreteState, bequest: FloatND
) -> FloatND:
    return bequest[pref_type] * jnp.log(wealth)


def _type_free_bequest(*, wealth: ContinuousState) -> FloatND:
    return 0.7 * jnp.log(wealth)


def _feasible(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 2.0


def _next_health(*, health: DiscreteState, pref_type: DiscreteState) -> FloatND:
    stay = 0.6 + 0.1 * pref_type
    return jnp.where(jnp.arange(2) == health, stay, 1.0 - stay)


def _alive(*, pref_type: DiscreteState, health: DiscreteState, age: float) -> FloatND:
    alive = 0.9 - 0.15 * pref_type - 0.1 * (1 - health)
    return jnp.where(age < _LAST_AGE - 1, alive, 0.0)


def _survival(
    *, pref_type: DiscreteState, health: DiscreteState, age: float
) -> FloatND:
    alive = _alive(pref_type=pref_type, health=health, age=age)
    return jnp.array([alive, 1.0 - alive])


def _survival_beside_outside(
    *, pref_type: DiscreteState, health: DiscreteState, age: float
) -> FloatND:
    alive = _alive(pref_type=pref_type, health=health, age=age)
    return jnp.array([alive, 0.0, 1.0 - alive])


def _outside_survival(age: float) -> FloatND:
    alive = jnp.where(age < _LAST_AGE - 1, 0.8, 0.0)
    return jnp.array([0.0, alive, 1.0 - alive])


def _reset_pref_type(pref_type: DiscreteState) -> DiscreteState:
    return jnp.zeros_like(pref_type)


def _model(
    *,
    typed_dead: bool,
    outside: bool = False,
    blocked: bool = False,
    budget: int | None = None,
    subject_width: int = 3,
    pref_law: Phased | None = None,
) -> Model:
    """Build a life cycle with type-dependent survival and health.

    Args:
        typed_dead: Whether the terminal bequest depends on the type, so that
            `dead` keeps `pref_type`; otherwise every type shares one dead value.
        outside: Whether a type-free regime `outside` is an admissible start;
            it never enters `work`. Requires a type-free `dead`.
        blocked: Whether `pref_type` is blocked, which also groups subjects.
        budget: Explicit device budget, or `None` to run unbudgeted.
        subject_width: Outer and inner subject width.
        pref_law: The law of `pref_type` in `work`; `None` declares the identity.

    Returns:
        The model.

    """
    wealth = LinSpacedGrid(start=1, stop=10, n_points=6)
    pref_type = DiscreteGrid(_PrefType)
    consumption = {"consumption": LinSpacedGrid(start=1, stop=3, n_points=5)}
    regimes = {
        "work": Regime(
            regime_transitions=StochasticTransition(
                func=_survival_beside_outside if outside else _survival
            ),
            states={
                "wealth": wealth,
                "pref_type": pref_type,
                "health": DiscreteGrid(_Health),
            },
            state_transitions={
                "wealth": _next_wealth,
                "pref_type": fixed_transition("pref_type")
                if pref_law is None
                else pref_law,
                "health": StochasticTransition(func=_next_health),
            },
            actions=consumption,
            functions={"utility": _work_utility},
            constraints={"feasible": _feasible},
        ),
        "dead": Regime(
            regime_transitions=None,
            states={"wealth": wealth, "pref_type": pref_type}
            if typed_dead
            else {"wealth": wealth},
            functions={"utility": _typed_bequest if typed_dead else _type_free_bequest},
        ),
    }
    if outside:
        regimes["outside"] = Regime(
            regime_transitions=StochasticTransition(func=_outside_survival),
            states={"wealth": wealth},
            state_transitions={"wealth": _next_wealth},
            actions=consumption,
            functions={"utility": _outside_utility},
            constraints={"feasible": _feasible},
        )
    return Model(
        regimes=regimes,
        ages=AgeGrid(start=0, inclusive_stop=_LAST_AGE, step="Y"),
        regime_id_class=_OutsideRegimeId if outside else _RegimeId,
        initial_nodes={0: ("work", "outside") if outside else "work"},
        edges={
            source: {
                source: AgeRange(exclusive_stop=_LAST_AGE - 1),
                "dead": AgeRange(exclusive_stop=_LAST_AGE),
            }
            for source in regimes
            if source != "dead"
        },
        execution_config=ExecutionConfig(
            invariant_block_widths={"pref_type": 1} if blocked else {},
            axis_widths={"subject": subject_width},
            device_memory_bytes=budget,
        ),
    )


def _params(*, typed_dead: bool, scale: float = 1.0) -> dict:
    return {
        "discount_factor": 0.9,
        "work": {"utility": {"weight": jnp.asarray([1.0, 1.4, 0.7]) * scale}},
        "dead": {"utility": {"bequest": jnp.asarray([0.4, 1.1, 2.3])}}
        if typed_dead
        else {},
    }


# Unbalanced: the middle type is empty, the first type is the largest.
_UNBALANCED = (0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0)


def _initial(
    *, codes: tuple[int, ...] = _UNBALANCED, outside_rows: tuple[int, ...] = ()
) -> dict[str, np.ndarray]:
    """Return initial conditions; `outside_rows` start in `outside` with code 7."""
    n = len(codes)
    pref = np.asarray(codes, dtype=np.int32)
    regime = np.full(n, 0, dtype=np.int32)
    for row in outside_rows:
        pref[row] = 7
        regime[row] = _OutsideRegimeId.outside
    return {
        "wealth": np.linspace(2.0, 9.0, n),
        "pref_type": pref,
        "health": np.arange(n, dtype=np.int32) % 2,
        "age": np.zeros(n),
        "regime_id": regime,
    }


def _leaf_bytes(leaf: object) -> tuple[str, tuple[int, ...], bytes]:
    array = np.asarray(leaf)
    return array.dtype.str, array.shape, array.tobytes()


def _assert_panels_identical(
    *, got: SimulationResult, want: SimulationResult, raw_rows: str = "all"
) -> None:
    """Require the same frame and the same raw bytes, signed zeros and NaNs included.

    `raw_rows="in_regime"` compares each regime-period only at the rows that
    occupy the regime, the rows the panel publishes.
    """
    got_frame = got.to_dataframe()
    want_frame = want.to_dataframe()
    pd.testing.assert_frame_equal(got_frame, want_frame, check_exact=True)
    for column in want_frame.columns:
        if pd.api.types.is_float_dtype(want_frame[column]):
            assert (
                got_frame[column].to_numpy().tobytes()
                == want_frame[column].to_numpy().tobytes()
            ), column
    assert jax.tree.structure(got.raw_results) == jax.tree.structure(want.raw_results)
    for regime, periods in want.raw_results.items():
        for period, data in periods.items():
            mask = np.asarray(data.in_regime)
            other = got.raw_results[regime][period]
            for field in dataclasses.fields(data):
                for got_leaf, want_leaf in zip(
                    jax.tree.leaves(getattr(other, field.name)),
                    jax.tree.leaves(getattr(data, field.name)),
                    strict=True,
                ):
                    if raw_rows == "in_regime":
                        got_leaf = np.asarray(got_leaf)[mask]  # noqa: PLW2901
                        want_leaf = np.asarray(want_leaf)[mask]  # noqa: PLW2901
                    assert _leaf_bytes(got_leaf) == _leaf_bytes(want_leaf), (
                        regime,
                        period,
                        field.name,
                    )


def _assert_general_panels_agree(
    *, got: SimulationResult, want: SimulationResult
) -> None:
    """Require singleton panels' structural bytes and values within eight ULP."""
    got_frame, want_frame = got.to_dataframe(), want.to_dataframe()
    assert got_frame.columns.equals(want_frame.columns)
    pd.testing.assert_frame_equal(
        got_frame.drop(columns="value"),
        want_frame.drop(columns="value"),
        check_exact=True,
    )
    for column in want_frame.columns:
        if column != "value" and pd.api.types.is_float_dtype(want_frame[column]):
            assert _leaf_bytes(got_frame[column].to_numpy()) == _leaf_bytes(
                want_frame[column].to_numpy()
            ), column
    test_config.assert_general_values_agree(
        got={0: {"value": got_frame["value"].to_numpy()}},
        expected={0: {"value": want_frame["value"].to_numpy()}},
    )
    assert jax.tree.structure(got.raw_results) == jax.tree.structure(want.raw_results)
    for regime, periods in want.raw_results.items():
        for period, data in periods.items():
            other = got.raw_results[regime][period]
            for field in dataclasses.fields(data):
                if field.name == "V_arr":
                    test_config.assert_general_values_agree(
                        got={period: {regime: other.V_arr}},
                        expected={period: {regime: data.V_arr}},
                    )
                    continue
                for got_leaf, want_leaf in zip(
                    jax.tree.leaves(getattr(other, field.name)),
                    jax.tree.leaves(getattr(data, field.name)),
                    strict=True,
                ):
                    assert _leaf_bytes(got_leaf) == _leaf_bytes(want_leaf), (
                        regime,
                        period,
                        field.name,
                    )


def _subject_grouping(*, result: SimulationResult) -> str | None:
    """Return the grouping state the call's plan reports."""
    assert result.plan_summary is not None
    return result.plan_summary.subject_grouping


def _archived_solution(
    *, model: Model, params: Mapping, directory: Path
) -> SolutionResult:
    """Solve once and reload from an archive, so two models read the same values."""
    path = model.solve(params=params, log_level="off").save(
        path=directory / f"solution-{len(list(directory.iterdir()))}"
    )
    solution = load_solution(path=path)
    assert isinstance(solution, SolutionResult)
    return solution


def _simulate(
    *,
    model: Model,
    params: Mapping,
    initial: Mapping[str, np.ndarray],
    solution: SolutionResult,
    seed: int = 7,
) -> SimulationResult:
    return model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=seed,
        log_level="off",
    )


def test_plan_groups_original_rows_by_code_and_restores_their_order() -> None:
    """Each chunk holds one code's original rows; positions restore public order.

    Rows of the empty middle code form no chunk, a code outside the grid joins
    the first code, a short tail repeats its group's last row, and every original
    row maps to the position its own output lands at.
    """
    from _lcm.simulation.subject_groups import (  # noqa: PLC0415
        SubjectGroupingRoute,
        plan_subject_groups,
    )

    route = SubjectGroupingRoute(
        state_name="pref_type", codes=(0, 1, 2), value_axis_names={}
    )
    codes = np.asarray((0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 7), dtype=np.int32)
    plan = plan_subject_groups(route=route, codes=codes, n_real=11, width=3)
    assert [(chunk.code, chunk.rows.tolist()) for chunk in plan.chunks] == [
        (0, [0, 2, 3]),
        (0, [5, 6, 8]),
        (0, [9, 10, 10]),
        (2, [1, 4, 7]),
    ]
    assert plan.positions.tolist() == [0, 9, 1, 2, 10, 3, 4, 11, 5, 6, 7]


def test_rowed_keys_are_the_full_population_keys_of_those_rows() -> None:
    """A chunk of arbitrary original rows draws exactly those rows' keys."""
    from _lcm.simulation.subject_groups import SubjectRows  # noqa: PLC0415

    key = jax.random.key(3)
    rows = np.asarray([9, 0, 4, 4], dtype=np.int32)
    _, full = generate_simulation_keys(key=key, names=["x"], n_initial_states=11)
    _, rowed = generate_simulation_keys(
        key=key,
        names=["x"],
        n_initial_states=11,
        subject_slice=SubjectRows(rows=rows),
    )
    np.testing.assert_array_equal(
        jax.random.key_data(rowed["key_x"]),
        jax.random.key_data(full["key_x"])[rows],
    )


@pytest.mark.parametrize("budget", [None, _BUDGET], ids=["unbudgeted", "budgeted"])
@pytest.mark.parametrize("typed_dead", [True, False], ids=["typed", "type_free"])
@pytest.mark.parametrize(
    "codes",
    [_UNBALANCED, (1,) * 7, (2, 1, 0, 2, 1, 0, 2, 1)],
    ids=["unbalanced_empty", "one_type", "balanced"],
)
def test_grouped_panel_equals_the_ungrouped_panel_bitwise(
    *,
    budget: int | None,
    typed_dead: bool,
    codes: tuple[int, ...],
    tmp_path: Path,
) -> None:
    """The same solution simulates to the same bytes grouped or ungrouped.

    Deaths, type-dependent health draws and short tails of each group's chunks
    all occur, and the empty middle type runs nothing.
    """
    params = _params(typed_dead=typed_dead)
    reference = _model(typed_dead=typed_dead, budget=budget)
    grouped = _model(typed_dead=typed_dead, budget=budget, blocked=True)
    solution = _archived_solution(model=reference, params=params, directory=tmp_path)
    initial = _initial(codes=codes)

    want = _simulate(model=reference, params=params, initial=initial, solution=solution)
    got = _simulate(model=grouped, params=params, initial=initial, solution=solution)

    assert _subject_grouping(result=got) == "pref_type"
    assert _subject_grouping(result=want) is None
    _assert_panels_identical(got=got, want=want)


@pytest.mark.parametrize("budget", [None, _BUDGET], ids=["unbudgeted", "budgeted"])
def test_type_free_starts_keep_their_panel_rows(
    *, budget: int | None, tmp_path: Path
) -> None:
    """Subjects starting where the type is irrelevant simulate the same rows.

    They hold a code outside the grid, never occupy a regime carrying the type,
    and read only the shared type-free values.
    """
    params = _params(typed_dead=False)
    reference = _model(typed_dead=False, outside=True, budget=budget)
    grouped = _model(typed_dead=False, outside=True, budget=budget, blocked=True)
    solution = _archived_solution(model=reference, params=params, directory=tmp_path)
    initial = _initial(outside_rows=(1, 5, 6))

    want = _simulate(model=reference, params=params, initial=initial, solution=solution)
    got = _simulate(model=grouped, params=params, initial=initial, solution=solution)

    assert _subject_grouping(result=got) == "pref_type"
    _assert_panels_identical(got=got, want=want, raw_rows="in_regime")


def test_a_population_without_the_type_column_simulates_grouped(
    tmp_path: Path,
) -> None:
    """With every subject outside, the type is never supplied and nothing changes."""
    params = _params(typed_dead=False)
    reference = _model(typed_dead=False, outside=True)
    grouped = _model(typed_dead=False, outside=True, blocked=True)
    solution = _archived_solution(model=reference, params=params, directory=tmp_path)
    initial = _initial(outside_rows=tuple(range(5)), codes=(0,) * 5)
    del initial["pref_type"]
    del initial["health"]

    want = _simulate(model=reference, params=params, initial=initial, solution=solution)
    got = _simulate(model=grouped, params=params, initial=initial, solution=solution)

    assert _subject_grouping(result=got) == "pref_type"
    _assert_panels_identical(got=got, want=want, raw_rows="in_regime")


def test_the_comparator_sees_a_changed_seed_and_changed_params(
    tmp_path: Path,
) -> None:
    """Positive controls: a different seed or parameter vector changes the panel.

    The grouped panel under changed parameters equals the ungrouped panel under
    the same changed parameters, so nothing stale is reused.
    """
    params = _params(typed_dead=True)
    changed = _params(typed_dead=True, scale=1.5)
    reference = _model(typed_dead=True)
    grouped = _model(typed_dead=True, blocked=True)
    solution = _archived_solution(model=reference, params=params, directory=tmp_path)
    changed_solution = _archived_solution(
        model=reference, params=changed, directory=tmp_path
    )
    initial = _initial()

    want = _simulate(model=reference, params=params, initial=initial, solution=solution)
    with pytest.raises(AssertionError):
        _assert_panels_identical(
            got=_simulate(
                model=grouped,
                params=params,
                initial=initial,
                solution=solution,
                seed=8,
            ),
            want=want,
        )
    got_changed = _simulate(
        model=grouped, params=changed, initial=initial, solution=changed_solution
    )
    assert _subject_grouping(result=got_changed) == "pref_type"
    with pytest.raises(AssertionError):
        _assert_panels_identical(got=got_changed, want=want)
    _assert_panels_identical(
        got=got_changed,
        want=_simulate(
            model=reference, params=changed, initial=initial, solution=changed_solution
        ),
    )


def test_blocked_solve_then_grouped_simulation_equals_the_unblocked_route() -> None:
    """End to end: blocked solve plus grouped simulation equals the plain route."""
    params = _params(typed_dead=True)
    reference = _model(typed_dead=True)
    grouped = _model(typed_dead=True, blocked=True)
    initial = _initial()

    want = _simulate(
        model=reference,
        params=params,
        initial=initial,
        solution=reference.solve(params=params, log_level="off"),
    )
    got = _simulate(
        model=grouped,
        params=params,
        initial=initial,
        solution=grouped.solve(params=params, log_level="off"),
    )

    assert _subject_grouping(result=got) == "pref_type"
    _assert_general_panels_agree(got=got, want=want)


@pytest.mark.parametrize("budget", [None, _BUDGET], ids=["unbudgeted", "budgeted"])
def test_a_group_moves_only_its_type_block_of_each_typed_continuation(
    *, budget: int | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every transfer of a typed continuation delivers one third of it.

    The transfer selects the group's code and removes the type axis, so the
    consumer receives, and the transfer accounting charges, the bytes of one
    type's block. Reads of a type-free value are never selected.
    """
    params = _params(typed_dead=True)
    reference = _model(typed_dead=True, budget=budget)
    grouped = _model(typed_dead=True, budget=budget, blocked=True)
    solution = _archived_solution(model=reference, params=params, directory=tmp_path)
    applied = []
    charged = []
    original_apply = value_reads.apply_value_transfer
    original_before = SimulationMemory.before_transfer

    def apply(*, value, transfer, **kwargs):
        result = original_apply(value=value, transfer=transfer, **kwargs)
        applied.append((transfer, np.asarray(value).nbytes, result.shape))
        return result

    def before(*memory, transfer, live_values):
        charged.append(transfer)
        return original_before(*memory, transfer=transfer, live_values=live_values)

    monkeypatch.setattr(value_reads, "apply_value_transfer", apply)
    monkeypatch.setattr(SimulationMemory, "before_transfer", before)
    _simulate(model=grouped, params=params, initial=_initial(), solution=solution)

    assert applied
    for transfer, stored_bytes, shape in applied:
        stored_shape = transfer.expected_shape
        assert transfer.selects
        assert transfer.view.selections[0].state_name == "pref_type"
        assert stored_shape[0] == _N_TYPES
        assert shape == transfer.consumer_shape == stored_shape[1:]
        assert transfer.cost.per_device_bytes * _N_TYPES == stored_bytes
    codes = {transfer.view.selections[0].codes for transfer, _, _ in applied}
    assert codes == {(0,), (2,)}
    if budget is not None:
        assert {id(transfer) for transfer in charged} == {
            id(transfer) for transfer, _, _ in applied
        }


def test_type_free_continuations_are_read_whole(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A shared type-free terminal is delivered unselected to every group."""
    params = _params(typed_dead=False)
    reference = _model(typed_dead=False)
    grouped = _model(typed_dead=False, blocked=True)
    solution = _archived_solution(model=reference, params=params, directory=tmp_path)
    applied = []
    original_apply = value_reads.apply_value_transfer

    def apply(*, value, transfer, **kwargs):
        applied.append(transfer)
        return original_apply(value=value, transfer=transfer, **kwargs)

    monkeypatch.setattr(value_reads, "apply_value_transfer", apply)
    _simulate(model=grouped, params=params, initial=_initial(), solution=solution)

    typed = [t for t in applied if t.target.regime == "work"]
    shared = [t for t in applied if t.target.regime == "dead"]
    assert typed
    assert all(t.selects for t in typed)
    assert all(t.view is None for t in shared)


def test_simulation_stays_ungrouped_when_the_simulate_phase_may_change_the_type(
    tmp_path: Path,
) -> None:
    """A simulate-phase reset of the type keeps the ungrouped route and its panel.

    The solve phase still preserves the type, so blocking the solve is accepted.
    """
    pref_law = Phased(solve=fixed_transition("pref_type"), simulate=_reset_pref_type)
    params = _params(typed_dead=True)
    reference = _model(typed_dead=True, pref_law=pref_law)
    blocked = _model(typed_dead=True, pref_law=pref_law, blocked=True)
    solution = _archived_solution(model=reference, params=params, directory=tmp_path)
    initial = _initial()

    want = _simulate(model=reference, params=params, initial=initial, solution=solution)
    got = _simulate(model=blocked, params=params, initial=initial, solution=solution)

    assert _subject_grouping(result=got) is None
    _assert_panels_identical(got=got, want=want)
