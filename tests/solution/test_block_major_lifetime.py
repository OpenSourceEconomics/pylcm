"""Block-major lifetime execution: one invariant code at a time, same results.

`ExecutionConfig(invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR)`
solves every component of the blocked state — one code — through all of its
periods before the next code starts. A combined `simulate` runs that code's
subjects while its values are still on the device. The component's values are
then retained on the host and its device buffers are deleted, so the device
holds one code's lifetime values at a time.

The public result keeps its contract: a complete logical `ValueStore` whose
entries assemble one value, on the layout an eager solve publishes, only when
read. Values and simulated panels equal the period-major blocked route byte for
byte. Against the unblocked route, structural outputs remain exact and published
values agree within eight ULP; the enumeration oracle is independent.
"""

import dataclasses
import gc
import logging
import weakref
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import tests.conftest as test_config
from _lcm.execution.core_program import core_program_graph
from _lcm.solution import backward_induction, block_major
from lcm import (
    AgeGrid,
    AgeRange,
    DiscreteGrid,
    ExecutionConfig,
    InvariantBlockSchedule,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    categorical,
    fixed_transition,
    load_solution,
)
from lcm.exceptions import ExecutionPlanningError, InvalidSimulationInputError
from lcm.result import SimulationResult
from lcm.solver_api import LoadState, SolutionResult, ValueStore
from lcm.tuning import _array_ulp_gap
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.simulation import test_type_grouped_simulation as life_cycle
from tests.solution import test_invariant_blocking as stage3
from tests.test_models import independent_types

_BLOCK_MAJOR = InvariantBlockSchedule.BLOCK_MAJOR
_PERIOD_MAJOR = InvariantBlockSchedule.PERIOD_MAJOR


def _config(
    *,
    schedule: InvariantBlockSchedule | None,
    budget: int | None = None,
    subject_width: int | None = None,
    axis_widths: Mapping[str, int] = MappingProxyType({}),
) -> ExecutionConfig:
    """Return an execution config: unblocked when `schedule` is `None`."""
    return ExecutionConfig(
        invariant_block_widths={} if schedule is None else {"pref_type": 1},
        invariant_block_schedule=_PERIOD_MAJOR if schedule is None else schedule,
        device_memory_bytes=budget,
        axis_widths={
            **axis_widths,
            **({} if subject_width is None else {"subject": subject_width}),
        },
    )


def _life_cycle_model(
    *,
    schedule: InvariantBlockSchedule | None,
    budget: int | None = None,
    pref_law: Phased | None = None,
) -> Model:
    """The grouped-simulation life cycle with a typed dead regime."""
    model = life_cycle._model(typed_dead=True, pref_law=pref_law)
    return Model(
        edges=model.edges,
        regimes=model.user_regimes,
        ages=model.ages,
        regime_id_class=life_cycle._RegimeId,
        initial_nodes={0: "work"},
        execution_config=_config(schedule=schedule, budget=budget, subject_width=3),
    )


def _workload(
    *, name: str, schedule: InvariantBlockSchedule | None
) -> tuple[Model, dict]:
    if name == "life_cycle":
        return (
            _life_cycle_model(schedule=schedule),
            life_cycle._params(typed_dead=True),
        )
    return stage3._workload(name=name, execution_config=_config(schedule=schedule))


_WORKLOADS = ("independent_types", "sector_typed_terminal", "life_cycle")


def _leaf_bytes(leaf: object) -> tuple[str, tuple[int, ...], bytes]:
    array = np.asarray(leaf)
    return array.dtype.str, array.shape, array.tobytes()


def _assert_value_bytes_equal(
    *, got: Mapping, want: Mapping, ordered: bool = True
) -> None:
    """Require the same coordinates and, at each, the same dtype, shape and bytes.

    `ordered` also requires each period to list its regimes in the same order,
    which an archive, written in sorted order, does not keep.
    """
    if ordered:
        assert {period: tuple(regimes) for period, regimes in got.items()} == {
            period: tuple(regimes) for period, regimes in want.items()
        }
    else:
        assert {(period, regime) for period in got for regime in got[period]} == {
            (period, regime) for period in want for regime in want[period]
        }
    mismatched = [
        (period, regime)
        for period, regimes in want.items()
        for regime in regimes
        if _leaf_bytes(got[period][regime]) != _leaf_bytes(want[period][regime])
    ]
    assert mismatched == []


def _solution(*, model: Model, params: Mapping) -> SolutionResult:
    return model.solve(params=params, log_level="off")


def _store(solution: SolutionResult) -> ValueStore:
    """Return a result's values as the store they are."""
    values = solution.values
    assert isinstance(values, ValueStore)
    return values


def _retained(solution: SolutionResult) -> block_major.RetainedComponentValues:
    """Return the host owner behind a block-major result's value entries."""
    values = _store(solution)
    period = next(iter(values))
    regime = next(iter(values[period]))
    entry = values._raw(period=period, regime=regime)
    owner = getattr(entry, "owner", None)
    assert isinstance(owner, block_major.RetainedComponentValues)
    return owner


class _Assemblies:
    """Count every full value a retained owner assembles on the host."""

    def __init__(self, *, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[int, str]] = []
        assemble = block_major.RetainedComponentValues.assemble_host_value

        calls = self.calls

        # keyword-only-exempt: library-callback=pytest.MonkeyPatch.setattr
        def counted(self: Any, *, period: int, regime: str) -> np.ndarray:
            calls.append((period, regime))
            return assemble(self, period=period, regime=regime)

        monkeypatch.setattr(
            block_major.RetainedComponentValues, "assemble_host_value", counted
        )


def test_execution_config_schedules_period_major_by_default() -> None:
    """Without a request the blocked solve keeps the period-major schedule."""
    assert ExecutionConfig().invariant_block_schedule is _PERIOD_MAJOR


def test_execution_config_refuses_a_schedule_that_is_not_a_member() -> None:
    """The schedule is an `InvariantBlockSchedule` member, never its spelling."""
    with pytest.raises(BeartypeCallHintParamViolation, match="InvariantBlockSchedule"):
        ExecutionConfig(invariant_block_schedule="block_major")  # ty: ignore[invalid-argument-type]


def test_block_major_without_a_blocked_state_is_refused_at_construction() -> None:
    """A component schedule needs a blocked state to take components of."""
    with pytest.raises(ExecutionPlanningError, match="invariant_block_widths"):
        stage3._independent_types_model(
            execution_config=ExecutionConfig(invariant_block_schedule=_BLOCK_MAJOR)
        )


def test_block_major_with_a_type_free_regime_is_refused_at_construction() -> None:
    """Every regime must carry the blocked state; a shared regime is named."""
    with pytest.raises(ExecutionPlanningError, match="'terminal'"):
        stage3._sector_model(
            typed_terminal=False, execution_config=_config(schedule=_BLOCK_MAJOR)
        )


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_block_major_values_equal_the_period_major_values_bitwise(
    workload: str,
) -> None:
    """Every published value is the period-major blocked solve's, byte for byte."""
    model, params = _workload(name=workload, schedule=_BLOCK_MAJOR)
    reference, _ = _workload(name=workload, schedule=_PERIOD_MAJOR)

    _assert_value_bytes_equal(
        got=_solution(model=model, params=params).values,
        want=_solution(model=reference, params=params).values,
    )


@pytest.mark.parametrize("workload", ["independent_types", "sector_typed_terminal"])
def test_block_major_values_agree_with_the_unblocked_values(workload: str) -> None:
    """Published values agree with the unblocked solve within eight ULP."""
    model, params = _workload(name=workload, schedule=_BLOCK_MAJOR)
    reference, _ = _workload(name=workload, schedule=None)

    test_config.assert_general_values_agree(
        got=_solution(model=model, params=params).values,
        expected=_solution(model=reference, params=params).values,
    )


def test_life_cycle_values_agree_with_the_unblocked_values_to_the_ulp() -> None:
    """On the life cycle, blocking itself places values on adjacent floats.

    Evaluating one type at a time compiles a kernel of a different width, whose
    vectorized arithmetic can land on a representable neighbour of the
    unblocked value; the period-major blocked solve does the same, and the
    block-major values equal its values bit for bit.
    """
    model, params = _workload(name="life_cycle", schedule=_BLOCK_MAJOR)
    reference, _ = _workload(name="life_cycle", schedule=None)
    got = _solution(model=model, params=params).values
    want = _solution(model=reference, params=params).values

    test_config.assert_general_values_agree(got=got, expected=want)


def test_block_major_values_match_the_enumerated_oracle() -> None:
    """Each type's values equal the independent enumeration of its own problem."""
    model, params = _workload(name="independent_types", schedule=_BLOCK_MAJOR)
    oracle, _ = independent_types.solve_by_enumeration()
    values = _solution(model=model, params=params).values

    np.testing.assert_array_almost_equal(
        np.stack([np.asarray(values[period]["working"]) for period in (0, 1, 2)]),
        np.asarray([oracle[period] for period in (0, 1, 2)]),
        decimal=test_config.DECIMAL_PRECISION,
    )


def test_block_major_values_follow_changed_params() -> None:
    """A second solve with new parameters reuses nothing from the first."""
    model, params = _workload(name="sector_typed_terminal", schedule=_BLOCK_MAJOR)
    reference, _ = _workload(name="sector_typed_terminal", schedule=None)
    changed_params = stage3._sector_params(typed_terminal=True, scale=1.3)
    base = _solution(model=model, params=params).values

    changed = _solution(model=model, params=changed_params).values

    test_config.assert_general_values_agree(
        got=changed, expected=_solution(model=reference, params=changed_params).values
    )
    assert _leaf_bytes(changed[0]["working"]) != _leaf_bytes(base[0]["working"])


def test_inspecting_a_block_major_result_assembles_no_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinates, load states and schemas are read without touching a block."""
    model, params = _workload(name="independent_types", schedule=_BLOCK_MAJOR)
    solution = _solution(model=model, params=params)
    assemblies = _Assemblies(monkeypatch=monkeypatch)

    coordinates = {
        (period, regime)
        for period in solution.values
        for regime in solution.values[period]
    }
    states = {
        _store(solution).load_state(period=period, regime=regime)
        for period, regime in coordinates
    }
    schemas = dict(solution.metadata.value_schemas)

    assert set(schemas) == coordinates
    assert len(solution.values) == len({period for period, _ in coordinates})
    assert 0 in solution.values
    assert "working" in solution.values[0]
    assert states == {LoadState.UNLOADED}
    assert assemblies.calls == []


def test_reading_one_value_assembles_only_that_entry_on_its_solve_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One read assembles one value, laid out like the period-major solve's value."""
    model, params = _workload(name="sector_typed_terminal", schedule=_BLOCK_MAJOR)
    reference, _ = _workload(name="sector_typed_terminal", schedule=_PERIOD_MAJOR)
    solution = _solution(model=model, params=params)
    want = _solution(model=reference, params=params).value(period=1, regime="working")
    assemblies = _Assemblies(monkeypatch=monkeypatch)

    got = solution.value(period=1, regime="working")

    assert assemblies.calls == [(1, "working")]
    assert isinstance(got, jax.Array)
    assert got.sharding == want.sharding
    assert _leaf_bytes(got) == _leaf_bytes(want)


def test_materializing_every_value_returns_the_period_major_values() -> None:
    """Without a budget, a full materialization equals the reference values."""
    model, params = _workload(name="life_cycle", schedule=_BLOCK_MAJOR)
    reference, _ = _workload(name="life_cycle", schedule=_PERIOD_MAJOR)

    got = _store(_solution(model=model, params=params)).materialize()

    _assert_value_bytes_equal(
        got=got, want=_solution(model=reference, params=params).values
    )


def _simulate(
    *,
    model: Model,
    params: Mapping,
    initial: Mapping[str, np.ndarray],
    solution: SolutionResult | None,
    seed: int = 7,
) -> SimulationResult:
    return model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=seed,
        log_level="off",
    )


_POPULATIONS = (
    pytest.param(life_cycle._UNBALANCED, id="unbalanced_empty"),
    pytest.param((1,) * 7, id="one_type"),
    pytest.param((0, 1, 2) * 3, id="balanced"),
)


def _reference_panels(
    *, params: Mapping, initial: Mapping[str, np.ndarray], combined: bool
) -> dict[str, SimulationResult]:
    """Simulate the period-major and the unblocked routes, split or combined."""
    panels = {}
    for label, schedule in (("period_major", _PERIOD_MAJOR), ("unblocked", None)):
        reference = _life_cycle_model(schedule=schedule)
        panels[label] = _simulate(
            model=reference,
            params=params,
            initial=initial,
            solution=None if combined else _solution(model=reference, params=params),
        )
    return panels


def _assert_panels_match_the_references(
    *, got: SimulationResult, references: Mapping[str, SimulationResult]
) -> None:
    """Require period-major bytes and exact unblocked structure with value-only ULP."""
    life_cycle._assert_panels_identical(got=got, want=references["period_major"])
    life_cycle._assert_general_panels_agree(got=got, want=references["unblocked"])


@pytest.fixture
def rounding_panel() -> tuple[Model, SimulationResult]:
    """Return a real one-subject panel for published-result rounding witnesses."""
    model = _life_cycle_model(schedule=None)
    result = _simulate(
        model=model,
        params=life_cycle._params(typed_dead=True),
        initial=life_cycle._initial(codes=(0,)),
        solution=None,
    )
    return model, result


def _create_rounding_panel(
    *,
    model: Model,
    result: SimulationResult,
    value_steps: int = 0,
    start_steps_below_one: int = 0,
    structural_field: str | None = None,
    structural_zero: bool = False,
) -> SimulationResult:
    """Publish a fixed value with an optional adjacent state or action witness."""
    data = result.raw_results["work"][0]
    one = np.ones_like(np.asarray(data.V_arr))
    value = one.copy()
    for _ in range(start_steps_below_one):
        value = np.nextafter(value, np.zeros_like(value))
    for _ in range(value_steps):
        value = np.nextafter(value, np.full_like(value, np.inf))
    wealth = np.ones_like(np.asarray(data.states["wealth"]))
    consumption = np.ones_like(np.asarray(data.actions["consumption"]))
    if structural_zero:
        wealth = np.zeros_like(wealth)
    if structural_field == "state":
        wealth = (
            -wealth
            if structural_zero
            else np.nextafter(wealth, np.full_like(wealth, np.inf))
        )
    elif structural_field == "action":
        consumption = np.nextafter(consumption, np.full_like(consumption, np.inf))
    changed = dataclasses.replace(
        data,
        V_arr=jnp.asarray(value),
        states=MappingProxyType({**data.states, "wealth": jnp.asarray(wealth)}),
        actions=MappingProxyType(
            {**data.actions, "consumption": jnp.asarray(consumption)}
        ),
    )
    raw = MappingProxyType(
        {
            **result.raw_results,
            "work": MappingProxyType({**result.raw_results["work"], 0: changed}),
        }
    )
    assert model.ages is not None
    return SimulationResult(
        raw_results=raw,
        regimes=model._regimes,
        flat_params=result.flat_params,
        period_to_regime_to_V_arr=result.period_to_regime_to_V_arr,
        ages=model.ages,
        simulation_output_dtypes=model.simulation_output_dtypes,
    )


@pytest.mark.parametrize("structural_field", ["state", "action"])
def test_general_panel_refuses_an_adjacent_structural_output(
    *, rounding_panel: tuple[Model, SimulationResult], structural_field: str
) -> None:
    """An allowed eight-ULP value gap never permits a one-ULP state or action gap."""
    model, result = rounding_panel
    want = _create_rounding_panel(model=model, result=result)
    got = _create_rounding_panel(
        model=model, result=result, value_steps=8, structural_field=structural_field
    )

    with pytest.raises(AssertionError):
        _assert_panels_match_the_references(
            got=got, references={"period_major": got, "unblocked": want}
        )


@pytest.mark.parametrize("start_steps_below_one", [0, 7])
def test_general_panel_accepts_an_eight_ulp_value_gap(
    *, rounding_panel: tuple[Model, SimulationResult], start_steps_below_one: int
) -> None:
    """Exactly eight value ULP are accepted while every structural bit agrees."""
    model, result = rounding_panel
    want = _create_rounding_panel(
        model=model, result=result, start_steps_below_one=start_steps_below_one
    )
    got = _create_rounding_panel(
        model=model,
        result=result,
        start_steps_below_one=start_steps_below_one,
        value_steps=8,
    )

    assert (
        _array_ulp_gap(
            got=np.asarray(got.raw_results["work"][0].V_arr),
            expected=np.asarray(want.raw_results["work"][0].V_arr),
        ),
        _assert_panels_match_the_references(
            got=got, references={"period_major": got, "unblocked": want}
        ),
    ) == (8.0, None)


def test_general_panel_refuses_a_nine_ulp_value_gap(
    *, rounding_panel: tuple[Model, SimulationResult]
) -> None:
    """A published value nine representable steps away exceeds the eight-ULP bound."""
    model, result = rounding_panel
    want = _create_rounding_panel(model=model, result=result)
    got = _create_rounding_panel(model=model, result=result, value_steps=9)

    with pytest.raises(AssertionError):
        _assert_panels_match_the_references(
            got=got, references={"period_major": got, "unblocked": want}
        )


def test_general_panel_refuses_a_state_signed_zero_change(
    *, rounding_panel: tuple[Model, SimulationResult]
) -> None:
    """An allowed value gap preserves the sign bit of a published float state."""
    model, result = rounding_panel
    want = _create_rounding_panel(model=model, result=result, structural_zero=True)
    got = _create_rounding_panel(
        model=model,
        result=result,
        value_steps=8,
        structural_field="state",
        structural_zero=True,
    )

    with pytest.raises(AssertionError):
        _assert_panels_match_the_references(
            got=got, references={"period_major": got, "unblocked": want}
        )


def test_general_panel_refuses_nine_value_steps_across_a_binade(
    *, rounding_panel: tuple[Model, SimulationResult]
) -> None:
    """A change in float spacing never admits more than eight representable steps."""
    model, result = rounding_panel
    want = _create_rounding_panel(model=model, result=result, start_steps_below_one=7)
    got = _create_rounding_panel(
        model=model, result=result, start_steps_below_one=7, value_steps=9
    )

    with pytest.raises(AssertionError):
        _assert_panels_match_the_references(
            got=got, references={"period_major": got, "unblocked": want}
        )


@pytest.mark.parametrize("codes", _POPULATIONS)
def test_solve_then_simulate_matches_the_reference_panels(
    codes: tuple[int, ...],
) -> None:
    """A block-major solve simulates to the period-major panel, byte for byte."""
    params = life_cycle._params(typed_dead=True)
    initial = life_cycle._initial(codes=codes)
    model = _life_cycle_model(schedule=_BLOCK_MAJOR)

    got = _simulate(
        model=model,
        params=params,
        initial=initial,
        solution=_solution(model=model, params=params),
    )

    _assert_panels_match_the_references(
        got=got,
        references=_reference_panels(params=params, initial=initial, combined=False),
    )


def test_simulating_a_block_major_solution_assembles_no_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulation reads each code's blocks, never a complete value."""
    params = life_cycle._params(typed_dead=True)
    model = _life_cycle_model(schedule=_BLOCK_MAJOR)
    solution = _solution(model=model, params=params)
    assemblies = _Assemblies(monkeypatch=monkeypatch)

    _simulate(
        model=model, params=params, initial=life_cycle._initial(), solution=solution
    )

    assert assemblies.calls == []


_LOOKUP_STATES = MappingProxyType(
    {
        "wealth": jnp.array([1.0, 4.5, 10.0, 2.5, 7.0, 1.0]),
        "pref_type": jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.int32),
        "health": jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32),
    }
)


def _lookup_column(
    *, schedule: InvariantBlockSchedule, period: int, column: str
) -> np.ndarray:
    params = life_cycle._params(typed_dead=True)
    model = _life_cycle_model(schedule=schedule)
    got = model.lookup_policy(
        params=params,
        solution=_solution(model=model, params=params),
        period=period,
        regime_name="work",
        states=_LOOKUP_STATES,
    )
    return np.asarray(got.value if column == "value" else got.actions[column])


@pytest.mark.parametrize("column", ["consumption", "value"])
@pytest.mark.parametrize("period", [0, 1, 2, 3])
def test_lookup_policy_on_a_block_major_result_equals_the_period_major_lookup(
    *, period: int, column: str
) -> None:
    """A policy lookup reads a block-major result's values like a period-major one."""
    got = _lookup_column(schedule=_BLOCK_MAJOR, period=period, column=column)
    want = _lookup_column(schedule=_PERIOD_MAJOR, period=period, column=column)
    assert _leaf_bytes(got) == _leaf_bytes(want)


@pytest.mark.parametrize("codes", _POPULATIONS)
def test_combined_simulation_matches_the_reference_routes(
    codes: tuple[int, ...],
) -> None:
    """Solving and simulating one code at a time gives the period-major panel.

    A code no subject holds is still solved, so the published solution is
    complete and holds the period-major values.
    """
    params = life_cycle._params(typed_dead=True)
    initial = life_cycle._initial(codes=codes)

    got = _simulate(
        model=_life_cycle_model(schedule=_BLOCK_MAJOR),
        params=params,
        initial=initial,
        solution=None,
    )

    references = _reference_panels(params=params, initial=initial, combined=True)
    _assert_panels_match_the_references(
        got=got,
        references=references,
    )
    assert isinstance(got.solution, SolutionResult)
    want_solution = references["period_major"].solution
    assert isinstance(want_solution, SolutionResult)
    _assert_value_bytes_equal(got=got.solution.values, want=want_solution.values)


def test_a_saved_block_major_solution_loads_and_simulates_identically(
    tmp_path: Path,
) -> None:
    """The archive round trip keeps every value and the simulated panel."""
    params = life_cycle._params(typed_dead=True)
    initial = life_cycle._initial()
    model = _life_cycle_model(schedule=_BLOCK_MAJOR)
    reference = _life_cycle_model(schedule=_PERIOD_MAJOR)
    want_solution = _solution(model=reference, params=params)

    path = _solution(model=model, params=params).save(path=tmp_path / "solution")
    loaded = load_solution(path=path)
    assert isinstance(loaded, SolutionResult)

    _assert_value_bytes_equal(
        got=loaded.values, want=want_solution.values, ordered=False
    )
    want = _simulate(
        model=reference, params=params, initial=initial, solution=want_solution
    )
    for consumer in (model, reference):
        life_cycle._assert_panels_identical(
            got=_simulate(
                model=consumer, params=params, initial=initial, solution=loaded
            ),
            want=want,
        )


def test_a_block_major_simulation_result_saves_and_loads_its_values(
    tmp_path: Path,
) -> None:
    """A combined result persists its complete values and its panel."""
    params = life_cycle._params(typed_dead=True)
    initial = life_cycle._initial()
    got = _simulate(
        model=_life_cycle_model(schedule=_BLOCK_MAJOR),
        params=params,
        initial=initial,
        solution=None,
    )
    want = _simulate(
        model=_life_cycle_model(schedule=_PERIOD_MAJOR),
        params=params,
        initial=initial,
        solution=None,
    )
    want_frame = want.to_dataframe()

    got.save(directory=tmp_path / "result")
    loaded = SimulationResult.load(directory=tmp_path / "result")

    _assert_value_bytes_equal(
        got=loaded.period_to_regime_to_V_arr,
        want=want.period_to_regime_to_V_arr,
        ordered=False,
    )
    assert loaded.to_dataframe().equals(want_frame)


def _record_component_blocks(
    *, monkeypatch: pytest.MonkeyPatch, fail_on_code: int | None = None
) -> list[tuple[int, jax.Array]]:
    """Record the device blocks of every retained component, optionally failing."""
    recorded: list[tuple[int, jax.Array]] = []
    retain = block_major.RetainedComponentValues.retain

    # keyword-only-exempt: library-callback=pytest.MonkeyPatch.setattr
    def recording(self: Any, *, code: int, blocks: Mapping, **kwargs: Any) -> None:
        recorded.extend(
            (code, block) for regimes in blocks.values() for block in regimes.values()
        )
        if code == fail_on_code:
            msg = "injected component failure"
            raise RuntimeError(msg)
        retain(self, code=code, blocks=blocks, **kwargs)

    monkeypatch.setattr(block_major.RetainedComponentValues, "retain", recording)
    return recorded


def test_every_component_releases_its_device_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once retained on the host, each code's device buffers are deleted."""
    model, params = _workload(name="independent_types", schedule=_BLOCK_MAJOR)
    recorded = _record_component_blocks(monkeypatch=monkeypatch)

    _solution(model=model, params=params)

    assert {code for code, _ in recorded} == {0, 1, 2}
    assert all(block.is_deleted() for _, block in recorded)


def test_a_failing_component_publishes_nothing_and_releases_its_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure in the second code raises, and that code's buffers are deleted."""
    model, params = _workload(name="independent_types", schedule=_BLOCK_MAJOR)
    recorded = _record_component_blocks(monkeypatch=monkeypatch, fail_on_code=1)

    with pytest.raises(RuntimeError, match="injected component failure"):
        _solution(model=model, params=params)

    assert {code for code, _ in recorded} == {0, 1}
    assert all(block.is_deleted() for _, block in recorded)


def test_a_block_major_result_outlives_its_model_and_frees_its_blocks() -> None:
    """The result reads its values after the model is gone, and owns them alone."""
    model, params = _workload(name="independent_types", schedule=_BLOCK_MAJOR)
    reference, _ = _workload(name="independent_types", schedule=None)
    want = _solution(model=reference, params=params).values
    solution = _solution(model=model, params=params)
    owner = weakref.ref(_retained(solution))
    del model
    gc.collect()

    _assert_value_bytes_equal(got=solution.values, want=want)

    del solution
    gc.collect()
    assert owner() is None


def test_the_retention_record_accounts_every_block() -> None:
    """Host bytes, device-to-host bytes and full-value bytes are explicit."""
    model, params = _workload(name="independent_types", schedule=_BLOCK_MAJOR)
    solution = _solution(model=model, params=params)
    value_bytes = sum(
        np.asarray(solution.values[period][regime]).nbytes
        for period in solution.values
        for regime in solution.values[period]
    )

    record = _retained(solution).retention_record()

    assert record.state_name == "pref_type"
    assert record.codes == (0, 1, 2)
    assert sum(record.host_bytes_by_code.values()) == value_bytes
    assert record.device_to_host_bytes == value_bytes
    assert record.host_to_device_bytes == 0


@categorical(ordered=False)
class _SixTypes:
    a: ScalarInt
    b: ScalarInt
    c: ScalarInt
    d: ScalarInt
    e: ScalarInt
    f: ScalarInt


@categorical(ordered=False)
class _LongRegimeId:
    working: ScalarInt
    terminal: ScalarInt


_LONG_AGES = AgeGrid(start=0, inclusive_stop=40, step="Y")
_LONG_WEALTH = LinSpacedGrid(start=0, stop=10, n_points=4000)


def _long_utility(
    *, consumption: ContinuousAction, pref_type: DiscreteState, weight: FloatND
) -> FloatND:
    return weight[pref_type] * jnp.log(1.0 + consumption)


def _long_bequest(*, wealth: ContinuousState, pref_type: DiscreteState) -> FloatND:
    return (1.0 + 0.1 * pref_type) * jnp.log(1.0 + wealth)


def _long_next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def _long_affordable(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> BoolND:
    return consumption <= wealth


def _long_lived_model(*, schedule: InvariantBlockSchedule, budget: int | None) -> Model:
    """Many periods and types, few choices: retained values dominate the memory."""
    last_age = _LONG_AGES.exact_values[-1]
    pref_type = DiscreteGrid(category_class=_SixTypes)
    working = Regime(
        states={"wealth": _LONG_WEALTH, "pref_type": pref_type},
        state_transitions={
            "wealth": _long_next_wealth,
            "pref_type": fixed_transition("pref_type"),
        },
        actions={"consumption": LinSpacedGrid(start=0, stop=1, n_points=2)},
        functions={"utility": _long_utility},
        constraints={"affordable": _long_affordable},
    )
    terminal = Regime(
        states={"wealth": _LONG_WEALTH, "pref_type": pref_type},
        functions={"utility": _long_bequest},
    )
    return Model(
        regimes={"working": working, "terminal": terminal},
        ages=_LONG_AGES,
        regime_id_class=_LongRegimeId,
        initial_nodes={0: "working"},
        edges={
            "working": {
                "working": AgeRange(exclusive_stop=last_age - 1),
                "terminal": last_age - 1,
            }
        },
        # Fixed widths lower the same programs with and without a budget, so
        # the budget only decides what is admitted.
        execution_config=_config(
            schedule=schedule,
            budget=budget,
            axis_widths={"cell": int(_LONG_WEALTH.n_points), "action_product": 2},
        ),
    )


_LONG_PARAMS = MappingProxyType(
    {
        "discount_factor": 0.95,
        "working": {"utility": {"weight": jnp.linspace(0.5, 1.5, 6)}},
    }
)


def _long_value_bytes() -> int:
    """Return the bytes of every value of the long-lived model at this precision."""
    item = jnp.zeros((), dtype=float).dtype.itemsize
    n_periods = len(_LONG_AGES.exact_values)
    # `working` is solved in every period but the last, `terminal` in the last.
    return n_periods * 6 * int(_LONG_WEALTH.n_points) * item


def test_block_major_solves_under_a_budget_the_period_major_schedule_refuses() -> None:
    """Holding one code's lifetime at a time admits a solve whose values do not fit.

    The budget is half of all retained values: the period-major schedule keeps
    every code's values on the device and is refused, the block-major schedule
    keeps one code's at a time and publishes the unbudgeted values.
    """
    budget = _long_value_bytes() // 2
    reference = _long_lived_model(schedule=_BLOCK_MAJOR, budget=None)
    want = _solution(model=reference, params=_LONG_PARAMS).values

    with pytest.raises(ExecutionPlanningError, match="bytes resident"):
        _solution(
            model=_long_lived_model(schedule=_PERIOD_MAJOR, budget=budget),
            params=_LONG_PARAMS,
        )
    got = _solution(
        model=_long_lived_model(schedule=_BLOCK_MAJOR, budget=budget),
        params=_LONG_PARAMS,
    ).values

    _assert_value_bytes_equal(got=got, want=want)


def test_materializing_every_value_above_the_budget_is_refused_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full materialization that cannot fit the device budget is refused.

    The refusal names the bytes and the budget, assembles nothing, and leaves
    one-at-a-time reads available.
    """
    budget = _long_value_bytes() // 2
    model = _long_lived_model(schedule=_BLOCK_MAJOR, budget=budget)
    solution = _solution(model=model, params=_LONG_PARAMS)
    assemblies = _Assemblies(monkeypatch=monkeypatch)

    with pytest.raises(ExecutionPlanningError, match=str(budget)):
        _store(solution).materialize()
    assert assemblies.calls == []

    value = solution.value(period=0, regime="working")
    assert value.shape == (6, int(_LONG_WEALTH.n_points))
    assert assemblies.calls == [(0, "working")]


def test_budgeted_block_major_simulation_is_refused_with_a_remedy() -> None:
    """Chunk admission against per-code residency is not served; the call says so."""
    model = _life_cycle_model(schedule=_BLOCK_MAJOR, budget=2**30)
    params = life_cycle._params(typed_dead=True)

    with pytest.raises(ExecutionPlanningError, match="device_memory_bytes=None"):
        _simulate(
            model=model, params=params, initial=life_cycle._initial(), solution=None
        )


def _simulate_another_models_result() -> None:
    """Simulate a block-major result with a budgeted model that did not solve it."""
    params = life_cycle._params(typed_dead=True)
    producer = _life_cycle_model(schedule=_BLOCK_MAJOR)
    consumer = _life_cycle_model(schedule=_BLOCK_MAJOR, budget=2**30)
    _simulate(
        model=consumer,
        params=params,
        initial=life_cycle._initial(),
        solution=_solution(model=producer, params=params),
    )


def test_budgeted_simulation_of_another_models_block_major_result_names_the_model() -> (
    None
):
    """A budgeted model refuses another instance's block-major result by its id."""
    with pytest.raises(
        InvalidSimulationInputError,
        match="model_instance_id does not match this Model",
    ):
        _simulate_another_models_result()


def test_refusing_another_models_block_major_result_assembles_no_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal of another instance's result reads none of its values."""
    assemblies = _Assemblies(monkeypatch=monkeypatch)

    with pytest.raises(InvalidSimulationInputError):
        _simulate_another_models_result()

    assert assemblies.calls == []


def test_block_major_simulation_without_grouping_is_refused() -> None:
    """A simulate phase that may change the type cannot read one code at a time."""
    pref_law = Phased(
        solve=fixed_transition("pref_type"), simulate=life_cycle._reset_pref_type
    )
    model = _life_cycle_model(schedule=_BLOCK_MAJOR, pref_law=pref_law)
    params = life_cycle._params(typed_dead=True)
    solution = _solution(model=model, params=params)

    with pytest.raises(ExecutionPlanningError, match="group"):
        _simulate(
            model=model,
            params=params,
            initial=life_cycle._initial(),
            solution=solution,
        )


def _count_compiles(
    *, monkeypatch: pytest.MonkeyPatch, model: Model, params: Mapping
) -> int:
    calls: list[object] = []
    compile_and_log = backward_induction._compile_and_log

    def counted(**kwargs: Any) -> object:
        calls.append(kwargs["lowering_key"])
        return compile_and_log(**kwargs)

    monkeypatch.setattr(backward_induction, "_compile_and_log", counted)
    _solution(model=model, params=params)
    monkeypatch.undo()
    return len(calls)


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_block_major_compiles_no_program_per_code(
    *, monkeypatch: pytest.MonkeyPatch, workload: str
) -> None:
    """Every code reuses the programs the first code compiled."""
    model, params = _workload(name=workload, schedule=_BLOCK_MAJOR)
    reference, _ = _workload(name=workload, schedule=_PERIOD_MAJOR)

    assert _count_compiles(
        monkeypatch=monkeypatch, model=model, params=params
    ) == _count_compiles(monkeypatch=monkeypatch, model=reference, params=params)


def test_component_programs_keep_the_original_type_code() -> None:
    """A component's bound program evaluates its original code at position zero."""
    model, _ = _workload(name="independent_types", schedule=_BLOCK_MAJOR)
    component = block_major.InvariantComponent(state_name="pref_type", start=2, code=2)

    regimes = block_major.component_regimes(regimes=model._regimes, component=component)
    (program,) = core_program_graph(
        kernel=regimes["working"].solution.period_kernels[0]
    ).values()

    assert program.name == "main[pref_type=2]"
    binding = program.invariant_binding
    assert binding is not None
    assert (binding.state_name, binding.start, binding.code, binding.family) == (
        "pref_type",
        0,
        2,
        "main",
    )
    assert regimes["working"].solution._base_state_action_space.states[
        "pref_type"
    ].tolist() == [2]


def _coverage() -> block_major.ComponentCoverage:
    return block_major.ComponentCoverage(
        state_name="pref_type",
        codes=(0, 1, 2),
        entries=frozenset({(0, "working"), (1, "terminal")}),
    )


def test_coverage_is_complete_only_with_every_code() -> None:
    """A manifest missing a code refuses to publish, naming the missing code."""
    coverage = _coverage().with_code(
        code=0, entries=frozenset({(0, "working"), (1, "terminal")})
    )
    coverage = coverage.with_code(
        code=2, entries=frozenset({(0, "working"), (1, "terminal")})
    )

    with pytest.raises(ExecutionPlanningError, match=r"\(1,\)"):
        coverage.fail_if_incomplete()


def test_coverage_refuses_a_repeated_code() -> None:
    """A code is covered at most once."""
    coverage = _coverage().with_code(
        code=1, entries=frozenset({(0, "working"), (1, "terminal")})
    )

    with pytest.raises(ExecutionPlanningError, match="twice"):
        coverage.with_code(code=1, entries=frozenset({(0, "working"), (1, "terminal")}))


def test_coverage_refuses_a_component_missing_an_entry() -> None:
    """Each code must cover exactly the solved domain."""
    with pytest.raises(ExecutionPlanningError, match="terminal"):
        _coverage().with_code(code=0, entries=frozenset({(0, "working")}))


def test_a_block_major_solve_logs_its_retention_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At debug level the solve reports what it retained and moved."""
    model, params = _workload(name="independent_types", schedule=_BLOCK_MAJOR)

    with caplog.at_level(logging.DEBUG, logger="lcm"):
        model.solve(params=params, log_level="debug")

    records = [
        getattr(record, "component_retention_record", None)
        for record in caplog.records
        if hasattr(record, "component_retention_record")
    ]
    assert len(records) == 1
    assert isinstance(records[0], block_major.ComponentRetentionRecord)
    assert records[0].codes == (0, 1, 2)
