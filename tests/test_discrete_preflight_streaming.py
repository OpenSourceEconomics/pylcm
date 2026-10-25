"""Complete deterministic-law validation uses bounded scratch and exact witnesses."""

import gc
import weakref
from collections.abc import Callable, Mapping
from itertools import product

import jax
import jax.numpy as jnp
import pytest

import _lcm.simulation.initial_conditions as initial_checks
import _lcm.transition_checks as checks
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.typing import EconFunctionArg, FlatParams, PytreeValue
from _lcm.utils.logging import get_call_logger
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    JointTransition,
    LinSpacedGrid,
    Model,
    Regime,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError, InvalidStateTransitionError
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteState,
    FloatND,
    IntND,
    ScalarInt,
)


def _literal_law(
    *, x: IntND, y: IntND, age: IntND, period: ScalarInt
) -> dict[str, IntND]:
    return {"next_kind": jnp.where((x >= 1) & (y == 2), age + period, 0)}


def _literal_feasible(*, x: IntND, y: IntND, z: IntND) -> BoolND:
    return x + y <= z


@pytest.mark.parametrize("width", [1, 3, 4])
@pytest.mark.parametrize("z", [(0, 1), (3, 4)])
def test_compact_witness_matches_literal_complete_ij_reference(
    *, width: int, z: tuple[int, int]
) -> None:
    """Late blocks and feasibility-only axes retain the original first point."""
    grids = {
        "x": jnp.arange(3, dtype=jnp.int32),
        "y": jnp.arange(4, dtype=jnp.int32),
        "z": jnp.asarray(z, dtype=jnp.int32),
    }
    kwargs = {"age": jnp.int32(1), "period": jnp.int32(1)}
    law = checks._NamedArgumentsCall(
        func=_literal_law, names=frozenset({"x", "y", "age", "period"})
    )
    feasible = checks._NamedArgumentsCall(
        func=_literal_feasible, names=frozenset({"x", "y", "z"})
    )
    run = jax.jit(
        lambda grid_args, scalar_kwargs: checks._discrete_off_grid_search(
            grid_args=grid_args,
            scalar_kwargs=scalar_kwargs,
            law=law,
            feasibility=feasible,
            name="next_kind",
            grid_names=tuple(grids),
            target_size=2,
            block_rows=width,
        )
    )
    found, point = run(grids, kwargs)
    expected = next(
        (
            row
            for row in product(range(3), range(4), z)
            if row[0] >= 1 and row[1] == 2 and row[0] + row[1] <= row[2]
        ),
        None,
    )
    assert bool(found) is (expected is not None)
    assert all(value.shape == () for value in point.values())
    if expected is not None:
        assert tuple(int(point[name]) for name in grids) == expected
    changed, _ = run(grids, {"age": jnp.int32(0), "period": jnp.int32(0)})
    assert not bool(changed)


def _constant_bad() -> dict[str, ScalarInt]:
    return {"next_kind": jnp.int32(2)}


def test_scalar_law_and_empty_declared_axis() -> None:
    """An empty product is not a sampled row and scalar laws keep empty points."""
    law = checks._NamedArgumentsCall(func=_constant_bad, names=frozenset())
    feasible = checks._NamedArgumentsCall(
        func=checks._every_cell_feasible, names=frozenset()
    )
    found, point = checks._discrete_off_grid_search(
        grid_args={},
        grid_names=(),
        scalar_kwargs={},
        law=law,
        feasibility=feasible,
        name="next_kind",
        target_size=2,
        block_rows=2,
    )
    assert bool(found)
    assert point == {}
    found, point = checks._discrete_off_grid_search(
        grid_args={"x": jnp.empty((0,), dtype=jnp.int32)},
        grid_names=("x",),
        scalar_kwargs={},
        law=law,
        feasibility=feasible,
        name="next_kind",
        target_size=2,
        block_rows=2,
    )
    assert not bool(found)
    assert point["x"].shape == ()


def _irrelevant_last(
    *, irrelevant: DiscreteState, wealth: ContinuousState
) -> DiscreteState:
    return jnp.where(wealth >= 3.0, 2, irrelevant).astype(jnp.int32)


def test_law_order_precedes_global_first_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The earlier named law owns the error even when another law fails earlier."""
    monkeypatch.setattr(checks, "_OFF_GRID_BLOCK_ROWS", 2)
    monkeypatch.setattr(checks, "_OFF_GRID_MAX_BLOCK_ROWS", 2)
    with pytest.raises(InvalidStateTransitionError) as error:
        _streaming_model(invalid=True, priority=True).validate_transitions(
            params={"discount_factor": 1.0}
        )
    assert "the law 'next_irrelevant'" in str(error.value)
    assert "{'irrelevant': 'low', 'kind': 'low', 'wealth': 3.0}" in str(error.value)


@pytest.mark.parametrize("width", [0, 1 << 31])
def test_unaddressable_local_offsets_refuse(*, width: int) -> None:
    with pytest.raises(ExecutionPlanningError, match="int32 block offsets"):
        checks._discrete_off_grid_search(
            grid_args={},
            scalar_kwargs={},
            law=_constant_bad,
            feasibility=checks._every_cell_feasible,
            name="next_kind",
            grid_names=(),
            target_size=2,
            block_rows=width,
        )


@categorical(ordered=True)
class _Two:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt


def _flow(*, wealth: ContinuousState, kind: DiscreteState) -> FloatND:
    return wealth + 1.0 * kind


def _terminal(*, kind: DiscreteState, irrelevant: DiscreteState) -> FloatND:
    return 1.0 * kind + 2.0 * irrelevant


def _late_bad(*, kind: DiscreteState, wealth: ContinuousState) -> DiscreteState:
    return jnp.where(wealth >= 1.0, 2, kind).astype(jnp.int32)


def _only_late(*, wealth: ContinuousState) -> BoolND:
    return wealth >= 2.0


def _zero_last_mass() -> FloatND:
    return jnp.asarray([1.0, 0.0])


def _kind_from_draw(*, draw: FloatND) -> DiscreteState:
    return jnp.where(draw[0] == 1, 2, 0).astype(jnp.int32)


def _streaming_model(
    *,
    invalid: bool,
    budget: int | None = None,
    joint: bool = False,
    priority: bool = False,
    infeasible_first: bool = False,
) -> Model:
    grid = DiscreteGrid(category_class=_Two)
    return Model(
        regimes={
            "source": Regime(
                states={
                    "irrelevant": grid,
                    "kind": grid,
                    "wealth": LinSpacedGrid(start=0.0, stop=3.0, n_points=4),
                },
                functions={"utility": _flow},
                constraints={"only_late": _only_late},
                state_transitions={
                    "irrelevant": _irrelevant_infeasible
                    if infeasible_first
                    else _irrelevant_last
                    if priority
                    else fixed_transition("irrelevant"),
                    **(
                        {}
                        if joint
                        else {
                            "kind": _late_bad if invalid else fixed_transition("kind")
                        }
                    ),
                    "wealth": fixed_transition("wealth"),
                },
                joint_transitions=(
                    {
                        "target": {
                            "draw": JointTransition(
                                support_size=2,
                                support=jnp.asarray([[0], [1]], dtype=jnp.int32),
                                probabilities=_zero_last_mass,
                                outputs={"kind": _kind_from_draw},
                            )
                        }
                    }
                    if joint
                    else {}
                ),
            ),
            "target": Regime(
                states={"irrelevant": grid, "kind": grid},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"source": Transition(law="target")},
        initial_nodes={0: "source"},
        regime_id_class=_RegimeId,
        enable_jit=True,
        execution_config=ExecutionConfig(device_memory_bytes=budget),
    )


def test_public_validator_restores_irrelevant_axes_and_first_feasible_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Splitting identity laws preserves the original diagnostic schema."""
    monkeypatch.setattr(checks, "_OFF_GRID_BLOCK_ROWS", 2)
    monkeypatch.setattr(checks, "_OFF_GRID_MAX_BLOCK_ROWS", 2)
    with pytest.raises(InvalidStateTransitionError) as error:
        _streaming_model(invalid=True).validate_transitions(
            params={"discount_factor": 1.0}
        )
    assert "the law 'next_kind'" in str(error.value)
    assert "{'irrelevant': 'low', 'kind': 'low', 'wealth': 2.0}" in str(error.value)


@pytest.mark.parametrize("invalid", [False, True])
def test_public_enabled_and_serial_paths_keep_the_actual_memory_owner(
    *,
    monkeypatch: pytest.MonkeyPatch,
    invalid: bool,
) -> None:
    """Real compact admission sees the same caller owner on invalid serial retry."""
    model = _streaming_model(invalid=invalid, budget=128 * 1024 * 1024)
    owners: list[SimulationMemory] = []
    admitted: list[SimulationMemory] = []
    original_owner = Model._transition_memory
    original_admit = checks._evaluate_admitted_transition_producer

    # keyword-only-exempt: primary-argument=self
    def observe_owner(
        self: Model,
        *,
        flat_params: FlatParams,
    ) -> SimulationMemory | None:
        owner = original_owner(self, flat_params=flat_params)
        assert owner is not None
        owners.append(owner)
        return owner

    def observe_admit(
        *,
        function: Callable[..., PytreeValue],
        arguments: Mapping[str, PytreeValue],
        memory: SimulationMemory,
    ) -> PytreeValue:
        admitted.append(memory)
        result = original_admit(function=function, arguments=arguments, memory=memory)
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
        ):
            assert all(value.shape == () for value in jax.tree.leaves(result))
        return result

    monkeypatch.setattr(Model, "_transition_memory", observe_owner)
    monkeypatch.setattr(checks, "_evaluate_admitted_transition_producer", observe_admit)
    monkeypatch.setattr(checks, "_OFF_GRID_BLOCK_ROWS", 2)
    monkeypatch.setattr(checks, "_OFF_GRID_MAX_BLOCK_ROWS", 2)
    if invalid:
        with pytest.raises(InvalidStateTransitionError, match="next_kind"):
            model.validate_transitions(params={"discount_factor": 1.0})
    else:
        model.validate_transitions(params={"discount_factor": 1.0})
    assert owners
    assert admitted
    assert all(owner is owners[0] for owner in admitted)


def test_zero_mass_declared_support_still_checks_its_raw_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared zero-mass event cannot hide an off-grid deterministic output."""
    monkeypatch.setattr(checks, "_OFF_GRID_BLOCK_ROWS", 1)
    monkeypatch.setattr(checks, "_OFF_GRID_MAX_BLOCK_ROWS", 1)
    with pytest.raises(InvalidStateTransitionError, match="next_kind"):
        _streaming_model(invalid=False, joint=True).validate_transitions(
            params={"discount_factor": 1.0}
        )


def test_public_tight_budget_refuses_before_large_row_dispatch() -> None:
    model = _streaming_model(invalid=False, budget=1)
    with pytest.raises(ExecutionPlanningError):
        model.validate_transitions(params={"discount_factor": 1.0})


def test_public_compact_width_retry_precedes_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused wide profile cannot execute; the admitted narrow route is real."""
    original_build = checks._build_discrete_off_grid_search
    original_admit = checks._evaluate_admitted_transition_producer
    widths_by_function: dict[int, int] = {}
    refused: list[int] = []
    dispatched: list[int] = []

    def candidates(*, n_rows: int) -> tuple[int, ...]:
        del n_rows
        return (4, 2)

    def observe_build(
        *,
        law: Callable[..., Mapping[str, FloatND | IntND]],
        feasibility: Callable[..., BoolND],
        name: str,
        grid_names: tuple[str, ...],
        limits: tuple[int, int],
    ) -> Callable[..., PytreeValue]:
        function = original_build(
            law=law,
            feasibility=feasibility,
            name=name,
            grid_names=grid_names,
            limits=limits,
        )
        widths_by_function[id(function)] = limits[1]
        return function

    def observe_admit(
        *,
        function: Callable[..., PytreeValue],
        arguments: Mapping[str, PytreeValue],
        memory: SimulationMemory,
    ) -> PytreeValue:
        width = widths_by_function.get(id(function))
        if width == 4:
            refused.append(width)
            raise ExecutionPlanningError("Controlled pre-dispatch width refusal")
        if width is not None:
            dispatched.append(width)
        return original_admit(function=function, arguments=arguments, memory=memory)

    monkeypatch.setattr(checks, "_off_grid_block_candidates", candidates)
    monkeypatch.setattr(checks, "_build_discrete_off_grid_search", observe_build)
    monkeypatch.setattr(checks, "_evaluate_admitted_transition_producer", observe_admit)
    _streaming_model(invalid=False, budget=128 * 1024 * 1024).validate_transitions(
        params={"discount_factor": 1.0}
    )
    assert refused
    assert dispatched
    assert set(dispatched) == {2}


def _irrelevant_infeasible(
    *, irrelevant: DiscreteState, wealth: ContinuousState
) -> DiscreteState:
    return jnp.where(wealth < 2.0, 2, irrelevant).astype(jnp.int32)


def test_infeasible_wider_witness_is_released_before_next_law_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The earlier law's false wider witness must leave both ledger and Python."""
    original = checks._find_discrete_law_off_grid
    references: list[weakref.ReferenceType[jax.Array]] = []
    checked: list[str] = []

    def observe(
        *,
        law: Callable[..., Mapping[str, FloatND | IntND]],
        feasibility: Callable[..., BoolND],
        name: str,
        target_size: int,
        grids: Mapping[str, FloatND | IntND],
        scalar_kwargs: Mapping[str, EconFunctionArg],
        memory: SimulationMemory | None,
        owner: ProfiledSimulationOperations | None,
    ) -> tuple[BoolND, Mapping[str, FloatND | IntND]]:
        if name == "next_kind" and references:
            gc.collect()
            assert all(reference() is None for reference in references)
            assert memory is not None
            assert memory.derived == ()
            checked.append(name)
        result = original(
            law=law,
            feasibility=feasibility,
            name=name,
            target_size=target_size,
            grids=grids,
            scalar_kwargs=scalar_kwargs,
            memory=memory,
            owner=owner,
        )
        if name == "next_irrelevant" and feasibility is not checks._every_cell_feasible:
            assert not bool(result[0])
            references.extend(weakref.ref(leaf) for leaf in jax.tree.leaves(result))
        return result

    monkeypatch.setattr(checks, "_find_discrete_law_off_grid", observe)
    _streaming_model(
        invalid=False, infeasible_first=True, budget=128 * 1024 * 1024
    ).validate_transitions(params={"discount_factor": 1.0})
    assert references
    assert checked


def test_unbudgeted_simulation_serial_retry_keeps_model_producer_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real simulation-input validation retries the invalid law in the same cache."""
    model = _streaming_model(invalid=True, budget=None)
    owner = model._simulate_entry_operations
    original = checks._find_discrete_law_off_grid
    observed: list[ProfiledSimulationOperations] = []

    def observe(
        *,
        law: Callable[..., Mapping[str, FloatND | IntND]],
        feasibility: Callable[..., BoolND],
        name: str,
        target_size: int,
        grids: Mapping[str, FloatND | IntND],
        scalar_kwargs: Mapping[str, EconFunctionArg],
        memory: SimulationMemory | None,
        owner: ProfiledSimulationOperations | None,
    ) -> tuple[BoolND, Mapping[str, FloatND | IntND]]:
        assert memory is None
        assert owner is model._simulate_entry_operations
        observed.append(owner)
        return original(
            law=law,
            feasibility=feasibility,
            name=name,
            target_size=target_size,
            grids=grids,
            scalar_kwargs=scalar_kwargs,
            memory=memory,
            owner=owner,
        )

    monkeypatch.setattr(checks, "_find_discrete_law_off_grid", observe)
    with pytest.raises(InvalidStateTransitionError, match="next_kind"):
        initial_checks.validate_simulation_inputs(
            initial_conditions={
                "age": jnp.asarray([0.0]),
                "regime_id": jnp.asarray(
                    [model.regime_names_to_ids["source"]], dtype=jnp.int32
                ),
                "irrelevant": jnp.asarray([0], dtype=jnp.int32),
                "kind": jnp.asarray([0], dtype=jnp.int32),
                "wealth": jnp.asarray([2.0]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=model._process_params(params={"discount_factor": 1.0}),
            ages=model._time,
            logger=get_call_logger(log_level="off", runtime_checks=True),
            execution=model._execution,
            producers=owner,
        )
    assert observed
    assert all(value is owner for value in observed)
