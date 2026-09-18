"""AC4: an unbudgeted exact-signature repeat reuses its static preparation.

With no device-memory budget configured, `SimulationRuntime.dispatch` splits a
cached, immutable preparation record -- the subject-extent descriptor, the
declared argument positions, the selected width map and the reusable call
structure, keyed by the complete abstract signature -- from a per-call binder
that rebuilds and re-places this call's own leaves every time.

The tests below instrument the operations themselves at their real imported
call sites, so an exact repeat is shown to reconstruct no static descriptor and
to run no width frontier, while a same-shaped new array is still bound as a
fresh value. Every signature change -- population, shape, dtype, weak type,
optional argument columns, period, declared program, typed static values,
PRNG configuration -- falls back to the full validated route.

Fields that are immutable for the life of one `SimulationRuntime` (the resolved
execution config with its explicit widths and device ids, `enable_jit` and the
ordered `subject_devices`) are deliberately absent from the record key: the
records live on the runtime that owns them, so a different configuration is a
different runtime with an empty cache. `test_a_new_runtime_shares_no_prepared_route`
enforces exactly that boundary.
"""

import contextlib
import dataclasses
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution import workspace_planning
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.simulation import chunk_admission
from _lcm.simulation import runtime as runtime_module
from _lcm.simulation.entry_inputs import capture_simulation_entry_inputs
from _lcm.simulation.program_types import subject_axis
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import (
    SimulationRuntime,
    _SimulationCandidateCompiler,
)
from benchmarks.asv._simulation_witnesses import multi_regime
from lcm.execution import ExecutionConfig
from lcm.typing import FloatND

# Operations whose repetition AC4 is about, named at the module attribute each
# caller actually reads, so a rename or a re-import cannot silently stop
# counting. `static_descriptor` and `subject_extent` are the static program
# construction, `width_frontier` the actual frontier function, `planner` the
# workspace preparation, `memory_report` the backend memory request, and
# `placement` the per-call binding whose *presence* the repeat must still show.
_INSTRUMENTED_SITES = (
    ("static_descriptor", runtime_module, "materialize_core_program"),
    ("subject_extent", runtime_module, "_with_subject_extent"),
    ("width_frontier", workspace_planning, "_workspace_width_candidates"),
    ("planner", runtime_module, "plan_workspace"),
    ("placement", runtime_module, "place_simulation_arguments"),
    ("memory_report", runtime_module, "compiler_memory_reservation"),
)

#: Counters that must be exactly zero on an exact-signature unbudgeted repeat.
_STATIC_SITES = (
    "static_descriptor",
    "subject_extent",
    "width_frontier",
    "planner",
    "memory_report",
    "lowering",
)


def _install_counters(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count each instrumented operation at its own imported call site."""
    counts = dict.fromkeys([name for name, _, _ in _INSTRUMENTED_SITES], 0)
    counts["lowering"] = 0
    for name, module, attribute in _INSTRUMENTED_SITES:
        original = getattr(module, attribute)

        def counted(
            *args: Any, _name: str = name, _original: Any = original, **kwargs: Any
        ) -> Any:
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, attribute, counted)

    original_call = _SimulationCandidateCompiler.__call__

    def counted_call(*args: Any, **kwargs: Any) -> Any:
        counts["lowering"] += 1
        return original_call(*args, **kwargs)

    monkeypatch.setattr(_SimulationCandidateCompiler, "__call__", counted_call)
    return counts


def _reset(counts: dict[str, int]) -> None:
    """Start a fresh measurement window over the same installed counters."""
    for name in counts:
        counts[name] = 0


def _small_unbudgeted_model():
    """Build a small multi-regime witness with no device-memory budget."""
    return multi_regime(execution_config=ExecutionConfig())


def _simulate(*, model, params, solution, initial_conditions, seed):
    """Run one public unbudgeted simulation."""
    return model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
        seed=seed,
    )


def _wealth(result) -> np.ndarray:
    """Read the published wealth column of one simulation result."""
    return result.to_dataframe()["wealth"].to_numpy()


def _increment_subject(*, state: FloatND) -> FloatND:
    """Advance the independently observed state by one unit."""
    return state + 1


def _unit_program(*, name: str = "simulate_transition") -> CoreProgram:
    """Declare one subject-valued program whose operand is a scalar per subject."""
    return CoreProgram(
        name=name,
        function=_SubjectTiled(func=_increment_subject, subject_arg_names=("state",)),
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name=name, subject_arg_names=("state",)
        ),
        requirements=CoreExecutionRequirements(
            tiled_axes=(subject_axis(state_names=("state",)),)
        ),
        output_roles="next_state",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


def _unit_runtime(*, width: int | None = None) -> SimulationRuntime:
    """Build an unbudgeted executor with an optional pinned subject tile."""
    return SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(0,),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({} if width is None else {"subject": width}),
            device_memory_bytes=None,
        ),
        enable_jit=True,
        subject_devices=(jax.devices()[0],),
    )


_UNIT_SUBJECTS = 64


def _unit_arguments(**overrides: object) -> dict[str, object]:
    """Bind one call's complete operands for the unit-level witness program."""
    arguments: dict[str, object] = {
        "state": jnp.arange(_UNIT_SUBJECTS, dtype=jnp.result_type(float))
    }
    arguments.update(overrides)
    return arguments


def test_unbudgeted_path_admits_no_residency_or_chunk_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unbudgeted simulation takes no chunk admission and no residency measure."""
    model, params, initial_conditions = _small_unbudgeted_model()
    solution = model.solve(params=params, log_level="off")

    entry_calls = []
    real_entry = capture_simulation_entry_inputs

    def counting_entry(**kwargs: Any) -> Any:
        entry_calls.append(kwargs)
        return real_entry(**kwargs)

    chunk_calls = []
    real_chunks = chunk_admission.prepare_simulation_chunks

    def counting_chunks(**kwargs: Any) -> Any:
        chunk_calls.append(kwargs)
        return real_chunks(**kwargs)

    residency_calls = []
    real_footprint = measure_buffer_footprint

    def counting_footprint(**kwargs: Any) -> Any:
        residency_calls.append(kwargs)
        return real_footprint(**kwargs)

    monkeypatch.setattr(
        "_lcm.simulation.entry_inputs.capture_simulation_entry_inputs",
        counting_entry,
    )
    monkeypatch.setattr(chunk_admission, "prepare_simulation_chunks", counting_chunks)
    monkeypatch.setattr(
        "_lcm.simulation.residency.measure_buffer_footprint", counting_footprint
    )

    _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )

    assert len(chunk_calls) == 0
    assert len(residency_calls) == 0


def test_a_cold_unbudgeted_dispatch_builds_its_static_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: a first-seen signature does all of the static work."""
    program = _unit_program()
    runtime = _unit_runtime()
    counts = _install_counters(monkeypatch)

    runtime.dispatch(
        program=program,
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )

    assert counts["static_descriptor"] == 1
    assert counts["subject_extent"] == 1
    assert counts["width_frontier"] >= 1
    assert counts["planner"] == 1
    assert counts["placement"] == 1
    assert counts["lowering"] >= 1
    assert len(runtime.routes) == 1


def test_an_exact_signature_repeat_reconstructs_no_static_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeat of one exact signature rebuilds no descriptor and no frontier."""
    program = _unit_program()
    runtime = _unit_runtime()
    counts = _install_counters(monkeypatch)

    first = runtime.dispatch(
        program=program,
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )
    _reset(counts)

    repeated = runtime.dispatch(
        program=program,
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )

    assert {name: counts[name] for name in _STATIC_SITES} == dict.fromkeys(
        _STATIC_SITES, 0
    )
    # The per-call binder still ran: the leaves were built and placed afresh.
    assert counts["placement"] == 1
    np.testing.assert_array_equal(np.asarray(first), np.asarray(repeated))


def test_a_repeat_binds_this_call_s_own_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-shaped new array is a fresh value, never the previous object."""
    program = _unit_program()
    runtime = _unit_runtime()
    counts = _install_counters(monkeypatch)

    runtime.dispatch(
        program=program,
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )
    _reset(counts)

    shifted = jnp.arange(_UNIT_SUBJECTS, dtype=jnp.result_type(float)) + 100.0
    result = runtime.dispatch(
        program=program,
        arguments=_unit_arguments(state=shifted),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )

    assert {name: counts[name] for name in _STATIC_SITES} == dict.fromkeys(
        _STATIC_SITES, 0
    )
    assert counts["placement"] == 1
    np.testing.assert_array_equal(np.asarray(result), np.asarray(shifted + 1))


def _changed_dtype() -> dict[str, object]:
    """Same population and shape, a different floating-point dtype."""
    return _unit_arguments(state=jnp.arange(_UNIT_SUBJECTS, dtype=jnp.float16))


def _weak_typed() -> dict[str, object]:
    """Same shape and dtype as the working precision, but a weakly typed leaf."""
    return _unit_arguments(state=jnp.full(_UNIT_SUBJECTS, 1.0))


def _extra_column() -> dict[str, object]:
    """An additional optional argument column the previous call did not carry."""
    return _unit_arguments(
        spare=jnp.zeros(_UNIT_SUBJECTS, dtype=jnp.result_type(float))
    )


def _typed_static_value() -> dict[str, object]:
    """A typed static leaf whose value, not shape, changed."""
    return _unit_arguments(replay_address=1)


def _prng_key() -> dict[str, object]:
    """A leaf carrying an extended PRNG-key dtype."""
    return _unit_arguments(state=jax.random.key(0))


@pytest.mark.parametrize(
    "changed",
    [_changed_dtype, _weak_typed, _extra_column, _typed_static_value, _prng_key],
    ids=["dtype", "weak_type", "optional_column", "typed_static_value", "prng_config"],
)
def test_a_changed_signature_re_enters_the_validated_route(
    *, monkeypatch: pytest.MonkeyPatch, changed: Any
) -> None:
    """Any abstract change other than a value falls back, never reusing the record."""
    program = _unit_program()
    runtime = _unit_runtime()
    counts = _install_counters(monkeypatch)

    runtime.dispatch(
        program=program,
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )
    _reset(counts)

    # A refused signature may legitimately fail downstream; what must not happen
    # is a silent reuse of the previous record, which the counters below settle.
    with contextlib.suppress(Exception):
        runtime.dispatch(
            program=program,
            arguments=changed(),
            period=0,
            n_subjects=_UNIT_SUBJECTS,
        )

    assert counts["static_descriptor"] == 1
    assert counts["subject_extent"] == 1


@pytest.mark.parametrize(
    ("period", "n_subjects"),
    [(1, _UNIT_SUBJECTS), (0, _UNIT_SUBJECTS * 2)],
    ids=["period", "population"],
)
def test_a_changed_period_or_population_re_enters_the_validated_route(
    *, monkeypatch: pytest.MonkeyPatch, period: int, n_subjects: int
) -> None:
    """Period program and population extent are part of the record's identity."""
    program = _unit_program()
    runtime = _unit_runtime()
    counts = _install_counters(monkeypatch)

    runtime.dispatch(
        program=program,
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )
    _reset(counts)

    runtime.dispatch(
        program=program,
        arguments=_unit_arguments(
            state=jnp.arange(n_subjects, dtype=jnp.result_type(float)),
        ),
        period=period,
        n_subjects=n_subjects,
    )

    assert counts["static_descriptor"] == 1
    assert counts["subject_extent"] == 1
    assert len(runtime.routes) == 2


def test_a_rebuilt_period_program_cannot_inherit_a_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record is published for one declared program object, never for a twin."""
    runtime = _unit_runtime()
    counts = _install_counters(monkeypatch)

    runtime.dispatch(
        program=_unit_program(),
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )
    _reset(counts)

    runtime.dispatch(
        program=_unit_program(),
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )

    assert counts["static_descriptor"] == 1
    assert counts["subject_extent"] == 1


def test_a_new_runtime_shares_no_prepared_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Widths, devices and JIT are immutable per runtime, so records never cross.

    `axis_widths`, `device_ids`, `enable_jit` and `subject_devices` cannot change
    within one `SimulationRuntime`; a different choice is a different runtime,
    and the record cache is owned by, and dies with, that runtime.
    """
    program = _unit_program()
    counts = _install_counters(monkeypatch)

    default_width = _unit_runtime()
    default_width.dispatch(
        program=program,
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )
    _reset(counts)

    pinned_width = _unit_runtime(width=8)
    assert pinned_width.routes == {}
    pinned_width.dispatch(
        program=program,
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )

    assert counts["static_descriptor"] == 1
    assert counts["subject_extent"] == 1
    assert counts["width_frontier"] >= 1
    (route,) = pinned_width.routes.values()
    assert dict(route.widths) == {"subject": 8}


def test_a_record_holds_no_caller_array() -> None:
    """A published record retains abstract description and compiled code only."""
    program = _unit_program()
    runtime = _unit_runtime()
    runtime.dispatch(
        program=program,
        arguments=_unit_arguments(),
        period=0,
        n_subjects=_UNIT_SUBJECTS,
    )

    (route,) = runtime.routes.values()
    retained = [
        leaf
        for field in dataclasses.fields(route)
        for leaf in jax.tree.leaves(getattr(route, field.name))
        if isinstance(leaf, jax.Array)
    ]
    assert retained == []


def test_a_public_warm_simulation_reuses_its_static_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated public simulation plans nothing again and still binds fresh leaves."""
    model, params, initial_conditions = _small_unbudgeted_model()
    solution = model.solve(params=params, log_level="off")
    counts = _install_counters(monkeypatch)

    first = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )
    assert counts["planner"] > 0
    assert counts["static_descriptor"] > 0
    _reset(counts)

    shifted = dict(initial_conditions)
    shifted["wealth"] = initial_conditions["wealth"] + 0.5
    repeated = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=shifted,
        seed=0,
    )

    assert counts["static_descriptor"] == 0
    assert counts["subject_extent"] == 0
    assert counts["planner"] == 0
    assert counts["lowering"] == 0
    assert counts["memory_report"] == 0
    assert counts["placement"] > 0

    # Fresh leaves: a same-shaped value change changes the published numbers,
    # and an identical repeat reproduces them exactly.
    assert not np.allclose(_wealth(first), _wealth(repeated))
    again = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=shifted,
        seed=0,
    )
    np.testing.assert_array_equal(_wealth(repeated), _wealth(again))


def test_a_changed_seed_keeps_its_own_stream() -> None:
    """A different PRNG seed is a different draw, not a reused warm result."""
    model, params, initial_conditions = _small_unbudgeted_model()
    solution = model.solve(params=params, log_level="off")

    first = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )
    other = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=7,
    )
    replay = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )
    np.testing.assert_array_equal(_wealth(first), _wealth(replay))
    assert not np.array_equal(_wealth(first), _wealth(other))


def test_a_grown_population_falls_back_and_stays_correct() -> None:
    """A shape change cannot dispatch a stale executable."""
    model, params, initial_conditions = _small_unbudgeted_model()
    solution = model.solve(params=params, log_level="off")

    baseline = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )
    _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )

    grown_initial_conditions = {
        key: jnp.concatenate([value, value[:2]])
        for key, value in initial_conditions.items()
    }
    grown_result = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=grown_initial_conditions,
        seed=0,
    )

    fresh_model, fresh_params, _ = _small_unbudgeted_model()
    fresh_solution = fresh_model.solve(params=fresh_params, log_level="off")
    fresh_result = _simulate(
        model=fresh_model,
        params=fresh_params,
        solution=fresh_solution,
        initial_conditions=grown_initial_conditions,
        seed=0,
    )

    np.testing.assert_array_equal(_wealth(grown_result), _wealth(fresh_result))
    replay = _simulate(
        model=model,
        params=params,
        solution=solution,
        initial_conditions=initial_conditions,
        seed=0,
    )
    np.testing.assert_array_equal(_wealth(baseline), _wealth(replay))
