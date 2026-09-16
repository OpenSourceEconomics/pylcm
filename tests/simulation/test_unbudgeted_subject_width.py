"""The unbudgeted subject width derives a byte bound instead of a fixed 4096 tile.

Follow-on 1 of the width lever (`width-lever-design.md`): without a declared
budget `_dispatch_widths` still derives a bound -- a constant cap on the bytes
of the live subject block -- and takes the widest admissible subject width
under it, instead of pinning the model-blind, device-blind 4096 default.
"""

import dataclasses
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    materialize_core_program,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.simulation.program_types import SimulationBuildContext, subject_axis
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.simulation.residency import DeviceBufferFootprint, measure_buffer_footprint
from _lcm.simulation.runtime import (
    _DEFAULT_UNBUDGETED_SUBJECT_WIDTH,
    _MIN_SUBJECT_ARGUMENT_BYTES,
    _UNBUDGETED_SUBJECT_BLOCK_BYTES,
    SimulationDispatchContext,
    SimulationRuntime,
    _dispatch_widths,
    _with_subject_extent,
)
from lcm.execution import ExecutionConfig
from lcm.typing import FloatND
from lcm_examples.precautionary_savings import create_model, get_params

# Widest tile the byte cap admits for a program lighter than the standing
# per-subject weight, which is every program in the repository's fixtures.
_LIGHT_PROGRAM_WIDTH = _UNBUDGETED_SUBJECT_BLOCK_BYTES // _MIN_SUBJECT_ARGUMENT_BYTES


def _empty_footprint() -> DeviceBufferFootprint:
    """Report no live buffers; the budgeted branch never reads this one."""
    return measure_buffer_footprint(tree={})


def _increment_subject(*, state: FloatND) -> FloatND:
    """Advance the independently observed state by one unit."""
    return state + 1


def _program() -> CoreProgram:
    """Declare one subject-valued program whose operand is a scalar per subject."""
    return CoreProgram(
        name="simulate_transition",
        function=_SubjectTiled(func=_increment_subject, subject_arg_names=("state",)),
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name="simulate_transition", subject_arg_names=("state",)
        ),
        requirements=CoreExecutionRequirements(
            tiled_axes=(subject_axis(state_names=("state",)),)
        ),
        output_roles="next_state",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


def _materialized(*, n_subjects: int, program: CoreProgram | None = None):
    """Materialize one witness program against a population of live scalar states."""
    return materialize_core_program(
        program=_with_subject_extent(
            program=_program() if program is None else program, n_subjects=n_subjects
        ),
        context=SimulationBuildContext(
            state_action_space=None,
            next_regime_to_V_arr={},
            next_regime_to_continuation={},
            flat_params={},
            period=0,
            ages=None,
            call_arguments={"state": jnp.zeros(n_subjects, dtype=jnp.float32)},
        ),
    )


def _with_heavy_operand(*, n_subjects: int, per_subject_leaf: int):
    """Replace the live operand by a shape-only leaf too heavy to materialize."""
    return dataclasses.replace(
        _materialized(n_subjects=n_subjects),
        arguments=MappingProxyType(
            {"state": jax.ShapeDtypeStruct((n_subjects, per_subject_leaf), jnp.float32)}
        ),
    )


def _runtime(*, width: int | None = None, budget: int | None = None):
    """Build an executor with an optional pinned subject tile and budget."""
    return SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(0,),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({} if width is None else {"subject": width}),
            device_memory_bytes=budget,
        ),
        enable_jit=True,
        subject_devices=(jax.devices()[0],),
    )


def test_a_light_population_above_the_old_default_reaches_its_full_extent() -> None:
    """20,000 scalar subjects fit the byte cap, so the whole population is one tile."""
    widths = _dispatch_widths(
        program=_materialized(n_subjects=20_000),
        configured=MappingProxyType({}),
        residency=None,
    )
    assert dict(widths) == {"subject": 20_000}


def test_the_derived_width_never_exceeds_the_byte_cap() -> None:
    """A population far above the cap is tiled at the cap, not at its extent."""
    widths = _dispatch_widths(
        program=_materialized(n_subjects=4 * _LIGHT_PROGRAM_WIDTH),
        configured=MappingProxyType({}),
        residency=None,
    )
    assert dict(widths) == {"subject": _LIGHT_PROGRAM_WIDTH}


def test_a_heavy_per_subject_operand_never_narrows_below_the_old_default() -> None:
    """A per-subject slice larger than the whole cap still gets the historical floor."""
    widths = _dispatch_widths(
        program=_with_heavy_operand(
            n_subjects=20_000,
            per_subject_leaf=2 * _UNBUDGETED_SUBJECT_BLOCK_BYTES // 4,
        ),
        configured=MappingProxyType({}),
        residency=None,
    )
    assert dict(widths) == {"subject": _DEFAULT_UNBUDGETED_SUBJECT_WIDTH}


def test_an_explicit_width_still_wins() -> None:
    """A pinned `axis_widths['subject']` is authoritative, as before."""
    widths = _dispatch_widths(
        program=_materialized(n_subjects=20_000),
        configured=MappingProxyType({"subject": 512}),
        residency=None,
    )
    assert dict(widths) == {"subject": 512}


def test_a_budgeted_residency_context_still_owns_the_width() -> None:
    """The reserved outer-chunk width is untouched by the unbudgeted rule."""
    widths = _dispatch_widths(
        program=_materialized(n_subjects=20_000),
        configured=MappingProxyType({}),
        residency=SimulationDispatchContext(
            live_footprint=_empty_footprint,
            budget_devices=(jax.devices()[0],),
            axis_widths=MappingProxyType({"subject": 1024}),
        ),
    )
    assert dict(widths) == {"subject": 1024}


def test_dispatch_and_the_prepared_route_pin_the_same_width() -> None:
    """A published prepared route carries exactly the width dispatch selected."""
    core = _program()
    program = _materialized(n_subjects=9_000, program=core)
    runtime = _runtime()
    resolved = _dispatch_widths(
        program=program, configured=runtime.execution.axis_widths, residency=None
    )
    runtime.dispatch(
        program=core,
        arguments={"state": jnp.arange(9_000, dtype=jnp.float32)},
        period=0,
        n_subjects=9_000,
    )
    assert len(runtime.routes) == 1
    (route,) = runtime.routes.values()
    assert dict(route.widths) == dict(resolved) == {"subject": 9_000}


@pytest.mark.parametrize("n_subjects", [9_000])
def test_results_are_identical_across_the_old_and_the_derived_width(
    *, n_subjects: int
) -> None:
    """Width is a lowering specialization; it moves no value and no RNG stream."""
    params = get_params(shock_type="rouwenhorst", sigma=0.2, rho=0.9)
    initial_conditions = {
        "age": jnp.full(n_subjects, 20.0),
        "wealth": jnp.full(n_subjects, 5.0),
        "income": jnp.zeros(n_subjects),
        "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
    }
    frames = []
    for width in (_DEFAULT_UNBUDGETED_SUBJECT_WIDTH, None):
        model = create_model(
            n_periods=5,
            shock_type="rouwenhorst",
            wealth_n_points=10,
            consumption_n_points=10,
            execution_config=ExecutionConfig()
            if width is None
            else ExecutionConfig(axis_widths={"subject": width}),
        )
        solution = model.solve(params=params, log_level="off")
        frames.append(
            model.simulate(
                params=params,
                initial_conditions=initial_conditions,
                solution=solution,
                log_level="off",
                seed=12345,
            ).to_dataframe(use_labels=False)
        )
    assert list(frames[0].columns) == list(frames[1].columns)
    for column in frames[0].columns:
        np.testing.assert_array_equal(
            frames[0][column].to_numpy(), frames[1][column].to_numpy()
        )
