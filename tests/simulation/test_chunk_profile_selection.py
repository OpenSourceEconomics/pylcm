"""Whole-chunk admission combines real compiler peaks and retained output banks."""

import importlib
from types import MappingProxyType
from typing import Any

import jax
import jax._src.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.simulate import _lookup_values_from_indices
from lcm.exceptions import ExecutionPlanningError


class _PlanningAllocatedError(AssertionError):
    """No candidate population may be allocated just to decide whether it fits."""


def _forbid_allocation(*_args: object, **_kwargs: object) -> object:
    raise _PlanningAllocatedError("Chunk selection allocated a population")


def test_whole_chunk_selection_uses_real_profiles_and_fresh_retained_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measured strict frontier controls selection without executing candidates."""
    device = jax.devices()[0]
    layout = jax.sharding.SingleDeviceSharding(device)
    grid = jnp.asarray([1.0, 3.0, 7.0])
    originals = jnp.zeros(1024, dtype=jnp.int32)
    operations = ProfiledSimulationOperations()
    compiled = {
        width: operations.prepare_abstract(
            function=_lookup_values_from_indices,
            arguments={
                "flat_indices": jax.ShapeDtypeStruct(
                    (width,), jnp.int32, sharding=layout
                ),
                "grids": MappingProxyType(
                    {
                        "consumption": jax.ShapeDtypeStruct(
                            grid.shape, grid.dtype, sharding=layout
                        )
                    }
                ),
            },
            subject_arg_names=("flat_indices",),
            devices=(device,),
        )
        for width in (1024, 256, 64)
    }
    # A complete retained output and its final assembly destination survive the
    # candidate's current decoder. This floor does not shrink with the chunk.
    retained_output_bank = 2 * originals.size * grid.dtype.itemsize
    totals = {
        width: grid.nbytes
        + originals.nbytes
        + retained_output_bank
        + width * originals.dtype.itemsize
        + profile.peak_bytes
        for width, profile in compiled.items()
    }
    assert totals[1024] > totals[256] > totals[64]
    budget = totals[256]
    planner = importlib.import_module("_lcm.simulation.chunk_planning")
    profiles = {
        width: planner.SimulationChunkProfile(
            n_subjects=width,
            padded_population=1024,
            stages=(
                planner.SimulationStageProfile(
                    name="decode_actions",
                    executable=profile.executable,
                    devices=(device,),
                ),
            ),
            fixed_reservation={device: width * originals.dtype.itemsize},
            output_reservation={device: retained_output_bank},
        )
        for width, profile in compiled.items()
    }
    requested: list[int] = []

    def profile_candidate(*, n_subjects: int) -> Any:
        requested.append(n_subjects)
        return profiles[n_subjects]

    live = measure_buffer_footprint(tree=(grid, originals))
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_allocation)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_allocation)
        with pytest.raises(_PlanningAllocatedError):
            jnp.zeros(3)
        selected = planner.plan_simulation_chunks(
            candidates=(1024, 256, 64),
            profile_candidate=profile_candidate,
            live=live,
            budget_bytes=budget,
            devices=(device,),
        )
    assert selected.profile is profiles[256]
    assert selected.required_bytes[device] == budget
    assert requested == [1024, 256]
    extra = jnp.zeros(totals[256] - totals[64], dtype=jnp.uint8)
    requested.clear()
    smaller = planner.plan_simulation_chunks(
        candidates=(1024, 256, 64),
        profile_candidate=profile_candidate,
        live=measure_buffer_footprint(tree=(grid, originals, extra)),
        budget_bytes=budget,
        devices=(device,),
    )
    assert smaller.profile is profiles[64]
    assert requested == [1024, 256, 64]
    with pytest.raises(ExecutionPlanningError, match="chunk"):
        planner.plan_simulation_chunks(
            candidates=(1024,),
            profile_candidate=profile_candidate,
            live=live,
            budget_bytes=budget,
            devices=(device,),
        )
    actual = selected.profile.stages[0].executable(
        flat_indices=jnp.zeros(256, dtype=jnp.int32),
        grids=MappingProxyType({"consumption": grid}),
    )
    np.testing.assert_array_equal(actual["consumption"], np.ones(256))
    np.testing.assert_array_equal(originals, np.zeros(1024, dtype=np.int32))
