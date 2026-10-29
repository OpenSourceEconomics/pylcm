"""Independent storage and allocation boundaries for the finite policy corridor."""

from collections.abc import Mapping
from typing import Never, Unpack, cast

import jax
import jax._src.core
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.chunk_admission as admission
import _lcm.simulation.simulate as simulation
from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.simulation.chunk_planning import SimulationChunkProfile
from _lcm.simulation.chunk_profile_inventory import ChunkProfileInventory
from _lcm.simulation.chunk_profiles import profile_simulation_chunk
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.typing import ShapeDtypePytree
from lcm.exceptions import ExecutionPlanningError
from tests.simulation._callback_types import (
    ChunkProfileInputs,
    InventoryCompiled,
)
from tests.simulation.test_finite_policy_budget import _inputs
from tests.simulation.test_population_allocation_budget import (
    _forbid_concrete,
    _UnadmittedAllocationError,
)


class _ProfileObservedError(Exception):
    """Stop the real public call immediately after inspecting its selected profile."""


@pytest.mark.parametrize("inner_width", [1, 3, 7])
def test_finite_bank_keeps_the_full_outer_extent_without_allocating(
    *, inner_width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inner tile cannot shrink the actual C-by-candidate preparation output."""
    model, params, initial = _inputs(discrete=False, budget=2**32, width=7)
    initial = {name: np.repeat(value[:1], 7) for name, value in initial.items()}
    solution = model.solve(params=params, log_level="off")
    policy = cast("OwnedSolutionView", solution._engine_view).simulation_policies[0][
        "alive"
    ]
    assert isinstance(policy, NNBEGMSimPolicy)
    candidate_count = policy.candidate_inner_action.shape[0]
    itemsize = policy.candidate_inner_action.dtype.itemsize
    observed: list[tuple[tuple[int, ...], ...]] = []
    compiled = ChunkProfileInventory.compiled

    def observe_bank(
        self: ChunkProfileInventory, **call: Unpack[InventoryCompiled]
    ) -> ShapeDtypePytree:
        result = compiled(self, **call)
        if call["name"] == "core:policy_prepare":
            bank = cast("tuple[jax.ShapeDtypeStruct, ...]", result)
            observed.append(tuple(leaf.shape for leaf in bank))
            assert sum(leaf.size * np.dtype(leaf.dtype).itemsize for leaf in bank) == (
                7 * candidate_count * (3 * itemsize + 1)
            )
        return result

    # keyword-only-exempt: library-callback=_ChunkProfiler.profile_widths
    def inspect_profile(
        self: admission._ChunkProfiler, *, n_subjects: int, widths: Mapping[str, int]
    ) -> Never:
        del widths
        assert n_subjects == 7
        with monkeypatch.context() as guard:
            guard.setattr(jax, "device_put", _forbid_concrete)
            guard.setattr(
                jax._src.core.EvalTrace, "process_primitive", _forbid_concrete
            )
            with pytest.raises(_UnadmittedAllocationError):
                jnp.zeros(3)
            profile_simulation_chunk(
                runtime=self.runtime,
                regimes=self.regimes,
                flat_params=self.call_inputs.flat_params,
                base_spaces=self.call_inputs.base_state_action_spaces,
                values=self.values,
                policies=self.policies,
                ages=self.ages,
                initial_conditions=self.initial_conditions,
                regime_names_to_ids=self.regime_names_to_ids,
                n_subjects=n_subjects,
                population=self.population,
                original_population=self.original_population,
                widths={"subject": inner_width},
                independent_taste=False,
                log_level="off",
            )
        assert observed == [((7, candidate_count),) * 4]
        raise _ProfileObservedError

    monkeypatch.setattr(ChunkProfileInventory, "compiled", observe_bank)
    monkeypatch.setattr(admission._ChunkProfiler, "profile_widths", inspect_profile)
    with pytest.raises(_ProfileObservedError):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            log_level="off",
            seed=17,
        )


@pytest.mark.parametrize("runtime_checks", [False, True])
def test_public_finite_bank_floor_refuses_before_any_chunk_allocation(
    *, monkeypatch: pytest.MonkeyPatch, runtime_checks: bool
) -> None:
    """A real bank lower bound cannot be hidden behind a raw compiler peak."""
    width = 8192
    reference, params, initial = _inputs(discrete=False, budget=None)
    solved = reference.solve(params=params, log_level="off")
    policy = cast("OwnedSolutionView", solved._engine_view).simulation_policies[0][
        "alive"
    ]
    assert isinstance(policy, NNBEGMSimPolicy)
    bank_bytes = (
        width
        * policy.candidate_inner_action.shape[0]
        * (3 * policy.candidate_inner_action.dtype.itemsize + 1)
    )
    model, params, _ = _inputs(discrete=False, budget=bank_bytes - 1, width=width)
    solution = model.solve(params=params, log_level="off")
    initial = {name: np.repeat(value[:1], width) for name, value in initial.items()}
    inspected: list[int] = []
    original = admission.profile_simulation_chunk

    def observe_profile(**call: Unpack[ChunkProfileInputs]) -> SimulationChunkProfile:
        inspected.append(call["n_subjects"])
        return original(**call)

    monkeypatch.setattr(admission, "profile_simulation_chunk", observe_profile)
    monkeypatch.setattr(simulation, "_simulate_subject_chunk", _forbid_concrete)
    # Numerical preflight can exhaust this budget before bank profiling begins.
    refusal = (
        "No workspace-width candidate fits" if runtime_checks else "no anchor map fits"
    )
    with pytest.raises(ExecutionPlanningError, match=refusal):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            log_level="off",
            runtime_checks=runtime_checks,
            seed=17,
        )
    assert inspected == ([] if runtime_checks else [width])
