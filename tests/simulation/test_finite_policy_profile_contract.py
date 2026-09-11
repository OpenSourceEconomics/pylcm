"""Independent storage and allocation boundaries for the finite policy corridor."""

import gc
import weakref
from typing import Any, cast

import jax
import jax._src.core
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.chunk_admission as admission
import _lcm.simulation.simulate as simulation
from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.simulation.chunk_profile_inventory import ChunkProfileInventory
from _lcm.simulation.chunk_profiles import profile_simulation_chunk
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.policy_diagnostics import dropped_candidate_counts
from _lcm.simulation.residency import measure_buffer_footprint, resident_bytes_by_device
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.solution.artifacts import OwnedSolutionView
from lcm.exceptions import ExecutionPlanningError
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

    def observe_bank(self: ChunkProfileInventory, **call: Any) -> object:
        result = compiled(self, **call)
        if call["name"] == "core:policy_prepare":
            bank = cast("tuple[jax.ShapeDtypeStruct, ...]", result)
            observed.append(tuple(leaf.shape for leaf in bank))
            assert sum(leaf.size * np.dtype(leaf.dtype).itemsize for leaf in bank) == (
                7 * candidate_count * (2 * itemsize + 2)
            )
        return result

    # keyword-only-exempt: library-callback=_ChunkProfiler.__call__
    def inspect_profile(self: admission._ChunkProfiler, *, n_subjects: int) -> object:
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
    monkeypatch.setattr(admission._ChunkProfiler, "__call__", inspect_profile)
    with pytest.raises(_ProfileObservedError):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            log_level="off",
            seed=17,
        )


def test_public_finite_bank_floor_refuses_before_any_chunk_allocation(
    monkeypatch: pytest.MonkeyPatch,
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
        * (2 * policy.candidate_inner_action.dtype.itemsize + 2)
    )
    model, params, _ = _inputs(discrete=False, budget=bank_bytes - 1, width=width)
    solution = model.solve(params=params, log_level="off")
    initial = {name: np.repeat(value[:1], width) for name, value in initial.items()}
    inspected: list[int] = []
    original = admission.profile_simulation_chunk

    def observe_profile(**call: Any) -> object:
        inspected.append(call["n_subjects"])
        return original(**call)

    monkeypatch.setattr(admission, "profile_simulation_chunk", observe_profile)
    monkeypatch.setattr(simulation, "_simulate_subject_chunk", _forbid_concrete)
    with pytest.raises(ExecutionPlanningError, match="No declared simulation chunk"):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            log_level="off",
            seed=17,
        )
    assert inspected == [width]


@pytest.mark.parametrize("refuse", [False, True])
def test_finite_diagnostic_counts_the_live_bank_and_admits_before_dispatch(
    *, refuse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host diagnostic sees actual preparation owners before allocating."""
    model, params, initial = _inputs(discrete=False, budget=2**32)
    solution = model.solve(params=params, log_level="off")
    banks: list[tuple[jax.Array, ...]] = []
    bank_refs: list[weakref.ReferenceType[jax.Array]] = []
    observed: list[bool] = []
    dispatch = SimulationRuntime.dispatch
    run = SimulationMemory.run

    def observe_prepare(self: SimulationRuntime, **call: Any) -> object:
        result = dispatch(self, **call)
        if call["program"].name == "simulate_policy_prepare":
            bank = cast("tuple[jax.Array, ...]", result)
            banks.append(bank)
            bank_refs.extend(weakref.ref(leaf) for leaf in bank)
        return result

    def inspect_count(self: SimulationMemory, **call: Any) -> object:
        if call["function"] is not dropped_candidate_counts:
            return run(self, **call)
        bank = banks.pop(0)
        expected = measure_buffer_footprint(tree=bank)
        missing = resident_bytes_by_device(
            live=expected, arguments=self.snapshot(), devices=tuple(expected.spans)
        )
        assert not any(missing.values()), "The finite diagnostic omitted its live bank"
        observed.append(True)
        if refuse:
            self.budget_bytes = 1
            with monkeypatch.context() as guard:
                guard.setattr(jax, "device_put", _forbid_concrete)
                guard.setattr(
                    jax._src.core.EvalTrace, "process_primitive", _forbid_concrete
                )
                return run(self, **call)
        return run(self, **call)

    monkeypatch.setattr(SimulationRuntime, "dispatch", observe_prepare)
    monkeypatch.setattr(SimulationMemory, "run", inspect_count)
    if refuse:
        with pytest.raises(ExecutionPlanningError):
            model.simulate(
                params=params,
                solution=solution,
                initial_conditions=initial,
                log_level="warning",
                seed=17,
            )
        assert observed == [True]
    else:
        result = model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            log_level="warning",
            seed=17,
        )
        assert observed == [True, True]
        assert result.n_subjects == 2
        jax.block_until_ready(result.period_to_regime_to_V_arr)
        gc.collect()
        assert all(ref() is None for ref in bank_refs)


def test_diagnostic_owner_oracle_rejects_an_omitted_preparation_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seeded missing host hold is detected by actual independently measured spans."""
    hold = SimulationMemory.hold

    # keyword-only-exempt: library-callback=SimulationMemory.hold
    def omit_bank(self: SimulationMemory, tree: object) -> None:
        if (
            isinstance(tree, tuple)
            and len(tree) == 4
            and all(isinstance(leaf, jax.Array) and leaf.ndim == 2 for leaf in tree)
        ):
            return
        hold(self, tree=tree)

    monkeypatch.setattr(SimulationMemory, "hold", omit_bank)
    with pytest.raises(AssertionError, match="diagnostic omitted its live bank"):
        test_finite_diagnostic_counts_the_live_bank_and_admits_before_dispatch(
            refuse=False, monkeypatch=monkeypatch
        )
