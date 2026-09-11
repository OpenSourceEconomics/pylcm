"""CPU assembly admits all retained chunks and every published record field."""

import dataclasses
from types import MappingProxyType

import jax
import jax._src.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.engine import PeriodRegimeSimulationData
from _lcm.simulation.initial_conditions import trim_pad_from_raw_results
from _lcm.simulation.simulate import _concatenate_chunk_results
from lcm.exceptions import ExecutionPlanningError
from tests.simulation.test_budget_lifecycle import _stateful_target_model
from tests.simulation.test_population_allocation_budget import (
    _forbid_concrete,
    _memory,
    _UnadmittedAllocationError,
)


def _record(*, start: int) -> PeriodRegimeSimulationData:
    return PeriodRegimeSimulationData(
        V_arr=jnp.asarray([[start, -0.0], [start + 1, jnp.nan]]),
        actions=MappingProxyType({"action": jnp.asarray([start, start + 1.0])}),
        states=MappingProxyType(
            {"state": jnp.asarray([start, start + 1], dtype=jnp.int32)}
        ),
        in_regime=jnp.asarray([True, False]),
        own_stakeholder=jnp.asarray([0, 1], dtype=jnp.int32),
        nested_policy_fallback=jnp.asarray([False, True]),
    )


def _leaves(record: PeriodRegimeSimulationData) -> list[jax.Array]:
    return jax.tree.leaves(
        tuple(getattr(record, f.name) for f in dataclasses.fields(record))
    )


@pytest.mark.parametrize("operation", ["concatenate", "trim"])
def test_final_assembly_refuses_before_allocation_and_preserves_every_field(
    *, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    first = _record(start=0)
    second = _record(start=2)
    originals = (_leaves(first), _leaves(second))
    chunks = [{"alive": {0: first}}, {"alive": {0: second}}]
    raw = MappingProxyType({"alive": MappingProxyType({0: first})})
    regimes = _stateful_target_model()._regimes
    low = _memory(inputs=originals, budget=1)
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_concrete)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_concrete)
        with pytest.raises(_UnadmittedAllocationError):
            jnp.zeros(3)
        if operation == "concatenate":
            with pytest.raises(ExecutionPlanningError):
                _concatenate_chunk_results(
                    chunk_results=chunks, regimes=regimes, memory=low
                )
        else:
            with pytest.raises(ExecutionPlanningError):
                trim_pad_from_raw_results(
                    raw_results=raw, original_n_subjects=1, memory=low
                )
    generous = _memory(inputs=originals, budget=1_000_000)
    if operation == "concatenate":
        actual = _concatenate_chunk_results(
            chunk_results=chunks, regimes=regimes, memory=generous
        )["alive"][0]
        expected = [
            np.concatenate([np.asarray(a), np.asarray(b)])
            for a, b in zip(*originals, strict=True)
        ]
    else:
        actual = trim_pad_from_raw_results(
            raw_results=raw, original_n_subjects=1, memory=generous
        )["alive"][0]
        expected = [np.asarray(a)[:1] for a in originals[0]]
    for observed, reference in zip(_leaves(actual), expected, strict=True):
        np.testing.assert_array_equal(observed, reference)
        assert observed.dtype == reference.dtype
        if np.issubdtype(reference.dtype, np.floating):
            np.testing.assert_array_equal(np.signbit(observed), np.signbit(reference))
    np.testing.assert_array_equal(first.V_arr, [[0.0, -0.0], [1.0, np.nan]])
    np.testing.assert_array_equal(second.V_arr, [[2.0, -0.0], [3.0, np.nan]])
