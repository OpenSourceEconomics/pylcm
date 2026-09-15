"""Native acceptance for planned, budgeted gated simulation."""

from typing import Any, Literal

import jax
import numpy as np
import pytest

from benchmarks.asv._simulation_witnesses import dissolution
from lcm.execution import ExecutionConfig
from lcm.persistence import load_solution


def _assert_raw_equal(*, actual: Any, expected: Any) -> None:
    assert jax.tree.structure(actual.raw_results) == jax.tree.structure(
        expected.raw_results
    )
    for got, want in zip(
        jax.tree.leaves(actual.raw_results),
        jax.tree.leaves(expected.raw_results),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("policy", ["legacy", "independent"])
def test_budgeted_gated_simulation_preserves_raw_results(
    policy: Literal["legacy", "independent"],
) -> None:
    baseline, params, initial = dissolution()
    expected = baseline.simulate(
        params=params, initial_conditions=initial, seed=6606, log_level="off"
    )
    model, params, initial = dissolution(
        execution_config=ExecutionConfig(
            devices=(0,),
            axis_widths={"subject": 2},
            device_memory_bytes=2**30,
            simulation_chunk_policy=policy,
        )
    )
    actual = model.simulate(
        params=params, initial_conditions=initial, seed=6606, log_level="off"
    )
    _assert_raw_equal(actual=actual, expected=expected)


def test_budgeted_gated_simulation_accepts_loaded_native_flags(tmp_path) -> None:
    source, params, initial = dissolution()
    solution = source.solve(params=params, log_level="off")
    loaded = load_solution(path=solution.save(path=tmp_path / "dissolution.lcm"))
    model, params, initial = dissolution(
        execution_config=ExecutionConfig(
            devices=(0,),
            axis_widths={"subject": 2},
            device_memory_bytes=2**30,
            simulation_chunk_policy="legacy",
        )
    )
    expected = source.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=6606,
        log_level="off",
    )
    actual = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=loaded,
        seed=6606,
        log_level="off",
    )
    _assert_raw_equal(actual=actual, expected=expected)
