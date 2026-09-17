"""A gated-edge model simulates over several subject devices, unchanged.

A collective regime whose value-dependent transition carries a gate reaches the
forward runtime as two extra planned programs: the gate fold, which folds the
next period's regime-level value grids, and the gate route, which sends each
subject through its own leg of the edge. Neither depends on any other subject,
so opting into `simulation_sharding="subjects"` partitions the population over
the configured devices and publishes exactly the rows one device publishes.

The witness runs in a four-CPU-device child process, because the partition under
test only exists where several devices do and a topology is pinned before JAX
initializes a backend.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tests.conftest
from lcm.execution import ExecutionConfig

_REPO_ROOT = Path(__file__).parent.parent.parent

#: Devices the child process runs on, and the subject partitions they take.
_N_DEVICES = 4

#: Subjects simulated, split over both stakeholder legs of the gated edge.
_N_SUBJECTS = 8


def _initial_conditions(*, model: Any) -> dict[str, Any]:
    """Seed both legs of the edge across the gated model's wage grid."""
    roles = model.stakeholder_names_to_ids
    return {
        "wage": jnp.asarray([1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0]),
        "age": jnp.zeros(_N_SUBJECTS),
        "regime_id": jnp.full(
            _N_SUBJECTS, model.regime_names_to_ids["married"], dtype=jnp.int32
        ),
        "own_stakeholder": jnp.asarray(
            [roles["f"], roles["m"]] * (_N_SUBJECTS // 2), dtype=jnp.int32
        ),
    }


def _simulated(*, execution_config: ExecutionConfig | None) -> Any:
    """Solve and simulate the collective dissolution model on one placement."""
    from benchmarks.asv._simulation_witnesses import dissolution  # noqa: PLC0415

    model, params, _ = dissolution(execution_config=execution_config)
    return model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions=_initial_conditions(model=model),
        seed=6606,
        log_level="off",
    )


def report_subject_sharded_gated_edges(*, decimal: int) -> dict[str, Any]:
    """Report how the subject-sharded gated simulation differs from one device.

    Args:
        decimal: Absolute agreement required of the published float leaves.

    Returns:
        Dictionary carrying any planning refusal and the per-leaf mismatches.

    """
    from lcm.exceptions import ExecutionPlanningError  # noqa: PLC0415

    try:
        actual = _simulated(
            execution_config=ExecutionConfig(
                devices=tuple(range(_N_DEVICES)),
                simulation_sharding="subjects",
            )
        )
    except ExecutionPlanningError as refusal:
        # A refusal publishes no rows at all, so it is the only mismatch there
        # is: report it under every comparison rather than leaving one empty.
        return {
            "refusal": str(refusal),
            "exact_mismatches": [str(refusal)],
            "value_mismatches": [str(refusal)],
            "n_subjects": None,
        }
    expected = _simulated(execution_config=None)

    exact: list[str] = []
    approximate: list[str] = []
    assert jax.tree.structure(actual.raw_results) == jax.tree.structure(
        expected.raw_results
    )
    leaves = zip(
        jax.tree.leaves_with_path(actual.raw_results),
        jax.tree.leaves(expected.raw_results),
        strict=True,
    )
    for (path, got), want in leaves:
        label = jax.tree_util.keystr(path)
        got_arr = np.asarray(got)
        want_arr = np.asarray(want)
        try:
            if got_arr.dtype.kind in "biu":
                np.testing.assert_array_equal(got_arr, want_arr)
            else:
                np.testing.assert_array_almost_equal(got_arr, want_arr, decimal=decimal)
        except AssertionError as mismatch:
            (exact if got_arr.dtype.kind in "biu" else approximate).append(
                f"{label}: {mismatch}"
            )
    return {
        "refusal": "",
        "exact_mismatches": exact,
        "value_mismatches": approximate,
        "n_subjects": actual.n_subjects,
    }


def _run_in_four_device_process(*, entry_point: str) -> dict[str, Any]:
    """Run one module-level report function on four CPU devices and return it.

    The child carries this run's float policy — `jax_enable_x64`, the matmul
    precision and the matching `DECIMAL_PRECISION` — because a topology pin
    needs a fresh process and a fresh process reads none of pytest's options.

    Args:
        entry_point: Name of a function in this module taking the tolerance in
            decimals and returning a JSON-serializable mapping.

    Returns:
        Dictionary of the report the child process produced.

    """
    code = (
        "import json, sys; import jax; "
        f"jax.config.update('jax_num_cpu_devices', {_N_DEVICES}); "
        "jax.config.update('jax_platform_name', 'cpu'); "
        f"jax.config.update('jax_enable_x64', {tests.conftest.X64_ENABLED!r}); "
        "jax.config.update('jax_default_matmul_precision', 'highest'); "
        "from tests.simulation.test_gated_edges_subject_sharding import "
        f"{entry_point} as entry; "
        "sys.stdout.write('@@' + json.dumps("
        f"entry(decimal={tests.conftest.DECIMAL_PRECISION!r})) + '@@')"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    _, _, rest = completed.stdout.partition("@@")
    payload, _, _ = rest.partition("@@")
    assert payload, completed.stdout
    return json.loads(payload)


@pytest.fixture(scope="module")
def sharded_gated_edges() -> dict[str, Any]:
    """Return the gated subject-sharding report from one four-device process."""
    return _run_in_four_device_process(entry_point="report_subject_sharded_gated_edges")


def test_a_gated_edge_model_accepts_subject_sharding(
    sharded_gated_edges: dict[str, Any],
) -> None:
    """The gate fold and gate route both declare what subject sharding needs."""
    assert sharded_gated_edges["refusal"] == ""


def test_subject_sharded_gated_routing_keeps_every_subject(
    sharded_gated_edges: dict[str, Any],
) -> None:
    """Padding the population over the devices publishes the seeded rows only."""
    assert sharded_gated_edges["n_subjects"] == _N_SUBJECTS


def test_subject_sharded_gated_routing_takes_the_same_branches(
    sharded_gated_edges: dict[str, Any],
) -> None:
    """Regime ids, roles and every other integer leaf are branch decisions."""
    assert sharded_gated_edges["exact_mismatches"] == []


def test_subject_sharded_gated_simulation_publishes_the_same_values(
    sharded_gated_edges: dict[str, Any],
) -> None:
    """Partitioning subjects moves no published float beyond the precision."""
    assert sharded_gated_edges["value_mismatches"] == []
