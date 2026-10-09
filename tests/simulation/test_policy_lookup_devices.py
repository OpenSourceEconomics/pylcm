"""A policy lookup gives the same rows however many devices the model has.

An empty query stands for one row, and one row cannot be split across devices.
A stateful query reads the next period's values wherever the solve left them,
gives its rows however many of them each subject device holds, and admits
copying the values under a memory budget before it copies them.
Each device count runs in its own subprocess with forced host devices, so the
witness does not depend on the ambient topology.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import jax
    import numpy as np

    from lcm import ExecutionConfig, LinSpacedGrid
    from tests.test_models.deterministic.regression import get_model, get_params

    n_devices = int(sys.argv[1])
    return_action_values = sys.argv[2] == "1"
    assert jax.device_count() == n_devices, jax.devices()

    grid = LinSpacedGrid(start=1, stop=3, n_points=3)
    model = get_model(
        n_periods=2,
        wealth_grid=grid,
        consumption_grid=grid,
        execution_config=ExecutionConfig(
            devices=tuple(device.id for device in jax.devices()),
            sharded_states=(),
            simulation_sharding="subjects",
            device_memory_bytes=2**30,
        ),
    )
    params = get_params(n_periods=2)
    solution = model.solve(params=params, log_level="off")
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=1,
        regime_name="dead",
        states={},
        return_action_values=return_action_values,
    )
    outputs = {"actions": sorted(got.actions), "value": np.asarray(got.value).tolist()}
    if return_action_values:
        outputs["Q"] = np.asarray(got.Q).tolist()
        outputs["F"] = np.asarray(got.F).tolist()
    print("LOOKUP-ROWS", json.dumps(outputs))
    """
)


_STATEFUL_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import jax
    import jax.numpy as jnp
    import numpy as np

    from tests.test_models.deterministic.regression import get_model, get_params

    from lcm import ExecutionConfig

    n_devices = int(sys.argv[1])
    return_action_values = sys.argv[2] == "1"
    execution = sys.argv[3]
    n_rows = int(sys.argv[4])
    assert jax.device_count() == n_devices, jax.devices()

    execution_config = {
        "default": None,
        "subjects": ExecutionConfig(
            devices=tuple(device.id for device in jax.devices()),
            sharded_states=(),
            simulation_sharding="subjects",
        ),
        "subjects-budgeted": ExecutionConfig(
            devices=tuple(device.id for device in jax.devices()),
            sharded_states=(),
            simulation_sharding="subjects",
            device_memory_bytes=2**30,
        ),
    }[execution]
    model = (
        get_model(n_periods=5)
        if execution_config is None
        else get_model(n_periods=5, execution_config=execution_config)
    )
    params = get_params(n_periods=5)
    solution = model.solve(params=params, log_level="off")
    wealth = model.state_grid(
        params=params, regime_name="working_life", state_name="wealth"
    )
    assert len(wealth) >= n_rows
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=3,
        regime_name="working_life",
        states={"wealth": jnp.asarray(wealth[:n_rows])},
        return_action_values=return_action_values,
    )
    outputs = {
        "actions": {
            name: np.asarray(values).tolist() for name, values in got.actions.items()
        },
        "value": np.asarray(got.value).tolist(),
    }
    if return_action_values:
        outputs["Q"] = np.asarray(got.Q).tolist()
        outputs["F"] = np.asarray(got.F).tolist()
    print("LOOKUP-ROWS", json.dumps(outputs))
    """
)


_ADMISSION_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import jax
    import jax.numpy as jnp
    import numpy as np

    import lcm.model
    from lcm import ExecutionConfig, LinSpacedGrid
    from lcm.exceptions import ExecutionPlanningError
    from tests.test_models.deterministic.regression import get_model, get_params

    n_devices = int(sys.argv[1])
    return_action_values = sys.argv[2] == "1"
    assert jax.device_count() == n_devices, jax.devices()
    budget = 2**22
    first, second = jax.devices()[:2]

    grid = LinSpacedGrid(start=1, stop=3, n_points=3)
    model = get_model(
        n_periods=3,
        wealth_grid=grid,
        consumption_grid=grid,
        execution_config=ExecutionConfig(
            devices=tuple(device.id for device in jax.devices()),
            sharded_states=(),
            simulation_sharding="subjects",
            device_memory_bytes=budget,
            device_memory_headroom_fraction=0.0,
        ),
    )
    params = get_params(n_periods=3)
    solution = model.solve(params=params, log_level="off")
    next_values = [
        np.asarray(value) for value in solution.values[1].values() if value.ndim
    ]
    assert next_values
    assert solution.values[1]["working_life"].devices() == {first}


    class _Copied(Exception):
        pass


    entered = []
    copies = []
    stop_at_copy = []
    place = jax.device_put
    lookup = lcm.model.Model._lookup_policy


    def _entered(self, **kwargs):
        entered.append(True)
        return lookup(self, **kwargs)


    def _recorded(x, *args, **kwargs):
        if isinstance(x, jax.Array) and any(
            x.shape == value.shape and np.array_equal(np.asarray(x), value)
            for value in next_values
        ):
            copies.append(x.shape)
            if stop_at_copy:
                raise _Copied
        return place(x, *args, **kwargs)


    lcm.model.Model._lookup_policy = _entered
    jax.device_put = _recorded
    dtype = jnp.zeros(()).dtype


    def run(n_actions, *, stop):
        entered.clear()
        copies.clear()
        stop_at_copy[:] = [True] if stop else []
        action_grids = {"consumption": place(jnp.ones(n_actions, dtype=dtype), second)}
        states = {"wealth": place(jnp.asarray([2.0, 3.0], dtype=dtype), first)}
        try:
            with jax.default_device(first):
                model.lookup_policy(
                    params=params,
                    solution=solution,
                    period=0,
                    regime_name="working_life",
                    states=states,
                    action_grids=action_grids,
                    return_action_values=return_action_values,
                )
        except _Copied:
            outcome = "copied"
        except ExecutionPlanningError:
            outcome = "refused"
        else:
            outcome = "returned"
        return {
            "outcome": outcome,
            "entered": bool(entered),
            "value_copies": len(copies),
        }


    assert run(1, stop=True)["outcome"] == "copied"
    low, high = 1, budget // dtype.itemsize
    while low < high:
        middle = (low + high + 1) // 2
        if run(middle, stop=True)["outcome"] == "copied":
            low = middle
        else:
            high = middle - 1
    outputs = {
        "small_grid": run(1, stop=False)["outcome"],
        "at_boundary": run(low, stop=True)["outcome"],
        "beyond_boundary": run(low + 1, stop=False),
    }
    print("LOOKUP-ROWS", json.dumps(outputs))
    """
)


_ALIGNED_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import jax
    import jax.numpy as jnp
    import numpy as np

    from lcm import ExecutionConfig
    from tests.test_models.deterministic.regression import get_model, get_params

    n_devices = int(sys.argv[1])
    return_action_values = sys.argv[2] == "1"
    assert jax.device_count() == n_devices, jax.devices()

    model = get_model(
        n_periods=5,
        execution_config=ExecutionConfig(
            devices=(0,),
            sharded_states=(),
            simulation_sharding="subjects",
            device_memory_bytes=2**30,
        ),
    )
    params = get_params(n_periods=5)
    solution = model.solve(params=params, log_level="off")
    next_values = [np.asarray(solution.values[3]["working_life"])]
    puts = []
    place = jax.device_put


    def _recorded(x, *args, **kwargs):
        if isinstance(x, jax.Array) and any(
            x.shape == value.shape and np.array_equal(np.asarray(x), value)
            for value in next_values
        ):
            puts.append(x.shape)
        return place(x, *args, **kwargs)


    jax.device_put = _recorded
    wealth = model.state_grid(
        params=params, regime_name="working_life", state_name="wealth"
    )
    got = model.lookup_policy(
        params=params,
        solution=solution,
        period=2,
        regime_name="working_life",
        states={"wealth": jnp.asarray(wealth[:2])},
        return_action_values=return_action_values,
    )
    outputs = {"rows": len(np.asarray(got.value)), "value_copies": len(puts)}
    print("LOOKUP-ROWS", json.dumps(outputs))
    """
)


_TRIM_SCRIPT = textwrap.dedent(
    """
    import json
    import sys
    from collections import Counter

    import jax
    import jax.numpy as jnp

    from lcm import ExecutionConfig, LinSpacedGrid
    from lcm.exceptions import ExecutionPlanningError
    from tests.test_models.deterministic.regression import get_model, get_params

    n_devices = int(sys.argv[1])
    return_action_values = sys.argv[2] == "1"
    n_rows = int(sys.argv[3])
    assert jax.device_count() == n_devices, jax.devices()
    budget = 4_718_592
    # Q's bytes do not depend on the float precision.
    n_actions = 800_000 // jnp.zeros(()).dtype.itemsize

    grid = LinSpacedGrid(start=1, stop=3, n_points=3)
    model = get_model(
        n_periods=3,
        wealth_grid=grid,
        consumption_grid=grid,
        execution_config=ExecutionConfig(
            devices=tuple(device.id for device in jax.devices()),
            sharded_states=(),
            simulation_sharding="subjects",
            device_memory_bytes=budget,
            device_memory_headroom_fraction=0.0,
        ),
    )
    params = get_params(n_periods=3)
    solution = model.solve(params=params, log_level="off")
    try:
        got = model.lookup_policy(
            params=params,
            solution=solution,
            period=1,
            regime_name="working_life",
            states={"wealth": jnp.linspace(1.0, 3.0, n_rows)},
            action_grids={"consumption": jnp.ones(n_actions)},
            return_action_values=return_action_values,
        )
    except ExecutionPlanningError:
        outputs = {"outcome": "refused"}
    else:
        per_device = Counter()
        for leaf in jax.tree.leaves((got.actions, got.value, got.Q, got.F)):
            for shard in leaf.addressable_shards:
                per_device[shard.device] += shard.data.nbytes
        outputs = {
            "outcome": "returned",
            "rows": len(got.value),
            "within_budget": max(per_device.values()) <= budget,
        }
    print("LOOKUP-ROWS", json.dumps(outputs))
    """
)


def _lookup_on_devices(
    *,
    n_devices: int,
    return_action_values: bool,
    script: str = _SCRIPT,
    extra_args: tuple[str, ...] = (),
) -> dict:
    env = {
        **os.environ,
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={n_devices}",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": str(int(jax.config.read("jax_enable_x64"))),
    }
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            script,
            str(n_devices),
            str(int(return_action_values)),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    (line,) = (
        line for line in result.stdout.splitlines() if line.startswith("LOOKUP-ROWS ")
    )
    return json.loads(line.removeprefix("LOOKUP-ROWS "))


@pytest.mark.parametrize("n_devices", [1, 2, 8])
def test_budgeted_stateless_lookup_returns_one_row_of_value(*, n_devices: int) -> None:
    """The terminal `dead` regime's empty query is one row of zero on any devices."""
    got = _lookup_on_devices(n_devices=n_devices, return_action_values=False)
    assert got == {"actions": [], "value": [0.0]}


@pytest.mark.parametrize("n_devices", [1, 2, 8])
def test_budgeted_stateless_lookup_returns_one_row_of_action_values(
    *, n_devices: int
) -> None:
    """With action values requested, `Q` and `F` carry the same one row."""
    got = _lookup_on_devices(n_devices=n_devices, return_action_values=True)
    assert got == {"actions": [], "value": [0.0], "Q": [0.0], "F": [True]}


@pytest.mark.parametrize("return_action_values", [False, True])
@pytest.mark.parametrize("n_devices", [2, 8])
def test_stateful_lookup_on_several_devices_matches_one_device(
    *, n_devices: int, return_action_values: bool
) -> None:
    """A default-config model's stateful lookup gives the one-device rows."""
    got = _lookup_on_devices(
        n_devices=n_devices,
        return_action_values=return_action_values,
        script=_STATEFUL_SCRIPT,
        extra_args=("default", "2"),
    )
    expected = _lookup_on_devices(
        n_devices=1,
        return_action_values=return_action_values,
        script=_STATEFUL_SCRIPT,
        extra_args=("default", "2"),
    )
    assert got == expected


@pytest.mark.parametrize("return_action_values", [False, True])
@pytest.mark.parametrize("execution", ["subjects", "subjects-budgeted"])
@pytest.mark.parametrize("n_devices", [2, 8])
def test_subject_sharded_lookup_of_uneven_rows_matches_one_device(
    *, n_devices: int, execution: str, return_action_values: bool
) -> None:
    """Three rows split over the subject devices give the three one-device rows."""
    got = _lookup_on_devices(
        n_devices=n_devices,
        return_action_values=return_action_values,
        script=_STATEFUL_SCRIPT,
        extra_args=(execution, "3"),
    )
    expected = _lookup_on_devices(
        n_devices=1,
        return_action_values=return_action_values,
        script=_STATEFUL_SCRIPT,
        extra_args=("default", "3"),
    )
    assert got == expected


@pytest.mark.parametrize("return_action_values", [False, True])
@pytest.mark.parametrize("n_devices", [2, 8])
def test_budgeted_lookup_refuses_a_value_copy_before_making_it(
    *, n_devices: int, return_action_values: bool
) -> None:
    """Admission refuses the next-period value copy before any copy is made.

    Growing a replacement action grid on the second device fills its budget.
    Up to some length the lookup admits copying the next-period values onto the
    subject devices; one action more, the call is admitted at entry but refuses
    with no value copied. A one-point grid returns rows.
    """
    got = _lookup_on_devices(
        n_devices=n_devices,
        return_action_values=return_action_values,
        script=_ADMISSION_SCRIPT,
    )
    assert got == {
        "small_grid": "returned",
        "at_boundary": "copied",
        "beyond_boundary": {"outcome": "refused", "entered": True, "value_copies": 0},
    }


@pytest.mark.parametrize("return_action_values", [False, True])
def test_budgeted_lookup_reads_values_in_place_on_one_device(
    *, return_action_values: bool
) -> None:
    """On one device the solved values are read where they are, without a copy."""
    got = _lookup_on_devices(
        n_devices=1,
        return_action_values=return_action_values,
        script=_ALIGNED_SCRIPT,
    )
    assert got == {"rows": 2, "value_copies": 0}


@pytest.mark.parametrize(
    ("n_rows", "expected"),
    [
        (3, {"outcome": "refused"}),
        (8, {"outcome": "returned", "rows": 8, "within_budget": True}),
    ],
)
def test_budgeted_lookup_admits_trimming_padded_rows_within_the_budget(
    *, n_rows: int, expected: dict
) -> None:
    """Returned action values fit the per-device budget, or the lookup refuses.

    Three rows padded to eight subject devices trim to a layout that holds every
    row of `Q` on each device, beyond the budget; eight rows stay split over the
    devices and fit it.
    """
    got = _lookup_on_devices(
        n_devices=8,
        return_action_values=True,
        script=_TRIM_SCRIPT,
        extra_args=(str(n_rows),),
    )
    assert got == expected
