"""A captured regime-period can be replayed on the device layout it ran on.

`replay_period` restores a capture's logical pytrees and lets the backend place
them, which makes any per-device figure taken from it incomparable with the
solve's. A layout-faithful replay instead puts every array back on its recorded
sharding and checks the compiled placements against the capture, so the run it
repeats is the run that happened.

These checks run on four forced host CPU devices. CPU proves the plumbing — that
descriptors round trip, that placements are rebuilt and validated, and that a
mismatch is refused — not GPU memory behaviour and not performance.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PRELUDE = """
import os
from pathlib import Path

import cloudpickle
import jax
import numpy as np

assert jax.device_count() == 4, jax.devices()

root = Path(os.environ["CAPTURE_ROOT"])
os.environ["LCM_CAPTURE_PERIOD"] = "alive@1"
os.environ["LCM_CAPTURE_DIR"] = str(root)

from _lcm.solution import period_capture, period_replay
from tests.test_models import nbegm_ride_along_toy as toy

model = toy.build_model(
    variant="brute",
    n_periods=4,
    n_liquid=24,
    n_consumption=16,
    n_savings=32,
    distributed_kind=True,
)
solution = model.solve(params=toy.build_params(), log_level="off")
directory = root / "alive@1"
payload_path = directory / period_capture._PAYLOAD_NAME


def replay_devices(order=None):
    # As many devices as the capture recorded, taken in the given order.
    with payload_path.open("rb") as stream:
        layouts = cloudpickle.load(stream)[period_capture.LAYOUTS_KEY]
    n_recorded = len(layouts.device_ids)
    pool = jax.devices() if order is None else [jax.devices()[i] for i in order]
    return pool[:n_recorded]


def rewrite(mutate):
    with payload_path.open("rb") as stream:
        payload = cloudpickle.load(stream)
    mutate(payload)
    with payload_path.open("wb") as stream:
        cloudpickle.dump(payload, stream)
"""


def _run(*, body: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run one check in a fresh four-device CPU interpreter."""
    env = {
        **os.environ,
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "JAX_PLATFORMS": "cpu",
        "CAPTURE_ROOT": str(tmp_path),
        "PYTHONPATH": os.pathsep.join((str(_REPO_ROOT / "src"), str(_REPO_ROOT))),
    }
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        timeout=900,
    )


def _stdout(*, body: str, tmp_path: Path) -> str:
    """Return the check's stdout, failing with its stderr when it did not finish."""
    result = _run(body=body, tmp_path=tmp_path)
    if result.returncode != 0:
        pytest.fail(result.stderr[-4000:])
    return result.stdout


def test_every_captured_array_leaf_round_trips_to_its_recorded_descriptor(tmp_path):
    """Restoring a capture's layout reproduces the descriptor of every array leaf."""
    body = """
    with payload_path.open("rb") as stream:
        payload = cloudpickle.load(stream)
    layouts = payload[period_capture.LAYOUTS_KEY]
    restored = period_replay._restore_recorded_layout(
        kernel_kwargs=payload["kernel_kwargs"],
        leaves=layouts.leaves,
        device_by_recorded_id=period_replay._device_substitution(
            recorded_ids=layouts.device_ids, devices=replay_devices()
        ),
    )
    observed = period_capture.describe_array_leaves(tree=restored)
    print("LEAVES", len(layouts.leaves), observed == layouts.leaves)
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-2:] == ["14", "True"]


def test_a_layout_replay_validates_every_recorded_compiled_sharding(tmp_path):
    """The replay compares each compiled placement against the recorded one."""
    body = """
    compared = []
    real = period_replay._assert_recorded_sharding

    def record(*, actual, recorded, label):
        compared.append(label)
        return real(actual=actual, recorded=recorded, label=label)

    period_replay._assert_recorded_sharding = record
    period_replay.replay_period_on_recorded_layout(
        directory=directory, devices=replay_devices()
    )
    print("COMPARED", sum("compiled" in label for label in compared))
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "14"


def test_a_mismatched_recorded_output_sharding_is_refused_naming_both_values(tmp_path):
    """A recorded placement the replay does not reproduce is refused, both named."""
    body = """
    import dataclasses

    def break_output_sharding(payload):
        layouts = payload[period_capture.LAYOUTS_KEY]
        core = layouts.cores["main"]
        name, descriptor = core.compiled_output_shardings[0]
        broken = dataclasses.replace(descriptor, partition_spec=(None, None))
        payload[period_capture.LAYOUTS_KEY] = dataclasses.replace(
            layouts,
            cores={
                "main": dataclasses.replace(
                    core, compiled_output_shardings=((name, broken),)
                )
            },
        )

    rewrite(break_output_sharding)
    try:
        period_replay.replay_period_on_recorded_layout(
            directory=directory, devices=replay_devices()
        )
    except ValueError as error:
        message = str(error)
    else:
        message = "NO-REFUSAL"
    print(
        "REFUSED",
        "partition_spec=(None, None)" in message
        and "partition_spec=('kind', None)" in message,
    )
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "True"


def test_a_wrong_device_count_is_refused_naming_recorded_and_given(tmp_path):
    """Replaying on fewer devices than were recorded refuses, naming both counts."""
    body = """
    try:
        period_replay.replay_period_on_recorded_layout(
            directory=directory, devices=replay_devices()[:-1]
        )
    except ValueError as error:
        message = str(error)
    else:
        message = "NO-REFUSAL"
    n_recorded = len(replay_devices())
    print(
        "REFUSED",
        f"recorded {n_recorded}" in message and f"given {n_recorded - 1}" in message,
    )
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "True"


def test_a_capture_on_a_device_subset_records_and_replays_only_that_subset(tmp_path):
    """A model solved on two of four devices replays on any two devices."""
    body = """
    import lcm

    subset = root / "subset"
    os.environ["LCM_CAPTURE_DIR"] = str(subset)
    toy.build_model(
        variant="brute",
        n_periods=4,
        n_liquid=24,
        n_consumption=16,
        n_savings=32,
        distributed_kind=True,
        execution_config=lcm.ExecutionConfig(devices=(2, 3)),
    ).solve(params=toy.build_params(), log_level="off")
    with (subset / "alive@1" / period_capture._PAYLOAD_NAME).open("rb") as stream:
        recorded = cloudpickle.load(stream)[period_capture.LAYOUTS_KEY].device_ids
    period_replay.replay_period_on_recorded_layout(
        directory=subset / "alive@1", devices=jax.devices()[:2]
    )
    print("RECORDED", *recorded)
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-3:] == ["RECORDED", "2", "3"]


def test_an_unsupported_route_is_refused_by_name(tmp_path):
    """A capture from a route this entry point cannot reinstate is refused by name."""
    body = """
    import dataclasses

    rewrite(
        lambda payload: payload.__setitem__(
            period_capture.LAYOUTS_KEY,
            dataclasses.replace(
                payload[period_capture.LAYOUTS_KEY], route="pkg.mod.OtherKernel"
            ),
        )
    )
    try:
        period_replay.replay_period_on_recorded_layout(
            directory=directory, devices=replay_devices()
        )
    except ValueError as error:
        message = str(error)
    else:
        message = "NO-REFUSAL"
    print("REFUSED", "'pkg.mod.OtherKernel'" in message)
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "True"


def test_a_capture_without_recorded_layouts_is_refused_by_name(tmp_path):
    """A capture predating the layout block is refused rather than half-honoured."""
    body = """
    rewrite(lambda payload: payload.pop(period_capture.LAYOUTS_KEY))
    try:
        period_replay.replay_period_on_recorded_layout(
            directory=directory, devices=jax.devices()
        )
    except ValueError as error:
        message = str(error)
    else:
        message = "NO-REFUSAL"
    print("REFUSED", "'layouts'" in message)
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "True"


def test_a_capture_without_recorded_layouts_still_replays_logically(tmp_path):
    """Dropping the layout block leaves the logical entry point working."""
    body = """
    rewrite(lambda payload: payload.pop(period_capture.LAYOUTS_KEY))
    replay = period_replay.replay_period(directory=directory)
    print(
        "LOGICAL",
        bool(
            np.array_equal(
                np.asarray(replay.output.value),
                np.asarray(solution.values[1]["alive"]),
            )
        ),
    )
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "True"


def test_a_layout_replay_reproduces_the_value_array_of_the_full_solve(tmp_path):
    """The layout-faithful replay returns exactly the array the solve published."""
    body = """
    replay = period_replay.replay_period_on_recorded_layout(
        directory=directory, devices=replay_devices()
    )
    print(
        "VALUE",
        bool(
            np.array_equal(
                np.asarray(replay.output.value),
                np.asarray(solution.values[1]["alive"]),
            )
        ),
    )
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "True"


def test_a_layout_replay_reports_scope_layout(tmp_path):
    """A replay that reinstated and validated the recorded layout says so."""
    body = """
    replay = period_replay.replay_period_on_recorded_layout(
        directory=directory, devices=replay_devices()
    )
    print("SCOPE", replay.scope)
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "layout"


def test_the_logical_entry_point_reports_scope_logical(tmp_path):
    """`replay_period` places by the backend's default rules and says so."""
    body = """
    replay = period_replay.replay_period(directory=directory)
    print("SCOPE", replay.scope)
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "logical"


def test_a_recorded_donation_is_reinstated_and_replays_on_its_layout(tmp_path):
    """A core whose capture records a donation is lowered donating and still matches.

    The GridSearch route declares no donation candidates, so a production capture
    of it never records one. The donating descriptor is therefore constructed for
    this check and lowered through the same routine the layout route uses, which
    is what makes the variant comparison a live branch rather than a constant.
    """
    body = """
    import dataclasses

    def record_a_donation(payload):
        layouts = payload[period_capture.LAYOUTS_KEY]
        core = layouts.cores["main"]
        payload[period_capture.LAYOUTS_KEY] = dataclasses.replace(
            layouts,
            cores={
                "main": dataclasses.replace(
                    core,
                    donated_arguments=("next_regime_to_V_arr",),
                    variant=period_capture.DONATING_VARIANT,
                )
            },
        )

    rewrite(record_a_donation)
    replay = period_replay.replay_period_on_recorded_layout(
        directory=directory, devices=replay_devices()
    )
    print("DONATING", replay.scope)
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "layout"


def test_a_mismatched_recorded_variant_is_refused_naming_both_values(tmp_path):
    """A recorded variant the replay does not reproduce is refused, both named."""
    body = """
    import dataclasses

    def claim_a_donation_without_naming_one(payload):
        layouts = payload[period_capture.LAYOUTS_KEY]
        core = layouts.cores["main"]
        payload[period_capture.LAYOUTS_KEY] = dataclasses.replace(
            layouts,
            cores={
                "main": dataclasses.replace(
                    core,
                    donated_arguments=(),
                    variant=period_capture.DONATING_VARIANT,
                )
            },
        )

    rewrite(claim_a_donation_without_naming_one)
    try:
        period_replay.replay_period_on_recorded_layout(
            directory=directory, devices=replay_devices()
        )
    except ValueError as error:
        message = str(error)
    else:
        message = "NO-REFUSAL"
    print(
        "REFUSED",
        "'non_donating' variant" in message and "recorded 'donating'" in message,
    )
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "True"


def test_a_recorded_donation_reaches_the_compiled_executable(tmp_path):
    """The reinstated donation reaches the executable as donated buffers.

    `next_regime_to_V_arr` carries one value array per next regime, so donating
    that one argument donates both of its leaves.
    """
    body = """
    real = period_replay._compile_cores_for_one_period
    seen = []

    def record(**kwargs):
        cores = real(**kwargs)
        seen.append(cores["main"].compiled.donate_argnums)
        return cores

    period_replay._compile_cores_for_one_period = record

    def record_a_donation(payload):
        import dataclasses

        layouts = payload[period_capture.LAYOUTS_KEY]
        core = layouts.cores["main"]
        payload[period_capture.LAYOUTS_KEY] = dataclasses.replace(
            layouts,
            cores={
                "main": dataclasses.replace(
                    core,
                    donated_arguments=("next_regime_to_V_arr",),
                    variant=period_capture.DONATING_VARIANT,
                )
            },
        )

    rewrite(record_a_donation)
    period_replay.replay_period_on_recorded_layout(
        directory=directory, devices=replay_devices()
    )
    print("DONATED", len(seen[0]))
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "2"


@pytest.mark.parametrize(
    ("continuous", "dtype", "cells", "order"),
    [
        (False, "float64", 24, (0, 1, 2, 3)),
        (True, "float32", 16, (0, 1, 2, 3)),
        (True, "float64", 24, (0, 1, 2, 3)),
        (False, "float32", 16, (0, 2, 1, 3)),
        (False, "float64", 24, (0, 2, 1, 3)),
        (False, "float64", 24, (0, 3, 2, 1)),
    ],
)
def test_capture_preserves_consumer_layout_under_device_substitution(
    *, tmp_path: Path, continuous: bool, dtype: str, cells: int, order: tuple[int, ...]
) -> None:
    """Replay genuine captures with full continuations and renamed physical devices."""
    body = f"""
    from dataclasses import replace
    from lcm import (
        DiscreteGrid, ExecutionConfig, LinSpacedGrid, Model, fixed_transition,
    )
    from lcm.solvers import GridSearch
    from tests.test_models.nbegm_common import (
        RegimeId, feasible, make_alive_dead_model, next_liquid_from_savings,
        savings, utility,
    )

    jax.config.update("jax_enable_x64", {dtype == "float64"!r})
    if {continuous!r}:
        liquid = LinSpacedGrid(start=0.1, stop=30.0, n_points={cells})
        template = make_alive_dead_model(
            n_periods=4, n_liquid={cells}, liquid_max=30.0, n_consumption=16,
            alive_functions={{"utility": utility, "tax": toy.tax,
                             "resources": toy.resources, "savings": savings}},
            liquid_law=next_liquid_from_savings, alive_solver=GridSearch(),
            constraints={{"feasible": feasible}},
            extra_states={{"kind": DiscreteGrid(category_class=toy.ConsumerKind)}},
            extra_state_transitions={{"kind": {{"alive": fixed_transition("kind")}}}},
            liquid_grid=liquid,
        )
        model = Model(
            regimes={{
                name: replace(regime, states={{
                    key: grid for key, grid in regime.states.items() if key != "liquid"
                }})
                for name, regime in template.user_regimes.items()
            }},
            states={{"liquid": liquid}}, ages=template.ages,
            regime_id_class=RegimeId,
            execution_config=ExecutionConfig(sharded_states=("liquid",)),
        )
    else:
        model = toy.build_model(
            variant="brute", n_periods=4, n_liquid={cells},
            n_consumption=16, n_savings=32, distributed_kind=True,
        )
    solution = model.solve(params=toy.build_params(), log_level="off")
    replay = period_replay.replay_period_on_recorded_layout(
        directory=directory, devices=replay_devices({order!r})
    )
    observed = np.asarray(replay.output.value)
    assert observed.dtype == np.dtype({dtype!r})
    np.testing.assert_array_equal(observed, np.asarray(solution.values[1]["alive"]))
    assert replay.scope == "layout"
    print("CONTEXT-MATCH")
    """
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "CONTEXT-MATCH"


_TWO_MESH_SOLVE = """
import dataclasses

import jax.numpy as jnp
from lcm import DiscreteGrid, ExecutionConfig, IrregSpacedGrid, Model
from tests.test_models.nbegm_common import RegimeId

# Under a memory budget, `simulate` commits the parameters to the subject mesh
# before it solves, while the regime keeps its own mesh named after its
# distributed state (`kind`). The captured period's inputs therefore span two
# meshes over the same devices under different axis names. The consumption grid
# takes its points at runtime, so the state-action space carries a parameter
# array on the subject mesh.
two_mesh = root / "two_mesh"
os.environ["LCM_CAPTURE_DIR"] = str(two_mesh)
template = toy.build_model(
    variant="brute",
    n_periods=4,
    n_liquid=24,
    n_consumption=16,
    n_savings=32,
    distributed_kind=True,
)
two_mesh_params = toy.build_params()
two_mesh_params["alive"]["consumption"] = {"points": jnp.linspace(0.1, 30.0, 16)}
simulated = Model(
    regimes={
        name: dataclasses.replace(
            regime,
            states={key: grid for key, grid in regime.states.items() if key != "kind"},
            actions={
                key: IrregSpacedGrid(n_points=16) if key == "consumption" else grid
                for key, grid in regime.actions.items()
            },
        )
        for name, regime in template.user_regimes.items()
    },
    states={"kind": DiscreteGrid(category_class=toy.ConsumerKind)},
    ages=template.ages,
    regime_id_class=RegimeId,
    execution_config=ExecutionConfig(
        sharded_states=("kind",), devices=(0, 1), device_memory_bytes=2**31
    ),
).simulate(
    params=two_mesh_params,
    initial_conditions={
        "liquid": jnp.array([5.0, 10.0]),
        "kind": jnp.array([0, 1]),
        "age": jnp.zeros(2),
        "regime_id": jnp.array([0, 0]),
    },
    log_level="off",
)
two_mesh_directory = two_mesh / "alive@1"
two_mesh_payload = two_mesh_directory / period_capture._PAYLOAD_NAME
with two_mesh_payload.open("rb") as stream:
    two_mesh_layouts = cloudpickle.load(stream)[period_capture.LAYOUTS_KEY]
mesh_names = {
    leaf.sharding.mesh_axis_names
    for leaf in two_mesh_layouts.leaves
    if leaf.sharding.mesh_axis_names is not None
}
assert len(mesh_names) == 2 and ("kind",) in mesh_names, mesh_names
two_mesh_devices = jax.devices()[: len(two_mesh_layouts.device_ids)]
"""


def test_a_capture_spanning_two_named_meshes_replays_under_the_strict_check(tmp_path):
    """Inputs on two same-device meshes with different axis names replay exactly.

    The replay passes the strict layout comparison and returns the value array
    the solve published, bit for bit.
    """
    body = _TWO_MESH_SOLVE + textwrap.dedent(
        """
    replay = period_replay.replay_period_on_recorded_layout(
        directory=two_mesh_directory, devices=two_mesh_devices
    )
    np.testing.assert_array_equal(
        np.asarray(replay.output.value),
        np.asarray(simulated.period_to_regime_to_V_arr[1]["alive"]),
    )
    print("STRICT-REPLAY", replay.scope)
    """
    )
    assert _stdout(body=body, tmp_path=tmp_path).split()[-2:] == [
        "STRICT-REPLAY",
        "layout",
    ]


def test_a_two_mesh_capture_with_a_different_recorded_input_spec_is_refused(tmp_path):
    """A recorded input partition spec the replay does not reproduce is refused."""
    body = _TWO_MESH_SOLVE + textwrap.dedent(
        """
    with two_mesh_payload.open("rb") as stream:
        payload = cloudpickle.load(stream)
    core = payload[period_capture.LAYOUTS_KEY].cores["main"]
    sharded = [
        index
        for index, (_, descriptor) in enumerate(core.compiled_input_shardings)
        if descriptor.partition_spec == ("kind",)
    ]
    assert sharded, core.compiled_input_shardings
    inputs = list(core.compiled_input_shardings)
    name, descriptor = inputs[sharded[0]]
    inputs[sharded[0]] = (name, dataclasses.replace(descriptor, partition_spec=()))
    payload[period_capture.LAYOUTS_KEY] = dataclasses.replace(
        payload[period_capture.LAYOUTS_KEY],
        cores={
            "main": dataclasses.replace(core, compiled_input_shardings=tuple(inputs))
        },
    )
    with two_mesh_payload.open("wb") as stream:
        cloudpickle.dump(payload, stream)
    try:
        period_replay.replay_period_on_recorded_layout(
            directory=two_mesh_directory, devices=two_mesh_devices
        )
    except ValueError as error:
        message = str(error)
    else:
        message = "NO-REFUSAL"
    print("REFUSED", name in message and "partition_spec=()" in message)
    """
    )
    assert _stdout(body=body, tmp_path=tmp_path).split()[-1] == "True"
