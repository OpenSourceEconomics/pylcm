"""The per-device bytes the plan keeps resident at each unit's position.

The single-artifact arithmetic and the walk of the schedule are tested
separately. Every claim about a genuine mesh runs in a subprocess with forced
host devices, so the per-device numbers are measured rather than assumed.
"""

import os
import subprocess
import sys
import textwrap
from collections.abc import Hashable
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.footprint import (
    ArtifactFootprint,
    ScheduledUnit,
    per_device_footprint,
    plan_resident_bytes,
    sharding_device_ids,
)
from _lcm.execution.liveness import PlannedInputLiveness
from lcm.exceptions import ExecutionPlanningError

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SHARDED_SCRIPT = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp

    from _lcm.execution.footprint import ArtifactFootprint, per_device_footprint

    assert jax.device_count() == 2, jax.devices()
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("kind",))
    sharded = jax.NamedSharding(mesh, jax.P("kind", None))
    array = jax.device_put(jnp.zeros((4, 8), dtype=jnp.int8), sharded)

    expected = ArtifactFootprint(
        bytes_per_device=2 * 8,
        device_ids=(devices[0].id, devices[1].id),
    )
    got = per_device_footprint(array=array)
    assert got == expected, (got, expected)
    print("FOOTPRINT-SHARDING-OK")
    """
)

_MESH_SCRIPT = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from _lcm.execution.footprint import (
        ArtifactFootprint,
        layout_footprint,
        per_device_footprint,
    )
    from lcm.exceptions import ExecutionPlanningError

    assert jax.device_count() == 4, jax.devices()
    devices = jax.devices()

    pair = jax.sharding.Mesh(devices[:2], ("kind",))
    replicated = jax.device_put(
        jnp.zeros((4, 8), dtype=jnp.int8), jax.NamedSharding(pair, jax.P())
    )
    expected = ArtifactFootprint(
        bytes_per_device=4 * 8,
        device_ids=tuple(sorted(device.id for device in devices[:2])),
    )
    got = per_device_footprint(array=replicated)
    assert got == expected, (got, expected)
    print("FOOTPRINT-REPLICATED-OK")

    square = jax.sharding.Mesh(np.asarray(devices).reshape(2, 2), ("x", "y"))
    partial = jax.device_put(
        jnp.zeros((4, 6), dtype=jnp.int8),
        jax.NamedSharding(square, jax.P("x", None)),
    )
    expected = ArtifactFootprint(
        bytes_per_device=2 * 6,
        device_ids=tuple(sorted(device.id for device in devices)),
    )
    got = per_device_footprint(array=partial)
    assert got == expected, (got, expected)
    print("FOOTPRINT-PARTIALLY-REPLICATED-OK")

    uneven = jax.NamedSharding(square, jax.P("x", None))
    try:
        uneven.shard_shape((3, 8))
    except ValueError as raw:
        raw_text = str(raw)
    else:
        raise AssertionError("shard_shape accepted a shape it cannot divide.")

    try:
        layout_footprint(sharding=uneven, shape=(3, 8), item_bytes=4)
    except ExecutionPlanningError as error:
        assert "(3, 8)" in str(error), str(error)
        assert raw_text in str(error), (raw_text, str(error))
        print("FOOTPRINT-INDIVISIBLE-OK")
    else:
        raise AssertionError("An indivisible layout was not refused.")
    """
)


def _run_forced_devices(*, script: str, device_count: int) -> str:
    """Run one script under forced host devices and return its combined output."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env={
            **os.environ,
            "XLA_FLAGS": (f"--xla_force_host_platform_device_count={device_count}"),
            "JAX_PLATFORMS": "cpu",
        },
        check=False,
        timeout=600,
    )
    output = f"{result.stdout}\n{result.stderr[-4000:]}"
    # A crashed script leaves no marker, so every claim below it would read as a
    # plain assertion failure; the exit status says which of the two it was.
    assert result.returncode == 0, output
    return output


@pytest.fixture(scope="module")
def mesh_script_output() -> str:
    """Report the output of the four-device mesh arithmetic script."""
    return _run_forced_devices(script=_MESH_SCRIPT, device_count=4)


def _unit(
    *,
    period: int,
    regime: str,
    devices: tuple[int, ...] = (0,),
    produces: tuple[Hashable, ...] = (),
    consumes: tuple[Hashable, ...] = (),
    output_bytes: int = 0,
) -> ScheduledUnit:
    """Build one scheduled unit with the defaults the walk tests share."""
    return ScheduledUnit(
        period=period,
        regime=regime,
        device_ids=devices,
        produces=produces,
        consumes=consumes,
        output_bytes_per_device=output_bytes,
    )


def _footprint(*, size: int, devices: tuple[int, ...] = (0,)) -> ArtifactFootprint:
    """Build one artifact footprint of the given size on the given devices."""
    return ArtifactFootprint(bytes_per_device=size, device_ids=devices)


def test_per_device_footprint_of_a_single_device_array_is_its_size() -> None:
    """An unsharded array is resident in full on its one device."""
    array = jnp.zeros((4, 8))

    assert per_device_footprint(array=array) == ArtifactFootprint(
        bytes_per_device=4 * 8 * array.dtype.itemsize,
        device_ids=(jax.devices()[0].id,),
    )


def test_per_device_footprint_multiplies_the_element_count_by_the_item_size() -> None:
    """The footprint is a byte count, not an element count."""
    array = jnp.zeros((3, 5), dtype=jnp.int8)

    assert per_device_footprint(array=array).bytes_per_device == 15


def test_per_device_footprint_of_a_zero_dimensional_array_is_one_element() -> None:
    """A scalar occupies exactly one element's bytes on its device."""
    array = jnp.asarray(2.0, dtype=jnp.float32)

    assert per_device_footprint(array=array).bytes_per_device == 4


def test_sharding_device_ids_are_reported_in_ascending_order() -> None:
    """The devices of a layout are named once each, lowest id first."""
    array = jnp.zeros((2, 2))

    assert sharding_device_ids(sharding=array.sharding) == (jax.devices()[0].id,)


def test_per_device_footprint_of_a_sharded_array_reports_one_shard() -> None:
    """A mesh-sharded array costs its shard, not its whole size, on each device."""
    output = _run_forced_devices(script=_SHARDED_SCRIPT, device_count=2)

    assert "FOOTPRINT-SHARDING-OK" in output


def test_a_replicated_array_costs_its_full_size_on_every_device(
    mesh_script_output: str,
) -> None:
    """Replication buys no space: each device holds the whole array."""
    assert "FOOTPRINT-REPLICATED-OK" in mesh_script_output


def test_a_partially_replicated_array_costs_a_shard_on_every_mesh_device(
    mesh_script_output: str,
) -> None:
    """An unsharded mesh axis replicates the shard rather than splitting it."""
    assert "FOOTPRINT-PARTIALLY-REPLICATED-OK" in mesh_script_output


def test_a_layout_whose_shape_does_not_divide_is_refused(
    mesh_script_output: str,
) -> None:
    """A shape no mesh axis divides evenly is a planning failure, named."""
    assert "FOOTPRINT-INDIVISIBLE-OK" in mesh_script_output


@pytest.mark.parametrize(
    ("bytes_per_device", "device_ids", "match"),
    [
        (8, (), "at least one device"),
        (8, (0, 0), "repeated device"),
        (-1, (0,), "cannot be negative"),
    ],
)
def test_an_invalid_artifact_footprint_is_refused(
    *, bytes_per_device: int, device_ids: tuple[int, ...], match: str
) -> None:
    """A footprint names a positive count of bytes on a set of distinct devices."""
    with pytest.raises(ValueError, match=match):
        ArtifactFootprint(bytes_per_device=bytes_per_device, device_ids=device_ids)


@pytest.mark.parametrize(
    ("output_bytes_per_device", "device_ids", "match"),
    [
        (0, (), "at least one device"),
        (0, (1, 1), "repeated device"),
        (-1, (0,), "cannot be negative"),
    ],
)
def test_an_invalid_scheduled_unit_is_refused(
    *, output_bytes_per_device: int, device_ids: tuple[int, ...], match: str
) -> None:
    """A unit names a positive output size and the distinct devices it runs on."""
    with pytest.raises(ValueError, match=match):
        ScheduledUnit(
            period=0,
            regime="a",
            device_ids=device_ids,
            produces=(),
            consumes=(),
            output_bytes_per_device=output_bytes_per_device,
        )


def test_a_retained_value_stays_resident_for_every_earlier_period() -> None:
    """Retained outputs accumulate backward, so period 0 sees every later value."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            (2, "a"): (),
            (1, "a"): ("V2",),
            (0, "a"): ("V1",),
        },
        retained_artifacts=("V2", "V1", "V0"),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                2: ((_unit(period=2, regime="a", produces=("V2",)),),),
                1: ((_unit(period=1, regime="a", produces=("V1",)),),),
                0: ((_unit(period=0, regime="a", produces=("V0",)),),),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType(
            {
                "V2": _footprint(size=10),
                "V1": _footprint(size=10),
                "V0": _footprint(size=10),
            }
        ),
    )

    assert (resident[(2, "a")], resident[(1, "a")], resident[(0, "a")]) == (0, 10, 20)


def test_a_released_artifact_leaves_the_footprint_after_its_last_consumer() -> None:
    """An unretained input is gone once the unit that reads it has committed."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(2, "a"): (), (1, "a"): ("leaf2",), (0, "a"): ()}
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                2: ((_unit(period=2, regime="a", produces=("leaf2",)),),),
                1: ((_unit(period=1, regime="a"),),),
                0: ((_unit(period=0, regime="a"),),),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType({"leaf2": _footprint(size=7)}),
    )

    assert (resident[(1, "a")], resident[(0, "a")]) == (7, 0)


def test_a_pinned_artifact_stays_resident_after_its_last_consumer() -> None:
    """A pin names consumers no dispatch declares, so zero does not release."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(1, "a"): ("leaf1",), (0, "a"): ()},
        pinned_artifacts=("leaf1",),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                1: ((_unit(period=1, regime="a", produces=("leaf1",)),),),
                0: ((_unit(period=0, regime="a"),),),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType({"leaf1": _footprint(size=7)}),
    )

    assert resident[(0, "a")] == 7


def test_an_alias_group_is_resident_until_its_last_member_closes() -> None:
    """One rolled buffer is released only when every key on it reaches zero."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(1, "a"): ("rolled",), (0, "a"): ("root",)},
        aliases=MappingProxyType({"rolled": "root"}),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                1: ((_unit(period=1, regime="a", produces=("rolled",)),),),
                0: ((_unit(period=0, regime="a"),),),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType({"rolled": _footprint(size=7)}),
    )

    assert resident[(0, "a")] == 7


def test_two_keys_of_one_alias_group_are_sized_as_one_buffer() -> None:
    """One shared buffer costs its largest key's bytes, never their sum."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(1, "a"): (), (0, "a"): ("rolled", "root")},
        aliases=MappingProxyType({"rolled": "root"}),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                1: ((_unit(period=1, regime="a", produces=("rolled", "root")),),),
                0: ((_unit(period=0, regime="a"),),),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType(
            {"rolled": _footprint(size=10), "root": _footprint(size=6)}
        ),
    )

    assert resident[(0, "a")] == 10


def test_a_shared_buffer_is_sized_once_and_gone_after_its_last_consumer() -> None:
    """The one buffer both keys name costs its largest key until both close."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(2, "a"): (), (1, "a"): ("rolled", "root"), (0, "a"): ()},
        aliases=MappingProxyType({"rolled": "root"}),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                2: ((_unit(period=2, regime="a", produces=("rolled", "root")),),),
                1: ((_unit(period=1, regime="a"),),),
                0: ((_unit(period=0, regime="a"),),),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType(
            {"rolled": _footprint(size=10), "root": _footprint(size=6)}
        ),
    )

    assert (resident[(1, "a")], resident[(0, "a")]) == (10, 0)


def test_a_shared_buffer_is_sized_per_device_by_the_key_present_there() -> None:
    """Each device pays its largest present key: both on one, one on the other."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            (1, "producer"): (),
            (0, "both"): ("rolled",),
            (0, "one"): ("root",),
        },
        aliases=MappingProxyType({"rolled": "root"}),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                1: (
                    (
                        _unit(
                            period=1,
                            regime="producer",
                            devices=(0, 1),
                            produces=("rolled", "root"),
                        ),
                    ),
                ),
                0: (
                    (
                        _unit(period=0, regime="both", devices=(0, 1)),
                        _unit(period=0, regime="one", devices=(1,)),
                    ),
                ),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType(
            {
                "rolled": _footprint(size=10, devices=(0,)),
                "root": _footprint(size=6, devices=(0, 1)),
            }
        ),
    )

    assert (resident[(0, "both")], resident[(0, "one")]) == (10, 6)


def test_a_concurrent_unit_on_the_same_device_counts_its_outputs() -> None:
    """Two units of one wave on one device each see the other's outputs."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(0, "a"): (), (0, "b"): ()}, retained_artifacts=("Va", "Vb")
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                0: (
                    (
                        _unit(period=0, regime="a", produces=("Va",), output_bytes=5),
                        _unit(period=0, regime="b", produces=("Vb",), output_bytes=3),
                    ),
                ),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType(
            {"Va": _footprint(size=5), "Vb": _footprint(size=3)}
        ),
    )

    assert (resident[(0, "a")], resident[(0, "b")]) == (3, 5)


def test_a_concurrent_unit_on_another_device_counts_nothing() -> None:
    """Disjoint submeshes share no allocator, so a neighbour adds no bytes."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(0, "a"): (), (0, "b"): ()}, retained_artifacts=("Va", "Vb")
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                0: (
                    (
                        _unit(
                            period=0,
                            regime="a",
                            devices=(0,),
                            produces=("Va",),
                            output_bytes=5,
                        ),
                        _unit(
                            period=0,
                            regime="b",
                            devices=(1,),
                            produces=("Vb",),
                            output_bytes=3,
                        ),
                    ),
                ),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType(
            {
                "Va": _footprint(size=5, devices=(0,)),
                "Vb": _footprint(size=3, devices=(1,)),
            }
        ),
    )

    assert (resident[(0, "a")], resident[(0, "b")]) == (0, 0)


def test_a_sharded_unit_reports_the_busiest_of_its_devices() -> None:
    """A unit on several devices is bounded by the device with the most resident."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(1, "single"): (), (0, "sharded"): ()},
        retained_artifacts=("V1", "S"),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                1: (
                    (_unit(period=1, regime="single", devices=(1,), produces=("V1",)),),
                ),
                0: (
                    (
                        _unit(
                            period=0,
                            regime="sharded",
                            devices=(0, 1),
                            produces=("S",),
                        ),
                    ),
                ),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType(
            {
                "V1": _footprint(size=9, devices=(1,)),
                "S": _footprint(size=2, devices=(0, 1)),
            }
        ),
    )

    assert resident[(0, "sharded")] == 9


def test_a_fold_output_enters_the_footprint_at_its_period_end() -> None:
    """A gated continuation folded at a period is resident for the period below."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            (1, "tgt"): (),
            (1, "src", "tgt"): ("Vt1",),
            (0, "src"): ("W1",),
        },
        retained_artifacts=("Vt1",),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                1: ((_unit(period=1, regime="tgt", produces=("Vt1",)),),),
                0: ((_unit(period=0, regime="src"),),),
            }
        ),
        fold_dispatches=MappingProxyType({(1, "src", "tgt"): "W1"}),
        ledger=ledger,
        footprints=MappingProxyType(
            {"Vt1": _footprint(size=4), "W1": _footprint(size=6)}
        ),
    )

    assert resident[(0, "src")] == 10


def test_an_artifact_without_a_footprint_occupies_nothing() -> None:
    """A key the ledger counts but no template sizes adds no bytes."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(1, "a"): (), (0, "a"): ()}, retained_artifacts=("scalar",)
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                1: ((_unit(period=1, regime="a", produces=("scalar",)),),),
                0: ((_unit(period=0, regime="a"),),),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType({}),
    )

    assert resident[(0, "a")] == 0


def test_a_unit_of_an_unknown_period_is_refused() -> None:
    """Every wave's units must carry the period their mapping key names."""
    with pytest.raises(ValueError, match="period"):
        plan_resident_bytes(
            waves_by_period=MappingProxyType({1: ((_unit(period=0, regime="a"),),)}),
            fold_dispatches=MappingProxyType({}),
            ledger=PlannedInputLiveness(dispatch_accesses={}),
            footprints=MappingProxyType({}),
        )


def test_a_footprint_the_ledger_does_not_know_is_refused() -> None:
    """Only artifacts of the immutable plan have a lifetime the walk can read."""
    with pytest.raises(ExecutionPlanningError, match="'stray'"):
        plan_resident_bytes(
            waves_by_period=MappingProxyType({0: ((_unit(period=0, regime="a"),),)}),
            fold_dispatches=MappingProxyType({}),
            ledger=PlannedInputLiveness(dispatch_accesses={(0, "a"): ()}),
            footprints=MappingProxyType({"stray": _footprint(size=1)}),
        )


def test_a_unit_whose_dispatch_the_ledger_does_not_plan_is_refused() -> None:
    """Every unit of the schedule commits, so the ledger must plan its dispatch."""
    with pytest.raises(ExecutionPlanningError, match="unplanned"):
        plan_resident_bytes(
            waves_by_period=MappingProxyType(
                {0: ((_unit(period=0, regime="unplanned"),),)}
            ),
            fold_dispatches=MappingProxyType({}),
            ledger=PlannedInputLiveness(dispatch_accesses={(0, "a"): ()}),
            footprints=MappingProxyType({}),
        )


def test_a_buffer_the_unit_is_handed_as_an_argument_is_not_charged() -> None:
    """A compiler peak counts an executable's arguments, so the walk must not."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(1, "a"): (), (0, "a"): ("V1",)},
        retained_artifacts=("V1",),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                1: ((_unit(period=1, regime="a", produces=("V1",)),),),
                0: ((_unit(period=0, regime="a", consumes=("V1",)),),),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType({"V1": _footprint(size=11)}),
    )

    assert resident[(0, "a")] == 0


def test_a_buffer_another_unit_is_handed_stays_charged() -> None:
    """Only the measured unit's own arguments leave its number."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(1, "a"): (), (0, "a"): ("V1",), (0, "b"): ()},
        retained_artifacts=("V1",),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                1: ((_unit(period=1, regime="a", produces=("V1",)),),),
                0: (
                    (_unit(period=0, regime="a", consumes=("V1",)),),
                    (_unit(period=0, regime="b"),),
                ),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType({"V1": _footprint(size=11)}),
    )

    assert resident[(0, "b")] == 11


def test_an_argument_on_another_device_leaves_that_devices_charge_standing() -> None:
    """A buffer is only free where the unit is actually handed it."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={(1, "a"): (), (0, "a"): ("rolled", "root")},
        aliases={"rolled": "root"},
        retained_artifacts=("root",),
    )
    resident = plan_resident_bytes(
        waves_by_period=MappingProxyType(
            {
                1: ((_unit(period=1, regime="a", produces=("rolled", "root")),),),
                0: (
                    (
                        _unit(
                            period=0,
                            regime="a",
                            devices=(0, 1),
                            consumes=("rolled",),
                        ),
                    ),
                ),
            }
        ),
        fold_dispatches=MappingProxyType({}),
        ledger=ledger,
        footprints=MappingProxyType(
            {
                "rolled": _footprint(size=10, devices=(0,)),
                "root": _footprint(size=6, devices=(0, 1)),
            }
        ),
    )

    assert resident[(0, "a")] == 6
