"""The static per-device footprint of one solve-lifetime artifact.

The sharded case runs in a subprocess with two forced host devices, so a
genuine mesh exists on CPU and the per-device claim is measured rather than
assumed.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.footprint import (
    ArtifactFootprint,
    ScheduledUnit,
    per_device_footprint,
)

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


def test_per_device_footprint_of_a_sharded_array_reports_one_shard() -> None:
    """A mesh-sharded array costs its shard, not its whole size, on each device."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SHARDED_SCRIPT],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env={
            **os.environ,
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "JAX_PLATFORMS": "cpu",
        },
        check=False,
        timeout=600,
    )

    assert "FOOTPRINT-SHARDING-OK" in result.stdout, result.stderr[-4000:]


def test_an_artifact_footprint_on_no_device_is_refused() -> None:
    """A footprint names the devices its bytes are resident on."""
    with pytest.raises(ValueError, match="at least one device"):
        ArtifactFootprint(bytes_per_device=8, device_ids=())


def test_an_artifact_footprint_with_a_repeated_device_is_refused() -> None:
    """Each device holds one shard, so a device is named once."""
    with pytest.raises(ValueError, match="repeated device"):
        ArtifactFootprint(bytes_per_device=8, device_ids=(0, 0))


def test_a_negative_artifact_footprint_is_refused() -> None:
    """Resident bytes are a count."""
    with pytest.raises(ValueError, match="cannot be negative"):
        ArtifactFootprint(bytes_per_device=-1, device_ids=(0,))


def test_a_scheduled_unit_on_no_device_is_refused() -> None:
    """A dispatch unit names the devices it runs on."""
    with pytest.raises(ValueError, match="at least one device"):
        ScheduledUnit(
            period=0,
            regime="a",
            device_ids=(),
            produces=(),
            output_bytes_per_device=0,
        )


def test_a_scheduled_unit_with_a_repeated_device_is_refused() -> None:
    """A unit names each of its devices once."""
    with pytest.raises(ValueError, match="repeated device"):
        ScheduledUnit(
            period=0,
            regime="a",
            device_ids=(1, 1),
            produces=(),
            output_bytes_per_device=0,
        )


def test_a_scheduled_unit_with_negative_output_bytes_is_refused() -> None:
    """Output bytes are a count."""
    with pytest.raises(ValueError, match="cannot be negative"):
        ScheduledUnit(
            period=0,
            regime="a",
            device_ids=(0,),
            produces=(),
            output_bytes_per_device=-1,
        )
