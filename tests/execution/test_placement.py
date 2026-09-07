"""Submesh placement: which devices each regime's nodes run on."""

import pytest

from _lcm.execution.placement import (
    PlacementRequest,
    SubmeshPlacement,
    mesh_size_for_extents,
    plan_submesh_placement,
)
from lcm.exceptions import PyLCMError


def _request(
    *,
    name: str,
    extents: tuple[int, ...] = (),
    active: tuple[int, ...] = (0, 1, 2),
    template_bytes: int = 8,
) -> PlacementRequest:
    return PlacementRequest(
        regime_name=name,
        distributed_extents=extents,
        active_periods=active,
        template_bytes=template_bytes,
    )


@pytest.mark.parametrize(
    ("extents", "n_devices", "expected"),
    [
        ((3,), 4, 3),
        ((4,), 4, 4),
        ((6,), 4, 3),
        ((8,), 4, 4),
        ((5,), 4, 1),
        ((2, 2), 4, 4),
        ((3,), 1, 1),
    ],
)
def test_mesh_size_is_the_largest_divisor_of_the_extent_that_fits(
    *, extents: tuple[int, ...], n_devices: int, expected: int
) -> None:
    """One distributed grid runs on the largest divisor of its extent that fits."""
    assert mesh_size_for_extents(extents=extents, n_devices=n_devices) == expected


def test_a_product_of_extents_beyond_the_devices_is_refused() -> None:
    """Several distributed grids scatter one point per device and need them all."""
    with pytest.raises(PyLCMError, match="must equal the number"):
        mesh_size_for_extents(extents=(4, 4), n_devices=4)


def test_a_three_valued_type_runs_on_three_of_four_devices() -> None:
    """The motivating case: extent three on four devices leaves one device idle."""
    placement = plan_submesh_placement(
        requests=(_request(name="working", extents=(3,)),), n_devices=4
    )

    assert placement.devices_for(regime_name="working") == (0, 1, 2)


def test_an_idle_device_is_filled_with_a_co_active_single_device_regime() -> None:
    """A single-device regime active beside the sharded one takes the idle device."""
    placement = plan_submesh_placement(
        requests=(
            _request(name="working", extents=(3,)),
            _request(name="retired"),
        ),
        n_devices=4,
    )

    assert placement.devices_for(regime_name="retired") == (3,)


def test_a_regime_never_co_active_with_another_stays_on_device_zero() -> None:
    """One regime per period is placed as a single-device solve is."""
    placement = plan_submesh_placement(
        requests=(
            _request(name="working", active=(0, 1)),
            _request(name="retired", active=(2, 3)),
        ),
        n_devices=4,
    )

    assert (
        placement.devices_for(regime_name="working"),
        placement.devices_for(regime_name="retired"),
    ) == ((0,), (0,))


def test_two_co_active_single_device_regimes_take_two_devices() -> None:
    """Independent single-device regimes of one period fill idle devices."""
    placement = plan_submesh_placement(
        requests=(_request(name="a"), _request(name="b")), n_devices=4
    )

    assert (
        placement.devices_for(regime_name="a"),
        placement.devices_for(regime_name="b"),
    ) == ((0,), (1,))


def test_without_an_idle_device_the_smallest_footprint_wins() -> None:
    """When the mesh covers every device, single nodes go where the least is planned."""
    placement = plan_submesh_placement(
        requests=(
            _request(name="sharded", extents=(4,), template_bytes=400),
            _request(name="first", template_bytes=50),
            _request(name="second", template_bytes=10),
        ),
        n_devices=4,
    )

    assert (
        placement.devices_for(regime_name="first"),
        placement.devices_for(regime_name="second"),
    ) == ((0,), (1,))


def test_a_full_mesh_and_device_zero_is_the_canonical_placement() -> None:
    """A plan every regime of which sits where a single-device solve puts it."""
    placement = plan_submesh_placement(
        requests=(
            _request(name="sharded", extents=(4,)),
            _request(name="single", active=(3,)),
        ),
        n_devices=4,
    )

    assert placement.is_canonical


def test_a_submesh_placement_is_not_canonical() -> None:
    """A regime on three of four devices departs from the single-device layout."""
    placement = plan_submesh_placement(
        requests=(_request(name="sharded", extents=(3,)),), n_devices=4
    )

    assert not placement.is_canonical


def test_the_placement_key_lists_every_regime_s_devices() -> None:
    """The key a lowering carries names each regime's device ids."""
    placement = plan_submesh_placement(
        requests=(_request(name="working", extents=(3,)), _request(name="retired")),
        n_devices=4,
    )

    assert placement.key == (("retired", (3,)), ("working", (0, 1, 2)))


def test_two_sharded_regimes_of_half_the_devices_take_disjoint_blocks() -> None:
    """Blocks are consecutive in declaration order while a full block is left."""
    placement = plan_submesh_placement(
        requests=(_request(name="a", extents=(2,)), _request(name="b", extents=(2,))),
        n_devices=4,
    )

    assert (
        placement.devices_for(regime_name="a"),
        placement.devices_for(regime_name="b"),
    ) == ((0, 1), (2, 3))


def test_a_single_device_yields_device_zero_for_every_regime() -> None:
    """On one device every regime, sharded or not, is placed on it."""
    placement = plan_submesh_placement(
        requests=(_request(name="a", extents=(3,)), _request(name="b")), n_devices=1
    )

    assert placement == SubmeshPlacement(
        device_ids_by_regime={"a": (0,), "b": (0,)}, n_devices=1
    )
