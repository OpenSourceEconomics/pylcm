"""Neutral ownership-aware partitions shared by grids and NBEGM."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.axis_boundaries import (
    AxisBoundary,
    axis_interval_indices,
    boundary_owner_for_feasible_region,
    effect_code,
    partition_effect_for_schedule_kind,
    resolve_axis_partition,
)


def test_resolved_partition_sorts_values_owners_and_effects_together() -> None:
    """Every boundary attribute follows its value into sorted liquid-axis order."""
    partition = resolve_axis_partition(
        start=jnp.float32(0.0),
        stop=jnp.float32(10.0),
        boundaries=(
            AxisBoundary(value=jnp.float32(7.0), owner="left", effect="feasibility"),
            AxisBoundary(
                value=jnp.float32(3.0),
                owner="right",
                effect="continuous_kink",
            ),
            AxisBoundary(
                value=jnp.float32(5.0),
                owner="left",
                effect="flat_budget",
            ),
        ),
    )

    np.testing.assert_array_equal(partition.values, np.array([3.0, 5.0, 7.0]))
    np.testing.assert_array_equal(
        partition.effect_codes,
        np.array(
            [
                effect_code("continuous_kink"),
                effect_code("flat_budget"),
                effect_code("feasibility"),
            ]
        ),
    )
    np.testing.assert_array_equal(
        axis_interval_indices(
            partition=partition,
            values=jnp.array([3.0, 5.0, 7.0]),
        ),
        np.array([1, 1, 2]),
    )


def test_resolved_partition_uses_one_ulp_for_each_open_side() -> None:
    """Effective bounds retain the threshold only on its declared owner."""
    partition = resolve_axis_partition(
        start=jnp.float32(0.0),
        stop=jnp.float32(10.0),
        boundaries=(
            AxisBoundary(value=jnp.float32(3.0), owner="right", effect="feasibility"),
            AxisBoundary(value=jnp.float32(7.0), owner="left", effect="feasibility"),
        ),
    )

    below_three = jnp.nextafter(jnp.float32(3.0), -jnp.inf)
    above_seven = jnp.nextafter(jnp.float32(7.0), jnp.inf)
    np.testing.assert_array_equal(
        partition.effective_starts,
        np.array([0.0, 3.0, above_seven], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        partition.effective_stops,
        np.array([below_three, 7.0, 10.0], dtype=np.float32),
    )


def test_coincident_opposite_owners_create_an_equality_only_interval() -> None:
    """An exact shared threshold remains distinct from both open neighbours."""
    boundary = jnp.float32(4.0)
    partition = resolve_axis_partition(
        start=jnp.float32(0.0),
        stop=jnp.float32(10.0),
        boundaries=(
            AxisBoundary(value=boundary, owner="right", effect="continuous_kink"),
            AxisBoundary(value=boundary, owner="left", effect="feasibility"),
        ),
    )
    values = jnp.array(
        [
            jnp.nextafter(boundary, -jnp.inf),
            boundary,
            jnp.nextafter(boundary, jnp.inf),
        ]
    )

    np.testing.assert_array_equal(
        axis_interval_indices(partition=partition, values=values),
        np.array([0, 1, 2]),
    )


@pytest.mark.parametrize(
    ("feasible_side", "includes_boundary", "owner"),
    [
        ("below", True, "left"),
        ("below", False, "right"),
        ("above", True, "right"),
        ("above", False, "left"),
    ],
)
def test_feasibility_comparison_selects_the_equality_owner(
    feasible_side, includes_boundary, owner
) -> None:
    """The feasible half-space owns equality exactly when its operator includes it."""
    assert (
        boundary_owner_for_feasible_region(
            feasible_side=feasible_side,
            includes_boundary=includes_boundary,
        )
        == owner
    )


@pytest.mark.parametrize(
    ("schedule_kind", "partition_effect"),
    [
        ("continuous_kink", "continuous_kink"),
        ("jump", "jump"),
        ("hard_constraint", "flat_budget"),
    ],
)
def test_schedule_boundary_kinds_normalize_to_partition_effects(
    schedule_kind, partition_effect
) -> None:
    """A flat budget is distinct from an externally compiled feasibility region."""
    assert partition_effect_for_schedule_kind(schedule_kind) == partition_effect


def test_partition_interval_lookup_supports_jit_and_vmap() -> None:
    """Resolved ownership is usable inside scalar and batched JAX programs."""
    partition = resolve_axis_partition(
        start=jnp.float32(0.0),
        stop=jnp.float32(10.0),
        boundaries=(
            AxisBoundary(value=jnp.float32(4.0), owner="left", effect="feasibility"),
        ),
    )
    lookup = jax.jit(
        jax.vmap(lambda value: axis_interval_indices(partition=partition, values=value))
    )

    np.testing.assert_array_equal(
        lookup(jnp.array([3.0, 4.0, 5.0])),
        np.array([0, 0, 1]),
    )
