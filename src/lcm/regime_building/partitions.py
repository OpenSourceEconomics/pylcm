"""Partition-dimension iteration and result stacking.

A partition dimension is a state declared via `state_transitions[name] = None`.
The Bellman backward induction never couples values across such a dimension,
so instead of vectorising over the partition axis (doubling memory per added
partition dim), pylcm compiles the reduced sub-model once and runs it once
per point in the Cartesian product of all partition grids.

This module provides the small set of helpers that model.solve / model.simulate
use to iterate over partition points, inject partition-scalar values into
`internal_params`, group subjects by their partition values at simulate time,
and stack sub-solve V-arrays back into the user-visible shape.

All helpers handle the empty-partition case transparently: a single iteration
with an empty scalar dict, no subject grouping, no axis stacking.
"""

from collections.abc import Iterator, Mapping
from itertools import product
from types import MappingProxyType

import jax.numpy as jnp
from jax import Array

from lcm.grids import DiscreteGrid
from lcm.typing import FloatND, InternalParams, RegimeName, ScalarInt

# Mapping of partition-dimension name to its discrete grid.
PartitionGrid = Mapping[str, DiscreteGrid]

# One concrete partition value per partition-dimension name.
PartitionPoint = Mapping[str, ScalarInt]


def iterate_partition_points(
    partition_grid: PartitionGrid,
) -> Iterator[PartitionPoint]:
    """Yield each point in the Cartesian product of partition grids.

    Empty `partition_grid` yields a single empty dict so callers can loop
    unconditionally.

    Args:
        partition_grid: Mapping of partition names to their discrete grids.

    Yields:
        Immutable mapping of partition name to scalar integer code for each
        point in the product.

    """
    if not partition_grid:
        yield MappingProxyType({})
        return

    names = tuple(partition_grid)
    codes_per_name = tuple(partition_grid[name].codes for name in names)
    for codes in product(*codes_per_name):
        yield MappingProxyType(
            {name: jnp.int32(code) for name, code in zip(names, codes, strict=True)}
        )


def inject_partition_scalars(
    *,
    internal_params: InternalParams,
    partition_point: PartitionPoint,
) -> InternalParams:
    """Return a copy of `internal_params` with partition scalars injected.

    Each partition value is added to every regime's flat params so that the
    compiled Q_and_F / max_Q_over_a closure receives it as a kwarg (the same
    channel user-supplied scalars flow through). No-op when the point is empty.

    Args:
        internal_params: Existing internal params, keyed by regime name.
        partition_point: Partition scalars to inject.

    Returns:
        New internal-params mapping with partition scalars added to each
        regime. The original is unchanged.

    """
    if not partition_point:
        return internal_params
    return MappingProxyType(
        {
            regime_name: MappingProxyType({**regime_params, **partition_point})
            for regime_name, regime_params in internal_params.items()
        }
    )


def stack_partition_V_arrays(
    *,
    sub_V_arrays: list[MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]],
    partition_grid: PartitionGrid,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Stack per-partition V-arrays along trailing axes.

    Partition axes are appended after all existing discrete/continuous state
    axes, in `partition_grid` insertion order. When `partition_grid` is empty,
    `sub_V_arrays` has a single entry and is returned as-is.

    Args:
        sub_V_arrays: One `period → regime → V` mapping per partition point,
            in the order produced by `iterate_partition_points`.
        partition_grid: Partition grids defining axis shape and order.

    Returns:
        A single `period → regime → V` mapping whose V-arrays carry the
        partition axes at the end.

    """
    if not partition_grid:
        (single,) = sub_V_arrays
        return single

    names = tuple(partition_grid)
    shape = tuple(len(partition_grid[name].codes) for name in names)

    periods = sub_V_arrays[0].keys()
    regimes = sub_V_arrays[0][next(iter(periods))].keys()

    stacked: dict[int, dict[RegimeName, FloatND]] = {}
    for period in periods:
        stacked[period] = {}
        for regime in regimes:
            per_point = [sub[period][regime] for sub in sub_V_arrays]
            stacked_arr = jnp.stack(per_point, axis=0).reshape(
                shape + per_point[0].shape
            )
            # Move the prepended partition axes to the end so the convention
            # is [non-partition state/action axes..., partition axes...].
            n_partition = len(shape)
            axes_in = tuple(range(stacked_arr.ndim))
            axes_out = axes_in[n_partition:] + axes_in[:n_partition]
            stacked[period][regime] = jnp.transpose(stacked_arr, axes_out)

    return MappingProxyType(
        {
            period: MappingProxyType(regime_dict)
            for period, regime_dict in stacked.items()
        }
    )


def group_subjects_by_partition(
    *,
    initial_conditions: Mapping[str, Array],
    partition_grid: PartitionGrid,
) -> Iterator[tuple[PartitionPoint, Array]]:
    """Yield (partition_point, subject_mask) for each partition group.

    Subjects are grouped by the tuple of their partition-state values read
    from `initial_conditions`. Empty `partition_grid` yields a single group
    containing all subjects.

    Args:
        initial_conditions: Flat mapping that must contain one array per
            partition name (integer category codes).
        partition_grid: Partition grids defining which columns to read.

    Yields:
        Tuples of (partition scalars, boolean subject mask).

    """
    if not partition_grid:
        n_subjects = initial_conditions["regime"].shape[0]
        yield MappingProxyType({}), jnp.ones(n_subjects, dtype=bool)
        return

    for point in iterate_partition_points(partition_grid):
        mask = jnp.ones_like(initial_conditions["regime"], dtype=bool)
        for name, scalar in point.items():
            mask = mask & (initial_conditions[name] == scalar)
        yield point, mask
