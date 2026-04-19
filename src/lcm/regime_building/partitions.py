"""Partition-dimension detection, iteration, and result stacking.

A partition dimension is a state declared via `state_transitions[name] = None`
on a `DiscreteGrid`. The Bellman backward induction never couples values
across such a dimension, so instead of vectorising over the partition axis
(doubling memory per added partition dim), pylcm compiles the reduced
sub-model once and runs it once per point in the Cartesian product of all
partition grids.

The module exposes two layers:

- **Model-building** (called from `process_regimes`):
  `detect_model_partitions`, `lift_partitions_from_regime`.
- **Solve / simulate runtime** (called from `Model.solve` / `Model.simulate`):
  `iterate_partition_points`, `inject_partition_scalars`,
  `stack_partition_V_arrays`, `group_subjects_by_partition`,
  `slice_V_at_partition_point`, `slice_initial_conditions`.

All helpers handle the empty-partition case transparently: a single
iteration with an empty scalar dict, no subject grouping, no axis stacking.
"""

import dataclasses
from collections.abc import Iterator, Mapping
from itertools import product
from types import MappingProxyType

import jax.numpy as jnp
from jax import Array

from lcm.grids import DiscreteGrid
from lcm.regime import Regime
from lcm.typing import FloatND, InternalParams, RegimeName, ScalarInt

# Mapping of partition-dimension name to its discrete grid.
PartitionGrid = Mapping[str, DiscreteGrid]

# One concrete partition value per partition-dimension name.
PartitionPoint = Mapping[str, ScalarInt]


def detect_model_partitions(
    *,
    regimes: Mapping[str, Regime],
) -> MappingProxyType[str, DiscreteGrid]:
    """Identify states that qualify as partition dimensions.

    A partition dimension is a discrete state whose value truly never
    changes — not just within one regime, but across the entire model.
    The qualification is:

    1. The state's grid is a `DiscreteGrid`.
    2. Every regime that includes the state declares `state_transitions[name]
       = None` for it (never via a non-None transition or per-target dict).
    3. All such regimes agree on the `DiscreteGrid.categories`.

    Why all three: if any regime carries a non-identity transition for the
    state, the state changes in that regime, so it cannot be modelled as a
    fixed partition dimension. If categories differ across regimes, the
    state behaves like different things in different regimes — again not a
    partition. Continuous fixed states are never lifted (identity
    transitions remain the correct implementation for them).

    Args:
        regimes: Mapping of regime names to user-facing regimes.

    Returns:
        Immutable mapping of partition name to its `DiscreteGrid`. Empty
        when no state in the model meets the criteria above.

    """
    candidates: dict[str, DiscreteGrid] = {}
    disqualified: set[str] = set()
    # A partition value must come from initial_conditions, which implies the
    # state exists in at least one non-terminal regime where subjects can
    # start. Target-only states (declared only in terminal regimes, populated
    # by per-target transitions at the boundary) are therefore excluded.
    seen_in_non_terminal: set[str] = set()

    for regime in regimes.values():
        if not regime.terminal:
            seen_in_non_terminal.update(regime.states)
        for name, grid in regime.states.items():
            if name in disqualified:
                continue
            if not isinstance(grid, DiscreteGrid):
                disqualified.add(name)
                candidates.pop(name, None)
                continue
            # Terminal regimes require `state_transitions` to be empty by
            # validation, so the absence of an entry is the terminal-regime
            # analogue of `None` — the value is carried through unchanged.
            if not regime.terminal and (
                regime.state_transitions.get(name, "MISSING") is not None
            ):
                disqualified.add(name)
                candidates.pop(name, None)
                continue
            existing = candidates.get(name)
            if existing is None:
                candidates[name] = grid
            elif existing.categories != grid.categories:
                disqualified.add(name)
                candidates.pop(name, None)
    return MappingProxyType(
        {
            name: grid
            for name, grid in candidates.items()
            if name in seen_in_non_terminal
        }
    )


def lift_partitions_from_regime(
    *,
    regime: Regime,
    partition_names: frozenset[str],
) -> tuple[Regime, MappingProxyType[str, DiscreteGrid]]:
    """Return a regime with the given partition states removed.

    Used after `detect_model_partitions` identifies which states qualify
    as partition dimensions across the whole model. Strips those states
    (and their `None` `state_transitions` entries) from the regime so
    downstream machinery sees only the varying state space.

    Args:
        regime: User-facing regime.
        partition_names: Names of partition states at the model level.
            States a regime does not include are ignored.

    Returns:
        Tuple of `(reduced_regime, partitions)` — the subset of
        `partition_names` this regime actually declared (so solve /
        simulate know which scalars to inject for this regime).

    """
    to_strip = {name for name in partition_names if name in regime.states}
    if not to_strip:
        return regime, MappingProxyType({})

    partitions: dict[str, DiscreteGrid] = {}
    for name in to_strip:
        grid = regime.states[name]
        assert isinstance(grid, DiscreteGrid)  # noqa: S101 — model-level check
        partitions[name] = grid
    reduced_states = MappingProxyType(
        {name: grid for name, grid in regime.states.items() if name not in to_strip}
    )
    reduced_state_transitions = MappingProxyType(
        {
            name: transition
            for name, transition in regime.state_transitions.items()
            if name not in to_strip
        }
    )
    reduced = dataclasses.replace(
        regime,
        states=reduced_states,
        state_transitions=reduced_state_transitions,
    )
    return reduced, MappingProxyType(partitions)


def iterate_partition_points(
    *,
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
    regime_partitions: Mapping[RegimeName, Mapping[str, DiscreteGrid]],
) -> InternalParams:
    """Return a copy of `internal_params` with partition scalars injected.

    For each regime, only partition values *declared on that regime* are
    injected — a regime whose functions never reference a partition name
    would otherwise receive it as an unexpected kwarg and the compiled
    closure's enforced signature would reject it.

    Args:
        internal_params: Existing internal params, keyed by regime name.
        partition_point: Partition scalars to inject (all partitions in
            the model-level grid).
        regime_partitions: Per-regime mapping of partition names → grid.
            Used to filter `partition_point` down to the subset each
            regime actually declares.

    Returns:
        New internal-params mapping with partition scalars added per
        regime. The original is unchanged.

    """
    if not partition_point:
        return internal_params
    return MappingProxyType(
        {
            regime_name: MappingProxyType(
                {
                    **regime_params,
                    **{
                        name: partition_point[name]
                        for name in regime_partitions.get(regime_name, {})
                        if name in partition_point
                    },
                }
            )
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
    n_partition = len(shape)

    stacked: dict[int, dict[RegimeName, FloatND]] = {}
    for period in sub_V_arrays[0]:
        stacked[period] = {}
        # Regimes may differ across periods (a regime only appears in
        # `sub[period]` when it is active there); take the union.
        regimes_this_period: list[RegimeName] = []
        seen: set[RegimeName] = set()
        for sub in sub_V_arrays:
            for regime in sub[period]:
                if regime not in seen:
                    seen.add(regime)
                    regimes_this_period.append(regime)
        for regime in regimes_this_period:
            per_point = [sub[period][regime] for sub in sub_V_arrays]
            stacked_arr = jnp.stack(per_point, axis=0).reshape(
                shape + per_point[0].shape
            )
            # Move the prepended partition axes to the end so the convention
            # is [non-partition state/action axes..., partition axes...].
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

    for point in iterate_partition_points(partition_grid=partition_grid):
        mask = jnp.ones_like(initial_conditions["regime"], dtype=bool)
        for name, scalar in point.items():
            mask = mask & (initial_conditions[name] == scalar)
        yield point, mask


def slice_V_at_partition_point(
    *,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
    partition_point: PartitionPoint,
    partition_grid: PartitionGrid,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Slice stacked V-arrays at a single partition point.

    The V-arrays returned by `Model.solve` carry partition axes appended
    after all discrete/continuous state axes (see
    `stack_partition_V_arrays`). To hand a slice to one simulation
    dispatch group, fix each partition axis at its corresponding scalar
    code and drop the axis.

    The stacked V-arrays violate `V._fail_if_interpolation_axes_are_not_last`
    because continuous state axes are no longer the trailing axes; this
    invariant is restored by slicing here. **`Model.simulate` must call
    this before handing the V-arrays to the per-subject interpolator.**

    Args:
        period_to_regime_to_V_arr: V-arrays with partition axes appended.
        partition_point: Scalar value per partition dimension.
        partition_grid: Partition grids defining name/position.

    Returns:
        V-arrays without partition axes.

    """
    if not partition_grid:
        return period_to_regime_to_V_arr

    names = tuple(partition_grid)
    codes_by_name = tuple(tuple(partition_grid[name].codes) for name in names)
    index_per_axis = tuple(
        codes_by_name[i].index(int(partition_point[name]))
        for i, name in enumerate(names)
    )

    sliced: dict[int, dict[RegimeName, FloatND]] = {}
    for period, regime_to_V in period_to_regime_to_V_arr.items():
        sliced[period] = {}
        for regime, V in regime_to_V.items():
            idx: tuple[slice | int, ...] = (slice(None),) * (
                V.ndim - len(partition_grid)
            ) + index_per_axis
            sliced[period][regime] = V[idx]
    return MappingProxyType({p: MappingProxyType(rd) for p, rd in sliced.items()})


def slice_initial_conditions(
    *,
    initial_conditions: Mapping[str, Array],
    subject_mask: Array,
) -> MappingProxyType[str, Array]:
    """Filter every array in `initial_conditions` by a boolean subject mask.

    Args:
        initial_conditions: Flat mapping of state names (plus `"regime"`) to
            subject-indexed arrays.
        subject_mask: Boolean 1-D array marking the subjects to keep.

    Returns:
        Immutable mapping with the same keys but arrays filtered to the
        selected subjects.

    """
    return MappingProxyType(
        {name: arr[subject_mask] for name, arr in initial_conditions.items()}
    )
