"""Partition-dimension detection, iteration, and result stacking.

A partition dimension is a `DiscreteGrid` state the user opted into
partition dispatch for by setting `dispatch=DispatchStrategy.PARTITION_SCAN`
or `PARTITION_VMAP` on the grid. The Bellman backward induction never
couples values across such a dimension, so pylcm lifts the dim out of
the state-action space and sweeps it at the kernel's top level — either
by `jax.lax.scan` (`PARTITION_SCAN`) or `jax.vmap` (`PARTITION_VMAP`).
See `docs/user_guide/dispatch.md` for the trade-offs.

The module exposes two layers:

- **Model-building** (called from `process_regimes`):
  `detect_model_partitions`, `model_partition_dispatch`,
  `lift_partitions_from_regime`.
- **Solve / simulate runtime** (called from `Model.solve` / `Model.simulate`):
  `iterate_partition_points`, `stack_partition_scalars`,
  `inject_partition_scalars`, `group_subjects_by_partition`,
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

from lcm.exceptions import ModelInitializationError
from lcm.grids import DiscreteGrid, DispatchStrategy
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
    """Identify states that the user opted into partition dispatch for.

    A partition dimension is a `DiscreteGrid` state the user opted into
    explicitly by setting `dispatch=DispatchStrategy.PARTITION_SCAN` or
    `PARTITION_VMAP` on the grid. The qualification is:

    1. The state's grid is a `DiscreteGrid` with a partition-lifted
       `dispatch`.
    2. Every regime that includes the state agrees on the
       `DispatchStrategy` and on the grid's `DiscreteGrid.categories`.
    3. Every regime that includes the state either omits it from
       `state_transitions` (terminal regimes) or sets
       `state_transitions[name] = None` (non-terminal regimes).
    4. The state appears in at least one non-terminal regime so an
       initial value can flow in through `initial_conditions`.

    Items 2-4 are *invariants* rather than heuristics — the detector
    raises `ModelInitializationError` if the user's model violates
    them rather than silently skipping the partition.

    Args:
        regimes: Mapping of regime names to user-facing regimes.

    Returns:
        Immutable mapping of partition name to its `DiscreteGrid`. Empty
        when no state in the model has a partition-lifted dispatch.

    Raises:
        ModelInitializationError: If a partition-lifted state has a
            non-identity transition, disagrees with another regime on
            `dispatch` or `categories`, lives only in terminal regimes,
            or appears on a non-`DiscreteGrid`.

    """
    # A partition value must come from `initial_conditions`, which implies
    # the state exists in at least one non-terminal regime where subjects
    # can start.
    seen_in_non_terminal: set[str] = set()
    for regime in regimes.values():
        if not regime.terminal:
            seen_in_non_terminal.update(regime.states)

    candidates: dict[str, DiscreteGrid] = {}
    for regime_name, regime in regimes.items():
        for name, grid in regime.states.items():
            if not isinstance(grid, DiscreteGrid):
                continue
            if not grid.dispatch.is_partition_lifted:
                continue
            _fail_if_partition_dim_has_non_identity_transition(
                regime=regime, regime_name=regime_name, state_name=name
            )
            existing = candidates.get(name)
            if existing is None:
                candidates[name] = grid
            else:
                _fail_if_partition_dim_disagrees_across_regimes(
                    existing=existing,
                    seen=grid,
                    regime_name=regime_name,
                    state_name=name,
                )

    _fail_if_partition_only_in_terminal_regimes(
        candidates=candidates, seen_in_non_terminal=seen_in_non_terminal
    )
    _fail_if_partition_dispatch_mixed(candidates=candidates)

    return MappingProxyType(candidates)


def model_partition_dispatch(
    *,
    partition_grid: PartitionGrid,
) -> DispatchStrategy | None:
    """Return the single `DispatchStrategy` the model's partition dims share.

    Returns `None` when the model has no partition-lifted dims. Any
    disagreement was already raised by `detect_model_partitions`, so
    picking the first grid's dispatch is safe.
    """
    if not partition_grid:
        return None
    # All dims agree — any representative works.
    first = next(iter(partition_grid.values()))
    return first.dispatch


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
    # Walk `regime.states` in the user-declared insertion order so the
    # resulting `partitions` (and the reduced states) retain a
    # deterministic axis ordering across processes. Using
    # `partition_names` as a membership set — not as the iteration
    # driver — avoids hash-randomised set ordering leaking into the
    # partition axis layout (visible as shape (2, 3, 10) vs (3, 2, 10)
    # swaps between runs under different `PYTHONHASHSEED` values).
    partitions: dict[str, DiscreteGrid] = {}
    reduced_states: dict[str, object] = {}
    for name, grid in regime.states.items():
        if name in partition_names:
            assert isinstance(grid, DiscreteGrid)  # noqa: S101 — model-level check
            partitions[name] = grid
        else:
            reduced_states[name] = grid

    if not partitions:
        return regime, MappingProxyType({})

    reduced_state_transitions = MappingProxyType(
        {
            name: transition
            for name, transition in regime.state_transitions.items()
            if name not in partitions
        }
    )
    reduced = dataclasses.replace(
        regime,
        states=MappingProxyType(reduced_states),
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


def stack_partition_scalars(
    *,
    internal_params: InternalParams,
    partition_grid: PartitionGrid,
    regime_partitions: Mapping[RegimeName, Mapping[str, DiscreteGrid]],
) -> tuple[InternalParams, tuple[int, ...]]:
    """Return `internal_params` with partition scalars attached as leading-axis arrays.

    Where `inject_partition_scalars` adds one scalar per partition name for a
    single point, `stack_partition_scalars` enumerates the Cartesian product
    of `partition_grid` and attaches the full column of values as a 1-D array
    of length $N = \\prod_i N_i$ (the flattened product size). A single
    `jax.vmap` over that axis dispatches the Bellman kernel once;
    `Model.solve` then calls `_reshape_leading_partition_axis` to restore
    the per-partition leading axes.

    The flatten order here **must** match the row-major unraveling performed
    by `_reshape_leading_partition_axis` downstream — both rely on
    `itertools.product` (used inside `iterate_partition_points`) yielding
    points with the last index varying fastest.

    Only partition names declared on a given regime are attached to that
    regime — same per-regime filtering as `inject_partition_scalars`.

    Args:
        internal_params: Base internal-params mapping (pre-injection).
        partition_grid: Partition grids defining the Cartesian product. Empty
            mapping returns `(internal_params, ())` unchanged.
        regime_partitions: Per-regime mapping of partition names → grid. Used
            to filter the stacked scalars down to the subset each regime
            declares.

    Returns:
        Tuple of `(stacked_params, partition_shape)`. `stacked_params` has
        the same regime/key structure as `internal_params`; declared
        partition names carry a 1-D array of length `prod(partition_shape)`.
        `partition_shape` is `()` when `partition_grid` is empty.

    """
    if not partition_grid:
        return internal_params, ()

    names = tuple(partition_grid)
    partition_shape = tuple(len(partition_grid[name].codes) for name in names)
    points = list(iterate_partition_points(partition_grid=partition_grid))

    stacked_per_name = {
        name: jnp.stack([point[name] for point in points]) for name in names
    }

    return (
        MappingProxyType(
            {
                regime_name: MappingProxyType(
                    {
                        **regime_params,
                        **{
                            name: stacked_per_name[name]
                            for name in regime_partitions.get(regime_name, {})
                            if name in stacked_per_name
                        },
                    }
                )
                for regime_name, regime_params in internal_params.items()
            }
        ),
        partition_shape,
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

    The V-arrays returned by `Model.solve` carry partition axes as the
    **leading** axes (canonical order: partitions > state axes). To hand
    a slice to one simulation dispatch group, fix each partition axis at
    its corresponding scalar code and drop the axis.

    The stacked V-arrays violate `V._fail_if_interpolation_axes_are_not_last`
    because continuous state axes are no longer the trailing axes; this
    invariant is restored by slicing here. **`Model.simulate` must call
    this before handing the V-arrays to the per-subject interpolator.**

    Args:
        period_to_regime_to_V_arr: V-arrays with partition axes as the
            leading axes (in `partition_grid` insertion order).
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
            idx: tuple[slice | int, ...] = index_per_axis + (slice(None),) * (
                V.ndim - len(partition_grid)
            )
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


def _fail_if_partition_only_in_terminal_regimes(
    *, candidates: Mapping[str, DiscreteGrid], seen_in_non_terminal: set[str]
) -> None:
    """Partition states must have an entry point via `initial_conditions`."""
    for name in candidates:
        if name not in seen_in_non_terminal:
            msg = (
                f"Partition state '{name}' appears only in terminal regimes. "
                f"An initial value must flow in through `initial_conditions`, "
                f"so the state must be declared on at least one non-terminal "
                f"regime."
            )
            raise ModelInitializationError(msg)


def _fail_if_partition_dispatch_mixed(
    *, candidates: Mapping[str, DiscreteGrid]
) -> None:
    """All partition-lifted dims in the model must share one DispatchStrategy.

    Supporting mixed strategies would need two different wrap primitives
    around the same kernel — defer that until a workload actually needs it.
    """
    strategies = {grid.dispatch for grid in candidates.values()}
    if len(strategies) > 1:
        msg = (
            f"Multiple partition-lifted DiscreteGrids in the model disagree on "
            f"`DispatchStrategy`: "
            f"{sorted(s.name for s in strategies)}. Pick one of "
            f"`PARTITION_SCAN` or `PARTITION_VMAP` and use it for every "
            f"partition-lifted dim."
        )
        raise ModelInitializationError(msg)


def _fail_if_partition_dim_has_non_identity_transition(
    *, regime: Regime, regime_name: str, state_name: str
) -> None:
    """Raise if a partition-lifted state has a non-identity transition.

    Two distinct user mistakes produce two distinct error messages:
    - state omitted from `state_transitions` entirely (a non-terminal regime
      must list every non-shock state);
    - state listed with a non-`None` (non-identity) transition.
    """
    if regime.terminal:
        # Terminal regimes must have empty `state_transitions` by separate
        # validation; nothing to check here — the value simply carries through.
        return
    if state_name not in regime.state_transitions:
        msg = (
            f"State '{state_name}' has dispatch=PARTITION_* (partition-lifted) "
            f"on its DiscreteGrid but regime '{regime_name}' omits it from "
            f"`state_transitions`. Partition-lifted states require an explicit "
            f"identity transition in every non-terminal regime — set "
            f"`state_transitions[{state_name!r}] = None` on regime "
            f"'{regime_name}' or change the grid's dispatch."
        )
        raise ModelInitializationError(msg)
    transition = regime.state_transitions[state_name]
    if transition is not None:
        msg = (
            f"State '{state_name}' has dispatch=PARTITION_* (partition-lifted) "
            f"on its DiscreteGrid but regime '{regime_name}' declares a "
            f"non-identity `state_transitions[{state_name!r}]` "
            f"({transition!r}). Partition-lifted states require the identity "
            f"transition in every regime — set "
            f"`state_transitions[{state_name!r}] = None` on regime "
            f"'{regime_name}' or change the grid's dispatch."
        )
        raise ModelInitializationError(msg)


def _fail_if_partition_dim_disagrees_across_regimes(
    *,
    existing: DiscreteGrid,
    seen: DiscreteGrid,
    regime_name: str,
    state_name: str,
) -> None:
    """Raise if two regimes declare the same partition state with mismatched specs."""
    if existing.categories != seen.categories:
        msg = (
            f"Partition state '{state_name}' has inconsistent categories "
            f"across regimes: {existing.categories} vs {seen.categories} "
            f"(regime '{regime_name}')."
        )
        raise ModelInitializationError(msg)
    if existing.dispatch is not seen.dispatch:
        msg = (
            f"Partition state '{state_name}' has inconsistent dispatch "
            f"strategies across regimes: {existing.dispatch} vs "
            f"{seen.dispatch} (regime '{regime_name}'). Every regime must "
            f"agree on a single `DispatchStrategy` for a given partition "
            f"state."
        )
        raise ModelInitializationError(msg)
