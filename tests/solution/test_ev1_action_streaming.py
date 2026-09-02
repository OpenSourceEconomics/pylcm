"""EV1 action streaming keeps branch semantics without a serial candidate scan."""

import itertools
import math
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from _lcm.solution.action_streaming import build_streaming_ev1_max_Q_over_a


def _numpy_ev1_oracle(
    *,
    Q_and_F: Callable[..., tuple[object, object]],
    action_names: tuple[str, ...],
    n_discrete_action_axes: int,
    action_grids: Mapping[str, np.ndarray],
    fixed_kwargs: Mapping[str, object],
    scale: float,
) -> float:
    """Enumerate branch maxima and their log-sum independently in NumPy."""
    grids = tuple(np.asarray(action_grids[name]) for name in action_names)
    discrete_grids = grids[:n_discrete_action_axes]
    continuous_grids = grids[n_discrete_action_axes:]
    continuous_coordinates = tuple(itertools.product(*continuous_grids))
    if not continuous_grids:
        continuous_coordinates = ((),)

    branch_values = []
    for discrete_coordinate in itertools.product(*discrete_grids):
        feasible_values = []
        for continuous_coordinate in continuous_coordinates:
            coordinate = (*discrete_coordinate, *continuous_coordinate)
            action_kwargs = dict(zip(action_names, coordinate, strict=True))
            value, feasible = Q_and_F(**fixed_kwargs, **action_kwargs)
            if bool(np.asarray(feasible)):
                feasible_values.append(float(np.asarray(value)))

        if not feasible_values:
            branch_values.append(-np.inf)
        elif np.isnan(feasible_values).any():
            branch_values.append(np.nan)
        else:
            branch_values.append(float(np.max(feasible_values)))

    branches = np.asarray(branch_values, dtype=np.float64)
    if np.isnan(branches).any() or np.isposinf(branches).any():
        return np.nan
    finite = np.isfinite(branches)
    if not finite.any():
        return -np.inf
    anchor = np.max(branches[finite])
    mass = np.sum(np.exp((branches[finite] - anchor) / scale), dtype=np.float64)
    return float(anchor + scale * np.log(mass))


def _nested_scan_lengths(closed_jaxpr: object) -> list[int]:  # noqa: C901
    """Collect scan lengths recursively from a closed action-core Jaxpr."""
    lengths: list[int] = []
    seen: set[int] = set()

    def visit(value: object) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))

        candidate = getattr(value, "jaxpr", value)
        equations = getattr(candidate, "eqns", None)
        if equations is not None:
            for equation in equations:
                if equation.primitive.name == "scan":
                    lengths.append(int(equation.params["length"]))
                for parameter in equation.params.values():
                    visit(parameter)
            return

        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(closed_jaxpr)
    return lengths


def test_ev1_lowering_has_no_block_width_candidate_scan() -> None:
    """A chunk may be scanned, but its candidates must reduce as a vector.

    The two branches have three chunks each, so an outer scan of length five is legal.
    A scan of length eight is the block-local candidate loop that serializes the
    reducer on GPU and turns this 34-cell product into one step per padded cell.
    """

    def Q_and_F(*, branch: jax.Array, choice: jax.Array):
        return branch - choice**2, jnp.ones((), dtype=bool)

    block_width = 8
    branches = jnp.arange(2, dtype=jnp.float32)
    choices = jnp.arange(17, dtype=jnp.float32)
    streamed = build_streaming_ev1_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("branch", "choice"),
        n_discrete_action_axes=1,
        block_width=block_width,
        scale=jnp.asarray(0.4),
    )

    closed_jaxpr = jax.make_jaxpr(streamed)(branch=branches, choice=choices)
    scan_lengths = _nested_scan_lengths(closed_jaxpr)

    assert block_width not in scan_lengths, (
        "EV1 lowering still scans every candidate inside each action chunk; "
        f"found scan lengths {scan_lengths}."
    )


def test_ev1_packs_pure_discrete_branches_into_vector_blocks() -> None:
    """Whole one-cell branches share blocks instead of becoming a scalar scan."""

    def Q_and_F(*, sector: jax.Array, status: jax.Array):
        return 3.0 * sector - status, jnp.ones((), dtype=bool)

    block_width = 8
    sectors = np.arange(3, dtype=np.float32)
    statuses = np.arange(5, dtype=np.float32)
    scale = 0.3
    streamed = build_streaming_ev1_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("sector", "status"),
        n_discrete_action_axes=2,
        block_width=block_width,
        scale=jnp.asarray(scale),
    )

    closed_jaxpr = jax.make_jaxpr(streamed)(
        sector=jnp.asarray(sectors),
        status=jnp.asarray(statuses),
    )
    scan_lengths = _nested_scan_lengths(closed_jaxpr)
    n_branches = sectors.size * statuses.size
    n_vector_blocks = math.ceil(n_branches / block_width)
    assert scan_lengths == [n_vector_blocks - 1]

    expected = _numpy_ev1_oracle(
        Q_and_F=Q_and_F,
        action_names=("sector", "status"),
        n_discrete_action_axes=2,
        action_grids={"sector": sectors, "status": statuses},
        fixed_kwargs={},
        scale=scale,
    )
    result = streamed(
        sector=jnp.asarray(sectors),
        status=jnp.asarray(statuses),
    )
    assert_allclose(result.smoothed_value, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("block_width", [5, 7, 11])
def test_ev1_vector_chunks_match_oracle_with_multiple_discrete_axes(
    block_width: int,
) -> None:
    """Branch-local chunks and their padded tails match direct enumeration."""

    def Q_and_F(*, sector, status, saving, hours, shift):
        value = (
            5.0 * sector + 2.0 * status - (saving - status) ** 2 + 0.125 * hours + shift
        )
        feasible = (saving != 0.5) | (hours == 0.0)
        return value, feasible

    grids = {
        "sector": np.asarray([0.0, 2.0], dtype=np.float32),
        "status": np.asarray([-1.0, 1.0, 3.0], dtype=np.float32),
        "saving": np.asarray([-2.0, 0.5, 3.0], dtype=np.float32),
        "hours": np.asarray([0.0, 2.0], dtype=np.float32),
    }
    scale = 0.37
    expected = _numpy_ev1_oracle(
        Q_and_F=Q_and_F,
        action_names=("sector", "status", "saving", "hours"),
        n_discrete_action_axes=2,
        action_grids=grids,
        fixed_kwargs={"shift": 0.25},
        scale=scale,
    )
    streamed = build_streaming_ev1_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("sector", "status", "saving", "hours"),
        n_discrete_action_axes=2,
        block_width=block_width,
        scale=jnp.asarray(scale),
    )

    result = streamed(
        **{name: jnp.asarray(grid) for name, grid in grids.items()},
        shift=jnp.asarray(0.25),
    )

    assert math.prod(len(grid) for grid in grids.values()) % block_width != 0
    assert_allclose(result.smoothed_value, expected, rtol=1e-5, atol=1e-6)


def test_ev1_packed_branches_still_require_scalar_q() -> None:
    """The extra branch-block dimension must not hide a vector-valued Q."""

    def Q_and_F(*, sector: jax.Array, status: jax.Array):
        return jnp.stack((sector, status)), jnp.ones((), dtype=bool)

    streamed = build_streaming_ev1_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("sector", "status"),
        n_discrete_action_axes=2,
        block_width=8,
        scale=jnp.asarray(0.3),
    )

    with pytest.raises(ValueError, match="requires scalar Q"):
        streamed(
            sector=jnp.arange(3, dtype=jnp.float32),
            status=jnp.arange(5, dtype=jnp.float32),
        )


def test_ev1_vector_chunks_keep_all_infeasible_distinct_from_nan() -> None:
    """Padding and empty branch maxima contribute zero EV1 mass."""

    def Q_and_F(*, branch, choice):
        return branch + choice, jnp.zeros((), dtype=bool)

    streamed = build_streaming_ev1_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("branch", "choice"),
        n_discrete_action_axes=1,
        block_width=4,
        scale=jnp.asarray(0.2),
    )
    result = streamed(
        branch=jnp.arange(3, dtype=jnp.float32),
        choice=jnp.arange(5, dtype=jnp.float32),
    )

    assert bool(jnp.isneginf(result.smoothed_value))
    assert not bool(jnp.isnan(result.smoothed_value))


def test_ev1_vector_chunks_preserve_signed_zero_tie_across_chunks() -> None:
    """A {-0, +0} branch maximum is +0 even when the tie crosses chunks."""

    def Q_and_F(*, branch, choice):
        del branch
        value = jnp.where(
            choice == 1,
            -jnp.zeros((), dtype=jnp.float32),
            jnp.where(choice == 2, jnp.zeros((), dtype=jnp.float32), -1.0),
        )
        return value, jnp.ones((), dtype=bool)

    streamed = build_streaming_ev1_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("branch", "choice"),
        n_discrete_action_axes=1,
        block_width=2,
        scale=jnp.asarray(0.5),
    )
    result = streamed(
        branch=jnp.asarray([0], dtype=jnp.int32),
        choice=jnp.arange(5, dtype=jnp.int32),
    )

    assert_array_equal(result.smoothed_value, jnp.asarray(0.0))
    assert not bool(jnp.signbit(result.smoothed_value))


@pytest.mark.parametrize("nan_is_feasible", [0, 1])
def test_ev1_vector_chunks_apply_feasibility_before_nan_propagation(
    nan_is_feasible: int,
) -> None:
    """Only a feasible NaN poisons its branch and the final EV1 log-sum."""

    def Q_and_F(*, branch, choice):
        is_nan_cell = (branch == 1) & (choice == 0)
        value = jnp.where(is_nan_cell, jnp.nan, 2.0 * branch - choice)
        feasible = jnp.where(
            is_nan_cell, nan_is_feasible == 1, jnp.ones((), dtype=bool)
        )
        return value, feasible

    branches = np.asarray([0.0, 1.0], dtype=np.float32)
    choices = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    scale = 0.4
    expected = _numpy_ev1_oracle(
        Q_and_F=Q_and_F,
        action_names=("branch", "choice"),
        n_discrete_action_axes=1,
        action_grids={"branch": branches, "choice": choices},
        fixed_kwargs={},
        scale=scale,
    )
    streamed = build_streaming_ev1_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("branch", "choice"),
        n_discrete_action_axes=1,
        block_width=4,
        scale=jnp.asarray(scale),
    )
    result = streamed(
        branch=jnp.asarray(branches),
        choice=jnp.asarray(choices),
    )

    if nan_is_feasible == 1:
        assert bool(jnp.isnan(result.smoothed_value))
    else:
        assert_allclose(result.smoothed_value, expected, rtol=1e-6, atol=1e-6)
