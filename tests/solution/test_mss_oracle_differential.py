"""The MSS envelope names the same owner as an exact rational reference.

`refine_envelope` decides ownership from stored IEEE operands with compensated
arithmetic and an integer comparator; the reference in `_mss_segment_oracle`
decides it in `fractions.Fraction` with a different control flow — it enumerates
every pairwise crossing, cuts the support at every breakpoint, and evaluates one
interior representative per open cell. Agreement between the two is therefore a
statement about the geometry rather than about either implementation.

The reference is built from the candidate arrays after they have been placed in
the working precision, so both routes read the same stored operands. Building it
from the generator's own float64 values instead would hand the reference a
geometry the production path never sees, and the comparison would report the
cast rather than the envelope.

Each branch carries a constant policy of its own, so the policy read back out of
a refined row names the owner: the reading is compared to the nearest branch
policy rather than to a tolerance, because which branch owns a cell is a
discrete fact and no tolerance is the right instrument for one.

These are ordinary geometries, generated so the runs contend rather than so any
margin is narrow, and they are a guard rather than a discriminator: the
deciding margins here are wide enough that a rounded comparison would call them
the same way. The witnesses whose margins are one representable step wide, or
exactly zero, are in `test_mss_decision_safety`.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope.mss import refine_envelope
from tests.solution._mss_segment_oracle import (
    Branch,
    all_breakpoints,
    interval_owners,
)

_N_BRANCHES = 3
_N_NODES = 3
_N_REFINED = 64
_NAMES = ("A", "B", "C")
_POLICY_OF = {"A": 1.0, "B": 2.0, "C": 3.0}
# The float64 bound this file was written against, kept as a floor so the
# double-precision leg is provably unchanged, widened by the working format's
# own resolution so the single-precision leg reports the geometry rather than
# the format. A crossing abscissa is an ordinary root of a well-conditioned
# affine gap here, so a few dozen representable steps is the honest reach.
_ABSCISSA_RTOL_FLOOR = 1e-9
_ABSCISSA_EPS_MULTIPLE = 256.0


def _candidate_arrays(
    *, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Lay out overlapping monotone runs as one candidate chain.

    Every run spans the same support and rises across it, so the runs genuinely
    contend rather than each owning a disjoint stretch, and their differing
    slopes and levels put the crossings inside the support rather than beyond
    it. The topology is declared through the branch labels, so no link ever
    bridges two runs.
    """
    rng = np.random.default_rng(seed=seed)
    grid: list[float] = []
    policy: list[float] = []
    value: list[float] = []
    label: list[float] = []
    for index in range(_N_BRANCHES):
        offsets = np.sort(rng.uniform(0.0, 0.4, size=_N_NODES - 2))
        positions = np.concatenate([[0.0], offsets, [1.0]])
        x_nodes = 1.0 + 4.0 * positions
        level = rng.uniform(-1.0, 1.0)
        slope = rng.uniform(0.2, 2.0)
        v_nodes = level + slope * (x_nodes - x_nodes[0])
        grid.extend(float(x) for x in x_nodes)
        value.extend(float(v) for v in v_nodes)
        policy.extend(_POLICY_OF[_NAMES[index]] for _ in x_nodes)
        label.extend(float(index) for _ in x_nodes)
    return (
        jnp.asarray(grid),
        jnp.asarray(policy),
        jnp.asarray(value),
        jnp.asarray(label),
    )


def _branches_from_arrays(
    *, grid: jnp.ndarray, policy: jnp.ndarray, value: jnp.ndarray
) -> tuple[Branch, ...]:
    """Read the reference's geometry off the arrays the solver is handed."""
    stored_grid = np.asarray(grid)
    stored_policy = np.asarray(policy)
    stored_value = np.asarray(value)
    branches = []
    for index, name in enumerate(_NAMES):
        start = index * _N_NODES
        stop = start + _N_NODES
        branches.append(
            Branch(
                name=name,
                x=tuple(Fraction(float(x)) for x in stored_grid[start:stop]),
                v=tuple(Fraction(float(v)) for v in stored_value[start:stop]),
                policy=tuple(Fraction(float(p)) for p in stored_policy[start:stop]),
            )
        )
    return tuple(branches)


def _resolvable_cells(
    *, branches: tuple[Branch, ...]
) -> list[tuple[Fraction, tuple[str, ...]]]:
    """Return each open cell's midpoint and sole owner, where there is one.

    A cell two branches own exactly carries no ownership fact to compare, and a
    cell narrower than the format can place a node inside cannot be resolved by
    any float representation of the envelope. Both are excluded, so what remains
    is what a refined row is answerable for.
    """
    cells = interval_owners(branches=branches, points=all_breakpoints(branches))
    resolvable = []
    for left, right, owners in cells:
        midpoint = (left + right) / 2
        width = float(right - left)
        spacing = float(np.spacing(np.float64(float(midpoint))))
        if len(owners) == 1 and width > 64.0 * spacing:
            resolvable.append((midpoint, owners))
    return resolvable


def _nearest_branch(*, reading: float) -> str:
    """Return the branch whose constant policy the reading is closest to."""
    return min(_POLICY_OF, key=lambda name: abs(reading - _POLICY_OF[name]))


def _abscissa_tolerance(*, abscissa: float, dtype: np.dtype) -> float:
    """Return how far an emitted crossing may sit from the exact breakpoint."""
    resolution = _ABSCISSA_EPS_MULTIPLE * float(np.finfo(dtype).eps)
    relative = max(_ABSCISSA_RTOL_FLOOR, resolution)
    return relative * max(1.0, abs(abscissa))


@pytest.mark.parametrize("seed", range(20))
def test_the_refined_row_names_the_exact_owner_of_every_open_cell(seed: int) -> None:
    """The policy read inside a cell is the policy of the branch that owns it."""
    grid, policy, value, segment_id = _candidate_arrays(seed=seed)
    branches = _branches_from_arrays(grid=grid, policy=policy, value=value)

    refined_grid, refined_policy, _refined_value, n_kept = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=_N_REFINED,
        segment_id=segment_id,
    )
    assert int(n_kept) <= _N_REFINED

    cells = _resolvable_cells(branches=branches)
    # A witness that resolves no cell would pass without comparing anything.
    assert len(cells) >= 2

    for midpoint, owners in cells:
        reading = float(
            interp_on_padded_grid(
                x_query=jnp.asarray(float(midpoint)),
                xp=refined_grid,
                fp=refined_policy,
            )
        )
        assert _nearest_branch(reading=reading) == owners[0]


@pytest.mark.parametrize("seed", range(20))
def test_the_refined_row_is_weakly_ascending_and_finite(seed: int) -> None:
    """Every published row is finite and no row goes back down the grid."""
    grid, policy, value, segment_id = _candidate_arrays(seed=seed)

    refined_grid, refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=_N_REFINED,
        segment_id=segment_id,
    )

    kept = int(n_kept)
    assert kept <= _N_REFINED
    live_grid = np.asarray(refined_grid)[:kept]
    assert np.all(np.diff(live_grid) >= 0.0)
    assert np.all(np.isfinite(live_grid))
    assert np.all(np.isfinite(np.asarray(refined_policy)[:kept]))
    assert np.all(np.isfinite(np.asarray(refined_value)[:kept]))


@pytest.mark.parametrize("seed", range(20))
def test_every_emitted_kink_sits_on_an_exact_breakpoint(seed: int) -> None:
    """A duplicated abscissa in the refined row is a crossing the geometry has."""
    grid, policy, value, segment_id = _candidate_arrays(seed=seed)
    branches = _branches_from_arrays(grid=grid, policy=policy, value=value)

    refined_grid, _refined_policy, _refined_value, n_kept = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=_N_REFINED,
        segment_id=segment_id,
    )

    kept = int(n_kept)
    live_grid = np.asarray(refined_grid)[:kept]
    breakpoints = np.array(
        sorted(float(point) for point in all_breakpoints(branches)), dtype=np.float64
    )
    duplicated = live_grid[:-1][live_grid[1:] == live_grid[:-1]]
    for abscissa in np.unique(duplicated):
        distance = float(np.min(np.abs(breakpoints - float(abscissa))))
        assert distance <= _abscissa_tolerance(
            abscissa=float(abscissa), dtype=live_grid.dtype
        )
