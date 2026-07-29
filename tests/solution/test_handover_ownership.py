"""A published switch lands on the first state the incoming link owns.

The refined row carries a switch as a duplicated abscissa holding both branches'
records, and the reader is right-continuous there, so the abscissa decides which
policy is published from that state onward. Placing it one representable state
too low publishes the incoming policy over ground the outgoing link still owns;
one too high leaves the outgoing policy in place after ownership has passed.

Ownership is a structural predicate, so it is asserted as a decision rather than
to a tolerance, and the decision is taken from the links' *stored endpoints* —
the lines themselves — not from anything the envelope computed on the way. Two
arbiters serve that: `certified_margin_sign`, which reports which link is above
at a state, and an exact `fractions.Fraction` solve of the two lines, which names
the crossing outright and so also covers the states where the margin predicate
is within its own resolution of zero.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.cell_hull import hull_owners
from _lcm.egm.upper_envelope.certified_sign import certified_margin_sign
from _lcm.egm.upper_envelope.segment_envelope import refine_envelope_exact
from tests.conftest import X64_ENABLED

# Two policy-consistent log-utility links: a run with consumption `c` has value
# slope `1 / c`. The outgoing run consumes 8, the incoming one 2.
_OUTGOING_POLICY = 8.0
_INCOMING_POLICY = 2.0
_OUTGOING_SLOPE = 1.0 / _OUTGOING_POLICY
_INCOMING_SLOPE = 1.0 / _INCOMING_POLICY
_OUTGOING_X0, _OUTGOING_X1 = 0.0, 1.0
_OUTGOING_V0 = 0.0

_REFINE = jax.jit(
    lambda grid, policy, value: refine_envelope_exact(
        endog_grid=grid, policy=policy, value=value, n_refined=8, max_runs=4
    )
)

# Abscissae of a row whose two links overlap across one node cell, with the
# cell's edges sitting on candidate nodes as they do in a solved row.
_LEVEL_ROW_GRID = (88.11295, 94.20885, 89.98355, 95.5522)
_LEVEL_ROW_N_REFINED = 16

_REFINE_LEVEL_ROW = jax.jit(
    lambda grid, policy, value: refine_envelope_exact(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=_LEVEL_ROW_N_REFINED,
        max_runs=4,
    )
)


def _working_dtypes():
    """The numpy/jax dtype pair matching the configured working precision."""
    return (np.float64, jnp.float64) if X64_ENABLED else (np.float32, jnp.float32)


def _owner_sign(jax_dtype, *, x_query: float, offset: float, incoming_v0: float) -> int:
    """`+1` where the outgoing link is above at `x_query`, `-1` where below."""
    return int(
        certified_margin_sign(
            a_x0=jnp.asarray(_OUTGOING_X0, jax_dtype),
            a_x1=jnp.asarray(_OUTGOING_X1, jax_dtype),
            a_v0=jnp.asarray(_OUTGOING_V0, jax_dtype),
            a_v1=jnp.asarray(_OUTGOING_V0 + _OUTGOING_SLOPE, jax_dtype),
            b_x0=jnp.asarray(offset, jax_dtype),
            b_x1=jnp.asarray(offset + 1.0, jax_dtype),
            b_v0=jnp.asarray(incoming_v0, jax_dtype),
            b_v1=jnp.asarray(incoming_v0 + _INCOMING_SLOPE, jax_dtype),
            x_query=jnp.asarray(x_query, jax_dtype),
        )
    )


def _published_switch(jax_dtype, *, offset: float, incoming_v0: float) -> float | None:
    """The abscissa at which the refined row hands the policy over, if it does."""
    grid = jnp.asarray(
        [_OUTGOING_X0, _OUTGOING_X1, offset, offset + 1.0], dtype=jax_dtype
    )
    policy = jnp.asarray(
        [_OUTGOING_POLICY, _OUTGOING_POLICY, _INCOMING_POLICY, _INCOMING_POLICY],
        dtype=jax_dtype,
    )
    value = jnp.asarray(
        [
            _OUTGOING_V0,
            _OUTGOING_V0 + _OUTGOING_SLOPE,
            incoming_v0,
            incoming_v0 + _INCOMING_SLOPE,
        ],
        dtype=jax_dtype,
    )
    refined_grid, refined_policy, _value, n_kept = _REFINE(grid, policy, value)
    n = int(n_kept)
    if n > 8:
        return None
    live_grid = np.asarray(refined_grid[:n])
    live_policy = np.asarray(refined_policy[:n])
    duplicated = [
        i
        for i in range(1, n)
        if live_policy[i] != live_policy[i - 1] and live_grid[i] == live_grid[i - 1]
    ]
    return float(live_grid[duplicated[0]]) if duplicated else None


def _misplaced_switches(offset: float, n_intercepts: int) -> list[tuple[float, str]]:
    """Sweep the incoming link's level and collect every mislocated handover."""
    dtype, jax_dtype = _working_dtypes()
    misplaced = []
    for step in range(n_intercepts):
        incoming_v0 = float(dtype(0.005 + step * 1e-5))
        event = _published_switch(jax_dtype, offset=offset, incoming_v0=incoming_v0)
        if event is None:
            continue
        predecessor = float(np.nextafter(dtype(event), dtype(-np.inf)))
        at_predecessor = _owner_sign(
            jax_dtype, x_query=predecessor, offset=offset, incoming_v0=incoming_v0
        )
        at_event = _owner_sign(
            jax_dtype, x_query=event, offset=offset, incoming_v0=incoming_v0
        )
        if at_predecessor != 1:
            misplaced.append((incoming_v0, "early"))
        elif at_event == 1:
            misplaced.append((incoming_v0, "late"))
    return misplaced


def _poisoned_level_steps(n_steps: int) -> list[int]:
    """Sweep the incoming link across states of a common level; collect refusals."""
    dtype, jax_dtype = _working_dtypes()
    level = dtype(1e5)
    state = dtype(np.spacing(level))
    grid = jnp.asarray(_LEVEL_ROW_GRID, dtype=jax_dtype)
    policy = jnp.asarray(
        [_OUTGOING_POLICY, _OUTGOING_POLICY, _INCOMING_POLICY, _INCOMING_POLICY],
        dtype=jax_dtype,
    )
    poisoned = []
    for step in range(1, n_steps + 1):
        value = jnp.asarray(
            [level, level, level + dtype(step) * state, level - state],
            dtype=jax_dtype,
        )
        *_arrays, n_kept = _REFINE_LEVEL_ROW(grid, policy, value)
        if int(n_kept) > _LEVEL_ROW_N_REFINED:
            poisoned.append(step)
    return poisoned


def test_handover_on_a_common_value_level_is_published_not_refused() -> None:
    """A row whose links differ by a few states of its own level still publishes.

    Where the value correspondence sits on a large common level, the rises that
    decide ownership are only a handful of representable states wide, and the
    crossing they imply lands on a state rather than between two. Locating it is
    then the easiest case, not the hardest, and the row must carry a published
    switch — refusing it would discard rows whose handover is not in doubt.
    """
    assert _poisoned_level_steps(n_steps=40) == []


@pytest.mark.parametrize("offset", [0.0, 0.1, 0.37])
def test_handover_is_the_first_state_the_incoming_link_owns(offset: float) -> None:
    """Across shifted supports, no published switch is a state too early or late.

    `offset` moves the incoming link's support relative to the outgoing one.
    At `0.0` the two share endpoints, which is the easy case: a link read at its
    own endpoint reads exactly. A shifted support forces every reading to travel,
    and the placement must survive that.
    """
    assert _misplaced_switches(offset, n_intercepts=60) == []


def _common_level_cells(n_cases: int):
    """Two-link node cells whose deciding rises are a few states of their level.

    Cell edges are candidate abscissae, so each link starts or ends on one. The
    slopes are expressed in states of the level per cell width, so the family
    means the same thing at either working precision.
    """
    dtype, jax_dtype = _working_dtypes()
    rng = np.random.default_rng(seed=0)

    left = rng.uniform(1.0, 100.0, n_cases).astype(dtype)
    width = rng.uniform(1e-3, 5.0, n_cases).astype(dtype)
    right = (left + width).astype(dtype)
    level = rng.uniform(1e4, 1e5, n_cases).astype(dtype)

    # The crossing sits within a few states of the cell's right edge, on either
    # side of it, which is where the walk and the located root can disagree.
    offsets = rng.integers(-6, 7, n_cases).astype(dtype)
    crossing = (right + offsets * np.spacing(right)).astype(dtype)

    per_state = (np.spacing(level) / width).astype(dtype)
    slope_a = (rng.uniform(-3.0, 3.0, n_cases) * per_state).astype(dtype)
    slope_b = (
        slope_a
        + rng.choice([-1.0, 1.0], n_cases) * rng.uniform(0.1, 3.0, n_cases) * per_state
    ).astype(dtype)

    x0a = (left - rng.uniform(0.1, 2.0, n_cases)).astype(dtype)
    x1a = right
    x0b = left
    x1b = (right + rng.uniform(0.1, 2.0, n_cases)).astype(dtype)
    endpoints = (
        np.stack([x0a, x1a, x0b, x1b], axis=-1),
        np.stack(
            [
                level + slope_a * (x0a - crossing),
                level + slope_a * (x1a - crossing),
                level + slope_b * (x0b - crossing),
                level + slope_b * (x1b - crossing),
            ],
            axis=-1,
        ).astype(dtype),
    )

    hull = jax.jit(
        jax.vmap(
            lambda cell_left, cell_right, grid, value: hull_owners(
                left=cell_left,
                right=cell_right,
                live=jnp.asarray([True, True]),
                low=jnp.asarray([0, 2], dtype=jnp.int32),
                high=jnp.asarray([1, 3], dtype=jnp.int32),
                endog_grid=grid,
                value=value,
                max_runs=2,
            )
        )
    )
    bounds, owners, refused = hull(
        jnp.asarray(left, jax_dtype),
        jnp.asarray(right, jax_dtype),
        jnp.asarray(endpoints[0], jax_dtype),
        jnp.asarray(endpoints[1], jax_dtype),
    )
    return {
        "left": left,
        "right": right,
        "grid": endpoints[0],
        "value": endpoints[1],
        "bounds": np.asarray(bounds),
        "owners": np.asarray(owners),
        "refused": np.asarray(refused),
    }


def _exact_handover(cells, case: int) -> float | None:
    """Smallest representable state at or above the exact crossing, clipped."""
    dtype, _jax_dtype = _working_dtypes()
    grid, value = cells["grid"][case], cells["value"][case]
    outgoing, incoming = (int(owner) for owner in cells["owners"][case])
    nodes = ((0, 1), (2, 3))

    def intercept_and_slope(link: int) -> tuple[Fraction, Fraction]:
        low, high = nodes[link]
        slope = (Fraction(float(value[high])) - Fraction(float(value[low]))) / (
            Fraction(float(grid[high])) - Fraction(float(grid[low]))
        )
        return Fraction(float(value[low])) - slope * Fraction(float(grid[low])), slope

    intercept_out, slope_out = intercept_and_slope(outgoing)
    intercept_in, slope_in = intercept_and_slope(incoming)
    if slope_out == slope_in:
        return None
    root = (intercept_out - intercept_in) / (slope_in - slope_out)

    state = dtype(float(root))
    while Fraction(float(state)) < root:
        state = np.nextafter(state, dtype(np.inf), dtype=dtype)
    while Fraction(float(np.nextafter(state, dtype(-np.inf), dtype=dtype))) >= root:
        state = np.nextafter(state, dtype(-np.inf), dtype=dtype)
    return min(
        max(float(state), float(cells["left"][case])), float(cells["right"][case])
    )


def test_handovers_on_a_common_value_level_are_never_refused() -> None:
    """No cell on a large common level withholds its handover.

    Where the value correspondence sits on a large common level, the rises that
    decide ownership are a handful of representable states wide and the crossing
    they imply lands on or beside a state rather than comfortably between two.
    That is the ordinary shape of a solved row near a kink, not a pathology, so
    every such cell must publish.
    """
    cells = _common_level_cells(n_cases=4000)
    hands_over = cells["owners"][:, 0] != cells["owners"][:, 1]
    assert int(np.sum(cells["refused"] & hands_over)) == 0


def test_published_handovers_match_the_exact_crossing_state() -> None:
    """Each published handover is the state an exact rational solve puts it at.

    The two links are lines with representable endpoints, so their crossing has
    an exact rational value and the state to publish is decided, not estimated.
    """
    cells = _common_level_cells(n_cases=4000)
    hands_over = cells["owners"][:, 0] != cells["owners"][:, 1]
    published = np.flatnonzero(hands_over & ~cells["refused"])[:200]
    mismatched = [
        case
        for case in published
        if _exact_handover(cells, int(case)) not in (None, cells["bounds"][case, 1])
    ]
    assert mismatched == []
