"""A published switch lands on the first state the incoming link owns.

The refined row carries a switch as a duplicated abscissa holding both branches'
records, and the reader is right-continuous there, so the abscissa decides which
policy is published from that state onward. Placing it one representable state
too low publishes the incoming policy over ground the outgoing link still owns;
one too high leaves the outgoing policy in place after ownership has passed.

Ownership is a structural predicate, so it is asserted as a decision rather than
to a tolerance, and the decision is taken by `certified_margin_sign` on the
links' *stored endpoints* — the lines themselves — not on anything the envelope
computed on the way.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.mark.parametrize("offset", [0.0, 0.1, 0.37])
def test_handover_is_the_first_state_the_incoming_link_owns(offset: float) -> None:
    """Across shifted supports, no published switch is a state too early or late.

    `offset` moves the incoming link's support relative to the outgoing one.
    At `0.0` the two share endpoints, which is the easy case: a link read at its
    own endpoint reads exactly. A shifted support forces every reading to travel,
    and the placement must survive that.
    """
    assert _misplaced_switches(offset, n_intercepts=60) == []
