"""The branch-faithful policy read opens only for a crossing-complete envelope.

`_envelope_publishes_crossings` decides whether simulation may replace its
grid-argmax with an off-grid read of the published row. That substitution is only
sound when the row carries every envelope switch it interpolates across: a switch
the refinement never emitted is a candidate the read can neither reach nor be
rescued from, because the canonical-Q safeguard rescores only the candidate that
was emitted against the finite action grid.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope.mss import refine_envelope
from _lcm.regime_building.processing import _envelope_publishes_crossings
from lcm import LinSpacedGrid
from lcm.solvers import DCEGM


def _solver(
    upper_envelope: Literal["exact", "fues", "rfc", "ltm", "mss"],
) -> DCEGM:
    """A minimal DC-EGM solver differing only in its upper-envelope backend."""
    return DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=8),
        upper_envelope=upper_envelope,
    )


def test_mss_omits_an_interior_owner_that_the_read_would_need():
    """MSS can publish a row whose interior owner was never enumerated.

    Three branches meet so that the middle one, `consumption == 3`, strictly owns
    `R = 0.8` without owning either bracketing candidate abscissa. The refinement
    emits no record for it, so a read at `R = 0.8` returns some other branch's
    policy.
    """
    refined_grid, refined_policy, _refined_value, _n_kept = refine_envelope(
        endog_grid=jnp.asarray([0.0, 1.0, 0.0, 1.0, 0.6, 1.2]),
        policy=jnp.asarray([10.0, 10.0, 3.0, 3.0, 1.0, 1.0]),
        value=jnp.asarray([1.0, 1.1, 0.84, 0.84 + 1 / 3, 0.8, 1.4]),
        n_refined=16,
    )
    got = float(
        interp_on_padded_grid(
            x_query=jnp.asarray(0.8), xp=refined_grid, fp=refined_policy
        )
    )
    assert not np.isclose(got, 3.0), (
        "if MSS ever enumerates this owner, this test has served its purpose and "
        "the gate may reopen"
    )


def test_gate_stays_closed_for_mss():
    """`"mss"` does not certify crossing completeness, so the read stays off.

    Pairs with the omitted-owner witness above: while a row can miss an interior
    owner, opening the read would publish a policy that is not merely imprecise
    but a different, strictly worse action than the one the envelope implies.
    """
    assert _envelope_publishes_crossings(_solver("mss")) is False


def test_gate_stays_closed_for_every_shipped_backend():
    """No shipped upper envelope is certified crossing-complete."""
    backends: tuple[Literal["exact", "fues", "rfc", "ltm", "mss"], ...] = (
        "exact",
        "fues",
        "rfc",
        "ltm",
        "mss",
    )
    for backend in backends:
        assert _envelope_publishes_crossings(_solver(backend)) is False, backend
