"""The branch-faithful policy read opens only for a crossing-complete envelope.

`_envelope_publishes_crossings` decides whether simulation may replace its
grid-argmax with an off-grid read of the published row. That substitution is only
sound when the row carries every envelope switch it interpolates across: a switch
the refinement never emitted is a candidate the read can neither reach nor be
rescued from, because the canonical-Q safeguard rescores only the candidate that
was emitted against the finite action grid.
"""

from typing import Literal

from _lcm.regime_building.processing import _envelope_publishes_crossings
from lcm import LinSpacedGrid
from lcm.solvers import DCEGM
from tests.envelope_configs import envelope_config


def _solver(
    envelope: Literal["exact", "fues", "rfc", "ltm"],
) -> DCEGM:
    """A minimal DC-EGM solver differing only in its upper-envelope backend."""
    return DCEGM(
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=8),
        envelope=envelope_config(envelope),
    )


def test_gate_stays_closed_for_every_shipped_backend():
    """No shipped upper envelope is certified crossing-complete."""
    backends: tuple[Literal["exact", "fues", "rfc", "ltm"], ...] = (
        "exact",
        "fues",
        "rfc",
        "ltm",
    )
    for backend in backends:
        assert _envelope_publishes_crossings(_solver(backend)) is False, backend
