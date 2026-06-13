"""Upper-envelope refinement of EGM candidate solutions.

Discrete choices make the EGM candidate value correspondence multi-valued in
non-concave regions; the algorithms here select the upper envelope and drop
dominated candidates. The EGM step obtains its backend through
`get_upper_envelope`, so additional algorithms slot in without touching the
step itself. Currently implemented: the Fast Upper-Envelope Scan
(`_lcm.egm.upper_envelope.fues`).
"""

from typing import Protocol, runtime_checkable

from _lcm.egm.upper_envelope.fues import refine_envelope
from lcm.solvers import DCEGM
from lcm.typing import Float1D, ScalarInt


@runtime_checkable
class UpperEnvelopeBackend(Protocol):
    """Refine one row of EGM candidates to its upper envelope."""

    def __call__(
        self,
        *,
        endog_grid: Float1D,
        policy: Float1D,
        value: Float1D,
    ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
        """Return refined (grid, policy, value) rows and the kept-point count.

        The refined rows are NaN-padded to a static length; a kept-point
        count exceeding that length signals overflow (the rows then hold a
        truncated prefix of the envelope).
        """
        ...


def get_upper_envelope(*, solver: DCEGM, n_refined: int) -> UpperEnvelopeBackend:
    """Build the upper-envelope backend selected by the solver configuration.

    Args:
        solver: The regime's DC-EGM solver configuration; `upper_envelope`
            selects the backend and the `fues_*` fields parametrize it.
        n_refined: Static length of the refined output rows.

    Returns:
        The configured backend.

    """
    if solver.upper_envelope == "fues":

        def fues_backend(
            *,
            endog_grid: Float1D,
            policy: Float1D,
            value: Float1D,
        ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
            """Run the FUES scan with the solver's thresholds."""
            return refine_envelope(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                n_refined=n_refined,
                jump_thresh=solver.fues_jump_thresh,
                n_points_to_scan=solver.fues_n_points_to_scan,
                scan_unroll=solver.fues_scan_unroll,
            )

        return fues_backend

    msg = f"Unknown upper-envelope backend: {solver.upper_envelope!r}."
    raise ValueError(msg)
