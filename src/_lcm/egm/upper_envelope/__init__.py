"""Upper-envelope refinement of EGM candidate solutions.

Discrete choices make the EGM candidate value correspondence multi-valued in
non-concave regions; the algorithms here select the upper envelope and drop
dominated candidates. The EGM step obtains its backend through
`get_upper_envelope`, so additional algorithms slot in without touching the
step itself. Currently implemented:

- the Fast Upper-Envelope Scan (`_lcm.egm.upper_envelope.fues`), a sequential
  scan that inserts exact segment-crossing points, and
- the Rooftop-Cut algorithm (`_lcm.egm.upper_envelope.rfc`), a parallel
  dominance test that only deletes points (no crossing insertion) and
  generalizes to multidimensional endogenous grids.

Both backends share one signature: they consume the candidate
`(endog_grid, policy, value)` rows plus the candidate supgradient
`marginal_utility` ($\\mu = \\partial v / \\partial R$, exact by the envelope
theorem) and return a NaN-padded weakly-ascending refined `(grid, policy,
value)` triple plus the kept-point count. FUES ignores the supgradient; RFC
uses it to build each point's tangent.
"""

from typing import Protocol, runtime_checkable

from _lcm.egm.upper_envelope.fues import refine_envelope as refine_envelope_fues
from _lcm.egm.upper_envelope.rfc import refine_envelope as refine_envelope_rfc
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
        marginal_utility: Float1D,
    ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
        """Return refined (grid, policy, value) rows and the kept-point count.

        The supgradient `marginal_utility` carries $\\mu = \\partial v /
        \\partial R$ per candidate — the exact value-row slope by the envelope
        theorem. The refined rows are NaN-padded to a static length; a
        kept-point count exceeding that length signals overflow (the rows then
        hold a truncated prefix of the envelope).
        """
        ...


def get_upper_envelope(*, solver: DCEGM, n_refined: int) -> UpperEnvelopeBackend:
    """Build the upper-envelope backend selected by the solver configuration.

    Args:
        solver: The regime's DC-EGM solver configuration; `upper_envelope`
            selects the backend and the `fues_*` / `rfc_*` fields parametrize
            it.
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
            marginal_utility: Float1D,
        ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
            """Run the FUES scan with the solver's thresholds.

            FUES recovers segment slopes from its own forward scan, so the
            candidate supgradient is not consumed.
            """
            del marginal_utility
            return refine_envelope_fues(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                n_refined=n_refined,
                jump_thresh=solver.fues_jump_thresh,
                n_points_to_scan=solver.fues_n_points_to_scan,
            )

        return fues_backend

    if solver.upper_envelope == "rfc":

        def rfc_backend(
            *,
            endog_grid: Float1D,
            policy: Float1D,
            value: Float1D,
            marginal_utility: Float1D,
        ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
            """Run the rooftop cut with the solver's thresholds."""
            return refine_envelope_rfc(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                marginal_utility=marginal_utility,
                n_refined=n_refined,
                search_radius=solver.rfc_search_radius,
                jump_thresh=solver.rfc_jump_thresh,
            )

        return rfc_backend

    msg = f"Unknown upper-envelope backend: {solver.upper_envelope!r}."
    raise ValueError(msg)
