"""Upper-envelope refinement of EGM candidate solutions.

Discrete choices make the EGM candidate value correspondence multi-valued in
non-concave regions; the algorithms here select the upper envelope and drop
dominated candidates. The EGM step obtains its backend through
`get_upper_envelope`, so additional algorithms slot in without touching the
step itself. Currently implemented:

- the Fast Upper-Envelope Scan (`_lcm.egm.upper_envelope.fues`), a sequential
  scan that inserts exact segment-crossing points,
- the Rooftop-Cut algorithm (`_lcm.egm.upper_envelope.rfc`), a parallel
  dominance test that only deletes points (no crossing insertion) and
  generalizes to multidimensional endogenous grids, and
- the local-upper-bound brute method (`_lcm.egm.upper_envelope.ltm`), an
  $O(K^2)$ dense segment scan that evaluates the envelope at every candidate
  abscissa (the quadratic baseline of Dobrescu & Shanker 2026), and
- HARK's EGM upper envelope (`_lcm.egm.upper_envelope.mss`), a left-to-right
  sweep that keeps the max-value branch at every abscissa *and* inserts the
  exact segment-crossing point (the `MSS` method of Dobrescu & Shanker 2026).

All backends share one signature: they consume the candidate
`(endog_grid, policy, value)` rows plus the candidate supgradient
`marginal_utility` ($\\mu = \\partial v / \\partial R$, exact by the envelope
theorem) and return a NaN-padded weakly-ascending refined `(grid, policy,
value)` triple plus the kept-point count. FUES, LTM, and MSS ignore the
supgradient (they recover slopes from the segments); RFC uses it to build each
point's tangent.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax.numpy as jnp

from _lcm.egm.interp import prepare_padded_grid
from _lcm.egm.upper_envelope.fues import (
    QueryBracket,
    refine_to_bracket,
)
from _lcm.egm.upper_envelope.fues import (
    refine_envelope as refine_envelope_fues,
)
from _lcm.egm.upper_envelope.ltm import refine_envelope as refine_envelope_ltm
from _lcm.egm.upper_envelope.mss import refine_envelope as refine_envelope_mss
from _lcm.egm.upper_envelope.rfc import refine_envelope as refine_envelope_rfc
from lcm.solvers import DCEGM
from lcm.typing import Float1D, ScalarFloat, ScalarInt


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
        savings: Float1D,
    ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
        """Return refined (grid, policy, value) rows and the kept-point count.

        The supgradient `marginal_utility` carries $\\mu = \\partial v /
        \\partial R$ per candidate — the exact value-row slope by the envelope
        theorem. `savings` carries each candidate's exogenous source savings
        (the savings node for Euler candidates, the borrowing limit for
        constrained ones); FUES uses it to judge savings monotonicity exactly,
        the other backends ignore it. The refined rows are NaN-padded to a
        static length; a kept-point count exceeding that length signals overflow
        (the rows then hold a truncated prefix of the envelope).
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
            savings: Float1D,
        ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
            """Run the FUES scan with the solver's thresholds.

            FUES recovers segment slopes from its own forward scan, so the
            candidate supgradient is not consumed; the exogenous source savings
            resolve the savings-monotonicity test exactly.
            """
            del marginal_utility
            return refine_envelope_fues(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                n_refined=n_refined,
                jump_thresh=solver.fues_jump_thresh,
                n_points_to_scan=solver.fues_n_points_to_scan,
                savings=savings,
                scan_unroll=solver.fues_scan_unroll,
            )

        return fues_backend

    if solver.upper_envelope == "rfc":

        def rfc_backend(
            *,
            endog_grid: Float1D,
            policy: Float1D,
            value: Float1D,
            marginal_utility: Float1D,
            savings: Float1D,
        ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
            """Run the rooftop cut with the solver's thresholds.

            The exogenous source savings are a FUES-only refinement; RFC judges
            monotonicity from its own geometry.
            """
            del savings
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

    if solver.upper_envelope == "ltm":

        def ltm_backend(
            *,
            endog_grid: Float1D,
            policy: Float1D,
            value: Float1D,
            marginal_utility: Float1D,
            savings: Float1D,
        ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
            """Run the brute local-upper-bound scan.

            LTM recovers segment slopes from the candidate chain, so neither the
            candidate supgradient nor the exogenous source savings are consumed.
            """
            del marginal_utility, savings
            return refine_envelope_ltm(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                n_refined=n_refined,
            )

        return ltm_backend

    if solver.upper_envelope == "mss":

        def mss_backend(
            *,
            endog_grid: Float1D,
            policy: Float1D,
            value: Float1D,
            marginal_utility: Float1D,
            savings: Float1D,
        ) -> tuple[Float1D, Float1D, Float1D, ScalarInt]:
            """Run HARK's EGM upper-envelope sweep with crossing insertion.

            MSS recovers segment slopes from the candidate chain, so neither the
            candidate supgradient nor the exogenous source savings are consumed.
            """
            del marginal_utility, savings
            return refine_envelope_mss(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                n_refined=n_refined,
            )

        return mss_backend

    msg = f"Unknown upper-envelope backend: {solver.upper_envelope!r}."
    raise ValueError(msg)


def get_bracket_finder(*, solver: DCEGM, n_refined: int) -> Callable[..., QueryBracket]:
    """Build the single-query bracket finder for the asset-row solve.

    The geometry-only counterpart of `get_upper_envelope` for asset-row mode,
    where the refined envelope is read at exactly one query per node: it returns
    the two bracketing envelope nodes (plus the first node and the kept count).
    It is deliberately not on the `UpperEnvelopeBackend` Protocol — the backend
    returns envelope geometry; the asset-row module owns the EGM economics
    (utility gradients, the borrowing limit, the constrained floor).

    The backends differ in how they reach that bracket:

    - `"fues"` streams it: `refine_to_bracket` runs the FUES scan and folds each
      step's emissions into an O(1) bracket-capture carry, so the NaN-padded
      `n_pad` envelope never materializes.
    - `"rfc"`, `"ltm"`, and `"mss"` do *not* stream: their dense scans have no
      sequential carry to fold a bracket out of, so the finder materializes the
      full refined envelope and locates the same
      `searchsorted(side="right")`-clamped bracket the row path would read. The
      published `(value, policy)` is therefore identical to a
      full-envelope-then-interpolate, but these asset-row paths do *not* yet get
      refine-to-query's `n_pad` memory win — a streamed dense bracket finder is
      future work.

    Args:
        solver: The regime's DC-EGM solver configuration; the `fues_*` / `rfc_*`
            fields parametrize the scan.
        n_refined: Static length of the refined envelope row the dense finder
            materializes before locating the bracket (unused by FUES, which
            streams). This is the `n_pad` overflow threshold the asset-row
            publish compares `n_kept` against.

    Returns:
        The configured bracket finder.

    """
    if solver.upper_envelope == "fues":

        def fues_bracket_finder(
            *,
            endog_grid: Float1D,
            policy: Float1D,
            value: Float1D,
            marginal_utility: Float1D,
            savings: Float1D,
            x_query: ScalarFloat,
        ) -> QueryBracket:
            """Run the streaming FUES scan with the solver's thresholds.

            FUES recovers segment slopes from its own forward scan, so the
            candidate supgradient is not consumed; the exogenous source savings
            resolve the savings-monotonicity test exactly.
            """
            del marginal_utility
            return refine_to_bracket(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                x_query=x_query,
                n_refined=n_refined,
                jump_thresh=solver.fues_jump_thresh,
                n_points_to_scan=solver.fues_n_points_to_scan,
                savings=savings,
                scan_unroll=solver.fues_scan_unroll,
            )

        return fues_bracket_finder

    if solver.upper_envelope == "rfc":

        def rfc_bracket_finder(
            *,
            endog_grid: Float1D,
            policy: Float1D,
            value: Float1D,
            marginal_utility: Float1D,
            savings: Float1D,
            x_query: ScalarFloat,
        ) -> QueryBracket:
            """Locate the query bracket from RFC's full refined envelope.

            Runs the rooftop cut to a full NaN-padded envelope row, then reads
            the bracket the row path would: the `searchsorted(side="right")`
            pair clamped to `[1, max(n_kept - 1, 1)]` (the
            `interp_on_prepared_grid` rule), so the published value cannot
            diverge from full-envelope-then-interpolate. The exogenous source
            savings are a FUES-only refinement.
            """
            del savings
            refined_grid, refined_policy, refined_value, n_kept = refine_envelope_rfc(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                marginal_utility=marginal_utility,
                n_refined=n_refined,
                search_radius=solver.rfc_search_radius,
                jump_thresh=solver.rfc_jump_thresh,
            )
            return _bracket_from_refined_row(
                refined_grid=refined_grid,
                refined_policy=refined_policy,
                refined_value=refined_value,
                n_kept=n_kept,
                x_query=x_query,
            )

        return rfc_bracket_finder

    if solver.upper_envelope == "ltm":

        def ltm_bracket_finder(
            *,
            endog_grid: Float1D,
            policy: Float1D,
            value: Float1D,
            marginal_utility: Float1D,
            savings: Float1D,
            x_query: ScalarFloat,
        ) -> QueryBracket:
            """Locate the query bracket from LTM's full refined envelope.

            Runs the brute scan to a full NaN-padded envelope row, then reads
            the bracket the row path would: the `searchsorted(side="right")`
            pair clamped to `[1, max(n_kept - 1, 1)]` (the
            `interp_on_prepared_grid` rule), so the published value cannot
            diverge from full-envelope-then-interpolate. Like RFC, LTM has no
            sequential carry to stream a bracket from, so it does not get
            refine-to-query's `n_pad` memory win; the exogenous source savings
            are a FUES-only refinement.
            """
            del marginal_utility, savings
            refined_grid, refined_policy, refined_value, n_kept = refine_envelope_ltm(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                n_refined=n_refined,
            )
            return _bracket_from_refined_row(
                refined_grid=refined_grid,
                refined_policy=refined_policy,
                refined_value=refined_value,
                n_kept=n_kept,
                x_query=x_query,
            )

        return ltm_bracket_finder

    if solver.upper_envelope == "mss":

        def mss_bracket_finder(
            *,
            endog_grid: Float1D,
            policy: Float1D,
            value: Float1D,
            marginal_utility: Float1D,
            savings: Float1D,
            x_query: ScalarFloat,
        ) -> QueryBracket:
            """Locate the query bracket from MSS's full refined envelope.

            Runs the HARK sweep to a full NaN-padded envelope row, then reads the
            bracket the row path would: the `searchsorted(side="right")` pair
            clamped to `[1, max(n_kept - 1, 1)]` (the `interp_on_prepared_grid`
            rule), so the published value cannot diverge from
            full-envelope-then-interpolate. Like RFC and LTM, MSS has no
            sequential carry to stream a bracket from, so it does not get
            refine-to-query's `n_pad` memory win; the exogenous source savings
            are a FUES-only refinement.
            """
            del marginal_utility, savings
            refined_grid, refined_policy, refined_value, n_kept = refine_envelope_mss(
                endog_grid=endog_grid,
                policy=policy,
                value=value,
                n_refined=n_refined,
            )
            return _bracket_from_refined_row(
                refined_grid=refined_grid,
                refined_policy=refined_policy,
                refined_value=refined_value,
                n_kept=n_kept,
                x_query=x_query,
            )

        return mss_bracket_finder

    msg = f"Unknown upper-envelope backend: {solver.upper_envelope!r}."
    raise ValueError(msg)


def _bracket_from_refined_row(
    *,
    refined_grid: Float1D,
    refined_policy: Float1D,
    refined_value: Float1D,
    n_kept: ScalarInt,
    x_query: ScalarFloat,
) -> QueryBracket:
    """Read the query bracket off a full NaN-padded refined envelope row.

    Reproduces the asset-row publish's bracket location node-for-node by reusing
    the `interp_on_prepared_grid` rule: `searchsorted(search_grid, x_query,
    side="right")` clamped to `[1, max(valid_length - 1, 1)]`, with the NaN tail
    treated as $+\\infty$. The gathered `(lower, upper)` pair, `first_grid`, and
    `n_kept` are exactly what `publish_node_from_bracket` consumes, so a
    full-envelope-then-interpolate and this bracket read publish identical
    `(value, policy)`.

    Args:
        refined_grid: Refined endogenous grid row (NaN-padded tail).
        refined_policy: Refined policy row, NaN-padded in lockstep.
        refined_value: Refined value row, NaN-padded in lockstep.
        n_kept: Number of envelope points kept; `> n_pad` signals overflow.
        x_query: The single point at which the envelope is read.

    Returns:
        The query bracket.

    """
    search_grid, valid_length = prepare_padded_grid(refined_grid)
    upper = jnp.clip(
        jnp.searchsorted(search_grid, x_query, side="right"),
        1,
        jnp.maximum(valid_length - 1, 1),
    ).astype(jnp.int32)
    lower = upper - 1
    dtype = refined_grid.dtype
    return QueryBracket(
        lower_grid=refined_grid[lower].astype(dtype),
        upper_grid=refined_grid[upper].astype(dtype),
        lower_policy=refined_policy[lower].astype(dtype),
        upper_policy=refined_policy[upper].astype(dtype),
        lower_value=refined_value[lower].astype(dtype),
        upper_value=refined_value[upper].astype(dtype),
        first_grid=refined_grid[0].astype(dtype),
        n_kept=n_kept,
    )
