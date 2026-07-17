"""The NEGM outer envelope equals the pointwise max of its candidate reads.

`thm:nnbegm` requires the published outer continuation to satisfy `V(q) = max_j
V_j(q)` at *every* query `q`, where each `V_j` is a candidate's conditional value
read through the parent's own interpolation convention (edge-clamped
Fritsch-Carlson-limited cubic Hermite, marginal row as node slopes). Taking the
maximum at the query — rather than at a shared node grid, republishing a single
interpolated row — is what makes this exact: a candidate that wins only on an
interval strictly between two nodes is read at its true value there instead of
being bridged upward, which interpolating an already-maximized row would do
(`thm:aggregate-bridge`).

The exactness gate reads an event-complete query set — every candidate knot,
every pairwise crossing on an overlap interval, every support boundary, and one
midpoint per inter-event gap — through both `outer_envelope_at_query` and an
independent host max-of-reads, and requires exact agreement, so a sub-spacing
island cannot hide between mesh points.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.outer_envelope import (
    build_stacked_outer_carry,
    outer_envelope_at_query,
)


def _host_envelope(
    candidate_endog: np.ndarray,
    candidate_value: np.ndarray,
    candidate_marginal: np.ndarray,
    x_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent host max-of-reads: value and winner's marginal per query.

    Each candidate is read one at a time through the same interpolation primitive
    the parent uses, masked to `-inf` below its own support, then the pointwise
    maximum and the winning candidate's marginal are taken in a plain loop — no
    shared vmap/argmax plumbing, so a bug in the reader under test cannot hide.
    """
    n_candidates = candidate_endog.shape[0]
    reads = np.full((n_candidates, x_query.shape[0]), -np.inf)
    marginal_reads = np.zeros((n_candidates, x_query.shape[0]))
    for j in range(n_candidates):
        endog = candidate_endog[j]
        lower = np.min(endog[np.isfinite(endog)])
        value_read = np.asarray(
            interp_on_padded_grid(
                x_query=jnp.asarray(x_query),
                xp=jnp.asarray(endog),
                fp=jnp.asarray(candidate_value[j]),
                fp_slopes=jnp.asarray(candidate_marginal[j]),
            )
        )
        marginal_reads[j] = np.asarray(
            interp_on_padded_grid(
                x_query=jnp.asarray(x_query),
                xp=jnp.asarray(endog),
                fp=jnp.asarray(candidate_marginal[j]),
            )
        )
        reads[j] = np.where(x_query < lower, -np.inf, value_read)
    winner = np.argmax(reads, axis=0)
    return reads.max(axis=0), marginal_reads[winner, np.arange(x_query.shape[0])]


def test_query_side_envelope_matches_host_max_of_reads_at_event_abscissae():
    """`outer_envelope_at_query` equals the pointwise max of candidate reads.

    Keeper `K(x) = x` and adjuster `A(x) = 1.5 - x` cross at `x = 0.75`. The
    event-complete query set — the shared knots `{0, 1, 2}`, the crossing `0.75`,
    the support boundary `0`, and a midpoint in each inter-event gap — is read
    both by the function under test and by an independent host max-of-reads; the
    two must agree exactly. At the crossing the envelope value is `0.75`, the true
    `max(K, A)`, never the value an interpolated single merged row would report.
    """
    candidate_endog = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    candidate_value = np.array([[0.0, 1.0, 2.0], [1.5, 0.5, -0.5]])
    candidate_marginal = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    events = np.array([0.0, 0.375, 0.75, 0.875, 1.0, 1.5, 2.0])

    value, marginal = outer_envelope_at_query(
        candidate_endog=jnp.asarray(candidate_endog),
        candidate_value=jnp.asarray(candidate_value),
        candidate_marginal=jnp.asarray(candidate_marginal),
        x_query=jnp.asarray(events),
    )
    host_value, host_marginal = _host_envelope(
        candidate_endog, candidate_value, candidate_marginal, events
    )

    np.testing.assert_allclose(np.asarray(value), host_value, atol=1e-9)
    np.testing.assert_allclose(np.asarray(marginal), host_marginal, atol=1e-9)
    np.testing.assert_allclose(
        float(value[events.tolist().index(0.75)]), 0.75, atol=1e-9
    )


def test_query_side_envelope_marginal_is_the_winner_slope_across_the_crossing():
    """The published marginal switches to the winner on each side of a crossing.

    With keeper `K(x) = x` (slope `+1`) winning above the crossing at `0.75` and
    adjuster `A(x) = 1.5 - x` (slope `-1`) winning below it, the winner-consistent
    marginal is `-1` just below `0.75` and `+1` just above — never an average.
    """
    candidate_endog = jnp.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    candidate_value = jnp.array([[0.0, 1.0, 2.0], [1.5, 0.5, -0.5]])
    candidate_marginal = jnp.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])

    _value, marginal = outer_envelope_at_query(
        candidate_endog=candidate_endog,
        candidate_value=candidate_value,
        candidate_marginal=candidate_marginal,
        x_query=jnp.array([0.7, 0.8]),
    )

    np.testing.assert_allclose(float(marginal[0]), -1.0, atol=1e-9)
    np.testing.assert_allclose(float(marginal[1]), 1.0, atol=1e-9)


def test_stacked_carry_lifts_candidates_into_common_coh_and_round_trips():
    """A stacked carry read at the query reproduces the exact `max_j V_j`.

    The keeper occupies coh space directly; the adjuster carries its value in its
    own resources space `R = coh - 0.5` (a credited cost of `0.5`), so its node
    grid `{0, 1, 2}` must be lifted to `{0.5, 1.5, 2.5}` before the max. Reading
    the stacked carry's single leading cell through `outer_envelope_at_query` must
    equal the max of the keeper read and the *lifted* adjuster read at every
    query — the lift is what puts both branches on the same coh axis.
    """
    keeper = EGMCarry(
        endog_grid=jnp.array([0.0, 1.0, 2.0])[None, :],
        value=jnp.array([0.0, 1.0, 2.0])[None, :],
        marginal_utility=jnp.array([1.0, 1.0, 1.0])[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    adjuster = EGMCarry(
        endog_grid=jnp.array([0.0, 1.0, 2.0])[None, :],
        value=jnp.array([1.0, 1.6, 2.0])[None, :],
        marginal_utility=jnp.array([0.8, 0.5, 0.3])[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )

    stacked = build_stacked_outer_carry(
        keeper_carry=keeper,
        adjuster_carries=(adjuster,),
        coh_shifts=jnp.asarray([[0.5]]),
    )
    # Leading shape is (durable=1, n_candidates=2); the trailing axis is the grid.
    assert stacked.endog_grid.shape == (1, 2, 3)
    np.testing.assert_allclose(
        np.asarray(stacked.endog_grid[0, 1]), np.array([0.5, 1.5, 2.5]), atol=1e-9
    )

    events = jnp.array([0.5, 1.0, 1.25, 1.5, 2.0, 2.5])
    value, marginal = outer_envelope_at_query(
        candidate_endog=stacked.endog_grid[0],
        candidate_value=stacked.value[0],
        candidate_marginal=stacked.marginal_utility[0],
        x_query=events,
    )
    host_value, host_marginal = _host_envelope(
        np.array([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]]),
        np.array([[0.0, 1.0, 2.0], [1.0, 1.6, 2.0]]),
        np.array([[1.0, 1.0, 1.0], [0.8, 0.5, 0.3]]),
        np.asarray(events),
    )

    np.testing.assert_allclose(np.asarray(value), host_value, atol=1e-9)
    np.testing.assert_allclose(np.asarray(marginal), host_marginal, atol=1e-9)


def test_query_below_every_candidate_support_is_masked_out():
    """A query below every candidate's first finite node yields `(-inf, 0)`.

    Both candidates' support starts at `x = 1.0`; a query at `0.5` is below both,
    so the envelope value is `-inf` (no candidate is feasible), never a clamped
    boundary value that would let an infeasible branch win spuriously — and the
    published marginal is exactly zero, matching the infeasible contract the
    parent's probability-weighted expectation relies on.
    """
    candidate_endog = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    candidate_value = jnp.array([[0.0, 1.0, 2.0], [0.5, 0.4, 0.3]])
    candidate_marginal = jnp.array([[1.0, 1.0, 1.0], [-0.1, -0.1, -0.1]])

    value, marginal = outer_envelope_at_query(
        candidate_endog=candidate_endog,
        candidate_value=candidate_value,
        candidate_marginal=candidate_marginal,
        x_query=jnp.array([0.5]),
    )

    assert float(value[0]) == float("-inf")
    assert float(marginal[0]) == 0.0


def test_exact_candidate_tie_publishes_the_right_continuous_marginal():
    """At an exact crossing the winner is the candidate that wins to the right.

    Keeper `K(q) = 1 - q` (marginal `-1`) and adjuster `A(q) = q` (marginal `+1`)
    tie at `q = 0.5`. Immediately to the right the adjuster wins, so the
    right-continuous envelope's marginal there is `+1`; publishing the keeper's
    `-1` (a first-index tie) would feed the wrong one-sided derivative into the
    parent's Euler inversion.
    """
    candidate_endog = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    candidate_value = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    candidate_marginal = jnp.array([[-1.0, -1.0], [1.0, 1.0]])

    value, marginal = outer_envelope_at_query(
        candidate_endog=candidate_endog,
        candidate_value=candidate_value,
        candidate_marginal=candidate_marginal,
        x_query=jnp.array([0.5]),
    )

    np.testing.assert_allclose(float(value[0]), 0.5, atol=1e-9)
    np.testing.assert_allclose(float(marginal[0]), 1.0, atol=1e-9)


def test_tie_at_a_support_edge_prefers_the_candidate_that_continues_right():
    """A candidate whose support ends at the tie point loses to one continuing.

    Candidate A spans `[0, 1]` with the steeper marginal but ends exactly at the
    query `q = 1`; candidate B spans `[1, 2]` and continues to the right. Both
    read value `1.0` at the query, but only B exists immediately to the right,
    so the right-continuous winner is B and the published marginal is B's.
    """
    candidate_endog = jnp.array([[0.0, 1.0], [1.0, 2.0]])
    candidate_value = jnp.array([[0.0, 1.0], [1.0, 1.5]])
    candidate_marginal = jnp.array([[5.0, 5.0], [0.5, 0.5]])

    value, marginal = outer_envelope_at_query(
        candidate_endog=candidate_endog,
        candidate_value=candidate_value,
        candidate_marginal=candidate_marginal,
        x_query=jnp.array([1.0]),
    )

    np.testing.assert_allclose(float(value[0]), 1.0, atol=1e-9)
    np.testing.assert_allclose(float(marginal[0]), 0.5, atol=1e-9)
