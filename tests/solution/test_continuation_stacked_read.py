"""The production child-carry read is exact on a stacked NEGM carry.

`_aggregate_child_choices` is the read every parent kernel runs per savings
node. For a stacked NEGM child it must collapse the candidate axis by the hard
max at the query — per durable node, *before* the passive blend — publishing the
winning candidate's gradient-scaled marginal, and mask each candidate to `-inf`
below its own first finite coh node. These tests pin that behavior against an
independent host computation through the same interpolation primitive.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import _aggregate_child_choices
from _lcm.egm.interp import interp_on_padded_grid, prepare_padded_grid
from tests.conftest import X64_ENABLED

# The host comparison reruns the identical interpolation arithmetic, so the two
# sides agree to the active float precision's roundoff, not better.
_READ_RTOL = 1e-12 if X64_ENABLED else 2e-5
_READ_ATOL = 1e-12 if X64_ENABLED else 2e-5


def _prepare(carry: EGMCarry):
    """Prepare the whole carry's search keys and valid lengths, as the kernel does."""
    n_rows = carry.endog_grid.shape[-1]
    flat_search, flat_valid = jax.vmap(prepare_padded_grid)(
        carry.endog_grid.reshape(-1, n_rows)
    )
    return (
        flat_search.reshape(carry.endog_grid.shape),
        flat_valid.reshape(carry.endog_grid.shape[:-1]),
    )


def _host_read(
    endog: np.ndarray, value: np.ndarray, marginal: np.ndarray, query: float
) -> tuple[float, float]:
    """Read one candidate row at `query`: Hermite value, linear marginal, -inf mask."""
    lower = float(np.min(endog[np.isfinite(endog)]))
    value_read = float(
        interp_on_padded_grid(
            x_query=jnp.asarray([query]),
            xp=jnp.asarray(endog),
            fp=jnp.asarray(value),
            fp_slopes=jnp.asarray(marginal),
        )[0]
    )
    marginal_read = float(
        interp_on_padded_grid(
            x_query=jnp.asarray([query]),
            xp=jnp.asarray(endog),
            fp=jnp.asarray(marginal),
        )[0]
    )
    if query < lower:
        return float("-inf"), marginal_read
    return value_read, marginal_read


def test_stacked_read_blends_the_nodewise_candidate_maximum():
    """The read equals `sum_k w_k max_j V_j(q; d_k)`, not `max_j sum_k w_k V_j`.

    Two durable nodes, two candidates whose winner *switches* across the nodes:
    at node 0 candidate A (value 2) beats B (value 0); at node 1 B (value 3)
    beats A (value 1). At the blend point `d' = 0.25` (weights 0.75 / 0.25) the
    nodewise maximum blends to `0.75 * 2 + 0.25 * 3 = 2.25`, whereas maximizing
    the blended candidates would give `max(0.75*2 + 0.25*1, 0.75*0 + 0.25*3) =
    1.75` — a strict lower bound. The read must publish 2.25.
    """
    n_pad = 3
    coh = jnp.array([0.0, 5.0, 10.0])
    flat = jnp.zeros(n_pad)

    def rows(value_a: float, value_b: float) -> jnp.ndarray:
        return jnp.stack([jnp.full((n_pad,), value_a), jnp.full((n_pad,), value_b)])

    # Leading shape (durable=2, candidates=2); flat value rows, zero marginals.
    carry = EGMCarry(
        endog_grid=jnp.broadcast_to(coh, (2, 2, n_pad)),
        value=jnp.stack([rows(2.0, 0.0), rows(1.0, 3.0)]),
        marginal_utility=jnp.broadcast_to(flat, (2, 2, n_pad)),
        taste_shock_scale=jnp.asarray(0.0),
    )
    prepared_search_grid, prepared_valid_length = _prepare(carry)

    smoothed_value, smoothed_marginal = _aggregate_child_choices(
        carry=carry,
        prepared_search_grid=prepared_search_grid,
        prepared_valid_length=prepared_valid_length,
        has_taste_shocks=False,
        child_index=(),
        child_passive_values=(jnp.asarray(0.25),),
        child_passive_grids=(jnp.asarray([0.0, 1.0]),),
        row_queries=jnp.asarray([4.0, 4.0]),
        row_gradients=jnp.asarray([1.0, 1.0]),
        n_outer_candidates=2,
    )

    np.testing.assert_allclose(float(smoothed_value), 2.25, atol=_READ_ATOL)
    np.testing.assert_allclose(float(smoothed_marginal), 0.0, atol=_READ_ATOL)


def test_stacked_read_masks_a_candidate_below_its_lifted_support():
    """A candidate is `-inf` below its own support even if its clamp would win.

    One durable node, two candidates. Candidate B carries the larger values but
    its lifted support starts at coh 6, above the query 4; the edge clamp would
    hand it its boundary value 9. The read must mask B to `-inf` there and
    publish candidate A's value and gradient-scaled marginal instead.
    """
    carry = EGMCarry(
        endog_grid=jnp.stack(
            [jnp.array([0.0, 5.0, 10.0]), jnp.array([6.0, 8.0, 10.0])]
        )[None, :, :],
        value=jnp.stack([jnp.array([1.0, 1.0, 1.0]), jnp.array([9.0, 9.5, 10.0])])[
            None, :, :
        ],
        marginal_utility=jnp.stack(
            [jnp.array([0.5, 0.5, 0.5]), jnp.array([2.0, 2.0, 2.0])]
        )[None, :, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    prepared_search_grid, prepared_valid_length = _prepare(carry)

    smoothed_value, smoothed_marginal = _aggregate_child_choices(
        carry=carry,
        prepared_search_grid=prepared_search_grid,
        prepared_valid_length=prepared_valid_length,
        has_taste_shocks=False,
        child_index=(),
        child_passive_values=(jnp.asarray(0.0),),
        child_passive_grids=(jnp.asarray([0.0]),),
        row_queries=jnp.asarray([4.0]),
        row_gradients=jnp.asarray([3.0]),
        n_outer_candidates=2,
    )

    np.testing.assert_allclose(float(smoothed_value), 1.0, atol=_READ_ATOL)
    # Winner A's marginal 0.5, scaled by the composed gradient 3.0.
    np.testing.assert_allclose(float(smoothed_marginal), 1.5, atol=_READ_ATOL)


def test_stacked_read_propagates_a_poisoned_candidate_row():
    """An all-NaN (poisoned) candidate row poisons the stacked read's value.

    A poisoned carry row marks an upstream overflow; the production read must
    keep the NaN through the candidate maximum — fail-loud for the runtime
    diagnostics — instead of masking the row as ordinarily infeasible and
    letting the live candidate win silently.
    """
    poisoned = jnp.full((3,), jnp.nan)
    live = jnp.array([0.0, 5.0, 10.0])
    carry = EGMCarry(
        endog_grid=jnp.stack([poisoned, live])[None, :, :],
        value=jnp.stack([poisoned, jnp.array([1.0, 2.0, 3.0])])[None, :, :],
        marginal_utility=jnp.stack([poisoned, jnp.array([0.5, 0.5, 0.5])])[None, :, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    prepared_search_grid, prepared_valid_length = _prepare(carry)

    smoothed_value, _ = _aggregate_child_choices(
        carry=carry,
        prepared_search_grid=prepared_search_grid,
        prepared_valid_length=prepared_valid_length,
        has_taste_shocks=False,
        child_index=(),
        child_passive_values=(jnp.asarray(0.0),),
        child_passive_grids=(jnp.asarray([0.0]),),
        row_queries=jnp.asarray([4.0]),
        row_gradients=jnp.asarray([1.0]),
        n_outer_candidates=2,
    )

    assert bool(jnp.isnan(smoothed_value))


def test_stacked_read_with_a_singleton_candidate_is_differentiable():
    """A singleton candidate keeps the read differentiable in the query.

    Asset-row mode publishes the Euler marginal by differentiating the
    continuation read, so a one-node candidate carry — a constant clamp in the
    query — must contribute a zero tangent, not a NaN one leaked from its
    padded bracket. The singleton's clamp value wins here, so the read's query
    derivative is exactly zero.
    """
    singleton = jnp.array([2.0, jnp.nan, jnp.nan])
    live = jnp.array([0.0, 5.0, 10.0])
    carry = EGMCarry(
        endog_grid=jnp.stack([singleton, live])[None, :, :],
        value=jnp.stack(
            [jnp.array([10.0, jnp.nan, jnp.nan]), jnp.array([1.0, 2.0, 3.0])]
        )[None, :, :],
        marginal_utility=jnp.stack(
            [jnp.array([0.0, jnp.nan, jnp.nan]), jnp.array([0.5, 0.5, 0.5])]
        )[None, :, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    prepared_search_grid, prepared_valid_length = _prepare(carry)

    def read_value(query):
        smoothed_value, _ = _aggregate_child_choices(
            carry=carry,
            prepared_search_grid=prepared_search_grid,
            prepared_valid_length=prepared_valid_length,
            has_taste_shocks=False,
            child_index=(),
            child_passive_values=(jnp.asarray(0.0),),
            child_passive_grids=(jnp.asarray([0.0]),),
            row_queries=query[None],
            row_gradients=jnp.asarray([1.0]),
            n_outer_candidates=2,
        )
        return smoothed_value

    value, derivative = jax.value_and_grad(read_value)(jnp.asarray(4.0))

    np.testing.assert_allclose(float(value), 10.0, atol=_READ_ATOL)
    np.testing.assert_allclose(float(derivative), 0.0, atol=_READ_ATOL)


def test_stacked_read_tie_publishes_the_right_continuous_marginal():
    """At an exact candidate crossing the production read selects the right winner.

    Keeper `K(q) = 1 - q` (marginal `-1`) and adjuster `A(q) = q` (marginal `+1`)
    tie at the query `q = 0.5`. The right-continuous envelope's marginal there is
    `+1` (times the composed gradient `2.0`); a first-index tie would publish the
    keeper's `-1` — the branch that loses immediately to the right — into the
    parent's Euler inversion.
    """
    carry = EGMCarry(
        endog_grid=jnp.stack([jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])])[
            None, :, :
        ],
        value=jnp.stack([jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])])[None, :, :],
        marginal_utility=jnp.stack([jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])])[
            None, :, :
        ],
        taste_shock_scale=jnp.asarray(0.0),
    )
    prepared_search_grid, prepared_valid_length = _prepare(carry)

    smoothed_value, smoothed_marginal = _aggregate_child_choices(
        carry=carry,
        prepared_search_grid=prepared_search_grid,
        prepared_valid_length=prepared_valid_length,
        has_taste_shocks=False,
        child_index=(),
        child_passive_values=(jnp.asarray(0.0),),
        child_passive_grids=(jnp.asarray([0.0]),),
        row_queries=jnp.asarray([0.5]),
        row_gradients=jnp.asarray([2.0]),
        n_outer_candidates=2,
    )

    np.testing.assert_allclose(float(smoothed_value), 0.5, atol=_READ_ATOL)
    np.testing.assert_allclose(float(smoothed_marginal), 2.0, atol=_READ_ATOL)


def test_stacked_read_tie_owner_follows_the_limited_value_slope():
    """At a tie the production read ranks by the value read's actual right slope.

    Candidate A carries a raw node marginal of `100` at the tie query, but its
    value row rises only by `0.1` per bracket, so the Fritsch-Carlson limiter
    caps the value read's right slope at three times the secant (`0.3`).
    Candidate B's value rises by `1.0` per bracket, so B actually wins
    immediately right of the tie and B's gradient-scaled marginal (`1.0 * 2.0`)
    must be published — ranking by the raw marginal would publish A's
    `100 * 2.0`.
    """
    carry = EGMCarry(
        endog_grid=jnp.broadcast_to(jnp.array([0.0, 1.0, 2.0]), (1, 2, 3)),
        value=jnp.stack([jnp.array([0.9, 1.0, 1.1]), jnp.array([0.0, 1.0, 2.0])])[
            None, :, :
        ],
        marginal_utility=jnp.stack(
            [jnp.array([0.1, 100.0, 0.1]), jnp.array([1.0, 1.0, 1.0])]
        )[None, :, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    prepared_search_grid, prepared_valid_length = _prepare(carry)

    smoothed_value, smoothed_marginal = _aggregate_child_choices(
        carry=carry,
        prepared_search_grid=prepared_search_grid,
        prepared_valid_length=prepared_valid_length,
        has_taste_shocks=False,
        child_index=(),
        child_passive_values=(jnp.asarray(0.0),),
        child_passive_grids=(jnp.asarray([0.0]),),
        row_queries=jnp.asarray([1.0]),
        row_gradients=jnp.asarray([2.0]),
        n_outer_candidates=2,
    )

    np.testing.assert_allclose(float(smoothed_value), 1.0, atol=_READ_ATOL)
    np.testing.assert_allclose(float(smoothed_marginal), 2.0, atol=_READ_ATOL)


def test_stacked_read_tie_with_equal_right_slopes_follows_the_curvature():
    """Equal first right derivatives at a tie resolve by the read's curvature.

    Both candidates tie at the query `q = 1` with limited right derivative
    exactly 3 (A's raw node slope 100 is limiter-capped, B's slope 3 passes),
    but B's Hermite piece curves less steeply downward, so B's read is strictly
    larger for every `q > 1`. The published marginal must be B's, scaled by the
    composed gradient (`3.0 * 2.0`), not the lower-index candidate A's raw 100.
    """
    carry = EGMCarry(
        endog_grid=jnp.broadcast_to(jnp.array([0.0, 1.0, 2.0]), (1, 2, 3)),
        value=jnp.stack([jnp.array([-1.0, 0.0, 1.0]), jnp.array([-2.0, 0.0, 2.0])])[
            None, :, :
        ],
        marginal_utility=jnp.stack(
            [jnp.array([1.0, 100.0, 1.0]), jnp.array([2.0, 3.0, 2.0])]
        )[None, :, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    prepared_search_grid, prepared_valid_length = _prepare(carry)

    smoothed_value, smoothed_marginal = _aggregate_child_choices(
        carry=carry,
        prepared_search_grid=prepared_search_grid,
        prepared_valid_length=prepared_valid_length,
        has_taste_shocks=False,
        child_index=(),
        child_passive_values=(jnp.asarray(0.0),),
        child_passive_grids=(jnp.asarray([0.0]),),
        row_queries=jnp.asarray([1.0]),
        row_gradients=jnp.asarray([2.0]),
        n_outer_candidates=2,
    )

    np.testing.assert_allclose(float(smoothed_value), 0.0, atol=_READ_ATOL)
    np.testing.assert_allclose(float(smoothed_marginal), 6.0, atol=_READ_ATOL)


def test_blend_of_a_dead_and_a_live_passive_node_keeps_the_infeasible_pair():
    """A `-inf` blended value carries an exactly-zero marginal.

    At the lower passive node every candidate's lifted support starts above the
    query, so that node reads the infeasible pair `(-inf, 0)`; the upper node is
    live with marginal `2.0`. With positive weights on both nodes the blended
    value is `-inf` — the cell is infeasible — so the published marginal must be
    exactly zero, not the finite average `0.5 * 2.0 = 1.0` that could be Euler-
    inverted before the `-inf` value kills the branch.
    """
    dead = jnp.array([10.0, 11.0, 12.0])
    live = jnp.array([0.0, 5.0, 10.0])
    carry = EGMCarry(
        endog_grid=jnp.stack([jnp.stack([dead, dead]), jnp.stack([live, live])]),
        value=jnp.ones((2, 2, 3)),
        marginal_utility=jnp.full((2, 2, 3), 2.0),
        taste_shock_scale=jnp.asarray(0.0),
    )
    prepared_search_grid, prepared_valid_length = _prepare(carry)

    smoothed_value, smoothed_marginal = _aggregate_child_choices(
        carry=carry,
        prepared_search_grid=prepared_search_grid,
        prepared_valid_length=prepared_valid_length,
        has_taste_shocks=False,
        child_index=(),
        child_passive_values=(jnp.asarray(0.5),),
        child_passive_grids=(jnp.asarray([0.0, 1.0]),),
        row_queries=jnp.asarray([4.0, 4.0]),
        row_gradients=jnp.asarray([1.0, 1.0]),
        n_outer_candidates=2,
    )

    assert float(smoothed_value) == float("-inf")
    assert float(smoothed_marginal) == 0.0


def test_stacked_read_matches_host_max_of_reads_on_curved_rows():
    """On curved rows the read equals the host nodewise-max-then-blend exactly.

    Two durable nodes, two candidates with genuinely interpolated (concave)
    value rows, distinct marginals, per-node queries and gradients. The host
    computes, per durable node, each candidate's Hermite value read and linear
    marginal read at that node's query, takes the nodewise max and the winner's
    gradient-scaled marginal, then blends with the passive weights. The
    production read must agree to numerical precision.
    """
    n_pad = 5
    coh = np.linspace(1.0, 9.0, n_pad)
    endog = np.broadcast_to(coh, (2, 2, n_pad)).copy()
    endog[:, 1, :] = coh + 0.5  # candidate B lifted by a credited cost
    value = np.empty((2, 2, n_pad))
    marginal = np.empty((2, 2, n_pad))
    for node, level in enumerate((0.0, 0.6)):
        value[node, 0] = np.sqrt(coh) + level
        marginal[node, 0] = 0.5 / np.sqrt(coh)
        value[node, 1] = 1.2 * np.sqrt(coh + 0.5) - 0.4 + level
        marginal[node, 1] = 0.6 / np.sqrt(coh + 0.5)

    carry = EGMCarry(
        endog_grid=jnp.asarray(endog),
        value=jnp.asarray(value),
        marginal_utility=jnp.asarray(marginal),
        taste_shock_scale=jnp.asarray(0.0),
    )
    prepared_search_grid, prepared_valid_length = _prepare(carry)
    queries = np.array([3.7, 6.2])
    gradients = np.array([1.05, 0.95])
    passive_value, passive_grid = 0.4, np.array([0.0, 1.0])

    smoothed_value, smoothed_marginal = _aggregate_child_choices(
        carry=carry,
        prepared_search_grid=prepared_search_grid,
        prepared_valid_length=prepared_valid_length,
        has_taste_shocks=False,
        child_index=(),
        child_passive_values=(jnp.asarray(passive_value),),
        child_passive_grids=(jnp.asarray(passive_grid),),
        row_queries=jnp.asarray(queries),
        row_gradients=jnp.asarray(gradients),
        n_outer_candidates=2,
    )

    expected_value = 0.0
    expected_marginal = 0.0
    weights = (1.0 - passive_value, passive_value)
    for node in range(2):
        reads = [
            _host_read(endog[node, j], value[node, j], marginal[node, j], queries[node])
            for j in range(2)
        ]
        winner = int(np.argmax([r[0] for r in reads]))
        expected_value += weights[node] * reads[winner][0]
        expected_marginal += weights[node] * reads[winner][1] * gradients[node]

    np.testing.assert_allclose(
        float(smoothed_value), expected_value, rtol=_READ_RTOL, atol=_READ_ATOL
    )
    np.testing.assert_allclose(
        float(smoothed_marginal), expected_marginal, rtol=_READ_RTOL, atol=_READ_ATOL
    )
