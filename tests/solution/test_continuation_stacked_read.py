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

    np.testing.assert_allclose(float(smoothed_value), 2.25, atol=1e-9)
    np.testing.assert_allclose(float(smoothed_marginal), 0.0, atol=1e-9)


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

    np.testing.assert_allclose(float(smoothed_value), 1.0, atol=1e-9)
    # Winner A's marginal 0.5, scaled by the composed gradient 3.0.
    np.testing.assert_allclose(float(smoothed_marginal), 1.5, atol=1e-9)


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

    np.testing.assert_allclose(float(smoothed_value), expected_value, rtol=1e-12)
    np.testing.assert_allclose(float(smoothed_marginal), expected_marginal, rtol=1e-12)
