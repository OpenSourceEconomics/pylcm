"""Spec for the NEGM stacked outer carry and its query-side envelope read.

The NEGM kernel publishes one continuation carry per durable state that retains
every outer candidate: the keeper and each adjuster outer-grid node, lifted into
common cash-on-hand (coh) space and stacked on a candidate axis. The parent read
takes the exact `max_j V_j(q)` over that axis at its own query, so it sees the
best durable choice at every coh — not only the keeper's, and not only at the
keeper's nodes. These tests exercise the lift + stack + query-side read in
isolation, with synthetic keeper and adjuster rows, so the envelope logic is
pinned independently of the toy model and oracle.
"""

from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.outer_envelope import build_stacked_outer_carry, outer_envelope_at_query
from _lcm.solution.solvers import _build_coh_shift_function
from _lcm.typing import EconFunction, EconFunctionsMapping
from lcm.exceptions import InvalidParamsError
from lcm.typing import Float1D, FloatND


def test_coh_shift_derives_from_the_declared_cost_alone() -> None:
    """The credited-cost shift evaluates only the declared cost DAG.

    With a declared `NEGM.outer_cost` the resources function is composed at
    model build as `base - cost`, so the shift is exactly
    `cost(z, z'_j) - cost(z, keep(z))` — nothing about the wider function set
    (a wage-only income node, params other states read) enters the evaluation,
    and no binding for them is demanded.
    """

    def income(wage: FloatND) -> FloatND:
        """A wage-only resources term, additively separable from the house cost."""
        return jnp.exp(wage)

    def housing_cost(housing: FloatND, next_housing: FloatND) -> FloatND:
        """Round-trip cost of moving the house from `housing` to `next_housing`."""
        return next_housing - housing

    shift_func = _build_coh_shift_function(
        functions=cast(
            "EconFunctionsMapping",
            MappingProxyType({"income": income, "housing_cost": housing_cost}),
        ),
        durable_state_name="housing",
        outer_post_decision="next_housing",
        no_adjustment_func=None,
        outer_cost_name="housing_cost",
    )

    durable_values = jnp.asarray([1.0, 2.0])
    outer_values = jnp.asarray([0.5, 1.5, 3.0])
    shifts = shift_func(durable_values=durable_values, outer_values=outer_values)

    # shift(z, z') = cost(z, z') - cost(z, keep(z) = z) = z' - z.
    expected = outer_values[None, :] - durable_values[:, None]
    assert shifts.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(shifts), np.asarray(expected), atol=1e-12)


def test_coh_shift_keeper_leg_uses_the_no_adjustment_level() -> None:
    """With a depreciating keeper, the shift references `keep(z)`, not `z`.

    The keeper core realises the outer post-decision at its no-adjustment level
    `keep(z) = z (1 - delta)`, so the credited-cost lift must difference the
    declared cost at that level: `shift(z, z') = cost(z, z') - cost(z,
    keep(z))`. With `housing_cost(next) = next - z` this is `z' - keep(z)` —
    larger than the identity reference by the depreciation `z - keep(z)`.
    """
    keep_rate = 0.9

    def housing_cost(housing: FloatND, next_housing: FloatND) -> FloatND:
        """Round-trip cost of moving the house from `housing` to `next_housing`."""
        return next_housing - housing

    def keep_housing(housing: FloatND) -> FloatND:
        """The keeper's depreciated no-adjustment level."""
        return keep_rate * housing

    shift_func = _build_coh_shift_function(
        functions=cast(
            "EconFunctionsMapping",
            MappingProxyType({"housing_cost": housing_cost}),
        ),
        durable_state_name="housing",
        outer_post_decision="next_housing",
        no_adjustment_func=cast("EconFunction", keep_housing),
        outer_cost_name="housing_cost",
    )

    durable_values = jnp.asarray([1.0, 2.0])
    outer_values = jnp.asarray([0.5, 1.5, 3.0])
    shifts = shift_func(durable_values=durable_values, outer_values=outer_values)

    expected = outer_values[None, :] - keep_rate * durable_values[:, None]
    np.testing.assert_allclose(np.asarray(shifts), np.asarray(expected), atol=1e-12)


def _crra_value_row(*, coh: Float1D, level: float) -> Float1D:
    """A concave, increasing synthetic value row plus a constant `level`."""
    return jnp.sqrt(coh) + level


def _identity_marginal(*, coh: Float1D) -> Float1D:
    """A positive, decreasing marginal row consistent with `sqrt` curvature."""
    return 0.5 / jnp.sqrt(coh)


def _read_stacked_cell(
    carry: EGMCarry, cell: tuple[int, ...], query: Float1D
) -> Float1D:
    """Read one leading cell of a stacked carry through the query-side envelope."""
    value, _marginal = outer_envelope_at_query(
        candidate_endog=carry.endog_grid[cell],
        candidate_value=carry.value[cell],
        candidate_marginal=carry.marginal_utility[cell],
        x_query=query,
    )
    return value


def test_adjuster_winning_by_delta_lifts_the_envelope_by_delta() -> None:
    """An adjuster that beats the keeper by `delta` lifts the read by `delta`.

    One durable state, one keeper row, one adjuster row. The adjuster's value is
    the keeper's plus a constant `delta` over a reachable coh interval, so the
    query-side envelope must equal the adjuster there: reading the stacked carry
    at a coh in that interval returns the keeper value plus `delta`.
    """
    n_pad = 40
    coh = jnp.linspace(1.0, 9.0, n_pad)
    delta = 0.25

    keeper = EGMCarry(
        endog_grid=coh[None, :],
        value=_crra_value_row(coh=coh, level=0.0)[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    # The adjuster's own resources grid is coh shifted down by its credited cost;
    # the shift maps it back into coh space. Its value beats the keeper by delta.
    shift = 2.0
    adjuster = EGMCarry(
        endog_grid=(coh - shift)[None, :],
        value=(_crra_value_row(coh=coh, level=delta))[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )

    carry = build_stacked_outer_carry(
        keeper_carry=keeper,
        adjuster_carries=(adjuster,),
        coh_shifts=jnp.asarray([[shift]]),  # one durable state, one adjuster
    )

    # The stacked carry keeps every candidate: keeper + one adjuster.
    assert carry.value.shape == (1, 2, n_pad)
    query = jnp.asarray([3.0, 5.0, 7.0])
    enveloped = _read_stacked_cell(carry, (0,), query)
    expected = jnp.sqrt(query) + delta
    np.testing.assert_allclose(np.asarray(enveloped), np.asarray(expected), atol=1e-3)


def test_adjuster_winning_only_on_a_subinterval_is_captured() -> None:
    """An adjuster winning only on a coh sub-interval lifts the envelope there.

    The adjuster beats the keeper by a hump centred in the coh range and is below
    it at both ends. The query-side envelope must equal the keeper outside the
    hump and the adjuster inside it, including at coh points strictly between the
    keeper's grid nodes — a nodewise maximum at the keeper's abscissae would miss
    the interior win.
    """
    n_pad = 60
    coh = jnp.linspace(1.0, 9.0, n_pad)
    # A hump peaking near coh = 5, negative at the ends: the adjuster wins only
    # in the middle of the coh range.
    hump = 0.5 - 0.08 * (coh - 5.0) ** 2
    keeper = EGMCarry(
        endog_grid=coh[None, :],
        value=_crra_value_row(coh=coh, level=0.0)[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    shift = 2.0
    adjuster_value = jnp.sqrt(coh) + hump
    adjuster = EGMCarry(
        endog_grid=(coh - shift)[None, :],
        value=adjuster_value[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )

    carry = build_stacked_outer_carry(
        keeper_carry=keeper,
        adjuster_carries=(adjuster,),
        coh_shifts=jnp.asarray([[shift]]),
    )

    # An off-node query in the winning interior: the envelope must equal the
    # adjuster (keeper + hump), strictly above the keeper.
    interior = jnp.asarray([4.7, 5.3])
    enveloped_interior = _read_stacked_cell(carry, (0,), interior)
    expected_interior = jnp.sqrt(interior) + (0.5 - 0.08 * (interior - 5.0) ** 2)
    np.testing.assert_allclose(
        np.asarray(enveloped_interior), np.asarray(expected_interior), atol=1e-3
    )
    assert bool(jnp.all(enveloped_interior > jnp.sqrt(interior)))

    # At the coh ends the keeper wins: the envelope equals the keeper value.
    ends = jnp.asarray([1.5, 8.5])
    enveloped_ends = _read_stacked_cell(carry, (0,), ends)
    np.testing.assert_allclose(
        np.asarray(enveloped_ends), np.asarray(jnp.sqrt(ends)), atol=1e-3
    )


def test_envelope_broadcasts_over_a_discrete_leading_axis() -> None:
    """A discrete state ahead of the durable axis is enveloped cell-by-cell.

    When the inner DC-EGM carries a discrete/process state (a Markov wage), the
    carry's leading axes are `(discrete, durable)` — the durable is the last
    leading axis, the discrete one precedes it. The credited shift depends only
    on the durable margin, so it broadcasts over the discrete axis, and the
    query-side envelope is taken independently per `(discrete, durable)` cell.
    The stacked carry must keep the discrete axis (here two nodes at different
    value levels) and lift each cell by the winning adjuster's gain, never
    mixing the two discrete nodes.
    """
    n_pad = 40
    n_discrete = 2
    coh = jnp.linspace(1.0, 9.0, n_pad)
    delta = 0.25
    shift = 2.0
    # One durable state, two discrete nodes at value levels 0.0 and 1.0.
    discrete_levels = jnp.asarray([0.0, 1.0])

    keeper = EGMCarry(
        endog_grid=jnp.broadcast_to(coh, (n_discrete, 1, n_pad)),
        value=jnp.stack(
            [_crra_value_row(coh=coh, level=lvl) for lvl in discrete_levels]
        )[:, None, :],
        marginal_utility=jnp.broadcast_to(
            _identity_marginal(coh=coh), (n_discrete, 1, n_pad)
        ),
        taste_shock_scale=jnp.asarray(0.0),
    )
    adjuster = EGMCarry(
        endog_grid=jnp.broadcast_to(coh - shift, (n_discrete, 1, n_pad)),
        value=jnp.stack(
            [_crra_value_row(coh=coh, level=lvl + delta) for lvl in discrete_levels]
        )[:, None, :],
        marginal_utility=jnp.broadcast_to(
            _identity_marginal(coh=coh), (n_discrete, 1, n_pad)
        ),
        taste_shock_scale=jnp.asarray(0.0),
    )

    carry = build_stacked_outer_carry(
        keeper_carry=keeper,
        adjuster_carries=(adjuster,),
        coh_shifts=jnp.asarray([[shift]]),  # one durable state, one adjuster
    )

    assert carry.value.shape == (n_discrete, 1, 2, n_pad)
    query = jnp.asarray([3.0, 5.0, 7.0])
    for node in range(n_discrete):
        enveloped = _read_stacked_cell(carry, (node, 0), query)
        expected = jnp.sqrt(query) + float(discrete_levels[node]) + delta
        np.testing.assert_allclose(
            np.asarray(enveloped), np.asarray(expected), atol=1e-3
        )


def test_keeper_wins_when_no_adjuster_improves() -> None:
    """With no adjuster beating the keeper, the read reproduces the keeper.

    Every adjuster value lies strictly below the keeper, so the query-side
    envelope is the keeper row itself: reading the stacked carry returns the
    keeper value.
    """
    n_pad = 40
    coh = jnp.linspace(1.0, 9.0, n_pad)

    keeper = EGMCarry(
        endog_grid=coh[None, :],
        value=_crra_value_row(coh=coh, level=0.0)[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    shift = 2.0
    adjuster = EGMCarry(
        endog_grid=(coh - shift)[None, :],
        value=_crra_value_row(coh=coh, level=-0.5)[None, :],
        marginal_utility=_identity_marginal(coh=coh)[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )

    carry = build_stacked_outer_carry(
        keeper_carry=keeper,
        adjuster_carries=(adjuster,),
        coh_shifts=jnp.asarray([[shift]]),
    )

    query = jnp.asarray([3.0, 5.0, 7.0])
    enveloped = _read_stacked_cell(carry, (0,), query)
    expected = jnp.sqrt(query)
    np.testing.assert_allclose(np.asarray(enveloped), np.asarray(expected), atol=1e-3)


def test_coh_shift_rejects_a_state_dependent_adjustment_wedge() -> None:
    """An adjustment cost scaled by a ride-along state has no constant lift.

    With `housing_cost = (1 + wage) * (next_housing - housing)`, the declared
    cost varies with a state the shift matrix cannot represent (it is indexed
    by durable and outer node only). Model build rejects this structurally; the
    shift builder itself must also refuse to evaluate the declaration instead
    of broadcasting the zero-wage cost to every wage.
    """

    def housing_cost(wage: FloatND, housing: FloatND, next_housing: FloatND) -> FloatND:
        """A wage-scaled cost of moving the house — not NEGM-liftable."""
        return (1.0 + wage) * (next_housing - housing)

    shift_func = _build_coh_shift_function(
        functions=cast(
            "EconFunctionsMapping",
            MappingProxyType({"housing_cost": housing_cost}),
        ),
        durable_state_name="housing",
        outer_post_decision="next_housing",
        no_adjustment_func=None,
        outer_cost_name="housing_cost",
    )

    with pytest.raises(InvalidParamsError, match="may read only"):
        shift_func(
            durable_values=jnp.asarray([1.0, 2.0]),
            outer_values=jnp.asarray([0.5, 3.0]),
        )


def test_coh_shift_is_zero_without_a_declared_cost() -> None:
    """With `outer_cost=None` every candidate shares the keeper's coh axis.

    A regime whose resources never read the outer post-decision declares no
    outer cost; the shift matrix is exactly zero for every (durable, outer)
    cell, so the lift is the identity.
    """

    def resources(liquid: FloatND) -> FloatND:
        """Liquid wealth only — independent of the outer choice."""
        return liquid

    shift_func = _build_coh_shift_function(
        functions=cast(
            "EconFunctionsMapping",
            MappingProxyType({"resources": resources}),
        ),
        durable_state_name="housing",
        outer_post_decision="next_housing",
        no_adjustment_func=None,
        outer_cost_name=None,
    )

    shifts = shift_func(
        durable_values=jnp.asarray([1.0, 2.0]),
        outer_values=jnp.asarray([0.5, 1.5, 3.0]),
    )

    assert shifts.shape == (2, 3)
    np.testing.assert_array_equal(np.asarray(shifts), np.zeros((2, 3)))
