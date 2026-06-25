"""FUES retains equal-value coincident-abscissa crossings.

When two value-correspondence branches cross exactly on an existing endogenous
grid point, the envelope has a genuine kink there: both branches attain the
same (maximal) value but carry different policies. The refinement must keep
both copies — the left branch valid up to the crossing and the right branch
valid beyond it — so a right-continuous read just past the crossing recovers
the right branch's policy rather than interpolating across the kink.

The host oracle `tests/solution/_envelope_oracle.exact_envelope` is the
reference: it consumes explicit branch labels and is exact for the
piecewise-linear correspondence.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope import fues, rfc
from tests.solution._envelope_oracle import exact_envelope

# FUES drops one copy of an equal-value coincident-abscissa crossing: the
# pre-scan dedup NaN-collapses every coincident point after the first
# regardless of value, so a node-aligned kink loses its second policy record.
# The fix is scan/kink-insertion-level (the on-node crossing path uses strict
# `<`/`>`), not a dedup tweak — retaining the duplicate alone breaks the scan's
# zero-width-interval handling. Tracked as P0.1 (#136).
_FUES_DROPS_NODE_CROSSING = pytest.mark.xfail(
    reason="FUES drops equal-value on-grid crossing copy; scan fix pending",
    strict=True,
)

# Two branches sampled on the shared grid R = (10, 11, 12):
# - branch A: policy 3, value (5/3, 2, 7/3), slope 1/3;
# - branch B: policy 0.5, value (0, 2, 4), slope 2.
# They cross exactly at R = 11, where both attain value 2. A wins below 11, B
# above; the envelope kink lands on the grid node R = 11.
_ENDOG = jnp.array([10.0, 11.0, 12.0, 10.0, 11.0, 12.0])
_POLICY = jnp.array([3.0, 3.0, 3.0, 0.5, 0.5, 0.5])
_VALUE = jnp.array([5 / 3, 2.0, 7 / 3, 0.0, 2.0, 4.0])
_SEGMENT = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
_MARGINAL = jnp.array([1 / 3, 1 / 3, 1 / 3, 2.0, 2.0, 2.0])
_N_REFINED = 20


def _kept(refine_output):
    """Return the non-NaN (grid, policy, value) prefix of a refinement output."""
    grid, policy, value, _ = refine_output
    grid = np.asarray(grid)
    keep = ~np.isnan(grid)
    return grid[keep], np.asarray(policy)[keep], np.asarray(value)[keep]


@_FUES_DROPS_NODE_CROSSING
def test_fues_retains_equal_value_crossing_on_grid_node():
    """FUES keeps both branch copies at an on-grid crossing, like RFC does.

    The refined envelope is `grid = [10, 11, 11, 12]` with the left branch
    (policy 3) then the right branch (policy 0.5) at the duplicated crossing
    node, so the published row carries the kink rather than collapsing it.
    """
    grid, policy, _ = _kept(
        fues.refine_envelope(
            endog_grid=_ENDOG,
            policy=_POLICY,
            value=_VALUE,
            n_refined=_N_REFINED,
            segment_id=_SEGMENT,
        )
    )

    np.testing.assert_allclose(grid, [10.0, 11.0, 11.0, 12.0])
    np.testing.assert_allclose(policy, [3.0, 3.0, 0.5, 0.5])


@_FUES_DROPS_NODE_CROSSING
def test_fues_matches_rfc_at_on_grid_crossing():
    """FUES and RFC publish the same refined envelope at the on-grid crossing."""
    fues_grid, fues_policy, fues_value = _kept(
        fues.refine_envelope(
            endog_grid=_ENDOG,
            policy=_POLICY,
            value=_VALUE,
            n_refined=_N_REFINED,
            segment_id=_SEGMENT,
        )
    )
    rfc_grid, rfc_policy, rfc_value = _kept(
        rfc.refine_envelope(
            endog_grid=_ENDOG,
            policy=_POLICY,
            value=_VALUE,
            marginal_utility=_MARGINAL,
            n_refined=_N_REFINED,
        )
    )

    np.testing.assert_allclose(fues_grid, rfc_grid)
    np.testing.assert_allclose(fues_policy, rfc_policy)
    np.testing.assert_allclose(fues_value, rfc_value)


def test_fues_envelope_value_matches_oracle_across_crossing():
    """The FUES envelope value equals the host oracle on both sides of the kink.

    On the grid nodes the published value is the branch maximum; the oracle
    confirms branch A wins at and below R = 11 and branch B above it.
    """
    grid, _, value = _kept(
        fues.refine_envelope(
            endog_grid=_ENDOG,
            policy=_POLICY,
            value=_VALUE,
            n_refined=_N_REFINED,
            segment_id=_SEGMENT,
        )
    )
    query = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    oracle_value, _, _ = exact_envelope(
        endog_grid=np.asarray(_ENDOG),
        value=np.asarray(_VALUE),
        policy=np.asarray(_POLICY),
        segment_id=np.asarray(_SEGMENT),
        x_query=query,
    )
    # Read the published row the same way the kernel does: max of the two
    # branch values at a duplicated node falls out of np.interp on the sorted
    # kept grid because the equal-value copies coincide.
    published = np.interp(query, grid, value)
    np.testing.assert_allclose(published, oracle_value, atol=1e-9)
