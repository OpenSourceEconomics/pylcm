"""2-D RFC region clouds must carry a KKT validity mask.

Each of the four region inverses (ucon/dcon/acon/con) solves the first-order
conditions *assuming* its own constraints bind. The solution is a genuine
candidate only where that region's complementary-slackness inequalities also
hold; elsewhere the formula still returns a finite point that is not a KKT
solution. The RFC selection currently filters candidates by finiteness alone, so
a KKT-inconsistent candidate can enter the rooftop-cut cloud and dominate a
valid one. Each region cloud must therefore expose a `valid_region` mask
enforcing its own inequalities.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.two_asset_inverse import invert_dcon_cloud


@pytest.mark.xfail(
    reason="RegionCloud has no valid_region KKT mask; finite-only filtering",
    strict=True,
)
def test_dcon_cloud_flags_kkt_inconsistent_candidate_invalid():
    """The `dcon` cloud marks a deposit-FOC-violating candidate invalid.

    With `beta = 1`, `crra = 2`, `w_a = w_b = 1` and a match rate `chi = 1`, the
    deposit-constrained inverse pins `d = 0` and returns `c = 1`, so
    `u'(c) = 1`. The `d = 0` corner is optimal only if
    `beta * w_b * (1 + chi) <= u'(c)`, i.e. `2 <= 1`, which fails — raising the
    deposit is profitable, so this point is not a KKT solution. The cloud must
    flag it via a `valid_region` mask that is `False` here.
    """
    one = jnp.ones((1,))
    cloud = invert_dcon_cloud(
        a=one * 5.0,
        b=one * 3.0,
        w_a=one,
        w_b=one,
        post_decision_value=one * 0.0,
        discount_factor=1.0,
        crra=2.0,
    )

    consumption = float(np.asarray(cloud.consumption)[0])
    marginal_utility = consumption ** (-2.0)
    deposit_foc_slack = 1.0 * 1.0 * (1.0 + 1.0) - marginal_utility
    assert deposit_foc_slack > 0.0  # the candidate genuinely violates the FOC

    # `valid_region` is the KKT mask this defect requires; it does not exist yet,
    # so the strict-xfail asserts the contract the fix must add.
    assert bool(np.asarray(cloud.valid_region)[0]) is False  # ty: ignore[unresolved-attribute]
