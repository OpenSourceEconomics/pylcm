"""NBEGM's discrete envelope takes the choice over several discrete actions jointly.

A regime may declare more than one discrete action whose branches shift only the
current budget and period utility. NBEGM solves the continuous subproblem per
ride cell and per element of the product of the discrete grids, then takes the
joint choice by the upper envelope over those branches, so the solved value
matches a dense brute solve that maximises over the same product.
"""

from collections.abc import Mapping

import numpy as np
import pytest

from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.test_models import nbegm_multi_discrete_toy as toy

# The certified comparison is the one that needs the native library; the
# ordinary cases decide in the working format and run anywhere.
_REQUIRES_KERNEL = pytest.mark.requires_exact_affine_kernel(
    reason=EXACT_KERNEL_SKIP_REASON
)

_ALIVE = "alive"
_TAX_EXEMPTION = 12.0
_LIQUID = np.linspace(0.1, 30.0, 40)
_AWAY_FROM_KINK = (
    (_LIQUID > 1.5) & (_LIQUID < 27.0) & (np.abs(_LIQUID - _TAX_EXEMPTION) > 0.5)
)


def _solve(
    *,
    variant: str,
    n_actions: int,
    n_consumption: int = 120,
    envelope_arithmetic: str = "certified",
) -> Mapping[int, Mapping]:
    model = toy.build_model(
        variant=variant,
        n_actions=n_actions,
        n_consumption=n_consumption,
        envelope_arithmetic=envelope_arithmetic,
    )
    return model.solve(params=toy.build_params(n_actions=n_actions), log_level="debug")


@pytest.mark.parametrize(
    ("n_actions", "n_branches", "envelope_arithmetic"),
    [
        pytest.param(2, 4, "certified", marks=_REQUIRES_KERNEL),
        (2, 4, "ordinary"),
        (3, 20, "ordinary"),
        pytest.param(3, 20, "certified", marks=_REQUIRES_KERNEL),
    ],
)
def test_nbegm_envelope_over_several_discrete_actions_matches_brute(
    *, n_actions: int, n_branches: int, envelope_arithmetic: str
) -> None:
    """`V` agrees with a dense brute solve when several discrete actions are declared.

    The envelope ranges over all `n_branches` combinations of the declared grids, so
    the agreement holds across the liquid interior at every income node, under
    either arithmetic and at both the narrow and the widest branch product.
    """
    nbegm = _solve(
        variant="nbegm", n_actions=n_actions, envelope_arithmetic=envelope_arithmetic
    )
    brute = _solve(variant="brute", n_actions=n_actions, n_consumption=1200)
    period = max(p for p in brute if _ALIVE in brute[p])
    nbegm_v = np.asarray(nbegm[period][_ALIVE])
    brute_v = np.asarray(brute[period][_ALIVE])
    assert nbegm_v.shape == brute_v.shape
    for node in range(brute_v.shape[0]):
        np.testing.assert_allclose(
            nbegm_v[node][_AWAY_FROM_KINK],
            brute_v[node][_AWAY_FROM_KINK],
            rtol=5e-3,
            atol=5e-3,
            err_msg=f"{n_branches} branches, income node {node}",
        )
