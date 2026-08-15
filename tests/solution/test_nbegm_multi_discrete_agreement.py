"""NBEGM's discrete envelope takes the choice over several discrete actions jointly.

A regime may declare more than one discrete action whose branches shift only the
current budget and period utility. NBEGM solves the continuous subproblem per
ride cell and per element of the product of the discrete grids, then takes the
joint choice by the upper envelope over those branches, so the solved value
matches a dense brute solve that maximises over the same product.
"""

import os
from collections.abc import Mapping

import numpy as np
import pytest

from tests.test_models import nbegm_multi_discrete_toy as toy

_ALIVE = "alive"
_BACKEND_OPT_OFF = "--xla_backend_optimization_level=0"
_NEEDS_OPT_OFF = pytest.mark.skipif(
    _BACKEND_OPT_OFF not in os.environ.get("XLA_FLAGS", ""),
    reason=(
        "The certified envelope's fused sign reduction does not leave XLA's backend"
        f" optimizer in bounded time, so these cases need `XLA_FLAGS={_BACKEND_OPT_OFF}`."
        " With it they take seconds; without it they exceed any CI timeout. The flag is"
        " not set suite-wide because it collapses every vmap width onto one bit pattern,"
        " which would mask the batch-width invariance the solve battery checks."
    ),
)
_TAX_EXEMPTION = 12.0
_LIQUID = np.linspace(0.1, 30.0, 40)
_AWAY_FROM_KINK = (
    (_LIQUID > 1.5) & (_LIQUID < 27.0) & (np.abs(_LIQUID - _TAX_EXEMPTION) > 0.5)
)


def _solve(
    variant: str,
    *,
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
        pytest.param(2, 4, "certified", marks=_NEEDS_OPT_OFF),
        (2, 4, "ordinary"),
        (3, 20, "ordinary"),
        pytest.param(3, 20, "certified", marks=_NEEDS_OPT_OFF),
    ],
)
def test_nbegm_envelope_over_several_discrete_actions_matches_brute(
    n_actions: int, n_branches: int, envelope_arithmetic: str
) -> None:
    """`V` agrees with a dense brute solve when several discrete actions are declared.

    The envelope ranges over all `n_branches` combinations of the declared grids, so
    the agreement holds across the liquid interior at every income node. Which
    arithmetic decides ownership is orthogonal to how many branches are enveloped
    over, so the widest branch product is checked under the cheaper comparison and
    the default comparison is checked at the narrower one.
    """
    nbegm = _solve(
        "nbegm", n_actions=n_actions, envelope_arithmetic=envelope_arithmetic
    )
    brute = _solve("brute", n_actions=n_actions, n_consumption=1200)
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
