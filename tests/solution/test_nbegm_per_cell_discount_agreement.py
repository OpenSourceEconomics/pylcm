"""NBEGM agreement with brute when the discount factor rides along per cell.

When the discount factor is a DAG function of a ride-along state (here `kind`,
standing in for a preference type) rather than the flat `H__discount_factor`
parameter, the case-piece solver must resolve the Euler weight per ride-along
slice. Its value function must still reproduce the dense-grid `GridSearch` value
across the asset interior and through the bracket kink, in every `kind` slice and
at every working age — with the two slices carrying *different* discount factors.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_ride_along_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the per-kind-discount ride-along toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        per_kind_discount=True,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(per_kind_discount=True), log_level="off")


def test_nbegm_matches_brute_with_per_kind_discount_every_age():
    """With a distinct discount factor per `kind` slice, the schedule solve equals
    brute in both slices, kink and all."""
    nbegm = _solve("nbegm")
    brute = _solve("brute", n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in nbegm[period]:
            continue
        brute_v = np.asarray(brute[period]["alive"])
        nbegm_v = np.asarray(nbegm[period]["alive"])
        for kind in range(brute_v.shape[0]):
            np.testing.assert_allclose(
                nbegm_v[kind, _INTERIOR],
                brute_v[kind, _INTERIOR],
                atol=2e-2,
                rtol=5e-3,
                err_msg=f"period={period} kind={kind}",
            )
