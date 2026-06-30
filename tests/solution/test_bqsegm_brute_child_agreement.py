"""BQSEGM agreement with brute when the continuation reads a brute-solved child.

A BQSEGM regime embedded in an otherwise-brute model transitions into brute
children that produce no EGM carry. The solver must read such a child from its
value array and reproduce the all-brute reference value across the asset interior
and through the bracket kink, in every `kind` slice.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import bqsegm_brute_child_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(
    young_variant: str,
    *,
    n_consumption: int = 120,
    old_discrete_action: bool = False,
) -> Mapping[int, Mapping]:
    """Solve the young→old→dead toy on the shared comparison grids."""
    model = toy.build_model(
        young_variant=young_variant,
        old_discrete_action=old_discrete_action,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(
        params=toy.build_params(old_discrete_action=old_discrete_action),
        log_level="off",
    )


def test_bqsegm_young_matches_brute_reading_a_brute_child():
    """`young` solved by BQSEGM equals `young` solved by brute, in both `kind`
    slices, when its continuation reads the brute `old` regime's value array."""
    bqsegm = _solve("bqsegm")
    brute = _solve("brute", n_consumption=1500)
    period = 0  # the only period `young` is active.
    brute_v = np.asarray(brute[period]["young"])
    bqsegm_v = np.asarray(bqsegm[period]["young"])
    for kind in range(brute_v.shape[0]):
        np.testing.assert_allclose(
            bqsegm_v[kind, _INTERIOR],
            brute_v[kind, _INTERIOR],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"kind={kind}",
        )


def test_bqsegm_young_matches_brute_reading_a_brute_child_with_discrete_action():
    """`young` solved by BQSEGM equals `young` solved by brute when the brute `old`
    child carries a discrete `work` choice — the child's value array is already
    maxed over `work`, so the continuation reads it without re-aggregating."""
    bqsegm = _solve("bqsegm", old_discrete_action=True)
    brute = _solve("brute", n_consumption=1500, old_discrete_action=True)
    period = 0  # the only period `young` is active.
    brute_v = np.asarray(brute[period]["young"])
    bqsegm_v = np.asarray(bqsegm[period]["young"])
    for kind in range(brute_v.shape[0]):
        np.testing.assert_allclose(
            bqsegm_v[kind, _INTERIOR],
            brute_v[kind, _INTERIOR],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"kind={kind}",
        )
