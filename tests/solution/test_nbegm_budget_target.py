"""NBEGM solves against a budget node named other than the convention `coh`.

The continuous-budget EGM path composes the regime's budget from the model DAG.
A real model names that node for its own domain (`resources`, `cash_on_hand`),
not the solver's convention `coh`. The `budget_target` field selects which DAG
node is the consumption budget, mirroring how `DCEGM` takes `resources=`. Solving
the tax toy with its budget node renamed to `resources` and
`NBEGM(budget_target="resources")` must reproduce the dense-grid `GridSearch`
value across the asset interior and through the bracket kink.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_tax_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the tax toy whose budget node is named `resources`."""
    model = toy.build_model(
        variant=variant,
        budget_name="resources",
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(
        params=toy.build_params(budget_name="resources"), log_level="off"
    )


def test_nbegm_with_renamed_budget_target_matches_brute_every_age():
    """A non-`coh` budget node solved by NBEGM equals brute at every working age."""
    nbegm = _solve("nbegm")
    brute = _solve("brute", n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in nbegm[period]:
            continue
        np.testing.assert_allclose(
            np.asarray(nbegm[period]["alive"])[_INTERIOR],
            np.asarray(brute[period]["alive"])[_INTERIOR],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period}",
        )
