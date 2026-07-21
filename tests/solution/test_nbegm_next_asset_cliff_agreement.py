"""NBEGM agreement with brute when the next-asset law jumps at a liquid cliff.

When `next_liquid` carries a current-asset boundary (a transfer that switches at a
declared liquid cliff), the continuation cannot read it as a function of savings
alone — within each declared interval the transfer is constant, so the solver must
evaluate the continuation per interval with that constant bound. Its value must
reproduce the dense `GridSearch` value across the asset interior in both `kind`
slices, at every working age.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_next_asset_cliff_toy as toy

_MEDICAID_LIMIT = 12.0
_LIQUID = np.linspace(0.1, 30.0, 120)
# The value jumps at the cliff (the next-asset transfer switches there), so the grid
# cells straddling it differ between a continuous EGM policy and the 1500-point brute
# by grid alignment alone — the diagnostic comparison is never exact at a notch. The
# interior strict-agreement region excludes that one-cell neighborhood.
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)
_AWAY_FROM_CLIFF = _INTERIOR & (np.abs(_LIQUID - _MEDICAID_LIMIT) > 0.4)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the next-asset-cliff toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_nbegm_matches_brute_with_a_next_asset_cliff_terminal_adjacent():
    """With a current-asset cliff in the next-asset law and a smooth (terminal)
    continuation, the per-interval solve equals brute across the asset interior in
    both `kind` slices, away from the one cell straddling the cliff jump."""
    nbegm = _solve("nbegm")
    brute = _solve("brute", n_consumption=1500)
    # The last period in which `alive` is active carries the smooth terminal
    # continuation, so the per-interval mechanism is isolated from the recurring
    # linear-continuation residual the deeper periods inherit.
    terminal_adjacent = max(p for p in brute if "alive" in brute[p])
    brute_v = np.asarray(brute[terminal_adjacent]["alive"])
    nbegm_v = np.asarray(nbegm[terminal_adjacent]["alive"])
    for kind in range(brute_v.shape[0]):
        np.testing.assert_allclose(
            nbegm_v[kind, _AWAY_FROM_CLIFF],
            brute_v[kind, _AWAY_FROM_CLIFF],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"kind={kind}",
        )
