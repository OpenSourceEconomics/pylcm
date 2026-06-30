"""BQSEGM agreement when the period utility is CES, not plain CRRA.

The M1 ACA regime's period utility is a CES aggregate of consumption and leisure,
so the Euler inversion `c = (u')^{-1}(m)` reads the ride-along leisure term and the
equivalence scale — a plain-CRRA inversion `c = (discount_factor * m) ** (-1/crra)`
recovers the wrong action. The BQSEGM ride-along solver must invert the *model's*
marginal utility, whether the BQSEGM ride-along solver inverts the
*model's* marginal utility — derived numerically from the period utility, since the
regime carries no `inverse_marginal_utility` function.

At the terminal-adjacent working period the continuation is the smooth bequest, so the
savings-space step is exact: BQSEGM reproduces the dense `GridSearch` value across the
liquid interior of every `wage` slice — through both the continuous-kink and the jump
budget — only when the CES inversion is honoured.
"""

from collections.abc import Mapping

import numpy as np
import pytest

from tests.test_models import bqsegm_ces_utility_toy as toy

_N_LIQUID = 120
_LIQUID = np.linspace(0.1, 30.0, _N_LIQUID)
_WAGE = np.linspace(0.5, 6.0, 6)
_FPL_CLIFF = 15.0
_EDGE = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _interior_for_wage(wage: float, *, breakpoint_kind: str) -> np.ndarray:
    """Grid-edge interior, dropping the cells straddling this slice's cliff preimage.

    A jump budget has a value discontinuity at the preimage `liquid = fpl_cliff - wage`;
    a continuous-kink budget only kinks there. Either way the straddling grid cells are
    excluded so the comparison is over regions where the dense brute is itself accurate.
    """
    preimage = _FPL_CLIFF - wage
    width = 0.75 if breakpoint_kind == "jump" else 0.4
    away_from_cliff = np.abs(_LIQUID - preimage) > width
    return _EDGE & away_from_cliff


def _solve(
    variant: str, *, breakpoint_kind: str, n_consumption: int = 160
) -> Mapping[int, Mapping]:
    """Solve the CES-utility ride-along toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        breakpoint_kind=breakpoint_kind,
        n_liquid=_N_LIQUID,
        n_wage=_WAGE.size,
        liquid_max=30.0,
        wage_max=6.0,
        n_savings=220,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    params = toy.build_params(breakpoint_kind=breakpoint_kind)
    return model.solve(params=params, log_level="off")


def _terminal_adjacent_period(solved: Mapping[int, Mapping]) -> int:
    """The largest period in which `alive` is active (its continuation is bequest)."""
    return max(period for period in solved if "alive" in solved[period])


@pytest.mark.parametrize("breakpoint_kind", ["continuous_kink", "jump"])
def test_bqsegm_ces_utility_matches_brute(breakpoint_kind: str) -> None:
    """The CES ride-along solve equals brute across every wage slice's budget.

    With the CES Euler inversion honoured (the marginal utility inverted numerically
    from the period utility), BQSEGM matches the dense `GridSearch` value across the
    liquid interior of every continuous `wage` slice, away from the cells straddling
    the slice's cliff preimage.
    """
    bqsegm = _solve("bqsegm", breakpoint_kind=breakpoint_kind)
    brute = _solve("brute", breakpoint_kind=breakpoint_kind, n_consumption=1800)
    period = _terminal_adjacent_period(bqsegm)
    brute_v = np.asarray(brute[period]["alive"])
    bqsegm_v = np.asarray(bqsegm[period]["alive"])
    for wage_idx in range(brute_v.shape[1]):
        interior = _interior_for_wage(
            float(_WAGE[wage_idx]), breakpoint_kind=breakpoint_kind
        )
        np.testing.assert_allclose(
            bqsegm_v[interior, wage_idx],
            brute_v[interior, wage_idx],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"kind={breakpoint_kind} wage_idx={wage_idx}",
        )
