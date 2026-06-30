"""BQSEGM agreement on a derived-income subsidy cliff with a continuous co-state.

The M1 ACA regime carries one Euler/liquid state alongside several other
continuous states — a deterministic co-state (AIME) and stochastic shock grids —
all of which the BQSEGM schedule solver must treat as ride-along axes integrated
by the continuation reader, with the liquid axis named explicitly. This toy is
the miniature: `liquid` is the Euler axis, `wage` a continuous ride-along
co-state, and a subsidy cliff declared on derived `gross_income = liquid + wage`
maps to a different liquid preimage in every `wage` slice.

At the terminal-adjacent working period the continuation is the smooth bequest,
so the savings-space jump step is exact: BQSEGM reproduces the dense `GridSearch`
value across the liquid interior of every `wage` slice, through each slice's own
cliff preimage.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import bqsegm_continuous_ride_along_toy as toy

_N_LIQUID = 120
_LIQUID = np.linspace(0.1, 30.0, _N_LIQUID)
_WAGE = np.linspace(0.5, 6.0, 6)
_FPL_CLIFF = 15.0
_EDGE = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _interior_for_wage(wage: float) -> np.ndarray:
    """Grid-edge interior minus the cells straddling this slice's cliff preimage."""
    preimage = _FPL_CLIFF - wage
    away_from_cliff = np.abs(_LIQUID - preimage) > 0.75
    return _EDGE & away_from_cliff


def _solve(variant: str, *, n_consumption: int = 160) -> Mapping[int, Mapping]:
    """Solve the continuous-ride-along toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=_N_LIQUID,
        n_wage=_WAGE.size,
        liquid_max=30.0,
        wage_max=6.0,
        n_savings=220,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def _terminal_adjacent_period(solved: Mapping[int, Mapping]) -> int:
    """The largest period in which `alive` is active (its continuation is bequest)."""
    return max(period for period in solved if "alive" in solved[period])


def test_bqsegm_continuous_ride_along_regime_builds() -> None:
    """A BQSEGM schedule regime with more than one continuous state builds when the
    liquid axis is named, treating the continuous co-state as a ride-along axis."""
    model = toy.build_model(variant="bqsegm")
    solved = model.solve(params=toy.build_params(), log_level="off")
    period = _terminal_adjacent_period(solved)
    assert np.asarray(solved[period]["alive"]).shape == (_N_LIQUID, _WAGE.size)


def test_bqsegm_continuous_co_state_matches_brute_across_the_cliff() -> None:
    """The jumped derived-income solve equals brute across every wage slice's cliff.

    At the terminal-adjacent working period the continuation is the smooth bequest,
    so the savings-space jump step is exact: BQSEGM matches the dense `GridSearch`
    value across the liquid interior of every continuous `wage` slice, through each
    slice's own cliff preimage (away from the straddling grid cells).
    """
    bqsegm = _solve("bqsegm")
    brute = _solve("brute", n_consumption=1800)
    period = _terminal_adjacent_period(bqsegm)
    brute_v = np.asarray(brute[period]["alive"])
    bqsegm_v = np.asarray(bqsegm[period]["alive"])
    for wage_idx in range(brute_v.shape[1]):
        interior = _interior_for_wage(float(_WAGE[wage_idx]))
        np.testing.assert_allclose(
            bqsegm_v[interior, wage_idx],
            brute_v[interior, wage_idx],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period} wage_idx={wage_idx}",
        )
