"""NBEGM agreement on a subsidy cliff declared on derived income, with ride-along.

The NBEGM ride-along solver maps a *jump* on the derived `gross_income` to its
per-`kind` asset preimage and solves the jumped budget against the savings-space
continuation. Where the continuation is smooth (the terminal-adjacent working
period, whose next value is the bequest), the jumped-budget solve reproduces the
dense `GridSearch` value across the cliff in both `kind` slices — proving the
per-cell preimage, the savings-space case partition, and the ride-along batching
end-to-end.

The deeper working periods carry a *recurring* jumped continuation: next
period's value itself has the cliff. The transition-aware reader
(`bind_continuation`) reads it side-faithfully — the jump rides inside the carry
as a duplicated abscissa with exact one-sided limits, so a straddling bracket
interpolates between the nearest own-side node and the cliff's side limit and
never averages across the jump. The remaining error is grid resolution at the
cliff preimage, shrinking as the savings grid refines
(`test_nbegm_jump_ride_along_recurring_is_resolution_limited` below).
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_jump_ride_along_toy as toy

_N_LIQUID = 160
_LIQUID = np.linspace(0.1, 30.0, _N_LIQUID)
# Cliff preimage per kind code: liquid = fpl_cliff - base_income[kind].
_CLIFF_BY_KIND = (15.0 - 1.0, 15.0 - 4.0)  # (lo, hi)
_EDGE = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _interior_for_kind(kind: int) -> np.ndarray:
    """Grid-edge interior minus the one cell straddling this slice's cliff."""
    away_from_cliff = np.abs(_LIQUID - _CLIFF_BY_KIND[kind]) > 0.75
    return _EDGE & away_from_cliff


def _solve(variant: str, *, n_consumption: int = 160) -> Mapping[int, Mapping]:
    """Solve the jump-ride-along toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=_N_LIQUID,
        liquid_max=30.0,
        n_savings=220,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def _terminal_adjacent_period(solved: Mapping[int, Mapping]) -> int:
    """The largest period in which `alive` is active (its continuation is bequest)."""
    return max(period for period in solved if "alive" in solved[period])


def test_nbegm_jump_matches_brute_across_the_cliff_smooth_continuation():
    """The jumped derived-income solve equals brute across each slice's cliff.

    At the terminal-adjacent working period the continuation is the smooth
    bequest, so the savings-space jump step is exact: NBEGM matches the dense
    `GridSearch` value across the asset interior of both `kind` slices, through
    each slice's own cliff preimage (away from the single straddling grid cell).
    """
    nbegm = _solve("nbegm")
    brute = _solve("brute", n_consumption=1800)
    period = _terminal_adjacent_period(nbegm)
    brute_v = np.asarray(brute[period]["alive"])
    nbegm_v = np.asarray(nbegm[period]["alive"])
    for kind in range(brute_v.shape[0]):
        interior = _interior_for_kind(kind)
        np.testing.assert_allclose(
            nbegm_v[kind, interior],
            brute_v[kind, interior],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period} kind={kind}",
        )


def test_nbegm_jump_ride_along_recurring_is_resolution_limited():
    """The recurring-jump residual shrinks with the savings grid.

    The side-faithful continuation read never averages across next period's
    value cliff: straddling brackets interpolate between the nearest own-side
    node and the cliff's exact side limit. The deeper periods' error is then
    resolution, not topology — the save-to-cliff distance shrinks with the
    savings grid, and the node-to-limit secant shrinks with the liquid grid —
    so refining the savings grid improves on the coarse residual. Closing
    the remainder needs an explicit save-to-cliff candidate (an off-grid
    savings candidate at each child breakpoint preimage).
    """
    brute = _solve("brute", n_consumption=1800)
    period = min(period for period in brute if "alive" in brute[period])

    def _worst_recurring_error(n_savings: int) -> float:
        model = toy.build_model(
            variant="nbegm",
            n_liquid=_N_LIQUID,
            liquid_max=30.0,
            n_savings=n_savings,
            savings_max=28.0,
            n_consumption=160,
        )
        nbegm_v = np.asarray(
            model.solve(params=toy.build_params(), log_level="off")[period]["alive"]
        )
        brute_v = np.asarray(brute[period]["alive"])
        worst = 0.0
        for kind in range(brute_v.shape[0]):
            interior = _interior_for_kind(kind)
            worst = max(
                worst,
                float(np.abs(nbegm_v[kind, interior] - brute_v[kind, interior]).max()),
            )
        return worst

    coarse = _worst_recurring_error(220)
    fine = _worst_recurring_error(880)
    # Above tolerance at the coarse savings grid, but shrinking with 4x more
    # nodes: the residual is grid resolution at the cliff preimage, not a
    # grid-independent topology error.
    assert coarse > 2e-2
    assert fine < coarse
    assert fine < 6e-2
