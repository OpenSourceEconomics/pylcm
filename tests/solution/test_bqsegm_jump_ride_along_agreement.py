"""BQSEGM agreement on a subsidy cliff declared on derived income, with ride-along.

The BQSEGM ride-along solver maps a *jump* on the derived `gross_income` to its
per-`kind` asset preimage and solves the jumped budget against the savings-space
continuation. Where the continuation is smooth (the terminal-adjacent working
period, whose next value is the bequest), the jumped-budget solve reproduces the
dense `GridSearch` value across the cliff in both `kind` slices — proving the
per-cell preimage, the savings-space case partition, and the ride-along batching
end-to-end.

The deeper working periods carry a *recurring* jumped continuation: next
period's value itself has the cliff, and the transition-aware reader
(`bind_continuation`) interpolates that next value linearly across the jump, so
an agent who would save to the eligible side reads a smoothed continuation. That
residual (a few percent of value, grid-independent — see
`test_bqsegm_jump_ride_along_recurring_needs_topology` below) is the
topology-preserving-continuation limitation, tracked separately; it is a property
of the continuation reader, not of the jump-budget step exercised here.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import bqsegm_jump_ride_along_toy as toy

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


def test_bqsegm_jump_matches_brute_across_the_cliff_smooth_continuation():
    """The jumped derived-income solve equals brute across each slice's cliff.

    At the terminal-adjacent working period the continuation is the smooth
    bequest, so the savings-space jump step is exact: BQSEGM matches the dense
    `GridSearch` value across the asset interior of both `kind` slices, through
    each slice's own cliff preimage (away from the single straddling grid cell).
    """
    bqsegm = _solve("bqsegm")
    brute = _solve("brute", n_consumption=1800)
    period = _terminal_adjacent_period(bqsegm)
    brute_v = np.asarray(brute[period]["alive"])
    bqsegm_v = np.asarray(bqsegm[period]["alive"])
    for kind in range(brute_v.shape[0]):
        interior = _interior_for_kind(kind)
        np.testing.assert_allclose(
            bqsegm_v[kind, interior],
            brute_v[kind, interior],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period} kind={kind}",
        )


def test_bqsegm_jump_ride_along_recurring_needs_topology():
    """The recurring-jump residual is grid-independent, not a resolution artifact.

    Refining the savings grid does not shrink the deeper periods' error, pinning
    it to the linear continuation read across next period's jump rather than to
    EGM resolution — the topology-preserving-continuation limitation. This test
    documents that the residual stays above the smooth-continuation tolerance and
    is insensitive to the savings grid, so a future topology-preserving reader has
    a concrete target to beat.
    """
    brute = _solve("brute", n_consumption=1800)
    period = min(period for period in brute if "alive" in brute[period])

    def _worst_recurring_error(n_savings: int) -> float:
        model = toy.build_model(
            variant="bqsegm",
            n_liquid=_N_LIQUID,
            liquid_max=30.0,
            n_savings=n_savings,
            savings_max=28.0,
            n_consumption=160,
        )
        bqsegm_v = np.asarray(
            model.solve(params=toy.build_params(), log_level="off")[period]["alive"]
        )
        brute_v = np.asarray(brute[period]["alive"])
        worst = 0.0
        for kind in range(brute_v.shape[0]):
            interior = _interior_for_kind(kind)
            worst = max(
                worst,
                float(np.abs(bqsegm_v[kind, interior] - brute_v[kind, interior]).max()),
            )
        return worst

    coarse = _worst_recurring_error(220)
    fine = _worst_recurring_error(880)
    # Above the smooth-continuation tolerance, and not closed by 4x more savings
    # nodes: the residual is the continuation topology, not EGM resolution.
    assert coarse > 2e-2
    assert fine > 2e-2
    assert abs(coarse - fine) < 5e-3
