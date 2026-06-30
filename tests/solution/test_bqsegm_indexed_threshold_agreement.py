"""BQSEGM agreement when a cliff threshold is read from a per-cell table.

The subsidy cliff sits at `gross_income == fpl_cliff[kind]`, a threshold pulled
from a length-2 table indexed by the ride-along `kind` state. Because the derived
income offset is kind-invariant, the cliff's per-slice asset preimage is driven
entirely by the table value. The BQSEGM ride-along solver must resolve each
cell's threshold from the table before mapping it to the asset preimage; where
the continuation is smooth (the terminal-adjacent working period, whose next
value is the bequest), the jumped-budget solve reproduces the dense `GridSearch`
value across each slice's own cliff.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import bqsegm_indexed_threshold_toy as toy

_N_LIQUID = 160
_LIQUID = np.linspace(0.1, 30.0, _N_LIQUID)
# Cliff preimage per kind code: liquid = fpl_cliff[kind] - base_income.
_CLIFF_BY_KIND = (14.0 - 5.0, 11.0 - 5.0)  # (lo, hi) = (9.0, 6.0)
_EDGE = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _interior_for_kind(kind: int) -> np.ndarray:
    """Grid-edge interior minus the one cell straddling this slice's cliff."""
    away_from_cliff = np.abs(_LIQUID - _CLIFF_BY_KIND[kind]) > 0.75
    return _EDGE & away_from_cliff


def _solve(variant: str, *, n_consumption: int = 160) -> Mapping[int, Mapping]:
    """Solve the indexed-threshold toy on the shared comparison grids."""
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


def test_bqsegm_indexed_threshold_matches_brute_across_each_cliff():
    """The per-cell threshold places each slice's cliff at the right asset preimage.

    At the terminal-adjacent working period the continuation is the smooth
    bequest, so the savings-space jump step is exact: BQSEGM matches the dense
    `GridSearch` value across the asset interior of both `kind` slices, through
    each slice's own cliff preimage `fpl_cliff[kind] - base_income` (away from the
    single straddling grid cell).
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
