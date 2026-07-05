"""NBEGM agreement when a cliff threshold lives inside a `MappingLeaf` param.

The subsidy cliff sits at `gross_income == fpl_cliff[kind]`, where the `fpl_cliff`
table is nested one level deeper, inside the `tax_schedule` mapping leaf
(`tax_schedule.data["fpl_cliff"]`) — the way the ACA model groups its bracket and
FPL schedules. The NBEGM ride-along solver must follow the
`tax_schedule.fpl_cliff` sub-key path into the leaf, read the per-cell threshold,
and map it to its asset preimage before solving. As with a bare indexed table, the
derived income offset is kind-invariant, so each slice's cliff preimage is driven
entirely by the table value; at the terminal-adjacent working period (whose
continuation is the smooth bequest) the jumped-budget solve reproduces the dense
`GridSearch` value across each slice's own cliff.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_mappingleaf_threshold_toy as toy

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
    """Solve the leaf-nested-threshold toy on the shared comparison grids."""
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


def test_nbegm_mappingleaf_threshold_matches_brute_across_each_cliff():
    """A sub-key threshold places each slice's cliff at the right asset preimage.

    With the cliff table nested in the `tax_schedule` mapping leaf, the
    ride-along solver follows the `tax_schedule.fpl_cliff` sub-key path, reads each
    cell's threshold, and maps it to the asset preimage `fpl_cliff[kind] -
    base_income`. At the terminal-adjacent working period the continuation is the
    smooth bequest, so the savings-space jump step is exact: NBEGM matches the
    dense `GridSearch` value across the asset interior of both `kind` slices,
    through each slice's own cliff preimage (away from the single straddling grid
    cell).
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
