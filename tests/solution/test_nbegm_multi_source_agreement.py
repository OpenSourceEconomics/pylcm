"""NBEGM agrees with brute when the budget kinks on several derived income vars.

The budget nets two taxes that bracket on two distinct monotone income concepts,
each offset differently by the ride-along `kind`, so their asset-space breakpoints
sit at different liquid points and reorder between slices. NBEGM must merge the
breakpoints declared across the two variables into one per-cell sorted partition
and match the dense-grid `GridSearch` value across the asset interior of each
slice.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from tests.test_models import nbegm_multi_source_jump_toy as jump_toy
from tests.test_models import nbegm_multi_source_toy as toy

_LIQUID = jnp.linspace(0.1, 30.0, 120)
# Stay clear of the per-kind breakpoints (preimages 11/14 for `a`, 12 for `b`)
# only at the grid edges; the interior spans across the kinks.
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the two-derived-variable budget toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_nbegm_merges_two_derived_variable_kinks_matching_brute() -> None:
    """A budget with kinks on two derived income vars equals brute in both slices."""
    nbegm = _solve("nbegm")
    brute = _solve("brute", n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in nbegm[period]:
            continue
        brute_v = np.asarray(brute[period]["alive"])
        nbegm_v = np.asarray(nbegm[period]["alive"])
        # Value is shaped (kind, liquid); compare the interior of each kind slice.
        for kind in range(brute_v.shape[0]):
            np.testing.assert_allclose(
                nbegm_v[kind, _INTERIOR],
                brute_v[kind, _INTERIOR],
                atol=2e-2,
                rtol=5e-3,
                err_msg=f"period={period} kind={kind}",
            )


# Jump preimage `cliff_b - base_b[kind]` is 12.0 in both slices; the kink preimage
# `kink_a - base_a[kind]` is 14.0 (lo) / 11.0 (hi), so the jump's sorted position
# swaps between slices.
_JUMP_PREIMAGE = (12.0, 12.0)


def _interior_away_from_jump(kind: int) -> np.ndarray:
    """Grid-edge interior minus the one cell straddling this slice's jump."""
    edge = (_LIQUID > 1.5) & (_LIQUID < 27.0)
    return np.asarray(edge & (jnp.abs(_LIQUID - _JUMP_PREIMAGE[kind]) > 0.75))


def _solve_jump(variant: str, *, n_consumption: int = 160) -> Mapping[int, Mapping]:
    """Solve the mixed jump-and-kink two-variable toy on the comparison grids."""
    model = jump_toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=220,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=jump_toy.build_params(), log_level="off")


def test_nbegm_merges_a_jump_and_kink_across_variables_matching_brute() -> None:
    """A jump and a kink on two income vars, reordering per slice, equals brute.

    At the terminal-adjacent working period the continuation is the smooth
    bequest, so the savings-space jump step is exact. The jump (on `income_b`) and
    the kink (on `income_a`) map to per-`kind` asset preimages whose order swaps
    between slices; NBEGM recovers the jump's position per cell and matches the
    dense `GridSearch` value across each slice's cliff.
    """
    nbegm = _solve_jump("nbegm")
    brute = _solve_jump("brute", n_consumption=1800)
    period = max(p for p in nbegm if "alive" in nbegm[p])
    brute_v = np.asarray(brute[period]["alive"])
    nbegm_v = np.asarray(nbegm[period]["alive"])
    for kind in range(brute_v.shape[0]):
        interior = _interior_away_from_jump(kind)
        np.testing.assert_allclose(
            nbegm_v[kind, interior],
            brute_v[kind, interior],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period} kind={kind}",
        )
