"""NBEGM agreement with brute when a deterministic co-state rides along.

The continuous-schedule NBEGM path must solve the 1-D liquid problem once per
ride-along `kind` slice, each against that slice's own budget (`base_income`
depends on `kind`) and continuation value. Its value function must reproduce the
dense-grid `GridSearch` value across the asset interior and through the bracket
kink, in every `kind` slice and at every working age.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_prepared_grid, prepare_padded_grid
from tests.test_models import nbegm_ride_along_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 1.5) & (_LIQUID < 27.0)


def _solve(variant: str, *, n_consumption: int = 120) -> Mapping[int, Mapping]:
    """Solve the ride-along tax toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_nbegm_matches_brute_in_every_ride_along_slice_every_age():
    """The schedule solve equals brute in both `kind` slices, kink and all."""
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


def test_value_is_invariant_to_envelope_cell_blocking():
    """The solved value does not depend on the envelope's ride-cell block size.

    `cell_block_size` streams both ride-along cores (continuation fan-out and
    envelope solve) over blocks of ride cells instead of vmapping the whole
    flattened mesh at once; padding cells are discarded after the scan, so the
    value function is identical for any block size, including one that does not
    divide the cell count.
    """
    reference = _solve("nbegm")
    for block_size in (1, 3):
        model = toy.build_model(
            variant="nbegm",
            n_liquid=120,
            liquid_max=30.0,
            n_savings=180,
            savings_max=28.0,
            nbegm_overrides={"cell_block_size": block_size},
        )
        blocked = model.solve(params=toy.build_params(), log_level="off")
        for period in reference:
            if "alive" not in reference[period]:
                continue
            np.testing.assert_allclose(
                np.asarray(blocked[period]["alive"]),
                np.asarray(reference[period]["alive"]),
                atol=1e-10,
                rtol=1e-10,
                err_msg=f"period={period} block_size={block_size}",
            )


def test_nbegm_matches_brute_with_per_kind_utility_curvature():
    """Per-`kind` CRRA routed through a DAG node yields brute-level accuracy.

    Each ride-along slice carries its own utility curvature
    (`crra_of_kind = crra_by_kind[kind]`), so the Euler inversion and the value
    assembly must both use the cell's own coefficient; using another slice's
    curvature would shift the whole value function.
    """
    nbegm_model = toy.build_model(
        variant="nbegm",
        per_kind_crra=True,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
    )
    brute_model = toy.build_model(
        variant="brute",
        per_kind_crra=True,
        n_liquid=120,
        liquid_max=30.0,
        n_consumption=1500,
    )
    params = toy.build_params(per_kind_crra=True)
    nbegm = nbegm_model.solve(params=params, log_level="off")
    brute = brute_model.solve(params=params, log_level="off")
    for period in brute:
        if "alive" not in brute[period] or "alive" not in nbegm[period]:
            continue
        brute_v = np.asarray(brute[period]["alive"])
        nbegm_v = np.asarray(nbegm[period]["alive"])
        for kind in range(brute_v.shape[0]):
            np.testing.assert_allclose(
                nbegm_v[kind, _INTERIOR],
                brute_v[kind, _INTERIOR],
                atol=2e-2,
                rtol=5e-3,
                err_msg=f"period={period} kind={kind}",
            )


def test_top_edge_query_matches_convention_oracle_on_a_coarse_liquid_grid():
    """The top liquid grid point carries the continuum optimum's value.

    The oracle is convention-matched (a dense consumption search against the
    same Hermite carry read NBEGM's continuation uses), so it isolates the
    candidate-set geometry at the grid edge: an Euler branch whose endogenous
    abscissae stop just inside the grid span must still bracket the boundary
    query (its preimage extends past the sampled grid), and an optimum at a
    continuation kink between Euler roots (a child-value interpolation node)
    must be carried by a savings-node candidate. A brute reference cannot
    serve here: its linear child-value read diverges from the Hermite read on
    a coarse child grid.
    """
    crra_by_kind = (3.84, 1.0)
    params = toy.build_params(
        per_kind_crra=True, crra_lo=crra_by_kind[0], crra_hi=crra_by_kind[1]
    )
    nbegm = toy.build_model(
        variant="nbegm",
        per_kind_crra=True,
        n_liquid=3,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
    ).solve(params=params, log_level="off")
    last_alive = max(period for period in nbegm if "alive" in nbegm[period])
    nbegm_v = np.asarray(nbegm[last_alive]["alive"])

    liquid_grid = jnp.array([0.1, 15.05, 30.0])
    dead_value = liquid_grid ** (1.0 - 2.0) / (1.0 - 2.0)
    dead_marginal = liquid_grid ** (-2.0)
    search_grid, valid_length = prepare_padded_grid(liquid_grid)
    base_income = (1.0, 4.0)
    for kind, crra in enumerate(crra_by_kind):
        coh_top = 30.0 + base_income[kind] - 0.3 * (30.0 - 12.0)
        consumption = jnp.linspace(1e-3, coh_top - 1e-6, 200_000)
        next_liquid = 1.03 * (coh_top - consumption) + 1.0
        continuation = interp_on_prepared_grid(
            x_query=next_liquid,
            search_grid=search_grid,
            valid_length=valid_length,
            xp=liquid_grid,
            fp=dead_value,
            fp_slopes=dead_marginal,
        )
        if crra == 1.0:
            utility = jnp.log(consumption)
        else:
            utility = consumption ** (1.0 - crra) / (1.0 - crra)
        oracle = float(jnp.max(utility + 0.95 * continuation))
        np.testing.assert_allclose(
            nbegm_v[kind, -1],
            oracle,
            atol=5e-2,
            rtol=5e-3,
            err_msg=f"top point, kind={kind}",
        )
