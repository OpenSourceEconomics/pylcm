"""Reading the continuation off a per-period grid costs no extra compilation.

The target's nodes reach the kernel as ordinary array arguments, not as part of
the compiled core's identity. An `AgeSpecializedGrid` has invariant shape by
contract, so one compiled program consumes every period's node values, and a
model whose grids move with age lowers no more cores than the same model with
static grids.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.one_asset_egm_step import egm_one_asset_step
from lcm import AgeSpecializedGrid, LinSpacedGrid
from lcm.solvers import EGM, Solver
from tests.solution._crra_preferences import crra_preferences
from tests.solution.test_egm_continuation_grid_provenance import (
    _N_PERIODS,
    _SAVINGS_GRID,
)
from tests.test_models.deterministic.ds_pension import get_model, get_params


def _reader_cache_entries(next_m_grids):
    """Trace-cache size after reading one affine `V` on several target grids."""
    liquid_grid = jnp.linspace(0.5, 3.0, 5)

    @jax.jit
    def read(next_value, next_marginal, next_liquid_grid):
        return egm_one_asset_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid_grid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=jnp.linspace(0.0, 2.0, 6),
            discount_factor=0.98,
            preferences=crra_preferences(2.0),
            return_liquid=0.02,
            income=1.0,
        ).value

    for next_liquid_grid in next_m_grids:
        next_value = 2.0 * next_liquid_grid
        next_marginal = jnp.full_like(next_liquid_grid, 2.0)
        read(next_value, next_marginal, next_liquid_grid).block_until_ready()
    return read._cache_size()  # ty: ignore[unresolved-attribute]


def test_numerically_different_target_grids_share_one_compiled_program():
    """Three same-shaped target grids trace the reader once.

    The nodes are data, not part of the program. Were they baked into the
    compiled core instead, every period of an age-specialized model would pay
    its own compilation.
    """
    n_points = 5
    grids = [jnp.linspace(0.5, stop, n_points) for stop in (3.0, 3.5, 4.0, 4.5, 5.0)]
    assert _reader_cache_entries(grids) == 1


def _lowered_core_identities(model):
    """Number of distinct kernel core objects the model would compile."""
    return len(
        {
            id(core)
            for regime in model._regimes.values()
            for kernel in regime.solution.period_kernels.values()
            for core in kernel.cores().values()
        }
    )


def test_an_age_specialized_model_compiles_no_more_cores_than_a_static_one():
    """Moving the liquid grid with age does not multiply compiled programs.

    Every period's nodes flow through one shared core, so the count is set by
    the number of genuinely distinct kernel branches — interior versus
    retirement boundary — and not by how many distinct grid signatures the
    lifecycle has.
    """
    solvers: dict[str, Solver] = {
        "retired": EGM(savings_grid=_SAVINGS_GRID),
    }
    static = get_model(n_periods=_N_PERIODS, solvers=solvers)
    moving = get_model(
        n_periods=_N_PERIODS,
        solvers=solvers,
        retired_liquid_grid=AgeSpecializedGrid(
            build=lambda age: LinSpacedGrid(
                start=0.1, stop=20.0 - 2.0 * float(age), n_points=12
            ),
            signature=float,
        ),
    )
    assert _lowered_core_identities(moving) == _lowered_core_identities(static)


def test_the_static_model_solves_reproducibly():
    """Two solves of one static-grid model publish bit-identical values.

    Threading the target's nodes in as runtime arguments must not make the
    result depend on anything that varies between calls. Equality, not a
    tolerance: the two solves run the same compiled program on the same
    inputs, so any difference at all would be a defect.
    """
    solvers: dict[str, Solver] = {
        "retired": EGM(savings_grid=_SAVINGS_GRID),
    }
    model = get_model(n_periods=_N_PERIODS, solvers=solvers)
    first = model.solve(params=get_params(), log_level="debug")
    second = model.solve(params=get_params(), log_level="debug")
    for period, regime_to_V in first.items():
        for regime_name, V_arr in regime_to_V.items():
            np.testing.assert_array_equal(
                np.asarray(V_arr), np.asarray(second[period][regime_name])
            )
