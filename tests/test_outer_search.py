"""Outer-search strategy configs and how `NNBEGM` carries them.

Covers the config-level validation of `FiniteOuterGrid` / `AdaptiveOuterMesh`,
that every knob they publish reaches the numerics, and that
`NNBEGM.outer_search` is mandatory: the strategy object is the only way to
describe the outer margin's candidates, so there is no field pair to keep
consistent.
"""

from collections.abc import Callable
from dataclasses import fields

import numpy as np
import pytest

from lcm import LinSpacedGrid
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import (
    NBEGM,
    NNBEGM,
    AdaptiveOuterMesh,
    FiniteOuterGrid,
    OuterSearch,
)
from tests.conftest import assert_agrees_to_ulp
from tests.test_models import n_nbegm_toy as toy
from tests.test_models.n_nbegm_toy import OUTER_GRID, SAVINGS_GRID

_MESH_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=17)
_TOY_PARAMS = {"discount_factor": 0.95}


def _make_nnbegm(*, outer_search: OuterSearch) -> NNBEGM:
    return NNBEGM(
        inner=NBEGM(savings_grid=SAVINGS_GRID),
        outer_search=outer_search,
    )


def test_finite_outer_grid_rejects_negative_batch_size() -> None:
    with pytest.raises(RegimeInitializationError, match="batch_size"):
        FiniteOuterGrid(grid=_MESH_GRID, batch_size=-1)


@pytest.mark.parametrize(
    "build",
    [
        lambda: AdaptiveOuterMesh(initial_grid=_MESH_GRID, max_nodes=1),
        lambda: AdaptiveOuterMesh(initial_grid=_MESH_GRID, max_refinement_rounds=-1),
        lambda: AdaptiveOuterMesh(initial_grid=_MESH_GRID, golden_iterations=0),
        lambda: AdaptiveOuterMesh(initial_grid=_MESH_GRID, batch_size=-2),
        lambda: AdaptiveOuterMesh(initial_grid=_MESH_GRID, value_atol=0.0),
        lambda: AdaptiveOuterMesh(initial_grid=_MESH_GRID, value_rtol=-1e-8),
    ],
)
def test_adaptive_outer_mesh_rejects_bad_config(
    build: Callable[[], AdaptiveOuterMesh],
) -> None:
    with pytest.raises(RegimeInitializationError):
        build()


def test_adaptive_outer_mesh_publishes_only_knobs_the_numerics_read() -> None:
    """Every `AdaptiveOuterMesh` field changes the computation.

    A public setting the solver never reads describes behaviour the model
    does not have: a caller who sets it gets the default silently. The field
    set is pinned here so a new knob has to arrive with a reader.
    """
    assert {f.name for f in fields(AdaptiveOuterMesh)} == {
        "initial_grid",
        "max_nodes",
        "max_refinement_rounds",
        "batch_size",
        "value_atol",
        "value_rtol",
        "outer_lipschitz_bound",
        "golden_iterations",
        "fail_closed",
    }


def test_outer_search_is_stored_as_given() -> None:
    """The field is the strategy — nothing normalizes or replaces it."""
    search = FiniteOuterGrid(grid=OUTER_GRID, batch_size=2)
    solver = _make_nnbegm(outer_search=search)
    assert solver.outer_search is search


def test_adaptive_outer_mesh_config_is_accepted_at_construction() -> None:
    """The strategy validates at config time; kernel wiring lands later."""
    solver = _make_nnbegm(outer_search=AdaptiveOuterMesh(initial_grid=_MESH_GRID))
    assert isinstance(solver.outer_search, AdaptiveOuterMesh)


def test_nnbegm_without_outer_search_raises_at_construction() -> None:
    """`outer_search` is mandatory: no default, no legacy fallback.

    A missing search strategy is a signature error, not a validation error
    raised from `__post_init__` — there is no longer a second field that could
    have supplied it.
    """
    with pytest.raises(TypeError, match="outer_search"):
        NNBEGM(  # ty: ignore[missing-argument]
            inner=NBEGM(savings_grid=SAVINGS_GRID),
        )


# Frozen from a solve of the two-period `n_nbegm` toy through the retired
# `outer_grid=OUTER_GRID, outer_batch_size=4` pair. Pins that
# `FiniteOuterGrid(grid=..., batch_size=...)` is the same computation and not
# merely an accepted one.
#
# Captured UNDER THE SUITE, which runs x64; the same solve from a plain script
# runs float32 and disagrees in the 8th digit, so a baseline harvested outside
# pytest would pin the wrong numbers.
#
# Compared in ULP of the working format, not to the bit and not to a fixed
# relative tolerance. Two reasons, and they point the same way. The pair agreed
# bit-for-bit when the baseline was taken, but rebuilding the exact-affine CPU
# kernel moved two of these entries by one ULP, so bit equality pins the `.so`
# build rather than the equivalence under test; and the numbers below are x64,
# while the suite also runs at `--precision=32`, where the same solve lands
# ~1e-7 away -- a fixed rtol either fails the fp32 leg or is too loose to say
# anything at x64. Measured gaps: head 1 ULP at x64, 2 at fp32; the sum is exact
# at both. `_FINITE_GRID_N_ULP` doubles that, and a genuine change in candidate
# generation moves these values by orders of magnitude more.
_FINITE_GRID_N_ULP = 4
_FINITE_GRID_BASELINE_HEAD = (
    -25.336494502711897,
    -19.638656410864485,
    -15.20980502450777,
    -12.968285372927772,
    -11.12194670592802,
    -9.684120807993004,
)
_FINITE_GRID_BASELINE_SUM = -2538.617102280524
_FINITE_GRID_BASELINE_SIZE = 240


def test_finite_outer_grid_reproduces_the_retired_legacy_pair_values() -> None:
    """`FiniteOuterGrid(grid=g, batch_size=b)` computes what `outer_grid=g,
    outer_batch_size=b` computed."""
    solution = toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=FiniteOuterGrid(grid=toy.OUTER_GRID, batch_size=4),
    ).solve(params=_TOY_PARAMS, log_level="off")

    values = np.concatenate(
        [
            np.asarray(solution[period][name]).ravel()
            for period in sorted(solution)
            for name in sorted(solution[period])
        ]
    )
    assert values.size == _FINITE_GRID_BASELINE_SIZE
    assert np.isfinite(values).all()
    assert_agrees_to_ulp(
        values[:6],
        np.asarray(_FINITE_GRID_BASELINE_HEAD, dtype=values.dtype),
        n_ulp=_FINITE_GRID_N_ULP,
        err_msg="first six solution values",
    )
    assert_agrees_to_ulp(
        values.sum(),
        np.asarray(_FINITE_GRID_BASELINE_SUM, dtype=values.dtype),
        n_ulp=_FINITE_GRID_N_ULP,
        err_msg="sum over all solution values",
    )
