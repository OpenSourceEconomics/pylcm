"""Outer-search strategy configs and their NNBEGM normalization.

Covers the config-level validation of `FiniteOuterGrid` / `AdaptiveOuterMesh`
/ `LegacyGoldenSection` and the legacy-field folding on `NNBEGM`: exactly one
of `outer_search` and `outer_grid` must be set, and the legacy pair resolves
to an equivalent `FiniteOuterGrid`.
"""

from collections.abc import Callable

import pytest

from _lcm.grids import ContinuousGrid
from lcm import (
    NBEGM,
    NNBEGM,
    AdaptiveOuterMesh,
    FiniteOuterGrid,
    LegacyGoldenSection,
    LinSpacedGrid,
    OuterSearch,
)
from lcm.exceptions import RegimeInitializationError
from tests.test_models.n_nbegm_toy import OUTER_GRID, SAVINGS_GRID

_MESH_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=17)


def _make_nnbegm(
    *,
    outer_search: OuterSearch | None = None,
    outer_grid: ContinuousGrid | None = None,
    outer_batch_size: int = 0,
) -> NNBEGM:
    return NNBEGM(
        inner=NBEGM(
            continuous_state="wealth",
            post_decision_function="liquid_savings",
            budget_target="resources",
            savings_grid=SAVINGS_GRID,
        ),
        outer_action="illiquid_investment",
        outer_post_decision="next_illiquid",
        outer_no_adjustment_candidate="keep_illiquid",
        outer_search=outer_search,
        outer_grid=outer_grid,
        outer_batch_size=outer_batch_size,
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
        lambda: AdaptiveOuterMesh(initial_grid=_MESH_GRID, outer_policy_atol=0.0),
        lambda: AdaptiveOuterMesh(initial_grid=_MESH_GRID, inner_policy_atol=-1.0),
    ],
)
def test_adaptive_outer_mesh_rejects_bad_config(
    build: Callable[[], AdaptiveOuterMesh],
) -> None:
    with pytest.raises(RegimeInitializationError):
        build()


def test_legacy_golden_section_rejects_inverted_domain() -> None:
    with pytest.raises(RegimeInitializationError, match="upper"):
        LegacyGoldenSection(
            lower=1.0,
            upper=0.0,
            iterations=40,
            tolerance=1e-8,
            endpoint_rule="fortran",
            tie_break="fortran",
        )


def test_legacy_fields_resolve_to_equivalent_finite_grid() -> None:
    solver = _make_nnbegm(outer_grid=OUTER_GRID, outer_batch_size=4)
    resolved = solver.resolved_outer_search
    assert resolved == FiniteOuterGrid(grid=OUTER_GRID, batch_size=4)


def test_explicit_outer_search_is_passed_through() -> None:
    search = FiniteOuterGrid(grid=OUTER_GRID, batch_size=2)
    solver = _make_nnbegm(outer_search=search)
    assert solver.resolved_outer_search is search


def test_adaptive_outer_mesh_config_is_accepted_at_construction() -> None:
    """The strategy validates at config time; kernel wiring lands later."""
    solver = _make_nnbegm(outer_search=AdaptiveOuterMesh(initial_grid=_MESH_GRID))
    assert isinstance(solver.resolved_outer_search, AdaptiveOuterMesh)


def test_missing_outer_search_and_grid_raise() -> None:
    with pytest.raises(RegimeInitializationError, match="outer_search"):
        _make_nnbegm()


def test_both_outer_search_and_grid_raise() -> None:
    with pytest.raises(RegimeInitializationError, match="not both"):
        _make_nnbegm(
            outer_search=FiniteOuterGrid(grid=OUTER_GRID),
            outer_grid=OUTER_GRID,
        )


def test_outer_batch_size_alongside_outer_search_raises() -> None:
    with pytest.raises(RegimeInitializationError, match="outer_batch_size"):
        _make_nnbegm(
            outer_search=FiniteOuterGrid(grid=OUTER_GRID),
            outer_batch_size=3,
        )
