"""Construction spec for the DS-2026 App.2 housing EGM-FUES discrete-grid model.

The EGM-FUES column of DS-2026 Table 3 solves the continuous-housing model by
1-D FUES nested over the housing grid (its Box 2). pylcm builds this as a
discrete-choice DC-EGM: the next-housing choice is discretised onto the housing
grid and solved as a discrete action, with the inner liquid-asset DC-EGM and the
discrete-choice upper envelope reproducing the EGM-FUES solve — the same shape as
Application 3's discrete-housing model with Application 2's separable-CES utility.

These checks build the model at a tiny grid and assert structure only.
"""

from lcm import DCEGM, DiscreteGrid, GridSearch, Model
from tests.test_models import ds_app2_housing_fues as fues


def test_dcegm_variant_builds_with_discrete_housing_choice():
    """The EGM-FUES variant builds with housing as a discrete action.

    The next-housing decision is a discrete `housing_choice` action over the
    housing-level alphabet, and the inner solver is a DC-EGM on liquid assets —
    the discrete-choice upper-envelope shape, not a continuous outer search.
    """
    model = fues.build_model(variant="dcegm", n_grid=6, n_housing=5, n_periods=4)
    assert isinstance(model, Model)
    working = model.user_regimes["working"]
    assert "housing_choice" in working.actions
    assert "consumption" in working.actions
    assert isinstance(working.solver, DCEGM)
    assert working.solver.continuous_state == "liquid"
    assert working.solver.continuous_action == "consumption"


def test_brute_variant_builds_with_grid_search():
    """The VFI twin builds with the same economics and a grid-search solver."""
    model = fues.build_model(variant="brute", n_grid=6, n_housing=5, n_periods=4)
    working = model.user_regimes["working"]
    assert isinstance(working.solver, GridSearch)
    assert {"consumption", "housing_choice"} <= set(working.actions)


def test_housing_levels_scale_with_n_housing():
    """The discrete housing alphabet has exactly `n_housing` levels."""
    model = fues.build_model(variant="dcegm", n_grid=6, n_housing=7, n_periods=4)
    housing_grid = model.user_regimes["working"].states["housing"]
    assert isinstance(housing_grid, DiscreteGrid)
    assert len(housing_grid.to_jax()) == 7


def test_three_regimes_with_terminal_dead():
    """The model carries working, retired, and a terminal dead regime."""
    model = fues.build_model(variant="dcegm", n_grid=6, n_housing=5, n_periods=4)
    assert set(model.regime_names_to_ids) == {"working", "retired", "dead"}
    assert model.user_regimes["dead"].transition is None
