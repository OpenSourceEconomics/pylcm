"""Construction spec for the DS-2026 Application 3 discrete-housing model (no tax).

The Dobrescu-Shanker (2026) Section 2.3 discrete-housing model maps onto pylcm's
DC-EGM solver as a discrete-choice consumption-savings problem:

- financial assets `assets` (`a >= 0`) are the continuous Euler state the Euler
  equation inverts on, and `consumption` (`c`) is the continuous action;
- the housing stock `housing` (`H`) is a discrete state carried as a
  value-function grid axis (`rent` plus the owned levels `own_h1..own_h5`);
- the own-vs-rent-and-level choice is a single discrete action `housing_choice`
  over the same categories, exactly like Application 1's work/retire labor-supply
  action — solved by the discrete-choice upper envelope (FUES/MSS/LTM) and by
  grid search (VFI);
- the next-period housing state equals the chosen code (`next_housing =
  housing_choice`), so the discrete action drives the discrete-state transition.

These checks build the model at a tiny grid and assert structure only — no
`.solve()` runs here. The ground truth for the DC-EGM mapping is the regime
processing pipeline the `Model` constructor runs at build, including the DC-EGM
contract validator: a model that builds *is* an accepted DC-EGM model. A
grid-search (brute/VFI) twin builds the same economics as the Table 4 oracle
method, so both solver variants are constructed and validated.
"""

from typing import Literal

import numpy as np
import pytest

from lcm import DiscreteGrid, Model
from lcm.solvers import DCEGM, GridSearch
from tests.test_models import ds_app3_discrete_housing


def test_dcegm_model_builds_at_small_grid_without_solving():
    """The DS App.3 discrete-housing model builds at a tiny grid as a `Model`.

    Building the model runs the full regime-processing pipeline, including the
    DC-EGM contract validator. A returned `Model` instance means the model fits
    the DC-EGM structure (one continuous Euler state, one continuous action,
    resources/savings/inverse-marginal-utility declared) — no solve is attempted.
    """
    model = ds_app3_discrete_housing.build_model(
        variant="dcegm", n_assets=20, n_wage_nodes=3, n_periods=4
    )
    assert isinstance(model, Model)


def test_brute_model_builds_at_small_grid_without_solving():
    """The grid-search (VFI) twin of the DS App.3 model builds at a tiny grid.

    The brute variant solves the same economics by grid search over the
    state-action product — the Table 4 VFI method — with no Euler machinery. A
    returned `Model` means the discrete-housing structure is grid-search-solvable.
    """
    model = ds_app3_discrete_housing.build_model(
        variant="brute", n_assets=20, n_wage_nodes=3, n_periods=4
    )
    assert isinstance(model, Model)


def test_dcegm_regime_carries_the_dcegm_solver_on_assets():
    """The working regime is driven by DC-EGM with assets as the Euler state.

    The `Model` constructor runs the DC-EGM contract validator on the regime; a
    successful build is its acceptance. The Euler state is the financial asset
    `assets` and the continuous action is `consumption`.
    """
    model = ds_app3_discrete_housing.build_model(
        variant="dcegm", n_assets=20, n_wage_nodes=3, n_periods=4
    )
    solver = model.user_regimes["working"].solver
    assert isinstance(solver, DCEGM)
    assert solver.continuous_state == "assets"
    assert solver.continuous_action == "consumption"


@pytest.mark.parametrize("upper_envelope", ["fues", "mss", "ltm"])
def test_dcegm_accepts_each_table4_upper_envelope(
    upper_envelope: Literal["fues", "mss", "ltm"],
):
    """Each Table 4 upper-envelope backend builds the discrete-housing regime.

    Table 4 reports FUES, MSS, and LTM alongside VFI. All three are 1-D
    upper-envelope refinements of the same DC-EGM discrete-choice solve, so the
    model must build under each.
    """
    model = ds_app3_discrete_housing.build_model(
        variant="dcegm",
        n_assets=20,
        n_wage_nodes=3,
        n_periods=4,
        upper_envelope=upper_envelope,
    )
    solver = model.user_regimes["working"].solver
    assert isinstance(solver, DCEGM)
    assert solver.upper_envelope == upper_envelope


def test_brute_regime_uses_grid_search():
    """The brute twin's working regime is solved by grid search (VFI)."""
    model = ds_app3_discrete_housing.build_model(
        variant="brute", n_assets=20, n_wage_nodes=3, n_periods=4
    )
    solver = model.user_regimes["working"].solver
    assert isinstance(solver, GridSearch)


def test_housing_is_a_discrete_state_and_the_choice_a_discrete_action():
    """Housing is a discrete state; the own/rent choice is a discrete action.

    The crux of the App.3 mapping (Q6): the held housing stock `housing` is a
    discrete state (a value-function carry axis), and the own-vs-rent-and-level
    decision is a discrete action `housing_choice` over the same categories — the
    DC-EGM discrete-choice envelope, not a continuous durable margin. The
    next-period housing equals the chosen code, so the action drives the state.
    """
    model = ds_app3_discrete_housing.build_model(
        variant="dcegm", n_assets=20, n_wage_nodes=3, n_periods=4
    )
    working = model.user_regimes["working"]

    housing_grid = working.states["housing"]
    choice_grid = working.actions["housing_choice"]
    assert isinstance(housing_grid, DiscreteGrid)
    assert isinstance(choice_grid, DiscreteGrid)
    # The held stock and the choice share the same categorical alphabet so the
    # deterministic `next_housing = housing_choice` transition maps codes 1:1.
    housing_categories = set(housing_grid.categories)
    choice_categories = set(choice_grid.categories)
    assert housing_categories == choice_categories
    assert "rent" in housing_categories


def test_expected_states_actions_and_wage_process_present():
    """The model carries the working/dead regimes with their variables.

    - financial assets `assets` are the continuous Euler state of the working
      regime, and housing `housing` is its discrete state,
    - consumption and the discrete housing choice are the working actions,
    - the Markov wage `wage` is a process state of the working regime,
    - the terminal `dead` regime carries the assets and housing states the
      bequest reads.
    """
    model = ds_app3_discrete_housing.build_model(
        variant="dcegm", n_assets=20, n_wage_nodes=3, n_periods=4
    )
    assert set(model.regime_names_to_ids) == {"working", "dead"}

    working = model.user_regimes["working"]
    assert {"assets", "housing", "wage"} <= set(working.states)
    assert {"consumption", "housing_choice"} <= set(working.actions)

    dead = model.user_regimes["dead"]
    assert dead.transition is None
    assert {"assets", "housing"} <= set(dead.states)


def test_build_params_threads_the_adjustment_cost_and_is_taxless():
    """`build_params(tau=...)` threads the adjustment cost; the tax is zero.

    The proportional adjustment cost `tau` enters the housing-flow function, and
    the capital-income tax hook enters the resources function. The no-tax variant
    fixes the tax to zero, so the tax parameter is present and zero.
    """
    params_low = ds_app3_discrete_housing.build_params(tau=0.07)
    params_high = ds_app3_discrete_housing.build_params(tau=0.12)
    resources_fn = ds_app3_discrete_housing.RESOURCES_FUNCTION_NAME
    assert params_low["working"]["housing_flow"]["tau"] == 0.07
    assert params_high["working"]["housing_flow"]["tau"] == 0.12
    assert params_low["working"][resources_fn]["capital_income_tax"] == 0.0


def test_terminal_bequest_weight_is_threaded():
    """`build_params(theta=...)` places the bequest weight on the dead regime.

    Table 7 (Fella replication) uses `theta = 0.5`; the bequest weight enters the
    terminal `dead` regime's utility.
    """
    params = ds_app3_discrete_housing.build_params(theta=0.5)
    assert params["dead"]["utility"]["theta"] == 0.5


def test_brute_solve_at_tiny_grid_yields_a_finite_value_function():
    """The grid-search (VFI) twin solves to a finite value function.

    T = 20 is short, so a tiny brute solve (small asset grid, 3 wage nodes,
    4 periods) runs locally. The value function over the (housing, wage, assets)
    grid must be finite everywhere — the discrete-housing structure (a discrete
    housing state, a discrete own/rent choice driving its transition, and a
    Markov wage) is correctly assembled and VFI-solvable.
    """
    model = ds_app3_discrete_housing.build_model(
        variant="brute", n_assets=20, n_wage_nodes=3, n_periods=4, n_consumption=20
    )
    params = ds_app3_discrete_housing.build_params(variant="brute", n_periods=4)
    solution = model.solve(params=params, log_level="off")
    working_V = np.asarray(solution[0]["working"])
    assert np.all(np.isfinite(working_V))


@pytest.mark.skip(
    reason=(
        "DC-EGM kernel scope gap on feat/dcegm: the solve raises "
        "NotImplementedError because (1) resources reads the Markov wage process "
        "state and (2) the discrete housing state reaches the terminal regime via "
        "a non-identity transition (next_housing = housing_choice). The model "
        "builds and is accepted by the DC-EGM contract; the VFI twin solves. Run "
        "the DC-EGM solve once the kernel covers process-state resources and "
        "non-identity discrete-state terminal carries."
    )
)
def test_discrete_housing_model_solves_under_dcegm():
    """The DS App.3 discrete-housing model solves under DC-EGM to a finite V.

    Skipped: the DC-EGM kernel on `feat/dcegm` does not yet cover a regime whose
    resources reads a stochastic process state, nor a discrete state that reaches
    a terminal regime through a non-identity (choice-driven) transition — both of
    which App.3 needs. The model still *builds* under DC-EGM (the contract
    accepts it); only the solve hits the gap.
    """
    model = ds_app3_discrete_housing.build_model(variant="dcegm", n_assets=1000)
    params = ds_app3_discrete_housing.build_params(variant="dcegm", theta=0.5)
    solution = model.solve(params=params, log_level="off")
    assert np.all(np.isfinite(np.asarray(solution[0]["working"])))
