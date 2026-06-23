"""Construction spec for the DS-2026 Application 2 housing NEGM model.

The Dobrescu-Shanker (2026) §2.2 housing model maps onto pylcm's nested-EGM
solver: a liquid consumption-savings margin the Euler equation inverts on
(the inner DC-EGM), plus an illiquid housing margin with a proportional
transaction cost the inner Euler cannot invert (the outer durable search). The
discrete adjust/keep choice `d ∈ {0, 1}` is the NEGM dual-core contract — the
solver builds a keeper kernel (housing held, `H' = H`) and an adjuster kernel
(housing swept over the outer grid, paying `(1 + τ)·H'`) from one regime, so it
is *not* a user-declared discrete action.

These checks build the model at a tiny grid and assert structure only — no
`.solve()` runs here. The 2-D-continuous + AR1-process NEGM solve OOMs a small
box, so the solve/accuracy/timing sweep is a separate gpu-01 job. The ground
truth for the NEGM mapping is `validate_negm_regimes`, which the `Model`
constructor runs at build: if the housing margin were Euler-coupled in a way
NEGM forbids, building the model would raise `ModelInitializationError`. A model
that builds therefore *is* an accepted NEGM model.
"""

import pytest

from lcm import NEGM, Model
from lcm.exceptions import ModelInitializationError
from lcm.solvers import DCEGM
from tests.test_models import ds_app2_housing


def test_model_builds_at_small_grid_without_solving():
    """The DS App.2 housing model builds at a tiny grid as a valid `Model`.

    Building the model runs the full regime-processing pipeline, including the
    NEGM contract validator. A returned `Model` instance means the housing
    margin fits the NEGM nesting contract — no solve is attempted.
    """
    model = ds_app2_housing.build_model(n_grid=5, n_periods=4)
    assert isinstance(model, Model)


def test_negm_contract_accepts_the_housing_regimes():
    """`validate_negm_regimes` accepts the working and retired NEGM regimes.

    The `Model` constructor runs `validate_negm_regimes` on the user regimes; a
    successful build is the validator's acceptance. The working and retired
    regimes both carry an `NEGM` solver whose inner is a `DCEGM` on liquid
    assets, so both are checked.
    """
    model = ds_app2_housing.build_model(n_grid=5, n_periods=4)
    working_solver = model.user_regimes["working"].solver
    retired_solver = model.user_regimes["retired"].solver
    assert isinstance(working_solver, NEGM)
    assert isinstance(retired_solver, NEGM)
    assert isinstance(working_solver.inner, DCEGM)
    assert working_solver.inner.continuous_state == "liquid"
    assert working_solver.outer_action == "housing_investment"


def test_expected_regimes_states_and_actions_are_present():
    """The model carries the working/retired/dead regimes with their variables.

    - liquid assets `liquid` and housing `housing` are the two continuous
      states of both non-terminal regimes,
    - the AR1 wage `wage` is a process state of the working regime (it drops at
      retirement),
    - consumption and the housing-investment outer action are the continuous
      actions of the working regime,
    - the terminal `dead` regime carries the two continuous states the bequest
      reads.
    """
    model = ds_app2_housing.build_model(n_grid=5, n_periods=4)
    assert set(model.regime_names_to_ids) == {"working", "retired", "dead"}

    working = model.user_regimes["working"]
    assert {"liquid", "housing", "wage"} <= set(working.states)
    assert {"consumption", "housing_investment"} <= set(working.actions)

    retired = model.user_regimes["retired"]
    assert {"liquid", "housing"} <= set(retired.states)
    assert "wage" not in retired.states

    dead = model.user_regimes["dead"]
    assert dead.transition is None
    assert {"liquid", "housing"} <= set(dead.states)


def test_keeper_and_adjuster_are_solver_internal_not_user_actions():
    """The adjust/keep choice is the NEGM dual core, not a discrete action.

    NEGM builds the keeper (`H' = H`) and adjuster (`H'` chosen, paying the
    transaction cost) from a single regime, so the model declares no `adjust`
    discrete action — the dual core lives inside the solver.
    """
    model = ds_app2_housing.build_model(n_grid=5, n_periods=4)
    working = model.user_regimes["working"]
    assert "adjust" not in working.actions
    solver = working.solver
    assert isinstance(solver, NEGM)
    assert solver.outer_no_adjustment_candidate is not None


def test_build_params_threads_the_transaction_cost():
    """`build_params(tau=...)` places the transaction cost on the housing law.

    The proportional transaction cost `τ` is the durable-margin friction; it
    enters the housing-cost function, so the params template carries it where
    that function reads it.
    """
    params_low = ds_app2_housing.build_params(tau=0.05)
    params_high = ds_app2_housing.build_params(tau=0.12)
    cost_fn = ds_app2_housing.HOUSING_COST_FUNCTION_NAME
    assert params_low["working"][cost_fn]["tau"] == 0.05
    assert params_high["working"][cost_fn]["tau"] == 0.12


@pytest.mark.skip(reason="gpu-01 only: 2-D+AR1 NEGM solve OOMs locally")
def test_housing_model_solves_on_gpu():
    """The DS App.2 housing model solves to a finite value function.

    Marked GPU-only: the 2-D-continuous + AR1-process NEGM solve exhausts a
    small box. Run on gpu-01.
    """
    model = ds_app2_housing.build_model(n_grid=250)
    params = ds_app2_housing.build_params(tau=0.05)
    solution = model.solve(params=params, log_level="off")
    assert solution[0]["working"] is not None


def test_euler_coupled_housing_law_would_be_rejected():
    """An Euler-coupled housing law trips the NEGM contract, as a guardrail.

    If the outer housing post-decision fed the inner liquid Euler-state
    transition, the inner Euler inversion would depend on the housing choice —
    the DS pension coupling NEGM forbids. The validator must reject such a
    variant with the 2-D-EGM pointer, confirming the accepted model is not
    accepted by accident.
    """
    with pytest.raises(ModelInitializationError, match="2-D EGM foundation"):
        ds_app2_housing.build_model(n_grid=5, n_periods=4, _euler_couple_housing=True)
