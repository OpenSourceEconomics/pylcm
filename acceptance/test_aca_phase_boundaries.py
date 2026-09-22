"""Exercise ACA age and schema boundaries using frozen production factory inputs.

These are structural helper tests, not a reduced or full production solve.
Select this module explicitly with the pinned ACA source overlays and set
ACA_PHASE_PACKET_ROOT to the original packet containing manifest.json and inputs/.
"""

# This separately selected integration probe asserts literal institutional boundaries.
# ruff: noqa: INP001, PLR2004, S101

import hashlib
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from aca_model.agent.labor_market import LaborSupply, LaggedLaborSupply
from aca_model.agent.preferences import PrefType
from aca_model.baseline.regimes._common import (
    REGIME_SPECS,
    Grids,
    RegimeId,
    build_actions,
    build_dead_regime,
    build_grids,
    build_state_transitions,
    build_states,
    make_active_func,
    make_targets,
    select_target_for_age,
)
from aca_model.environment.social_security import ClaimedSS
from aca_slurm._simulate import _assemble, _load_inputs
from aca_slurm.config import A40_GRID_CONFIG

from lcm import DiscreteGrid, MarkovTransition


@pytest.fixture(scope="module")
def production_grids() -> Grids:
    """Build production grids from all eleven identity-checked factory inputs."""
    packet = Path(os.environ["ACA_PHASE_PACKET_ROOT"])
    manifest = json.loads((packet / "manifest.json").read_text())
    hashes = {
        name: hashlib.sha256((packet / "inputs" / name).read_bytes()).hexdigest()
        for name in manifest["inputs"]
    }
    assert len(hashes) == 11
    assert hashes == manifest["inputs"]
    files = {
        "ss": "social_security_params",
        "tax": "tax_params",
        "ssi": "ssi_medicaid_params",
        "hi": "health_insurance_params",
        "pension": "pension_params",
        "wage": "wage_params",
        "transition": "transition_probs",
        "pref": "preference_start_values",
        "env": "environment_constants",
        "hcc_insurer": "hcc_insurer_params",
        "initdist": "initial_conditions",
    }
    inputs = _load_inputs(
        **{
            f"{key}_path": packet / "inputs" / f"{name}.pkl"
            for key, name in files.items()
        }
    )
    fixed, _ = _assemble(inputs=inputs, grid_config=A40_GRID_CONFIG)
    grids = build_grids(
        grid_config=A40_GRID_CONFIG,
        fixed_params=fixed,
        wage_params=inputs.wage,
        pref_type_grid=DiscreteGrid(PrefType),
        consumption_dollars_points=None,
    )
    assert grids.assets.to_jax().shape == (24,)
    assert grids.aime.to_jax().shape == (38,)
    return grids


@pytest.mark.parametrize(
    ("source_age", "source", "target"),
    [
        (61, "retiree_nomc_inelig_canwork", "retiree_nomc_choose_canwork"),
        (64, "retiree_nomc_choose_canwork", "retiree_oamc_choose_canwork"),
        (69, "retiree_oamc_choose_canwork", "retiree_oamc_forced_canwork"),
        (71, "retiree_oamc_forced_canwork", "retiree_oamc_forced_forcedout"),
    ],
)
def test_age_boundary_changes_active_regime_and_selects_target(
    *,
    source_age: int,
    source: str,
    target: str,
) -> None:
    """Each institutional age boundary selects the newly active target regime."""
    source_active = make_active_func(REGIME_SPECS[source])
    target_active = make_active_func(REGIME_SPECS[target])
    assert source_active(source_age)
    assert not source_active(source_age + 1)
    assert not target_active(source_age)
    assert target_active(source_age + 1)
    own, _ = make_targets(source)
    assert int(
        select_target_for_age(
            next_age=source_age + 1,
            mc_next=False,
            tgts=own,
        )
    ) == int(getattr(RegimeId, target))


def test_claimed_status_enters_unclaimed_and_disappears_at_forced_claim(
    production_grids: Grids,
) -> None:
    """Claim status enters unclaimed at 62 and ceases to be a state at 70."""
    before = "retiree_nomc_inelig_canwork"
    choose = "retiree_nomc_choose_canwork"
    forced = "retiree_oamc_forced_canwork"
    states_before = build_states(spec=REGIME_SPECS[before], grids=production_grids)
    states_choose = build_states(spec=REGIME_SPECS[choose], grids=production_grids)
    states_forced = build_states(spec=REGIME_SPECS[forced], grids=production_grids)
    assert "claimed_ss" not in states_before
    np.testing.assert_array_equal(states_choose["claimed_ss"].to_jax(), [0, 1])
    assert "claimed_ss" not in states_forced
    assert "claim_ss" in build_actions(
        spec=REGIME_SPECS[choose], grids=production_grids
    )
    assert "claim_ss" not in build_actions(
        spec=REGIME_SPECS[forced], grids=production_grids
    )
    entry = build_state_transitions(REGIME_SPECS[before])["claimed_ss"][choose]
    assert int(entry()) == int(ClaimedSS.no)
    laws = build_state_transitions(REGIME_SPECS[choose])["claimed_ss"]
    assert forced not in laws
    absorbing = laws[choose]
    assert int(
        absorbing(claim_ss=jnp.int32(ClaimedSS.no), claimed_ss=jnp.int32(ClaimedSS.yes))
    ) == int(ClaimedSS.yes)


def test_health_transition_changes_from_three_source_states_to_two_targets(
    production_grids: Grids,
) -> None:
    """Age-65 health transitions retain all source rows and two target categories."""
    before = "retiree_nomc_choose_canwork"
    after = "retiree_oamc_choose_canwork"
    source = build_states(spec=REGIME_SPECS[before], grids=production_grids)
    target = build_states(spec=REGIME_SPECS[after], grids=production_grids)
    np.testing.assert_array_equal(source["health"].to_jax(), [0, 1, 2])
    np.testing.assert_array_equal(target["health"].to_jax(), [0, 1])
    law = build_state_transitions(REGIME_SPECS[before])["health"][after]
    assert isinstance(law, MarkovTransition)
    probabilities = jnp.array([[[0.25, 0.75], [0.5, 0.5], [0.875, 0.125]]])
    for health in range(3):
        np.testing.assert_array_equal(
            law(
                health=jnp.int32(health),
                period=jnp.int32(0),
                health_trans_probs_cross=probabilities,
            ),
            np.asarray(probabilities)[0, health],
        )


def test_leaving_tied_insurance_enters_lagged_labor_from_current_action(
    production_grids: Grids,
) -> None:
    """Tied-to-nongroup entry initializes labor history from the chosen hours."""
    tied = "tied_nomc_choose_canwork"
    nongroup = "nongroup_nomc_choose_canwork"
    source = build_states(spec=REGIME_SPECS[tied], grids=production_grids)
    target = build_states(spec=REGIME_SPECS[nongroup], grids=production_grids)
    assert "lagged_labor_supply" not in source
    np.testing.assert_array_equal(target["lagged_labor_supply"].to_jax(), [0, 1])
    _, nongroup_targets = make_targets(tied)
    assert int(
        select_target_for_age(next_age=64, mc_next=False, tgts=nongroup_targets)
    ) == int(getattr(RegimeId, nongroup))
    law = build_state_transitions(REGIME_SPECS[tied])["lagged_labor_supply"][nongroup]
    np.testing.assert_array_equal(
        law(
            labor_supply=jnp.array(
                [LaborSupply.do_not_work, LaborSupply.h1000, LaborSupply.h2500],
                dtype=jnp.int32,
            )
        ),
        [
            LaggedLaborSupply.did_not_work,
            LaggedLaborSupply.worked,
            LaggedLaborSupply.worked,
        ],
    )
    forcedout = "nongroup_oamc_forced_forcedout"
    final_states = build_states(spec=REGIME_SPECS[forcedout], grids=production_grids)
    assert "lagged_labor_supply" not in final_states
    assert "log_ft_wage_res" not in final_states
    assert "labor_supply" not in build_actions(
        spec=REGIME_SPECS[forcedout], grids=production_grids
    )


def test_last_living_age_enters_dead_and_terminal_has_no_carried_pension() -> None:
    """The final age routes into an absorbing terminal regime without pension carry."""
    living = "retiree_oamc_forced_forcedout"
    active = make_active_func(REGIME_SPECS[living])
    assert active(94)
    assert not active(95)
    own, _ = make_targets(living)
    assert int(select_target_for_age(next_age=95, mc_next=False, tgts=own)) == int(
        RegimeId.dead
    )
    assert int(select_target_for_age(next_age=96, mc_next=False, tgts=own)) == int(
        RegimeId.dead
    )
    dead = build_dead_regime()
    assert dead.transition is None
    assert dead.states["pension_wealth"] is None
    assert not dead.actions
    assert dead.active(95)
