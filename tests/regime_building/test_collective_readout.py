"""Unit tests for the collective-regime (E1) value readout.

The readout is the mathematical heart of the "collective regimes" extension: given
per-stakeholder action-value arrays and a feasibility mask, choose the action that
maximizes the household scalarization Σ_s λ_s Q^s over the feasible set, then read
off each stakeholder's OWN Q at that common argmax (eqs. 10-12 of Eckstein-Keane-
Lifshitz 2019). See `pylcm-extension-collective-regimes.md` §2 (E1).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.collective import collective_readout


def test_readout_reads_own_Q_at_household_argmax_single_state():
    # Two stakeholders, three actions, one state cell. Household weights 1/2 each.
    # Q^f favours action 0; Q^m favours action 2; the joint objective picks the
    # action maximizing 1/2 (Q^f + Q^m).
    q_f = jnp.array([10.0, 4.0, 0.0])
    q_m = jnp.array([0.0, 3.0, 9.0])
    #   O = [5.0, 3.5, 4.5]  -> argmax = action 0
    feas = jnp.array([True, True, True])

    values, divorce = collective_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feas,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(0,),
    )

    # At a* = 0, each stakeholder reads its OWN Q, not the scalarization.
    assert values["f"] == pytest.approx(10.0)
    assert values["m"] == pytest.approx(0.0)
    # The two values differ -> genuinely per-stakeholder (not the shared objective).
    assert values["f"] != values["m"]
    assert bool(divorce) is False


def test_feasibility_mask_changes_the_joint_choice():
    # Same Q as above but action 0 is infeasible -> objective over {1,2} = [3.5, 4.5]
    # -> a* = action 2 -> V^f = 0.0, V^m = 9.0.
    q_f = jnp.array([10.0, 4.0, 0.0])
    q_m = jnp.array([0.0, 3.0, 9.0])
    feas = jnp.array([False, True, True])

    values, divorce = collective_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feas,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(0,),
    )
    assert values["f"] == pytest.approx(0.0)
    assert values["m"] == pytest.approx(9.0)
    assert bool(divorce) is False


def test_empty_feasible_set_sets_divorce_flag():
    q_f = jnp.array([10.0, 4.0, 0.0])
    q_m = jnp.array([0.0, 3.0, 9.0])
    feas = jnp.array([False, False, False])

    _, divorce = collective_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feas,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(0,),
    )
    # No feasible action -> the empty-set / divorce marker, distinct from -inf value.
    assert bool(divorce) is True


def test_batched_over_states():
    # Two state cells x three actions. Distinct argmax per cell.
    # cell 0: O = 1/2([1,0,0]+[0,0,3]) = [0.5,0,1.5] -> a*=2 -> Vf=0, Vm=3
    # cell 1: O = 1/2([5,0,0]+[0,0,1]) = [2.5,0,0.5] -> a*=0 -> Vf=5, Vm=0
    q_f = jnp.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    q_m = jnp.array([[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]])
    feas = jnp.ones((2, 3), dtype=bool)

    values, divorce = collective_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feas,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(1,),
    )
    np.testing.assert_allclose(np.asarray(values["f"]), [0.0, 5.0])
    np.testing.assert_allclose(np.asarray(values["m"]), [3.0, 0.0])
    np.testing.assert_array_equal(np.asarray(divorce), [False, False])


def test_unequal_weights_shift_the_choice():
    # With weight 0.9 on m, the objective tilts to m's preferred action.
    q_f = jnp.array([10.0, 0.0])
    q_m = jnp.array([0.0, 10.0])
    feas = jnp.array([True, True])
    #   O = [0.1*10, 0.9*10] = [1.0, 9.0] -> a*=1 -> Vf=0, Vm=10
    values, _ = collective_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feas,
        weights={"f": 0.1, "m": 0.9},
        action_axes=(0,),
    )
    assert values["f"] == pytest.approx(0.0)
    assert values["m"] == pytest.approx(10.0)


def test_two_action_axes_are_flattened_consistently():
    # Actions on a 2x2 grid; verify the gather uses the same flatten order as argmax.
    # O = 1/2(q_f+q_m); pick the joint (i,j) maximizing O, read each Q there.
    q_f = jnp.array([[1.0, 2.0], [3.0, 0.0]])
    q_m = jnp.array([[0.0, 0.0], [1.0, 8.0]])
    #   O = [[0.5,1.0],[2.0,4.0]] -> argmax at (1,1) -> Vf=0.0, Vm=8.0
    feas = jnp.ones((2, 2), dtype=bool)
    values, _ = collective_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feas,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(0, 1),
    )
    assert values["f"] == pytest.approx(0.0)
    assert values["m"] == pytest.approx(8.0)


def test_mismatched_keys_raise():
    with pytest.raises(ValueError, match="identical keys"):
        collective_readout(
            stakeholder_Q={"f": jnp.array([1.0]), "m": jnp.array([1.0])},
            feasibility=jnp.array([True]),
            weights={"f": 0.5},
            action_axes=(0,),
        )
