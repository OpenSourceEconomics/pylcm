"""Unit tests for the collective-regime value readout.

The readout is the mathematical heart of the "collective regimes" extension: given
per-stakeholder action-value arrays and a feasibility mask, choose the action that
maximizes the household scalarization Σ_s λ_s Q^s over the feasible set, then read
off each stakeholder's OWN Q at that common argmax (eqs. 10-12 of Eckstein-Keane-
Lifshitz 2019). See `pylcm-extension-collective-regimes.md` §2.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.collective import (
    _weighted_sum,
    collective_readout,
)


def test_readout_reads_own_Q_at_household_argmax_single_state():
    # Two stakeholders, three actions, one state cell. Household weights 1/2 each.
    # Q^f favours action 0; Q^m favours action 2; the joint objective picks the
    # action maximizing 1/2 (Q^f + Q^m).
    q_f = jnp.array([10.0, 4.0, 0.0])
    q_m = jnp.array([0.0, 3.0, 9.0])
    #   O = [5.0, 3.5, 4.5]  -> argmax = action 0
    feas = jnp.array([True, True, True])

    values, dissolution = collective_readout(
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
    assert bool(dissolution) is False


def test_feasibility_mask_changes_the_joint_choice():
    # Same Q as above but action 0 is infeasible -> objective over {1,2} = [3.5, 4.5]
    # -> a* = action 2 -> V^f = 0.0, V^m = 9.0.
    q_f = jnp.array([10.0, 4.0, 0.0])
    q_m = jnp.array([0.0, 3.0, 9.0])
    feas = jnp.array([False, True, True])

    values, dissolution = collective_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feas,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(0,),
    )
    assert values["f"] == pytest.approx(0.0)
    assert values["m"] == pytest.approx(9.0)
    assert bool(dissolution) is False


def test_empty_feasible_set_sets_dissolution_flag():
    q_f = jnp.array([10.0, 4.0, 0.0])
    q_m = jnp.array([0.0, 3.0, 9.0])
    feas = jnp.array([False, False, False])

    _, dissolution = collective_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feas,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(0,),
    )
    # No feasible action -> the empty-set / dissolution marker, distinct from -inf.
    assert bool(dissolution) is True


def test_batched_over_states():
    # Two state cells x three actions. Distinct argmax per cell.
    # cell 0: O = 1/2([1,0,0]+[0,0,3]) = [0.5,0,1.5] -> a*=2 -> Vf=0, Vm=3
    # cell 1: O = 1/2([5,0,0]+[0,0,1]) = [2.5,0,0.5] -> a*=0 -> Vf=5, Vm=0
    q_f = jnp.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    q_m = jnp.array([[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]])
    feas = jnp.ones((2, 3), dtype=bool)

    values, dissolution = collective_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feas,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(1,),
    )
    np.testing.assert_allclose(np.asarray(values["f"]), [0.0, 5.0])
    np.testing.assert_allclose(np.asarray(values["m"]), [3.0, 0.0])
    np.testing.assert_array_equal(np.asarray(dissolution), [False, False])


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


def test_terminal_e1_end_to_end_with_real_utilities():
    """Terminal E1 milestone: real per-stakeholder utilities -> correct V^s.

    Mirrors exactly what the terminal collective kernel will do (slice 1b): build one
    Q^s per stakeholder by evaluating its utility over the action product for each
    state, then hand the stacked Q^s to `collective_readout`. Here the "action" is a
    binary work choice and the "state" is a wage; the wife weights leisure, the
    husband weights household consumption. No engine mutation — proves the numerics.
    """
    # Two states (wages), one binary action a in {0=leisure, 1=work}.
    wage = jnp.array([10.0, 40.0])  # state grid
    work = jnp.array([0.0, 1.0])  # action grid

    # Consumption (public good): earnings from whoever works. Wife's utility values
    # her own leisure highly; husband's values consumption. Both see the same C.
    def consumption(w, a):  # (state, action) -> C
        return w * a

    def u_wife(w, a):
        return consumption(w, a) + 30.0 * (1.0 - a)  # strong leisure taste

    def u_husband(w, a):
        return 2.0 * consumption(w, a)  # values consumption, indifferent to leisure

    # Evaluate each stakeholder's utility over the action product, per state:
    # shape (n_states, n_actions).
    q_f = jax.vmap(lambda w: jax.vmap(lambda a: u_wife(w, a))(work))(wage)
    q_m = jax.vmap(lambda w: jax.vmap(lambda a: u_husband(w, a))(work))(wage)
    feas = jnp.ones((2, 2), dtype=bool)

    values, dissolution = collective_readout(
        stakeholder_Q={"f": q_f, "m": q_m},
        feasibility=feas,
        weights={"f": 0.5, "m": 0.5},
        action_axes=(1,),
    )

    # Hand-check. O = 1/2 (u_f + u_m).
    # wage=10: a=0 -> u_f=30, u_m=0,  O=15;  a=1 -> u_f=10, u_m=20, O=15 -> tie -> a=0.
    #          a*=0 -> V_f=30, V_m=0.
    # wage=40: a=0 -> u_f=30, u_m=0,  O=15;  a=1 -> u_f=40, u_m=80, O=60 -> a=1.
    #          a*=1 -> V_f=40, V_m=80.
    np.testing.assert_allclose(np.asarray(values["f"]), [30.0, 40.0])
    np.testing.assert_allclose(np.asarray(values["m"]), [0.0, 80.0])
    np.testing.assert_array_equal(np.asarray(dissolution), [False, False])
    # The household trades off: at the low wage it keeps the wife home (her leisure
    # taste dominates the joint objective); at the high wage her work wins jointly.


def test_mismatched_keys_raise():
    with pytest.raises(ValueError, match="identical keys"):
        collective_readout(
            stakeholder_Q={"f": jnp.array([1.0]), "m": jnp.array([1.0])},
            feasibility=jnp.array([True]),
            weights={"f": 0.5},
            action_axes=(0,),
        )


def test_multi_party_household_choice_is_invariant_to_stakeholder_order():
    """A relabeling of three stakeholders cannot select a summation order.

    The exact objectives are ``[1, 1/2]``. A declaration-order left fold yields
    either ``[1, 1/2]`` or ``[0, 1/2]`` because the first action cancels two large
    terms. The public API accepts multi-party households, so all six permutations
    must retain the exact-side action and one objective bit pattern.
    """
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    large = jnp.asarray(2.0 ** (53 if jax.config.jax_enable_x64 else 24), dtype=dtype)
    stakeholder_q = {
        "a": jnp.asarray([2.0 * large, 1.0], dtype=dtype),
        "b": jnp.asarray([-4.0 * large, 0.0], dtype=dtype),
        "c": jnp.asarray([4.0, 0.0], dtype=dtype),
    }
    weights = {"a": 0.5, "b": 0.25, "c": 0.25}
    policies = set()
    patterns = set()

    for order in itertools.permutations(stakeholder_q):
        q = {name: stakeholder_q[name] for name in order}
        w = {name: weights[name] for name in order}
        # `order` and `w` are bound as defaults rather than captured: the lambda is
        # rebuilt each iteration, and a late-bound closure would silently jit the
        # last permutation for every one of them.
        objective = jax.jit(
            lambda *arrays, _order=order, _w=w: _weighted_sum(
                stakeholder_Q=dict(zip(_order, arrays, strict=True)),
                weights=_w,
            )
        )(*(q[name] for name in order))
        policies.add(int(jnp.argmax(objective)))
        patterns.add(np.asarray(objective).tobytes())

    assert policies == {0}
    assert len(patterns) == 1
