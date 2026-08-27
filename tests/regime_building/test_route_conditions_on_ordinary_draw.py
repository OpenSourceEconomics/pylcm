"""`route_gated_edges` conditions on the ORDINARY next-regime draw, not just
on `subjects_in_regime`.

Masking the `jnp.where` overrides in `route_gated_edges`
(`_lcm.simulation.gated_routing`) with `subjects_in_regime` alone would, for
EVERY subject in the source regime, force-override the row's
`new_subject_regime_ids` (the ordinary, already-drawn next regime —
`calculate_next_regime_membership`'s output) through a declared edge's gate,
regardless of what that ordinary draw actually selected. Two concrete failure
modes follow:

(a) A row whose ordinary draw selected some OTHER, edge-unrelated regime is
    still routed through the edge (to the target when its gate happens to be
    open at the row's candidate target state, or to the edge's OWN fallback
    when closed) -- discarding the row's real ordinary draw.
(b) With MULTIPLE gated edges declared on one source, each edge's iteration
    masks on the same `subjects_in_regime` and unconditionally overwrites
    `routed_ids` -- so a row whose ordinary draw picked edge B's target gets
    clobbered by edge A's (declaration-order-dependent) unconditional write,
    and vice versa.

No other collective-regimes test can see this, because EKL's own topology
declares the gated edge's target via an ordinary `MarkovTransition` at
PROBABILITY 1 (see module docstrings in `test_collective_regime_simulate.py`
/ `test_row_split_synthetic.py`: "offer arrives with certainty; only consent
is modeled") -- so `ordinary_draw == target_id` holds for every row the gate
could possibly touch, and the condition is vacuously true there.

`route_gated_edges` therefore threads an IMMUTABLE snapshot of the ordinary
draw (`ordinary_draw_ids`, captured before the edge loop) through each edge's
routing mask AND its fallback-state-write mask:
`subjects_in_regime & (ordinary_draw_ids == target_id)`. Only a row whose
ordinary draw actually selected THIS edge's target is routed -- and has its
fallback state written -- by this edge's gate; every other row keeps its
ordinary draw (and its other state slots) untouched. Using the immutable
snapshot (not the successively-updated `routed_ids`) makes multiple edges
order-independent.

These tests call `route_gated_edges` directly (kernel level, no `simulate()`
loop) so `new_subject_regime_ids` / `subjects_in_regime` / candidate
`next_states` can be hand-crafted to isolate the masking logic.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from _lcm.regime_building.collective import NO_ROLE
from _lcm.simulation.gated_routing import (
    route_gated_edges,
    substitute_gated_edge_continuations,
)
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import (
    DiscreteGrid,
    GatedEdge,
    LinSpacedGrid,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.regime_building.test_collective_regime_simulate import (
    _solve_and_process,
    _solve_consent,
)

_BETA = 0.95


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _same_period_mappings_for(
    regime, *, regimes, flat_params, solution, dissolution_flags
):
    """Fold exactly like `simulate()` does before `route_gated_edges`.

    Returns the per-target same-period value mapping (target V / `D` /
    reference-V arrays) `route_gated_edges` recomputes its gate from, rather
    than a boolean gate array.
    """
    base_state_action_spaces = {
        name: r.solution.state_action_space(regime_params=flat_params[name])
        for name, r in regimes.items()
    }
    _substituted, same_period_mappings = substitute_gated_edge_continuations(
        regime=regime,
        regime_name=next(n for n, r in regimes.items() if r is regime),
        regimes=regimes,
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        flat_params=flat_params,
    )
    return same_period_mappings


def test_unrelated_ordinary_draw_is_not_force_routed_through_an_open_gate():
    """A row whose ordinary draw picked an edge-unrelated regime stays there.

    The row drew `single_m_terminal` and must keep it even though the edge's
    gate, evaluated at the row's candidate `married_terminal` state, is OPEN.
    Masking only on `subjects_in_regime` (True for both rows here) would
    force-route BOTH rows to `married_terminal`; only the row whose ordinary
    draw actually IS `married_terminal` may be routed by this edge.

    Also proves the state-write conditioning: `single_f_terminal`'s (the
    edge's OWN fallback regime) state slot for the unrelated row must NOT be
    overwritten by this edge's fallback projector either.
    """
    _ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_consent()
    )
    single_f = regimes["single_f"]
    same_period_mappings = _same_period_mappings_for(
        single_f,
        regimes=regimes,
        flat_params=flat_params,
        solution=solution,
        dissolution_flags=dissolution_flags,
    )
    assert "married_terminal" in same_period_mappings

    target_id = regime_names_to_ids["married_terminal"]
    unrelated_id = regime_names_to_ids["single_m_terminal"]

    # Sentinel pre-existing state for the fallback regime -- row 1 (unrelated
    # draw) must keep it untouched; row 0 (ordinary draw == target) is not
    # this test's concern for the fallback slot (gate is open for it too).
    states = MappingProxyType(
        {
            "married_terminal": MappingProxyType({"wage": jnp.array([1.0, 1.0])}),
            "single_f_terminal": MappingProxyType(
                {"wage": jnp.array([-999.0, -999.0])}
            ),
        }
    )
    new_subject_regime_ids = jnp.array([target_id, unrelated_id], dtype=jnp.int32)
    subjects_in_regime = jnp.array([True, True])

    routed_states, routed_ids, _routed_roles = route_gated_edges(
        # The source is simulated at period 0, so the gate is decided on
        # the value it would enter at period 1.
        fold_period=1,
        regime=single_f,
        same_period_mappings=same_period_mappings,
        next_states=states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=new_subject_regime_ids,
        subjects_in_regime=subjects_in_regime,
        flat_params=flat_params,
        own_stakeholder=jnp.full_like(subjects_in_regime, NO_ROLE, dtype=jnp.int32),
        new_own_stakeholder=jnp.full_like(subjects_in_regime, NO_ROLE, dtype=jnp.int32),
    )

    # Row 0 (ordinary draw == target, gate open): routed to target, unchanged.
    # Row 1 (ordinary draw == unrelated regime, gate open): stays UNRELATED,
    # NOT force-routed to the target.
    np.testing.assert_array_equal(np.asarray(routed_ids), [target_id, unrelated_id])

    # Row 1's fallback-regime state slot must be untouched (sentinel intact);
    # this edge does not own row 1 at all.
    np.testing.assert_allclose(
        np.asarray(routed_states["single_f_terminal"]["wage"])[1], -999.0
    )


def test_ordinary_draw_is_target_routes_exactly_as_before_open_and_closed():
    """A row whose ordinary draw IS the edge's target routes on the gate alone.

    Control case with a probability-1 offer: open gate -> target, closed gate ->
    own fallback. This is the shape every other collective-simulate test
    exercises.
    """
    _ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_consent()
    )
    single_f = regimes["single_f"]
    same_period_mappings = _same_period_mappings_for(
        single_f,
        regimes=regimes,
        flat_params=flat_params,
        solution=solution,
        dissolution_flags=dissolution_flags,
    )
    target_id = regime_names_to_ids["married_terminal"]
    fallback_id = regime_names_to_ids["single_f_terminal"]

    # wage=1 -> gate open; wage=2 -> gate closed (consent fixture, see
    # test_consent_routing_simulate_matches_gate_exactly). `single_f_terminal`
    # (the fallback regime) needs a pre-existing state slot for
    # `_advance_states_for_subjects` to merge into.
    next_states = MappingProxyType(
        {
            "married_terminal": MappingProxyType({"wage": jnp.array([1.0, 2.0])}),
            "single_f_terminal": MappingProxyType({"wage": jnp.array([0.0, 0.0])}),
        }
    )
    new_subject_regime_ids = jnp.array([target_id, target_id], dtype=jnp.int32)
    subjects_in_regime = jnp.array([True, True])

    _states, routed_ids, _routed_roles = route_gated_edges(
        # The source is simulated at period 0, so the gate is decided on
        # the value it would enter at period 1.
        fold_period=1,
        regime=single_f,
        same_period_mappings=same_period_mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=new_subject_regime_ids,
        subjects_in_regime=subjects_in_regime,
        flat_params=flat_params,
        own_stakeholder=jnp.full_like(subjects_in_regime, NO_ROLE, dtype=jnp.int32),
        new_own_stakeholder=jnp.full_like(subjects_in_regime, NO_ROLE, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(np.asarray(routed_ids), [target_id, fallback_id])


# Multiple gated edges from one source -- order independence. A row drawn to
# edge B's target must not be clobbered by an unconditional overwrite from
# edge A, regardless of declaration order.

_WAGE_2 = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)


def _prob_half(age: FloatND) -> FloatND:
    return 0.5 * jnp.ones_like(age, dtype=float)


def _gate_always_open(wage: ContinuousState) -> BoolND:
    return jnp.ones_like(wage, dtype=bool)


def _u_src(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage * work


def _u_target_a(wage: ContinuousState) -> FloatND:
    return wage + 10.0


def _u_target_b(wage: ContinuousState) -> FloatND:
    return wage + 20.0


def _u_fallback_a(wage: ContinuousState) -> FloatND:
    return wage + 1.0


def _u_fallback_b(wage: ContinuousState) -> FloatND:
    return wage + 2.0


def _make_dual_edge_regimes(*, edge_order: tuple[str, str]) -> dict[str, Regime]:
    """A singleton source with TWO gated edges (targets `a`/`b`), each with its
    own fallback. `edge_order` controls the `gated_edges` declaration order
    -- both orderings must yield the identical, order-independent routing.
    """
    edges = {
        "target_a": GatedEdge(
            gate=_gate_always_open,
            legs={
                "only": StakeholderRoute(
                    fallback=ProjectedRegimeValue(
                        regime="fallback_a", projection={"wage": _identity_wage}
                    )
                )
            },
        ),
        "target_b": GatedEdge(
            gate=_gate_always_open,
            legs={
                "only": StakeholderRoute(
                    fallback=ProjectedRegimeValue(
                        regime="fallback_b", projection={"wage": _identity_wage}
                    )
                )
            },
        ),
    }
    src = Regime(
        transition={
            "target_a": MarkovTransition(_prob_half),
            "target_b": MarkovTransition(_prob_half),
        },
        active=lambda age: age < 1,
        states={"wage": _WAGE_2},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={name: edges[name] for name in edge_order},
    )
    target_a = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_2},
        functions={"utility": _u_target_a},
    )
    target_b = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_2},
        functions={"utility": _u_target_b},
    )
    fallback_a = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_2},
        functions={"utility": _u_fallback_a},
    )
    fallback_b = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_2},
        functions={"utility": _u_fallback_b},
    )
    return {
        "src": src,
        "target_a": target_a,
        "target_b": target_b,
        "fallback_a": fallback_a,
        "fallback_b": fallback_b,
    }


def _solve_dual_edge(*, edge_order: tuple[str, str]):
    ages = AgeGrid(start=0, stop=2, step="Y")
    regime_names = ["src", "target_a", "target_b", "fallback_a", "fallback_b"]
    regimes_dict = _make_dual_edge_regimes(edge_order=edge_order)
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=regime_names
    )
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "target_a": MappingProxyType({}),
            "target_b": MappingProxyType({}),
            "fallback_a": MappingProxyType({}),
            "fallback_b": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    dissolution_flags = _bi_result.dissolution_flags
    return regimes, regime_names_to_ids, flat_params, solution, dissolution_flags


def _route_dual_edge(*, edge_order: tuple[str, str]):
    regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_dual_edge(edge_order=edge_order)
    )
    src = regimes["src"]
    same_period_mappings = _same_period_mappings_for(
        src,
        regimes=regimes,
        flat_params=flat_params,
        solution=solution,
        dissolution_flags=dissolution_flags,
    )
    assert set(same_period_mappings) == {"target_a", "target_b"}

    target_a_id = regime_names_to_ids["target_a"]
    target_b_id = regime_names_to_ids["target_b"]

    # Row 0's ordinary draw picked target_a; row 1's picked target_b. Both
    # candidate states (for BOTH targets) are wage=1.0; the gate is
    # unconditionally open, so this isolates the routing MASK, not the gate.
    next_states = MappingProxyType(
        {
            "target_a": MappingProxyType({"wage": jnp.array([1.0, 1.0])}),
            "target_b": MappingProxyType({"wage": jnp.array([1.0, 1.0])}),
            "fallback_a": MappingProxyType({"wage": jnp.array([0.0, 0.0])}),
            "fallback_b": MappingProxyType({"wage": jnp.array([0.0, 0.0])}),
        }
    )
    new_subject_regime_ids = jnp.array([target_a_id, target_b_id], dtype=jnp.int32)
    subjects_in_regime = jnp.array([True, True])

    _states, routed_ids, _routed_roles = route_gated_edges(
        # The source is simulated at period 0, so the gate is decided on
        # the value it would enter at period 1.
        fold_period=1,
        regime=src,
        same_period_mappings=same_period_mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=new_subject_regime_ids,
        subjects_in_regime=subjects_in_regime,
        flat_params=flat_params,
        own_stakeholder=jnp.full_like(subjects_in_regime, NO_ROLE, dtype=jnp.int32),
        new_own_stakeholder=jnp.full_like(subjects_in_regime, NO_ROLE, dtype=jnp.int32),
    )
    return np.asarray(routed_ids), target_a_id, target_b_id


def test_multiple_gated_edges_route_order_independently():
    """Two gated edges from one source route independently of declaration order.

    Declared as (target_a, target_b), an unconditional overwrite from edge_a
    (processed first) would force EVERY subject to target_a, then edge_b's would
    force EVERY subject to target_b -- BOTH rows ending at target_b, the LAST
    declared edge winning and clobbering row 0's real target_a draw.

    Instead row 0 (drew target_a) stays at target_a, edge_b's mask
    (`ordinary_draw == target_b_id`) excluding it, and row 1 (drew target_b)
    ends at target_b. Declaring the edges in the OPPOSITE order gives the
    identical result.
    """
    routed_ids_ab, target_a_id, target_b_id = _route_dual_edge(
        edge_order=("target_a", "target_b")
    )
    np.testing.assert_array_equal(routed_ids_ab, [target_a_id, target_b_id])

    routed_ids_ba, target_a_id2, target_b_id2 = _route_dual_edge(
        edge_order=("target_b", "target_a")
    )
    assert target_a_id2 == target_a_id
    assert target_b_id2 == target_b_id
    np.testing.assert_array_equal(routed_ids_ba, [target_a_id, target_b_id])
