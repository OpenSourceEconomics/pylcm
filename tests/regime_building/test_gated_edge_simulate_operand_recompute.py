"""Repro/regression: simulate F1 -- recompute the gate from VALUE OPERANDS,
never by interpolating the solve-side fold's baked BOOLEAN gate array.

Pre-fix, `route_gated_edges` (`_lcm.simulation.gated_routing`) decided
routing by interpolating the fold's grid-level boolean `gate` array (cast to
float) at the subject's realized candidate target state, then thresholding
the interpolated float at 0.5. That does not commute with the gate's own
(nonlinear) predicate: "interpolate the {0,1}-valued grid, then threshold"
and "interpolate the VALUE operands, then apply the predicate" agree at
every grid node by construction, but can disagree at any off-grid point
whenever the value operands' difference does not cross zero at the same
point the boolean ramp crosses 0.5.

Mirrors the reviewer's counterexample: a target with a 2-point grid
`x in {0.0, 1.0}`, a consent-style gate `V_target(x) > V_ref(x)` whose
boolean value at the two nodes is `[False, True]` (so the OLD boolean-grid
interpolation is, independent of the actual values, a linear ramp from 0 to
1 across `[0, 1]`, thresholding to "open iff `x > 0.5`" for ANY such
fixture) -- but whose VALUE operands are engineered so the true zero-crossing
of `V_target(x) - V_ref(x)` sits at `x = 2/3`, not `x = 0.5`:

    V_target(x) = 1 + 2*x      -> V_target(0) = 1, V_target(1) = 3
    V_ref(x)    = 2 + 0.5*x    -> V_ref(0)    = 2, V_ref(1)    = 2.5

    d(x) = V_target(x) - V_ref(x) = -1 + 1.5*x   =>  d(x) = 0  at  x = 2/3

At the realized (off-grid) candidate state `x = 0.6` (strictly between 0.5
and 2/3):

    - OLD (interpolate boolean, threshold 0.5): 0.6 > 0.5  -> gate OPEN
      (route to the target) -- WRONG, this fixture's own boolean grid never
      says so at a genuine value comparison.
    - NEW (interpolate operands, apply predicate): d(0.6) = -0.1 < 0
      -> gate CLOSED (route to the fallback) -- the faithful answer.

`test_route_open_gate_is_recomputed_from_operands_not_from_interpolated_boolean`
proves the CURRENT code produces the faithful (CLOSED) answer, and
cross-checks -- using the SAME production interpolation kernel
(`_lcm.regime_building.ndimage.map_coordinates` + the grid's own
`get_coordinate`) rather than a hand-derived number -- that the OLD
boolean-interpolate-then-threshold recipe would have produced the OPPOSITE
(OPEN) answer for this exact fixture, so the two recipes are shown to
genuinely disagree here (not merely asserted to).
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from _lcm.regime_building.ndimage import map_coordinates
from _lcm.simulation.gated_routing import (
    route_gated_edges,
    substitute_gated_edge_continuations,
)
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, LinSpacedGrid, categorical
from lcm.ages import AgeGrid
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.regime_building.test_collective_regime_simulate import _solve_and_process

_BETA = 0.95
_X2 = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)  # nodes {0.0, 1.0}
_AGES = AgeGrid(start=0, stop=2, step="Y")


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity_x(x: ContinuousState) -> ContinuousState:
    return x


def _next_x_offgrid(x: ContinuousState) -> FloatND:
    """Every subject's candidate `x` for `target`/`ref` is the OFF-GRID 0.6,
    regardless of the current `x` -- the realized point the repro probes."""
    return jnp.full_like(x, 0.6)


def _u_src(x: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.zeros_like(x) * work


def _u_target(x: ContinuousState) -> FloatND:
    return 1.0 + 2.0 * x  # V_target(0) = 1, V_target(1) = 3


def _u_ref(x: ContinuousState) -> FloatND:
    return 2.0 + 0.5 * x  # V_ref(0) = 2, V_ref(1) = 2.5


def _u_fallback(x: ContinuousState) -> FloatND:
    return jnp.zeros_like(x)


def _value_gate(V_target: FloatND, V_ref: FloatND) -> BoolND:
    return V_target > V_ref


def _make_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": _X2},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_value_gate,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="fallback", projection={"x": _identity_x}
                        )
                    )
                },
                gate_refs={
                    "V_ref": SamePeriodRef(
                        regime="ref", projection={"x": _identity_x}
                    )
                },
            )
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": _X2},
        functions={"utility": _u_target},
    )
    ref = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": _X2},
        functions={"utility": _u_ref},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": _X2},
        functions={"utility": _u_fallback},
    )
    return {"src": src, "target": target, "ref": ref, "fallback": fallback}


def _solve_fixture():
    regimes_dict = _make_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=_AGES, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "target": MappingProxyType({}),
            "ref": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    solution, _sim_policies, dissolution_flags = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return regimes, regime_names_to_ids, flat_params, solution, dissolution_flags


def test_route_open_gate_is_recomputed_from_operands_not_from_interpolated_boolean():
    """Fail-pre/pass-post repro (simulate F1).

    Confirms (a) the two recipes GENUINELY disagree for this fixture at
    `x = 0.6` (via the production interpolation kernel, not a hand-derived
    number), and (b) the CURRENT `route_gated_edges` matches the faithful
    (value-operand recompute) answer, not the old boolean-interpolation one.
    """
    regimes, regime_names_to_ids, flat_params, solution, _dissolution_flags = (
        _solve_fixture()
    )
    src = regimes["src"]
    target_id = regime_names_to_ids["target"]
    fallback_id = regime_names_to_ids["fallback"]

    # (a) Cross-check with the SAME production interpolation kernel that the
    # OLD (removed) `GATE_ARR_NAME` mechanism used: interpolate the grid-level
    # BOOLEAN gate (evaluated exactly at the two nodes, cast to float) at the
    # realized x = 0.6, and threshold at 0.5.
    gate_at_nodes = np.array(
        [
            _u_target(jnp.asarray(0.0)) > _u_ref(jnp.asarray(0.0)),
            _u_target(jnp.asarray(1.0)) > _u_ref(jnp.asarray(1.0)),
        ]
    )
    np.testing.assert_array_equal(gate_at_nodes, [False, True])
    old_style_interp = map_coordinates(
        input=jnp.asarray(gate_at_nodes, dtype=float),
        coordinates=jnp.asarray([_X2.get_coordinate(jnp.asarray(0.6))]),
    )
    old_style_gate_open = bool(np.asarray(old_style_interp) > 0.5)
    assert old_style_gate_open, (
        "Sanity check on the repro fixture itself: the OLD boolean-"
        "interpolate-then-threshold recipe must say OPEN at x=0.6 for this "
        "test to be a genuine counterexample."
    )

    # The faithful (value-operand) answer at the SAME realized point: exact,
    # since V_target/V_ref are each linear on this 2-point grid.
    faithful_gate_open = bool(
        _u_target(jnp.asarray(0.6)) > _u_ref(jnp.asarray(0.6))
    )
    assert faithful_gate_open is False
    assert faithful_gate_open != old_style_gate_open, (
        "The two recipes must genuinely disagree at x=0.6 for this to be a "
        "repro, not merely a regression pin."
    )

    # (b) The CURRENT code, exercised at the kernel level (mirrors
    # `test_route_conditions_on_ordinary_draw.py`): hand-craft the realized
    # candidate states exactly like `calculate_next_states` would have
    # produced them (x = 0.6 for `target`; `fallback`'s slot is an
    # untouched-by-this-assertion placeholder, overwritten by the leg
    # projector regardless of its initial value).
    base_state_action_spaces = {
        name: regime.solution.state_action_space(regime_params=flat_params[name])
        for name, regime in regimes.items()
    }
    _substituted, same_period_mappings = substitute_gated_edge_continuations(
        regime=src,
        regime_name="src",
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=MappingProxyType({}),
        flat_params=flat_params,
    )
    assert "target" in same_period_mappings

    next_states = MappingProxyType(
        {
            "target": MappingProxyType({"x": jnp.array([0.6])}),
            "fallback": MappingProxyType({"x": jnp.array([-999.0])}),
        }
    )
    new_subject_regime_ids = jnp.array([target_id], dtype=jnp.int32)
    subjects_in_regime = jnp.array([True])

    _states, routed_ids = route_gated_edges(
        regime=src,
        same_period_mappings=same_period_mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=new_subject_regime_ids,
        subjects_in_regime=subjects_in_regime,
        flat_params=flat_params,
    )

    # Faithful (post-fix) answer: gate CLOSED at x=0.6 -> routed to the
    # fallback, NOT the target the old interpolate-then-threshold recipe
    # would have (wrongly) selected.
    np.testing.assert_array_equal(np.asarray(routed_ids), [fallback_id])
