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

CHARACTERIZATION (F1 residual, deliberately NOT fixed)
------------------------------------------------------
That fix replaced boolean interpolation with VALUE-OPERAND interpolation; it
did NOT make the gate faithful. `V_target` is an ALREADY-MAXIMIZED object and
interpolation does not commute with a `max`, so `interp(V_grid) != max_a Q`
off-grid whenever the target's V curves (see
`_lcm.regime_building.gated_edges.get_edge_simulate_gate_evaluator`'s
docstring, which documents this residual and declines to fix it).

The repro above is BLIND to that residual by construction: `_u_target` is
affine (`1 + 2*x`) and its regime declares no `actions`, so `V_target` is
exactly affine and 2-point linear interpolation is exact. The tests at the
bottom of this file therefore add an ACTIONED, CURVED target -- two affine
actions `u = x` and `u = 0.7 - 0.3*x` on the same `{0, 1}` grid, whose upper
envelope kinks at `x = 7/13` -- and PIN THE MEASURED RESIDUAL rather than
assert faithfulness:

    V_grid            = [max(0, 0.7), max(1, 0.4)] = [0.7, 1.0]
    interp(V_grid)(x) = 0.7 + 0.3*x          <- what the gate actually reads
    max_a Q(x)        = max(x, 0.7 - 0.3*x)  <- what a faithful gate would read

    at the NODES x in {0, 1}: the two AGREE exactly (0.7, 1.0).
    at x = 0.6:               interp = 0.88, true max_a Q = 0.6 -- a residual
                              of 0.28, enough to flip routing for any gate
                              threshold in (0.6, 0.88).

`test_gate_value_read_is_exact_at_grid_nodes_for_an_actioned_curved_target`
and `test_gate_value_read_off_grid_is_interp_of_V_not_true_max_a_Q` MEASURE
the gate's implied value read (by bisecting the gate's own threshold param
against the production evaluator, never a reimplementation of it) and assert
it equals `interp(V_grid)` -- exactly at nodes, and NOT `max_a Q` between
them. `test_offgrid_residual_flips_routing_of_the_real_router` then shows the
residual is not merely numeric: it changes which regime the real
`route_gated_edges` sends a subject to.

These are CHARACTERIZATION tests: they pin a KNOWN, ACCEPTED defect. They are
expected to fail if someone makes the gate faithful (recomputing `max_a Q`) —
that is the signal to delete them together with the residual, not a
regression.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from _lcm.regime_building.gated_edges import SOURCE_PARAMS
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.Q_and_F import SAME_PERIOD_V_ARG
from _lcm.simulation.gated_routing import (
    _call_vmapped_with_accepted_kwargs,
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
from tests.regime_building.test_simulate_gate_param_and_leg_selection import (
    exposed_param_name,
)

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
                    "V_ref": SamePeriodRef(regime="ref", projection={"x": _identity_x})
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
    faithful_gate_open = bool(_u_target(jnp.asarray(0.6)) > _u_ref(jnp.asarray(0.6)))
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


# ==================================================================================
# CHARACTERIZATION: the F1 residual (interp(V) != max_a Q), deliberately NOT fixed.
# ==================================================================================

# Two AFFINE actions whose upper envelope is CURVED (kinked) -- the minimal
# fixture in which "interpolate the stored V" and "recompute max_a Q" differ.
#
#   u(x, hold)  = 0.7 - 0.3*x      u(x, invest) = x
#   max_a Q(x)  = max(x, 0.7 - 0.3*x),   kink at x = 0.7/1.3 = 7/13 ~ 0.538462
#   V_grid      = [0.7, 1.0]  on x in {0, 1}
#   interp(V_grid)(x) = 0.7 + 0.3*x
_KINK_X = 0.7 / 1.3
_CURVED_V_GRID = (0.7, 1.0)


@categorical(ordered=True)
class Invest:
    hold: ScalarInt  # code 0
    invest: ScalarInt  # code 1


def _u_curved_target(x: ContinuousState, invest: DiscreteAction) -> FloatND:
    return jnp.where(invest == 1, x, 0.7 - 0.3 * x)


def _true_max_a_Q(x: float) -> float:
    """The faithful object: the target's realized upper envelope at `x`."""
    return float(max(x, 0.7 - 0.3 * x))


def _interp_of_V_grid(x: float) -> float:
    """What the gate actually reads: linear interpolation of the STORED,
    already-maximized V over the 2-point grid `{0, 1}`."""
    lo, hi = _CURVED_V_GRID
    return float(lo + (hi - lo) * x)


def _threshold_gate(V_target: FloatND, gate_threshold: FloatND) -> BoolND:
    """A gate whose only nonlinearity is the comparison itself, so the gate's
    flip point in `gate_threshold` IS the value operand the gate read."""
    return V_target > gate_threshold


def _make_curved_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": _X2},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_threshold_gate,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="fallback", projection={"x": _identity_x}
                        )
                    )
                },
            )
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": _X2},
        actions={"invest": DiscreteGrid(Invest)},
        functions={"utility": _u_curved_target},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": _X2},
        functions={"utility": _u_fallback},
    )
    return {"src": src, "target": target, "fallback": fallback}


def _solve_curved_fixture(*, gate_threshold: float):
    regimes_dict = _make_curved_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=_AGES, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {
                    "H__discount_factor": jnp.asarray(_BETA),
                    "gate_threshold": jnp.asarray(gate_threshold),
                }
            ),
            "target": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    solution, _sim_policies, _dissolution_flags = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return regimes, regime_names_to_ids, flat_params, solution


def _measure_gate_value_read(*, x: float) -> float:
    """Return the value operand the PRODUCTION gate evaluator read at `x`.

    The gate is `V_target > gate_threshold`, so the threshold at which the
    gate flips from open to closed IS whatever `V_target` the evaluator
    obtained. Bisecting on the threshold reads that number out of the real
    evaluator without reimplementing any part of it.
    """
    regimes, _ids, flat_params, solution = _solve_curved_fixture(gate_threshold=0.0)
    src = regimes["src"]
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
    evaluator = src.gated_edge_simulate_gate_evaluators["target"]

    # The gate's own param is SOURCE-owned, and the evaluator exposes it under a
    # namespace-qualified leaf (F2) — ask the published provenance for the name
    # rather than hard-coding the qualification scheme here.
    threshold_arg = exposed_param_name(
        evaluator, qname="gate_threshold", namespace=SOURCE_PARAMS
    )

    def _gate_open(threshold: float) -> bool:
        value = _call_vmapped_with_accepted_kwargs(
            evaluator,
            batched_kwargs={"x": jnp.array([x])},
            static_kwargs={
                threshold_arg: jnp.asarray(threshold),
                SAME_PERIOD_V_ARG: same_period_mappings["target"],
            },
            axis_size=1,
        )
        return bool(np.asarray(value)[0])

    lo, hi = -1.0, 3.0
    assert _gate_open(lo), "bisection bracket must start with the gate OPEN"
    assert not _gate_open(hi), "bisection bracket must end with the gate CLOSED"
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _gate_open(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_curved_target_V_grid_is_the_upper_envelope_at_the_nodes():
    """Fixture ground truth: the solved target V really is `[0.7, 1.0]`, i.e.
    the maximized envelope of the two affine actions at the two nodes."""
    _regimes, _ids, _flat_params, solution = _solve_curved_fixture(gate_threshold=0.0)
    np.testing.assert_allclose(
        np.asarray(solution[1]["target"]), _CURVED_V_GRID, rtol=1e-6
    )
    # ...and the two recipes coincide there, which is exactly why the node
    # assertions below are a meaningful exactness check and not a tautology.
    np.testing.assert_allclose(
        [_true_max_a_Q(0.0), _true_max_a_Q(1.0)], _CURVED_V_GRID, rtol=1e-6
    )


def test_gate_value_read_is_exact_at_grid_nodes_for_an_actioned_curved_target():
    """CHARACTERIZATION: on grid NODES the gate's value read is exact.

    Even with an actioned, curved target the interpolation residual vanishes
    at the nodes (the interpolant reproduces the stored V there by
    construction), so the gate agrees with the true `max_a Q` to float32
    resolution. This is the half of the residual's characterization that is
    GOOD news, and it is what bounds the defect: it is an off-grid,
    O(h^2)-in-the-cell effect, not a wholesale wrong read.
    """
    for node, expected in zip((0.0, 1.0), _CURVED_V_GRID, strict=True):
        measured = _measure_gate_value_read(x=node)
        # Exact against the stored V...
        np.testing.assert_allclose(measured, expected, atol=1e-6)
        # ...and, at a node, exact against the FAITHFUL object too.
        np.testing.assert_allclose(measured, _true_max_a_Q(node), atol=1e-6)


def test_gate_value_read_off_grid_is_interp_of_V_not_true_max_a_Q():
    """CHARACTERIZATION: off-grid the gate reads `interp(V)`, NOT `max_a Q`.

    This test PINS THE KNOWN DEFECT. It deliberately does not assert the
    faithful answer, because the production code does not produce it: at
    x = 0.6 the gate reads 0.88 (the interpolant of the stored, already-
    maximized V) where a faithful `max_a Q` recompute would read 0.6.
    """
    x = 0.6
    measured = _measure_gate_value_read(x=x)

    interp_value = _interp_of_V_grid(x)  # 0.88
    true_value = _true_max_a_Q(x)  # 0.6
    assert x > _KINK_X, (
        "the probe point must sit on the far side of the envelope's kink, or "
        "the two recipes would not differ here"
    )
    assert abs(interp_value - true_value) > 0.25, (
        "fixture sanity: the two recipes must genuinely disagree at this point"
    )

    # What the code DOES: interpolate the stored V.
    np.testing.assert_allclose(measured, interp_value, atol=1e-5)
    # What a faithful gate WOULD do -- and demonstrably does not.
    assert abs(measured - true_value) > 0.25, (
        "The F1 residual is documented as live: if the gate now recomputes "
        "max_a Q, delete this characterization test together with the "
        "residual (see this module's docstring)."
    )


def test_offgrid_residual_flips_routing_of_the_real_router():
    """CHARACTERIZATION: the residual changes an actual routing decision.

    With a gate threshold of 0.75 -- strictly between the true `max_a Q(0.6)`
    = 0.6 and the interpolated 0.88 -- a faithful gate would route the
    subject to the FALLBACK, while `route_gated_edges` routes it to the
    TARGET. Pinned, not fixed.
    """
    threshold = 0.75
    x = 0.6
    assert _true_max_a_Q(x) < threshold < _interp_of_V_grid(x)

    regimes, regime_names_to_ids, flat_params, solution = _solve_curved_fixture(
        gate_threshold=threshold
    )
    src = regimes["src"]
    target_id = regime_names_to_ids["target"]
    fallback_id = regime_names_to_ids["fallback"]

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
    next_states = MappingProxyType(
        {
            "target": MappingProxyType({"x": jnp.array([x])}),
            "fallback": MappingProxyType({"x": jnp.array([-999.0])}),
        }
    )
    _states, routed_ids = route_gated_edges(
        regime=src,
        same_period_mappings=same_period_mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.array([target_id], dtype=jnp.int32),
        subjects_in_regime=jnp.array([True]),
        flat_params=flat_params,
    )
    # The gate opens because interp(V)(0.6) = 0.88 > 0.75 -- a faithful
    # max_a Q(0.6) = 0.6 < 0.75 would have closed it and routed to the
    # fallback. THIS ASSERTION PINS THE DEFECT, it does not bless it.
    np.testing.assert_array_equal(np.asarray(routed_ids), [target_id])
    assert int(np.asarray(routed_ids)[0]) != int(fallback_id)
