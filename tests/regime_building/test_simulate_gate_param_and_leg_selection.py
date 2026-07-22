"""Repros/regressions for three simulate-side gated-edge routing defects.

All three live on the path `_lcm.simulation.gated_routing.route_gated_edges`
walks once per declared gated edge; each is exercised here at the kernel
level (the `test_route_conditions_on_ordinary_draw.py` /
`test_gated_edge_simulate_operand_recompute.py` idiom: build + solve a
miniature, then call `substitute_gated_edge_continuations` and
`route_gated_edges` directly on hand-crafted realized states).

**F2 — each of the gate evaluator's args must be resolved against the ONE
namespace that owns it.** The evaluator
(`_lcm.regime_building.gated_edges.get_edge_simulate_gate_evaluator`)
publishes an `arg_provenance` precisely because its extra args are NOT
distinguishable by name: `get_V_interpolator`'s runtime grid helpers are named
after the STATE with no regime qualification
(`_lcm.regime_building.V._get_coordinate_finder` ->
`qname_from_tree_path((state_name, "points"))` -> `x__points`). A SOURCE
regime that happens to declare a state named `x` therefore contributes an
identically named param, and the pre-fix single merged dict
(`{**flat_params[target_name], **flat_params[regime.name]}`, source last =
source wins) silently handed the TARGET's interpolator the SOURCE's grid
points. `test_gate_reads_target_grid_points_not_the_source_s_same_named_ones`
proves the current code reads the target's grid and — by calling
`_call_vmapped_with_accepted_kwargs` with exactly the old merged dict, in
this same process — that the old recipe returns the OPPOSITE gate.

An intermediate revision published two frozensets of UNQUALIFIED names
(`args_from_target_params` / `args_from_source_params`) and had the router
build two filtered dicts, merged source-last. That did not fix it, on two
counts, and both are pinned here:

- The sets are not DISJOINT — the target's `x__points` and a source param of
  the same qname both land in them — so the merge still let the source win the
  collision exactly as before. No merge ORDER can be right: one keyword
  argument cannot carry two regimes' arrays. Exposed param leaves are now
  namespace-QUALIFIED, which is what makes the two distinct
  (`test_gate_evaluator_provenance_partitions_its_signature`).
- Nor were they COMPLETE. A gate-ref reader's args were assigned WHOLESALE to
  the target, though they mix target candidate states, SOURCE-declared
  projection params, and the reference regime's own interpolation helpers —
  three provenances, not two
  (`test_gate_ref_projection_param_is_bound_from_the_source_not_the_target`).

**F3 — a STATELESS gated target must not crash `vmap`.** With no states of
its own (a terminal scrap-value regime), the evaluator's batched kwargs are
empty after filtering, and `jax.vmap` cannot infer a batch size from zero
batched arguments ("vmap wrapped function must be passed at least one
argument containing an array or axis_size must be specified"). `axis_size`
is now threaded explicitly from the population size.

**F4 — `_select_own_leg`'s sole-leg exemption must key on the leg's ROLE,
not on arity.** The validator accepts a ONE-element `stakeholders` tuple, and
`processing.py`'s `leg_order = [(s, s) for s in source_stakeholders]` gives
that sole leg `source_stakeholder="f"`, not `None`. The pre-fix
`len(legs) > 1` guard therefore let a typo'd `own_stakeholder` fall through
silently on exactly the single-stakeholder COLLECTIVE source the raise
exists to protect.
"""

from inspect import signature
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.gated_edges import (
    SOURCE_PARAMS,
    TARGET_PARAMS,
    ResolvedEdgeLeg,
    ResolvedSamePeriodRef,
)
from _lcm.regime_building.Q_and_F import SAME_PERIOD_PARAMS_ARG, SAME_PERIOD_V_ARG
from _lcm.simulation.gated_routing import (
    _bind_provenance_params,
    _call_vmapped_with_accepted_kwargs,
    _select_own_leg,
    route_gated_edges,
    substitute_gated_edge_continuations,
)
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, IrregSpacedGrid, LinSpacedGrid, categorical
from lcm.ages import AgeGrid
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.regime_building.test_collective_regime_simulate import _solve_and_process

_BETA = 0.95
_AGES = AgeGrid(start=0, stop=2, step="Y")

# The realized, OFF-GRID candidate `x` every subject lands on in the F2 repro.
_REALIZED_X = 0.6

# The target's own grid points (supplied at runtime, hence the collision-prone
# `x__points` param) and the SOURCE's identically named ones. Both grids are
# declared `IrregSpacedGrid(n_points=2)`, so both regimes' flat params carry a
# key literally named `x__points` — the whole point of the repro.
_TARGET_POINTS = (0.0, 1.0)
_SOURCE_POINTS = (0.0, 10.0)

# Gate threshold, in the SOURCE's namespace. V_target is 1 + 2x on the
# target's own {0, 1} grid, so:
#   - read on the TARGET's points, x=0.6 -> V = 2.2 > 2.0  -> gate OPEN.
#   - read on the SOURCE's points (the pre-fix bug), x=0.6 collapses to grid
#     coordinate 0.06 -> V = 1.12 < 2.0 -> gate CLOSED (misroute).
_GATE_THRESHOLD = 2.0


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity_x(x: ContinuousState) -> ContinuousState:
    return x


def _next_x_offgrid(x: ContinuousState) -> FloatND:
    return jnp.full_like(x, _REALIZED_X)


def _u_src(x: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.zeros_like(x) * work


def _u_target(x: ContinuousState) -> FloatND:
    return 1.0 + 2.0 * x  # V_target(0) = 1, V_target(1) = 3


def _u_fallback(x: ContinuousState) -> FloatND:
    return jnp.zeros_like(x)


def _threshold_gate(V_target: FloatND, gate_threshold: FloatND) -> BoolND:
    """Gate reading an operand from the TARGET's namespace (`V_target`, via the
    target-grid interpolator) and a param from the SOURCE's (`gate_threshold`)
    — so both `args_from_target_params` and `args_from_source_params` are
    non-empty and each must be resolved against its own namespace."""
    return V_target > gate_threshold


def _make_f2_regimes() -> dict[str, Regime]:
    """Source and target BOTH declare a continuous state named `x` on a
    runtime-points `IrregSpacedGrid`, with DIFFERENT points."""
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": IrregSpacedGrid(n_points=2)},
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
        states={"x": IrregSpacedGrid(n_points=2)},
        functions={"utility": _u_target},
    )
    # Fixed grid: the fallback's V is read by the solve-side fold through the
    # SOURCE's params, so a runtime-points fallback grid would confound this
    # repro with a second (solve-side) namespace question.
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_fallback},
    )
    return {"src": src, "target": target, "fallback": fallback}


def _solve_f2_fixture():
    regimes_dict = _make_f2_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=_AGES, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {
                    "H__discount_factor": jnp.asarray(_BETA),
                    "gate_threshold": jnp.asarray(_GATE_THRESHOLD),
                    # The collision: the SOURCE's own `x` grid points, named
                    # exactly like the target's.
                    "x__points": jnp.asarray(_SOURCE_POINTS),
                }
            ),
            "target": MappingProxyType({"x__points": jnp.asarray(_TARGET_POINTS)}),
            "fallback": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    _dissolution_flags = _bi_result.dissolution_flags
    return regimes, regime_names_to_ids, flat_params, solution


def _f2_same_period_mappings(regimes, flat_params, solution):
    base_state_action_spaces = {
        name: regime.solution.state_action_space(regime_params=flat_params[name])
        for name, regime in regimes.items()
    }
    _substituted, same_period_mappings = substitute_gated_edge_continuations(
        regime=regimes["src"],
        regime_name="src",
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=MappingProxyType({}),
        flat_params=flat_params,
    )
    return same_period_mappings


def exposed_param_name(evaluator, *, qname: str, namespace: str) -> str:
    """The signature leaf under which `evaluator` exposes `namespace`'s `qname`.

    Shared with `test_gated_edge_simulate_operand_recompute.py`: a test that
    calls a gate evaluator directly must name its params the way the evaluator
    exposes them, and asking the published provenance is how — hard-coding the
    qualification scheme in every test would pin the naming instead of the
    contract.
    """
    for exposed, provenance in evaluator.arg_provenance.params.items():
        if provenance == (namespace, qname):
            return exposed
    msg = (
        f"No argument of provenance {(namespace, qname)}; the evaluator has "
        f"{dict(evaluator.arg_provenance.params)}."
    )
    raise AssertionError(msg)


def test_gate_evaluator_provenance_partitions_its_signature():
    """The published provenance names ONE namespace per argument, and covers
    every argument of the evaluator's signature.

    This is the contract `route_gated_edges` binds against, and both halves are
    load-bearing (see the module docstring):

    - DISJOINT: the target's runtime-grid helper and the source's identically
      named param are distinct LEAVES, so neither can clobber the other. The
      pre-fix frozensets carried the qname `x__points` in both sets, which no
      merge order can resolve.
    - COMPLETE: no argument is left for the router to guess a namespace for.
    """
    regimes, _ids, _flat_params, _solution = _solve_f2_fixture()
    evaluator = regimes["src"].gated_edge_simulate_gate_evaluators["target"]
    provenance = evaluator.arg_provenance

    # The target's grid points and the gate's own threshold are each attributed
    # to exactly one namespace...
    target_points = exposed_param_name(
        evaluator, qname="x__points", namespace=TARGET_PARAMS
    )
    threshold = exposed_param_name(
        evaluator, qname="gate_threshold", namespace=SOURCE_PARAMS
    )
    assert target_points != threshold

    # ...and the source ALSO declares an `x__points` (that is the collision this
    # fixture is built on). Were the source's own points ever needed by this
    # evaluator, they would be a DIFFERENT leaf — the qname alone is not an
    # identity, which is precisely what the two frozensets got wrong.
    assert target_points != f"__source_param__{'x__points'}"

    # Every non-engine argument of the actual signature is classified.
    signature_args = set(signature(evaluator).parameters)
    engine_args = {SAME_PERIOD_V_ARG, SAME_PERIOD_PARAMS_ARG}
    classified = provenance.states | set(provenance.params)
    assert not (signature_args - engine_args - classified)
    assert not (provenance.states & set(provenance.params))
    assert classified <= signature_args | engine_args


def test_gate_reads_target_grid_points_not_the_source_s_same_named_ones():
    """Fail-pre/pass-post repro (F2, the disjointness half).

    Proves (a) the pre-fix and post-fix BINDINGS genuinely disagree for this
    fixture, in this same process and on the production evaluator — the source
    files are never touched — and (b) the CURRENT `route_gated_edges` produces
    the faithful (target-grid) answer.

    The pre-fix call cannot be replayed verbatim any more, and that is the fix
    working rather than a gap in the proof: both the ORIGINAL merged dict and
    the intermediate two-frozenset revision fed the target's interpolator
    through a leaf named by the bare qname `x__points`, which the source's
    identically named param won. That leaf no longer exists — the evaluator
    exposes the target's points and the source's under distinct
    namespace-qualified names, so no caller can deliver one where the other
    belongs. What IS replayed here is the VALUE the pre-fix binding delivered:
    the target's interpolation helper fed the source's points, which is exactly
    what the merge produced and what `_bind_provenance_params` now cannot.
    """
    regimes, regime_names_to_ids, flat_params, solution = _solve_f2_fixture()
    src = regimes["src"]
    target_id = regime_names_to_ids["target"]
    fallback_id = regime_names_to_ids["fallback"]
    same_period_mappings = _f2_same_period_mappings(regimes, flat_params, solution)
    assert "target" in same_period_mappings

    # Ground truth: V_target is 1 + 2x, exactly linear on the target's own
    # {0, 1} grid, so its interpolant at the realized x = 0.6 is exact.
    v_target_grid = np.asarray(solution[1]["target"])
    np.testing.assert_allclose(v_target_grid, [1.0, 3.0], rtol=1e-6)
    faithful_v = float(_u_target(jnp.asarray(_REALIZED_X)))
    assert faithful_v > _GATE_THRESHOLD  # -> gate OPEN

    # What the SOURCE's points would make of the same realized value: 0.6 on
    # the points (0, 10) is grid coordinate 0.06, i.e. V ~ 1.12 -- CLOSED.
    evaluator = src.gated_edge_simulate_gate_evaluators["target"]
    candidate_target_states = {"x": jnp.array([_REALIZED_X])}

    # The collision this fixture is built on is real: BOTH regimes' flat params
    # carry a key literally named `x__points`, with different values. That is
    # what made a single merged dict — and a filter by unqualified names —
    # unable to serve both.
    assert "x__points" in flat_params["target"]
    assert "x__points" in flat_params["src"]
    assert not np.array_equal(
        np.asarray(flat_params["target"]["x__points"]),
        np.asarray(flat_params["src"]["x__points"]),
    )

    points_arg = exposed_param_name(
        evaluator, qname="x__points", namespace=TARGET_PARAMS
    )
    threshold_arg = exposed_param_name(
        evaluator, qname="gate_threshold", namespace=SOURCE_PARAMS
    )

    def _gate(points) -> bool:
        """The production evaluator with the target's interpolator fed `points`."""
        return bool(
            np.asarray(
                _call_vmapped_with_accepted_kwargs(
                    evaluator,
                    batched_kwargs=candidate_target_states,
                    static_kwargs={
                        points_arg: jnp.asarray(points),
                        threshold_arg: jnp.asarray(_GATE_THRESHOLD),
                        SAME_PERIOD_V_ARG: same_period_mappings["target"],
                    },
                    axis_size=1,
                )
            )[0]
        )

    # (a) The PRE-FIX binding's VALUE: the target's interpolation helper fed the
    # SOURCE's points, exactly what `{**flat_params[target], **flat_params[src]}`
    # (source last) delivered, and what the intermediate revision's frozensets
    # still delivered (their intersection on `x__points` left the merge to
    # decide). x=0.6 on the points (0, 10) is grid coordinate 0.06, V ~ 1.12.
    assert not _gate(_SOURCE_POINTS), (
        "Sanity check on the repro fixture itself: with the source's "
        "x__points=(0, 10) driving the target's interpolation the gate must "
        "close, or this test is not a genuine counterexample."
    )
    # ...and the POINTS are the mechanism, not some incidental property: the
    # identical call with the target's own points (the only change) opens it.
    assert _gate(_TARGET_POINTS)

    # The post-fix, provenance-bound recipe on the SAME evaluator and the SAME
    # realized point: the target's own points win, the gate opens. Nothing is
    # merged — each leaf is looked up in exactly the one namespace that owns it.
    new_style_gate = jnp.asarray(
        _call_vmapped_with_accepted_kwargs(
            evaluator,
            batched_kwargs=candidate_target_states,
            static_kwargs={
                **_bind_provenance_params(
                    evaluator.arg_provenance,
                    flat_params=flat_params,
                    source_name="src",
                    target_name="target",
                ),
                SAME_PERIOD_V_ARG: same_period_mappings["target"],
            },
            axis_size=1,
        )
    )
    assert bool(np.asarray(new_style_gate)[0])
    assert bool(np.asarray(new_style_gate)[0]) != _gate(_SOURCE_POINTS), (
        "The two bindings must genuinely disagree for this to be a repro, not "
        "merely a regression pin."
    )
    # The provenance binder is what delivers the target's own points to the
    # target's interpolator, with the source's same-named param present in
    # `flat_params` throughout and unable to reach that leaf.
    bound = _bind_provenance_params(
        evaluator.arg_provenance,
        flat_params=flat_params,
        source_name="src",
        target_name="target",
    )
    np.testing.assert_array_equal(
        np.asarray(bound[points_arg]), np.asarray(_TARGET_POINTS)
    )

    # (b) The CURRENT production router must produce the faithful answer:
    # routed to the TARGET, not to the fallback the merged dict selected.
    next_states = MappingProxyType(
        {
            "target": MappingProxyType({"x": jnp.array([_REALIZED_X])}),
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
    np.testing.assert_array_equal(np.asarray(routed_ids), [target_id])
    assert int(np.asarray(routed_ids)[0]) != int(fallback_id)


# ----------------------------------------------------------------------------------
# F3: a STATELESS gated target leaves `vmap` with zero batched args.
# ----------------------------------------------------------------------------------


def _u_stateless_target() -> FloatND:
    return jnp.asarray(1.0)


def _u_stateless_fallback() -> FloatND:
    return jnp.asarray(0.0)


def _stateless_gate(V_target: FloatND) -> BoolND:
    # V_target = 1.0 for the terminal stateless target -> OPEN.
    return V_target > 0.5


def _make_f3_regimes() -> dict[str, Regime]:
    """A 3-regime model whose gated target is STATELESS (terminal scrap value)."""
    src = Regime(
        transition={"stateless_target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _identity_x},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "stateless_target": GatedEdge(
                gate=_stateless_gate,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="stateless_fallback", projection={}
                        )
                    )
                },
            )
        },
    )
    stateless_target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": _u_stateless_target},
    )
    stateless_fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": _u_stateless_fallback},
    )
    return {
        "src": src,
        "stateless_target": stateless_target,
        "stateless_fallback": stateless_fallback,
    }


def _solve_f3_fixture():
    regimes_dict = _make_f3_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=_AGES, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "stateless_target": MappingProxyType({}),
            "stateless_fallback": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    _dissolution_flags = _bi_result.dissolution_flags
    return regimes, regime_names_to_ids, flat_params, solution


def test_stateless_gated_target_routes_without_vmap_axis_size_error():
    """Fail-pre/pass-post repro (F3).

    A stateless target's evaluator accepts no batched arg at all, so the
    pre-fix `jax.vmap(f)(batched)` with `batched == {}` raised
    `ValueError: vmap wrapped function must be passed at least one argument
    containing an array or axis_size must be specified`. The routing call
    below must simply work, and produce a per-subject gate of the right
    length.
    """
    regimes, regime_names_to_ids, flat_params, solution = _solve_f3_fixture()
    src = regimes["src"]
    target_id = regime_names_to_ids["stateless_target"]

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
    assert "stateless_target" in same_period_mappings

    n_subjects = 3
    next_states = MappingProxyType(
        {
            "stateless_target": MappingProxyType({}),
            "stateless_fallback": MappingProxyType({}),
        }
    )
    _states, routed_ids = route_gated_edges(
        regime=src,
        same_period_mappings=same_period_mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.array([target_id] * n_subjects, dtype=jnp.int32),
        subjects_in_regime=jnp.array([True] * n_subjects),
        flat_params=flat_params,
    )
    # V_target = 1.0 > 0.5 -> gate OPEN for every subject; the routing is
    # per-subject (length n_subjects), not a collapsed scalar.
    np.testing.assert_array_equal(np.asarray(routed_ids), [target_id] * n_subjects)


def test_vmap_without_axis_size_is_the_pre_fix_failure_for_a_stateless_target():
    """Proves the F3 test above is fail-pre, not just a pin.

    Reproduces the PRE-FIX call (`jax.vmap(f)(batched)`, no `axis_size`) on
    the SAME evaluator and the SAME empty batched kwargs, in this process —
    the production source is never reverted.
    """
    regimes, _ids, flat_params, solution = _solve_f3_fixture()
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
    evaluator = src.gated_edge_simulate_gate_evaluators["stateless_target"]
    static = {SAME_PERIOD_V_ARG: same_period_mappings["stateless_target"]}

    def _call_one_subject(one_subject_kwargs):
        return evaluator(**one_subject_kwargs, **static)

    # The pre-fix body, verbatim except for the missing `axis_size`.
    with pytest.raises(ValueError, match="at least one argument"):
        jax.vmap(_call_one_subject)({})

    # And the post-fix body, on the identical inputs, does not raise.
    result = _call_vmapped_with_accepted_kwargs(
        evaluator,
        batched_kwargs={},
        static_kwargs=static,
        axis_size=4,
    )
    assert np.asarray(result).shape == (4,)


# ----------------------------------------------------------------------------------
# F4: `_select_own_leg`'s sole-leg exemption keys on the ROLE, not on arity.
# ----------------------------------------------------------------------------------


def _leg(source_stakeholder: str | None, fallback_regime: str) -> ResolvedEdgeLeg:
    return ResolvedEdgeLeg(
        source_stakeholder=source_stakeholder,
        target_component_index=None,
        fallback=ResolvedSamePeriodRef(
            regime=fallback_regime, projection={}, stakeholder_index=None
        ),
    )


def _old_select_own_leg(legs, own_stakeholder):
    """A local copy of the PRE-FIX guard (`len(legs) > 1`), for the
    fail-pre proof. Never imported from production — the production
    `_select_own_leg` is the fixed one."""
    if own_stakeholder is not None:
        for leg in legs:
            if leg.source_stakeholder == own_stakeholder:
                return leg
        if len(legs) > 1:
            raise ValueError("own_stakeholder does not match any leg")
    return legs[0]


def test_one_leg_collective_source_raises_on_unknown_own_stakeholder():
    """Fail-pre/pass-post repro (F4a).

    A ONE-element `stakeholders=("f",)` source is legal, and
    `processing.py`'s `leg_order = [(s, s) for s in source_stakeholders]`
    gives its sole leg `source_stakeholder="f"` — NOT `None`. A typo'd
    `own_stakeholder` must therefore raise, even though there is only one
    leg.
    """
    legs = (_leg("f", "single_f"),)

    with pytest.raises(ValueError, match="does not match any"):
        _select_own_leg(legs, "typo")

    # The PRE-FIX guard, reproduced here: `len(legs) > 1` is False, so the
    # typo fell through to legs[0] silently.
    assert _old_select_own_leg(legs, "typo") is legs[0]


def test_one_leg_singleton_source_still_falls_back_without_raising():
    """Regression pin (F4b): the LEGITIMATE sole-leg exemption survives.

    A SINGLETON source's one leg carries `source_stakeholder=None` and so
    structurally never matches a non-`None` `own_stakeholder`; that is the
    common, correct case, not an error.
    """
    legs = (_leg(None, "single"),)
    assert _select_own_leg(legs, "f") is legs[0]
    assert _select_own_leg(legs, None) is legs[0]


def test_one_leg_collective_source_selects_the_matching_leg():
    """Regression pin (F4c): a MATCHING own-role on a one-leg collective
    source selects that leg (the raise does not over-fire)."""
    legs = (_leg("f", "single_f"),)
    assert _select_own_leg(legs, "f") is legs[0]
    # `own_stakeholder=None` keeps the legacy first-declared-leg convention.
    assert _select_own_leg(legs, None) is legs[0]


def test_multi_leg_collective_source_unknown_own_stakeholder_still_raises():
    """Regression pin: the multi-leg case the pre-fix guard DID cover keeps
    raising — the fix is a widening, not a replacement."""
    legs = (_leg("f", "single_f"), _leg("m", "single_m"))
    with pytest.raises(ValueError, match="does not match any"):
        _select_own_leg(legs, "typo")
    assert _select_own_leg(legs, "m") is legs[1]
