"""Argument PROVENANCE across the regimes a gated edge relates (F2/F3/F4).

One defect class, three symptoms: an argument of an edge-side callable was
bound from a regime that does not own it, because the callable did not record
— and the caller could not tell — which regime does.

The three provenances an edge-side callable mixes, all in one signature:

1. The TARGET's candidate states, and (simulate only) the TARGET's own V/`D`
   interpolation grid.
2. The SOURCE's params: the gate predicate's own free params, and the free
   params of the source-declared gate-ref / fallback projections.
   `backward_induction._evaluate_edge_fold` binds every param the SOLVE-side
   fold needs from `flat_params[source]`, so this is not a preference between
   two merges — it is what makes simulate evaluate the same object solve folded.
3. Each REFERENCE regime's own interpolation grid, for a gate ref's or a leg
   fallback's read of that regime's V.

Covered here:

- **F2 (completeness).** `gated_edges.py`'s previous revision assigned every
  gate-ref reader argument WHOLESALE to the target namespace, though a reader's
  signature mixes all three of the above. A SOURCE-declared projection param was
  therefore read out of the TARGET's params — silently, when the target happened
  to declare the same qname, and as a crash when it did not.
  (F2's disjointness half — two regimes' identically named `x__points` — is
  pinned in `test_simulate_gate_param_and_leg_selection.py`.)
- **F3.** `route_gated_edges` called every fallback projector with
  `{**candidate_target_states, **flat_params[target]}` while the fold projected
  the very same coordinate with `flat_params[source]`. The row entered the right
  fallback REGIME at a STATE the solved policy never priced, and carried it into
  the next period.
- **F4.** `_build_same_period_ref_reader` exposed the reference regime's
  interpolation helpers under the PREFIXED coordinate name
  (`__same_period_ref__x__points`), which no params template emits and no caller
  supplies, so any reference to a runtime irregular grid raised a
  missing-argument error — on all four consumers of that reader. Coordinate
  variables stay prefixed; the PARAMS are now resolved against the reference
  regime's own namespace (`SAME_PERIOD_PARAMS_ARG`).

Fixture discipline: every one of these is only a repro if the two candidate
bindings genuinely disagree, so each fixture gives the two regimes DIFFERENT
values for the contested name (`_SRC_SHIFT` != `_TARGET_SHIFT`, `_REF_POINTS`
!= `_SRC_POINTS`) and asserts the disagreement itself.
"""

from inspect import signature
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.gated_edges import (
    SOURCE_PARAMS,
    TARGET_PARAMS,
    ResolvedSamePeriodRef,
    _reached_target_param_leaves,
    _reject_gate_operand_state_name_collision,
)
from _lcm.regime_building.Q_and_F import (
    _REF_STATE_PREFIX,
    SAME_PERIOD_PARAMS_ARG,
    SAME_PERIOD_V_ARG,
    _build_same_period_ref_reader,
)
from _lcm.regime_building.V import create_v_interpolation_info, get_V_interpolator
from _lcm.simulation.gated_routing import (
    _bind_provenance_params,
    _call_vmapped_with_accepted_kwargs,
    route_gated_edges,
    substitute_gated_edge_continuations,
)
from _lcm.solution.backward_induction import solve
from _lcm.utils.functools import get_union_of_args
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, IrregSpacedGrid, LinSpacedGrid, categorical
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.regime_building.test_collective_regime_simulate import _solve_and_process
from tests.regime_building.test_simulate_gate_param_and_leg_selection import (
    exposed_param_name,
)

_BETA = 0.95
_AGES = AgeGrid(start=0, stop=2, step="Y")

# The realized, OFF-GRID candidate `x` every subject lands on.
_REALIZED_X = 0.6


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


def _u_identity(x: ContinuousState) -> FloatND:
    """V(x) = x — linear, hence interpolated EXACTLY on any grid containing the
    read point's bracketing nodes, so every number below is a clean equality
    rather than an interpolation tolerance."""
    return x


def _solve_fixture(regimes_dict, flat_params):
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=_AGES, regime_names=list(regimes_dict)
    )
    solution, _sim_policies, _dissolution_flags = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return regimes, regime_names_to_ids, solution


def _same_period_mappings(regimes, flat_params, solution):
    base_state_action_spaces = {
        name: regime.solution.state_action_space(regime_params=flat_params[name])
        for name, regime in regimes.items()
    }
    _substituted, mappings = substitute_gated_edge_continuations(
        regime=regimes["src"],
        regime_name="src",
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=MappingProxyType({}),
        flat_params=flat_params,
    )
    return mappings


# ----------------------------------------------------------------------------------
# F2 (completeness): a SOURCE-declared gate-ref projection's free param.
# ----------------------------------------------------------------------------------

# The contested qname. The SOURCE declares the projection, so `shift` is the
# source's parameter; the target's identically named one exists only to make the
# pre-fix misbinding SILENT rather than a crash (both halves are asserted below).
_SRC_SHIFT = 0.1
_TARGET_SHIFT = 0.9


def _project_to_shift(shift: FloatND) -> FloatND:
    """The gate ref's projection: read the reference regime's V at `shift`.

    Declared on the SOURCE (inside its `gated_edges`), so `shift` is a parameter
    of the SOURCE — that is the namespace the solve-side fold binds it from.
    """
    return shift


def _ref_gate(V_target: FloatND, ref_v: FloatND) -> BoolND:
    """`V_target(0.6) = 0.6` vs the reference value AT the projected point.

    With `V_ref(x) = x`, `ref_v` IS whichever `shift` the projection was given:
    the source's 0.1 (gate OPEN, `0.6 > 0.1`) or the target's 0.9 (gate CLOSED,
    `0.6 > 0.9`) — the reviewer's exact counterexample.
    """
    return V_target > ref_v


def _make_shift_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_ref_gate,
                gate_refs={
                    "ref_v": SamePeriodRef(
                        regime="refregime", projection={"x": _project_to_shift}
                    )
                },
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
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    refregime = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    return {
        "src": src,
        "target": target,
        "refregime": refregime,
        "fallback": fallback,
    }


def _shift_flat_params(*, target_declares_shift: bool = True):
    target_params = (
        {"shift": jnp.asarray(_TARGET_SHIFT)} if target_declares_shift else {}
    )
    return MappingProxyType(
        {
            "src": MappingProxyType(
                {
                    "H__discount_factor": jnp.asarray(_BETA),
                    "shift": jnp.asarray(_SRC_SHIFT),
                }
            ),
            "target": MappingProxyType(target_params),
            "refregime": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )


def test_gate_ref_projection_param_is_bound_from_the_source_not_the_target():
    """Fail-pre/pass-post repro (F2, the completeness half).

    A gate-ref reader's arguments have THREE provenances, and the previous
    revision published all of them as target-owned
    (`args_from_target_params = ... | union(gate_ref_args)`), so the router
    filtered this SOURCE-declared projection param out of `flat_params[target]`.

    Proven here on the production evaluator, in this process: binding `shift`
    from the target's namespace (the pre-fix behaviour, replayed by feeding that
    value into the leaf) CLOSES the gate, binding it from the source's OPENS it,
    and the two differ. The published provenance says SOURCE — matching the
    solve-side fold, which bound this same argument from `flat_params["src"]`.
    """
    flat_params = _shift_flat_params()
    regimes, regime_names_to_ids, solution = _solve_fixture(
        _make_shift_regimes(), flat_params
    )
    src = regimes["src"]
    evaluator = src.gated_edge_simulate_gate_evaluators["target"]
    mappings = _same_period_mappings(regimes, flat_params, solution)

    # The provenance attributes `shift` to the SOURCE, and to nothing else.
    shift_arg = exposed_param_name(evaluator, qname="shift", namespace=SOURCE_PARAMS)
    assert (TARGET_PARAMS, "shift") not in evaluator.arg_provenance.params.values()

    # The fixture is a genuine counterexample only if the two namespaces
    # disagree about the contested qname.
    assert float(flat_params["src"]["shift"]) != float(flat_params["target"]["shift"])

    def _gate(shift_value: float) -> bool:
        return bool(
            np.asarray(
                _call_vmapped_with_accepted_kwargs(
                    evaluator,
                    batched_kwargs={"x": jnp.array([_REALIZED_X])},
                    static_kwargs={
                        shift_arg: jnp.asarray(shift_value),
                        SAME_PERIOD_V_ARG: mappings["target"],
                        SAME_PERIOD_PARAMS_ARG: MappingProxyType(
                            {
                                name: flat_params[name]
                                for name in ("target", "refregime", "fallback")
                            }
                        ),
                    },
                    axis_size=1,
                )
            )[0]
        )

    # V_target(0.6) = 0.6: open against the source's 0.1, closed against the
    # target's 0.9. Fail-pre / pass-post, on the same evaluator.
    assert _gate(_SRC_SHIFT)
    assert not _gate(_TARGET_SHIFT)

    # The production router binds it from the SOURCE, so the row is routed to
    # the target — not to the fallback the pre-fix target-binding selected.
    next_states = MappingProxyType(
        {
            "target": MappingProxyType({"x": jnp.array([_REALIZED_X])}),
            "refregime": MappingProxyType({"x": jnp.array([0.0])}),
            "fallback": MappingProxyType({"x": jnp.array([-999.0])}),
        }
    )
    _states, routed_ids = route_gated_edges(
        regime=src,
        same_period_mappings=mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.array(
            [regime_names_to_ids["target"]], dtype=jnp.int32
        ),
        subjects_in_regime=jnp.array([True]),
        flat_params=flat_params,
    )
    np.testing.assert_array_equal(
        np.asarray(routed_ids), [regime_names_to_ids["target"]]
    )
    assert int(np.asarray(routed_ids)[0]) != int(regime_names_to_ids["fallback"])


def test_gate_ref_projection_param_absent_from_the_target_still_routes():
    """Fail-pre/pass-post repro (F2, the crash half).

    The misclassification was only SILENT because the target happened to declare
    the same qname. A target that does not — the ordinary case, since `shift`
    belongs to the source's edge declaration and nothing obliges the target to
    have heard of it — made the pre-fix filter (`{name: value for name, value in
    flat_params[target].items() if name in from_target}`) yield nothing, and the
    evaluator was then called without a required argument. A valid topology
    CRASHED. The source-bound provenance simply routes it.
    """
    flat_params = _shift_flat_params(target_declares_shift=False)
    regimes, regime_names_to_ids, solution = _solve_fixture(
        _make_shift_regimes(), flat_params
    )
    src = regimes["src"]
    evaluator = src.gated_edge_simulate_gate_evaluators["target"]
    mappings = _same_period_mappings(regimes, flat_params, solution)

    assert "shift" not in flat_params["target"]
    shift_arg = exposed_param_name(evaluator, qname="shift", namespace=SOURCE_PARAMS)

    # The PRE-FIX binding, replayed on the production evaluator: `shift` filtered
    # out of the target's (empty) params is simply not passed, and the call dies.
    with pytest.raises(Exception, match="missing required argument"):
        _call_vmapped_with_accepted_kwargs(
            evaluator,
            batched_kwargs={"x": jnp.array([_REALIZED_X])},
            static_kwargs={
                SAME_PERIOD_V_ARG: mappings["target"],
                SAME_PERIOD_PARAMS_ARG: MappingProxyType(
                    {
                        name: flat_params[name]
                        for name in ("target", "refregime", "fallback")
                    }
                ),
            },
            axis_size=1,
        )

    # ...and the provenance binder finds it in the namespace that owns it.
    bound = _bind_provenance_params(
        evaluator.arg_provenance,
        flat_params=flat_params,
        source_name="src",
        target_name="target",
    )
    assert float(bound[shift_arg]) == _SRC_SHIFT

    next_states = MappingProxyType(
        {
            "target": MappingProxyType({"x": jnp.array([_REALIZED_X])}),
            "refregime": MappingProxyType({"x": jnp.array([0.0])}),
            "fallback": MappingProxyType({"x": jnp.array([-999.0])}),
        }
    )
    _states, routed_ids = route_gated_edges(
        regime=src,
        same_period_mappings=mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.array(
            [regime_names_to_ids["target"]], dtype=jnp.int32
        ),
        subjects_in_regime=jnp.array([True]),
        flat_params=flat_params,
    )
    np.testing.assert_array_equal(
        np.asarray(routed_ids), [regime_names_to_ids["target"]]
    )


# ----------------------------------------------------------------------------------
# F3: the fallback projector must project the coordinate the FOLD projected.
# ----------------------------------------------------------------------------------

_PROJ_SRC_SHIFT = 1.0
_PROJ_TARGET_SHIFT = 9.0


def _old_style_call(func, kwargs):
    """The PRE-FIX router's projector call, for the fail-pre proof.

    A local copy of the removed `gated_routing._call_with_accepted_kwargs`
    (which the router used as
    `_call_with_accepted_kwargs(projector, {**candidate_target_states,
    **flat_params[target_name]})`), kept here rather than in production: with the
    router now binding by provenance, that helper has no caller left, and dead
    engine code retained only to serve a test is worse than four lines here.
    Mirrors `_old_select_own_leg` in
    `test_simulate_gate_param_and_leg_selection.py`.
    """
    accepted = set(signature(func).parameters)
    return func(**{name: value for name, value in kwargs.items() if name in accepted})


def _project_x_plus_shift(x: ContinuousState, shift: FloatND) -> FloatND:
    """The leg's fallback projection: `z = x + shift`, `shift` the SOURCE's."""
    return x + shift


def _always_closed_gate(V_target: FloatND) -> BoolND:
    """A gate that is CLOSED on the whole target grid.

    `Wbar` then equals the CLOSED branch everywhere — `V_fallback(pi(x))` — and
    with `V_fallback(z) = z` (linear, so interpolated exactly) the fold's own
    `Wbar` array literally IS the coordinate the fold projected. That is what
    makes the solve-vs-simulate equality below an observation rather than a
    reimplementation: nothing here recomputes `pi`.
    """
    return V_target < -1e10


def _u_fallback_identity(z: ContinuousState) -> FloatND:
    return z


def _make_projector_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_always_closed_gate,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="fallback", projection={"z": _project_x_plus_shift}
                        )
                    )
                },
            )
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"z": LinSpacedGrid(start=0.0, stop=10.0, n_points=11)},
        functions={"utility": _u_fallback_identity},
    )
    return {"src": src, "target": target, "fallback": fallback}


def _projector_flat_params():
    return MappingProxyType(
        {
            "src": MappingProxyType(
                {
                    "H__discount_factor": jnp.asarray(_BETA),
                    "shift": jnp.asarray(_PROJ_SRC_SHIFT),
                }
            ),
            "target": MappingProxyType({"shift": jnp.asarray(_PROJ_TARGET_SHIFT)}),
            "fallback": MappingProxyType({}),
        }
    )


def test_simulate_projector_equals_the_solve_folds_projected_coordinate():
    """Fail-pre/pass-post repro (F3) — the actual contract.

    The fold's `Wbar` (gate closed everywhere, `V_fallback(z) = z`) IS the
    coordinate the SOLVE side projected, cell by cell. The simulate-side
    projector must produce the same number at the same target state; it did not,
    because the router bound `shift` from the target while the fold bound it from
    the source. The pre-fix call is replayed below — `{**states,
    **flat_params[target]}`, filtered to the projector's signature — and
    disagrees.
    """
    flat_params = _projector_flat_params()
    regimes, _ids, solution = _solve_fixture(_make_projector_regimes(), flat_params)
    projector = regimes["src"].gated_edge_leg_projectors["target"][0]

    # The two namespaces genuinely disagree about `shift`.
    assert float(flat_params["src"]["shift"]) != float(flat_params["target"]["shift"])

    # What SOLVE projected, read off the fold's own output: Wbar on the target's
    # {0, 1} grid.
    edge_wbar = np.asarray(_same_period_wbar(regimes, flat_params, solution))
    np.testing.assert_allclose(
        edge_wbar, [0.0 + _PROJ_SRC_SHIFT, 1.0 + _PROJ_SRC_SHIFT], atol=1e-6
    )

    # What SIMULATE projects at those same target states, through the production
    # router's own binding.
    target_states = {"x": jnp.array([0.0, 1.0])}
    simulated = projector(
        **{name: target_states[name] for name in projector.arg_provenance.states},
        **_bind_provenance_params(
            projector.arg_provenance,
            flat_params=flat_params,
            source_name="src",
            target_name="target",
        ),
    )
    # THE CONTRACT: solve's projected coordinate == simulate's, at every cell.
    np.testing.assert_allclose(np.asarray(simulated["z"]), edge_wbar, atol=1e-6)

    # The PRE-FIX recipe, replayed on the production projector: the removed
    # `target_kwargs = {**candidate_target_states, **flat_params[target_name]}`,
    # filtered to the projector's signature exactly as the removed
    # `_call_with_accepted_kwargs` did. It disagrees with the fold by (9.0 - 1.0).
    old_style = _old_style_call(projector, {**target_states, **flat_params["target"]})
    np.testing.assert_allclose(
        np.asarray(old_style["z"]),
        [0.0 + _PROJ_TARGET_SHIFT, 1.0 + _PROJ_TARGET_SHIFT],
        atol=1e-6,
    )
    assert not np.allclose(np.asarray(old_style["z"]), edge_wbar), (
        "The pre-fix binding must genuinely disagree with the fold, or this is "
        "not a repro."
    )


def _same_period_wbar(regimes, flat_params, solution):
    """The edge's folded `Wbar` on the target grid, from the production fold."""
    substituted, _mappings = substitute_gated_edge_continuations(
        regime=regimes["src"],
        regime_name="src",
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces={
            name: regime.solution.state_action_space(regime_params=flat_params[name])
            for name, regime in regimes.items()
        },
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=MappingProxyType({}),
        flat_params=flat_params,
    )
    return substituted["target"]


def test_router_writes_the_fold_consistent_fallback_state():
    """Fail-pre/pass-post repro (F3) at the router level.

    The end of the causal chain the equality above pins: the row enters the
    fallback regime, and the state it enters with must be the one the solved
    policy priced (`x + 1.0`), not the target-bound `x + 9.0` the pre-fix router
    stored and carried into the next period. The pre-fix number is asserted to
    be a DIFFERENT one, so this is not a pin on an arbitrary value.
    """
    flat_params = _projector_flat_params()
    regimes, regime_names_to_ids, solution = _solve_fixture(
        _make_projector_regimes(), flat_params
    )
    src = regimes["src"]
    mappings = _same_period_mappings(regimes, flat_params, solution)

    next_states = MappingProxyType(
        {
            "target": MappingProxyType({"x": jnp.array([_REALIZED_X])}),
            "fallback": MappingProxyType({"z": jnp.array([-999.0])}),
        }
    )
    states, routed_ids = route_gated_edges(
        regime=src,
        same_period_mappings=mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.array(
            [regime_names_to_ids["target"]], dtype=jnp.int32
        ),
        subjects_in_regime=jnp.array([True]),
        flat_params=flat_params,
    )
    # Gate closed by construction -> the row goes to the fallback...
    np.testing.assert_array_equal(
        np.asarray(routed_ids), [regime_names_to_ids["fallback"]]
    )
    # ...at the SOURCE-shifted coordinate the fold projected.
    np.testing.assert_allclose(
        np.asarray(states["fallback"]["z"]), [_REALIZED_X + _PROJ_SRC_SHIFT], atol=1e-6
    )
    assert not np.allclose(
        np.asarray(states["fallback"]["z"]), [_REALIZED_X + _PROJ_TARGET_SHIFT]
    )


# ----------------------------------------------------------------------------------
# F4: a REFERENCE regime's runtime irregular grid.
# ----------------------------------------------------------------------------------

# The reference regime's own points, and the SOURCE's identically named ones.
# Both regimes declare a runtime-points `IrregSpacedGrid`, so both flat-param
# mappings carry a key literally named `x__points` / `z__points`.
_REF_POINTS = (0.0, 1.0)
_SRC_POINTS = (0.0, 10.0)


def test_prefixed_reference_grid_param_is_satisfiable_by_no_regime():
    """Why F4 CRASHED, from production code alone (the fail-pre proof).

    `get_V_interpolator(state_prefix=_REF_STATE_PREFIX)` derives a runtime
    irregular grid's helper name from the COORDINATE variable it was handed, so
    it asks for `__same_period_ref__x__points`. The reader used to expose that
    name verbatim as an outer argument — and no params template anywhere emits
    it: `_add_runtime_grid_params` names the very same quantity `x__points`, in
    the reference regime's own template. The argument was unsatisfiable by every
    regime in the model, which is exactly the missing-argument error all four
    consumers of the reader raised.
    """
    regimes_dict = _make_ref_grid_regimes()
    flat_params = _ref_grid_flat_params()
    regimes, _ids, _solution = _solve_fixture(regimes_dict, flat_params)

    interpolator = get_V_interpolator(
        v_interpolation_info=create_v_interpolation_info(regimes_dict["refregime"]),
        state_prefix=_REF_STATE_PREFIX,
        V_arr_name="__v__",
    )
    prefixed = f"{_REF_STATE_PREFIX}x__points"
    assert prefixed in get_union_of_args([interpolator])

    # No regime's params carry it — not the reference's, not the reader's own.
    for regime_name, params in flat_params.items():
        assert prefixed not in params, regime_name
    # The reference regime DOES carry the unprefixed qname; separating the
    # coordinate variable from the parameter qname is the whole fix.
    assert "x__points" in flat_params["refregime"]

    # And the production reader no longer exposes the unsatisfiable name.
    fold = regimes["src"].gated_edge_folds["target"]
    assert prefixed not in get_union_of_args([fold])
    assert SAME_PERIOD_PARAMS_ARG in get_union_of_args([fold])


def _gate_ref_only(ref_v: FloatND) -> BoolND:
    """`V_ref` read at the projected coordinate 0.6, on the REFERENCE's grid.

    `V_ref(x) = x`, so on the reference's own points (0, 1) the read is 0.6 and
    the gate OPENS; on the source's identically named (0, 10) the coordinate
    collapses to 0.06 and it CLOSES.
    """
    return ref_v > 0.5


def _project_realized(x: ContinuousState) -> FloatND:
    return jnp.full_like(jnp.asarray(x, dtype=float), _REALIZED_X)


def _make_ref_grid_regimes() -> dict[str, Regime]:
    """The gate ref's reference regime carries a RUNTIME irregular grid, and the
    source declares an identically named state on a DIFFERENT runtime grid."""
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": IrregSpacedGrid(n_points=2)},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_gate_ref_only,
                gate_refs={
                    "ref_v": SamePeriodRef(
                        regime="refregime", projection={"x": _project_realized}
                    )
                },
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
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    refregime = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": IrregSpacedGrid(n_points=2)},
        functions={"utility": _u_identity},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    return {"src": src, "target": target, "refregime": refregime, "fallback": fallback}


def _ref_grid_flat_params():
    return MappingProxyType(
        {
            "src": MappingProxyType(
                {
                    "H__discount_factor": jnp.asarray(_BETA),
                    # The collision: the SOURCE's own `x` grid points, named
                    # exactly like the reference regime's.
                    "x__points": jnp.asarray(_SRC_POINTS),
                }
            ),
            "target": MappingProxyType({}),
            "refregime": MappingProxyType({"x__points": jnp.asarray(_REF_POINTS)}),
            "fallback": MappingProxyType({}),
        }
    )


def test_gate_ref_reads_the_reference_regimes_own_runtime_grid():
    """Fail-pre/pass-post (F4 + the reference-provenance half).

    Pre-fix this topology could not run at all (see
    `test_prefixed_reference_grid_param_is_satisfiable_by_no_regime`). Post-fix
    it solves and routes — and the reader interpolates on the REFERENCE
    regime's points, not on the source's identically named ones: binding the
    reference's grid from the reading regime's namespace (the merge that F4's
    prefix-stripping would have invited, and the reason F4 does not subsume F2)
    is replayed here by handing the reader a params mapping in which
    `refregime`'s points are the source's, and it returns the OPPOSITE gate.
    """
    flat_params = _ref_grid_flat_params()
    regimes, regime_names_to_ids, solution = _solve_fixture(
        _make_ref_grid_regimes(), flat_params
    )
    src = regimes["src"]
    evaluator = src.gated_edge_simulate_gate_evaluators["target"]
    mappings = _same_period_mappings(regimes, flat_params, solution)

    # The two namespaces genuinely disagree about `x__points`.
    assert not np.array_equal(
        np.asarray(flat_params["refregime"]["x__points"]),
        np.asarray(flat_params["src"]["x__points"]),
    )

    def _gate(ref_points) -> bool:
        return bool(
            np.asarray(
                _call_vmapped_with_accepted_kwargs(
                    evaluator,
                    batched_kwargs={"x": jnp.array([_REALIZED_X])},
                    static_kwargs={
                        SAME_PERIOD_V_ARG: mappings["target"],
                        SAME_PERIOD_PARAMS_ARG: MappingProxyType(
                            {
                                "target": flat_params["target"],
                                "refregime": MappingProxyType(
                                    {"x__points": jnp.asarray(ref_points)}
                                ),
                                "fallback": flat_params["fallback"],
                            }
                        ),
                    },
                    axis_size=1,
                )
            )[0]
        )

    assert _gate(_REF_POINTS), "0.6 on the reference's own (0, 1) -> V_ref = 0.6"
    assert not _gate(_SRC_POINTS), (
        "0.6 on the source's (0, 10) collapses to coordinate 0.06 -> V_ref ~ "
        "0.06; the two namespaces must genuinely disagree here, or the "
        "reference-provenance half of this test is vacuous."
    )

    # The production router picks the reference regime's own points.
    next_states = MappingProxyType(
        {
            "target": MappingProxyType({"x": jnp.array([_REALIZED_X])}),
            "refregime": MappingProxyType({"x": jnp.array([0.0])}),
            "fallback": MappingProxyType({"x": jnp.array([-999.0])}),
        }
    )
    _states, routed_ids = route_gated_edges(
        regime=src,
        same_period_mappings=mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.array(
            [regime_names_to_ids["target"]], dtype=jnp.int32
        ),
        subjects_in_regime=jnp.array([True]),
        flat_params=flat_params,
    )
    np.testing.assert_array_equal(
        np.asarray(routed_ids), [regime_names_to_ids["target"]]
    )


def _u_fallback_z(z: ContinuousState) -> FloatND:
    return z


def _make_fallback_grid_regimes() -> dict[str, Regime]:
    """The LEG FALLBACK's regime carries a runtime irregular grid — the reader
    `get_edge_fold` builds for the CLOSED branch (a third consumer of
    `_build_same_period_ref_reader`, on the solve side)."""
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_always_closed_gate,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="fallback", projection={"z": _identity_x_to_z}
                        )
                    )
                },
            )
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"z": IrregSpacedGrid(n_points=3)},
        functions={"utility": _u_fallback_z},
    )
    return {"src": src, "target": target, "fallback": fallback}


def _identity_x_to_z(x: ContinuousState) -> FloatND:
    return x


# The fallback regime's own (irregular!) points. `V_fallback(z) = z` is linear,
# but the GRID is not uniform, so reading it at 0.5 on these points and reading
# it at 0.5 on a different regime's points give different coordinates — and the
# value is exact either way (a linear function on any grid), which keeps the
# comparison a clean equality.
_FALLBACK_POINTS = (0.0, 0.25, 1.0)


def test_leg_fallback_reader_reads_the_fallback_regimes_own_runtime_grid():
    """Fail-pre/pass-post (F4, the solve-side leg-fallback reader).

    The fourth consumer of `_build_same_period_ref_reader`, and the one that runs
    at SOLVE time inside `get_edge_fold`. With the gate closed everywhere, `Wbar`
    IS this reader's output: `V_fallback(pi(x)) = pi(x) = x`, exactly, whatever
    the fallback's grid spacing — but only if the reader interpolated on the
    FALLBACK's own runtime points. Pre-fix the fold could not be called at all
    (its signature carried `__same_period_ref__z__points`, which nothing
    supplies).
    """
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "target": MappingProxyType({}),
            "fallback": MappingProxyType({"z__points": jnp.asarray(_FALLBACK_POINTS)}),
        }
    )
    regimes, _ids, solution = _solve_fixture(_make_fallback_grid_regimes(), flat_params)

    prefixed = f"{_REF_STATE_PREFIX}z__points"
    fold = regimes["src"].gated_edge_folds["target"]
    assert prefixed not in get_union_of_args([fold])

    wbar = np.asarray(_same_period_wbar(regimes, flat_params, solution))
    # Wbar = V_fallback(x) = x on the target's {0, 1} grid — exact only because
    # the read used the fallback's own irregular points.
    np.testing.assert_allclose(wbar, [0.0, 1.0], atol=1e-6)


# ----------------------------------------------------------------------------------
# F4: the ORDINARY E2 same-period ref (the reader's fourth consumer, and the
# only one that runs inside a regime's own compiled kernel).
# ----------------------------------------------------------------------------------

# The reference regime `single_f`'s own points, and the READING regime's
# identically named ones. Both declare a state `wage` on a runtime irregular
# grid, so both flat-param mappings carry a key literally named `wage__points`.
_SINGLE_POINTS = (0.0, 10.0)
_MARRIED_POINTS = (0.0, 1.0)


def _u_wage(wage: ContinuousState) -> FloatND:
    return wage


def _u_married_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 5.0 + 0.0 * wage * work


def _u_married_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.0 * wage * work


def _ir_f(Q_f: FloatND, V_single_f_ref: FloatND) -> BoolND:
    """`Q_f = 5` against the reference value read at the married node.

    At the married node `wage = 1.0`, projected onto `single_f`'s own points
    (0, 10), the read is coordinate 0.1 -> `V_single_f = 1.0`, so the constraint
    holds and the cell is feasible. Read on the MARRIED regime's identically
    named points (0, 1) instead, the same value lands on coordinate 1.0 ->
    `V_single_f = 10.0`, the mask empties, and the cell publishes `D = True`
    with `V = -inf`. Two very different models, one qname.
    """
    return Q_f >= V_single_f_ref


def _project_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _make_e2_ref_grid_regimes() -> dict[str, Regime]:
    single_f = Regime(
        transition=None,
        active=lambda age: age < 1,
        states={"wage": IrregSpacedGrid(n_points=2)},
        functions={"utility": _u_wage},
    )
    married = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": IrregSpacedGrid(n_points=2)},
        state_transitions={"wage": _identity_x_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_married_f, "utility_m": _u_married_m},
        value_constraints={"ir_f": _ir_f},
        same_period_refs={
            "V_single_f_ref": SamePeriodRef(
                regime="single_f", projection={"wage": _project_wage}
            )
        },
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": IrregSpacedGrid(n_points=2)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_married_m, "utility_m": _u_married_m},
    )
    return {
        "single_f": single_f,
        "married": married,
        "married_terminal": married_terminal,
    }


def _identity_x_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def test_e2_same_period_ref_reads_the_reference_regimes_own_runtime_grid():
    """Fail-pre/pass-post (F4, the ordinary E2 consumer).

    The one consumer of `_build_same_period_ref_reader` that lives inside a
    regime's own compiled kernel, which is why its reference-grid params could
    not simply be added to that kernel's arguments: they belong to ANOTHER
    regime's namespace, and the kernel is splatted with only its own. They ride
    in `SAME_PERIOD_PARAMS_ARG` beside the same-period V arrays instead.

    Pre-fix this topology raised a missing-argument error for
    `__same_period_ref__wage__points`. Post-fix it solves, and the number proves
    WHICH namespace won: the reference regime's points, not the reading regime's
    identically named ones (which would empty the mask — the assertion below is
    the difference between a feasible cell and a dissolved household).
    """
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes_dict = _make_e2_ref_grid_regimes()
    flat_params = MappingProxyType(
        {
            "single_f": MappingProxyType({"wage__points": jnp.asarray(_SINGLE_POINTS)}),
            "married": MappingProxyType(
                {
                    "H__discount_factor": jnp.asarray(_BETA),
                    "wage__points": jnp.asarray(_MARRIED_POINTS),
                }
            ),
            "married_terminal": MappingProxyType(
                {"wage__points": jnp.asarray(_MARRIED_POINTS)}
            ),
        }
    )
    # The two namespaces genuinely disagree about the contested qname.
    assert not np.array_equal(
        np.asarray(flat_params["single_f"]["wage__points"]),
        np.asarray(flat_params["married"]["wage__points"]),
    )

    regimes, _ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )
    solution, _sim_policies, dissolution_flags = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )

    # V_single_f = wage on ITS OWN points (0, 10).
    np.testing.assert_allclose(np.asarray(solution[0]["single_f"]), [0.0, 10.0])

    # At married's node wage = 1.0 the reference read is 1.0 (coordinate 0.1 on
    # single_f's points) -> `5 >= 1` holds -> feasible, V_f = 5, D = False.
    # Had the read used married's own points, it would have been 10.0 -> `5 >=
    # 10` fails -> empty mask -> D = True and V_f = -inf.
    V_married = np.asarray(solution[0]["married"])
    D_married = np.asarray(dissolution_flags[0]["married"])
    np.testing.assert_allclose(V_married[1, 0], 5.0)
    assert not bool(D_married[1])
    assert np.isfinite(V_married[1, 0])

    # The fixture is only a repro if the contested binding actually decides the
    # outcome, so exercise the production reader itself on the SAME solved
    # `V_single_f` and vary ONLY which regime's points resolve its grid.
    reader = _build_same_period_ref_reader(
        ref=ResolvedSamePeriodRef(
            regime="single_f",
            projection={"wage": _project_wage},
            stakeholder_index=None,
        ),
        v_interpolation_info=create_v_interpolation_info(regimes_dict["single_f"]),
        functions=MappingProxyType({}),
        deterministic_transitions=MappingProxyType({}),
    )
    same_period_V = MappingProxyType({"single_f": solution[0]["single_f"]})

    def _ref_value(points) -> float:
        return float(
            reader(
                wage=jnp.asarray(1.0),
                **{
                    SAME_PERIOD_V_ARG: same_period_V,
                    SAME_PERIOD_PARAMS_ARG: MappingProxyType(
                        {
                            "single_f": MappingProxyType(
                                {"wage__points": jnp.asarray(points)}
                            )
                        }
                    ),
                },
            )
        )

    assert _ref_value(_SINGLE_POINTS) == 1.0  # the reference regime's own grid
    assert _ref_value(_MARRIED_POINTS) == 10.0  # the reading regime's — the bug
    # ...and 10.0 is exactly the value that empties the mask (`5 >= 10` fails),
    # so the solved `D = False` above is the reference regime's grid winning and
    # nothing else.
    assert _ref_value(_MARRIED_POINTS) > 5.0
    assert _ref_value(_SINGLE_POINTS) <= 5.0


# ----------------------------------------------------------------------------------
# Round-4 audit F2: a param introduced by the TARGET regime's OWN functions, read
# by a source-declared gate, is mis-owned as source (and would collapse with a
# same-named source param). Origin-preserving edge compilation is deferred, so the
# builder FENCES this topology instead of silently misbinding it.
# ----------------------------------------------------------------------------------
_HELPER_TARGET_SCALE = 0.9


def _target_scaled_x(x: ContinuousState, target_scale: FloatND) -> FloatND:
    """A helper declared in the TARGET regime's functions.

    `target_scale` is a parameter the TARGET regime binds from
    `flat_params[target]` — NOT a parameter the source edge declares. The current
    collective-edge provenance binds every non-injected gate argument from
    `flat_params[source]`, so this leaf would be evaluated from the wrong
    namespace; the builder must reject the topology rather than misbind it.
    """
    return x * target_scale


def _gate_reads_target_helper(V_target: FloatND, target_scaled_x: FloatND) -> BoolND:
    """A source gate that routes through the target regime's `target_scaled_x`."""
    return V_target > target_scaled_x


def _make_target_helper_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_gate_reads_target_helper,
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
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity, "target_scaled_x": _target_scaled_x},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    return {"src": src, "target": target, "fallback": fallback}


def test_gate_reaching_a_target_function_param_is_rejected_not_misbound():
    """Round-4 F2 fence: building an edge whose gate reads a target-regime function
    with a free dynamic parameter must raise, rather than silently binding that
    parameter from the source namespace."""
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "target": MappingProxyType(
                {"target_scale": jnp.asarray(_HELPER_TARGET_SCALE)}
            ),
            "fallback": MappingProxyType({}),
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"target_scale.*introduced by the TARGET regime's own functions",
    ):
        _solve_fixture(_make_target_helper_regimes(), flat_params)


# ----------------------------------------------------------------------------------
# Round-5 audit. The round-4 fence covered only the concatenated gate predicate and
# was keyed on GLOBAL target-DAG leaf names, so three topologies slipped through:
#   F1 - a gate-REFERENCE projection reaches a target helper param (never fenced:
#        the readers are compiled on a separate path from the gate predicate).
#   F2 - the fence over-rejects a valid DIRECT source param merely because an
#        UNRELATED target helper reuses the qname (global union, not the consumer's
#        own ancestor closure).
#   F3 - a target function/transition NODE whose name collides with an injected
#        gate-ref key shadows the injected reference value in the concatenated DAG.
# The fix makes the fence ancestry-aware (seeded on each consumer's OWN args) and
# applies it to every gate-ref / fallback projection, plus an injected-name
# collision guard. Fixtures give the two candidate bindings DIFFERENT values so the
# misbinding each finding describes is a genuine repro, not a coincidence.
# ----------------------------------------------------------------------------------


def _project_through_target_helper(target_scaled_x: FloatND) -> FloatND:
    """A gate-REFERENCE projection routed through the target regime's own helper.

    The projected coordinate is `target_scaled_x(x, target_scale)`, so
    `target_scale` is a TARGET-owned param reached by a gate-ref READER — the
    fourth target-DAG-concatenating path the round-4 direct-gate fence never
    inspected (round-5 F1).
    """
    return target_scaled_x


def _gate_ref_value_only(V_target: FloatND, scaled_ref: FloatND) -> BoolND:
    return V_target > scaled_ref


def _make_gate_ref_target_helper_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_gate_ref_value_only,
                gate_refs={
                    "scaled_ref": SamePeriodRef(
                        regime="refregime",
                        projection={"x": _project_through_target_helper},
                    )
                },
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
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity, "target_scaled_x": _target_scaled_x},
    )
    refregime = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    return {"src": src, "target": target, "refregime": refregime, "fallback": fallback}


def test_gate_ref_projection_reaching_a_target_param_is_rejected_not_misbound():
    """Round-5 F1: a gate-ref projection reaching a target helper's param must raise.

    The round-4 fence only inspected the concatenated gate predicate; a gate-ref
    reader is compiled separately (`_build_same_period_ref_reader`) and its args
    were classified source-owned, so the target-owned `target_scale` was bound
    from the source in both the solve fold and the simulate gate. Fail-pre the
    edge builds silently; post-fix construction raises.
    """
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "target": MappingProxyType(
                {"target_scale": jnp.asarray(_HELPER_TARGET_SCALE)}
            ),
            "refregime": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"target_scale.*introduced by the TARGET regime's own functions",
    ):
        _solve_fixture(_make_gate_ref_target_helper_regimes(), flat_params)


# F2: the fence must be ancestry-aware — it must not reject a valid direct source
# param merely because an UNRELATED target helper reuses the name. This is a
# property of the leaf-set computation itself, so it is pinned as a unit test on
# `_reached_target_param_leaves` (a full-solve fixture would instead exercise
# pylcm's function-param qualification `helper__param`, which cannot collide with
# a bare source qname and so cannot reproduce the finding at all).
def _reached_helper(x: ContinuousState, target_scale: FloatND) -> FloatND:
    """A target node the consumer DOES reach — contributes `target_scale`."""
    return x * target_scale


def _unrelated_helper(y: ContinuousState, shift: FloatND) -> FloatND:
    """A target node the consumer does NOT reach — its `shift` must stay clean."""
    return y * shift


def test_fence_leaf_set_is_ancestry_aware_not_global_name_matching():
    """Round-5 F2: the fence returns only the target params a consumer REACHES.

    The round-4 fence unioned the free args of every target-DAG function and
    rejected on a bare name match, so a gate declaring `shift` directly was
    rejected merely because an unrelated target helper also had a `shift`. The
    ancestry-aware replacement walks the consumer's own closure: a gate that
    reaches `reached_helper` (hence `target_scale`) but declares `shift` as its
    OWN source param yields exactly `{target_scale}` — never `shift`.
    """
    dag_pool = {
        "reached_helper": _reached_helper,
        "unrelated_helper": _unrelated_helper,
    }
    state_names = frozenset({"x", "y"})

    # A gate reaching `reached_helper` and declaring `shift` directly.
    reached = _reached_target_param_leaves(
        dag_pool, ("V_target", "reached_helper", "shift"), state_names
    )
    assert reached == frozenset({"target_scale"})
    assert "shift" not in reached  # the unrelated helper's param is NOT contested

    # A gate declaring only `shift` directly reaches no target node at all.
    assert (
        _reached_target_param_leaves(dag_pool, ("V_target", "shift"), state_names)
        == frozenset()
    )

    # Fail-pre proof: the removed global-union fence WOULD have flagged `shift`,
    # because `unrelated_helper` contributes it to the whole-pool leaf set.
    global_leaves: set[str] = set()
    for fn in dag_pool.values():
        global_leaves |= set(get_union_of_args([fn]))
    global_leaves -= set(dag_pool) | state_names
    assert "shift" in global_leaves


# F3: an injected gate-ref key that collides with a target function name must be
# rejected, not silently shadowed by the target node in the concatenated DAG.
def _target_outside(x: ContinuousState) -> FloatND:
    return jnp.full_like(jnp.asarray(x, dtype=float), 0.9)


def _gate_reads_outside(V_target: FloatND, outside: FloatND) -> BoolND:
    return V_target > outside


def _make_gate_ref_name_collision_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_gate_reads_outside,
                gate_refs={
                    # Injected operand named exactly like the target's `outside`
                    # function below: the concatenated DAG resolves the gate's
                    # `outside` arg to the target NODE (0.9), not this ref (~0.6).
                    "outside": SamePeriodRef(
                        regime="refregime", projection={"x": _project_realized}
                    )
                },
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
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity, "outside": _target_outside},
    )
    refregime = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    return {"src": src, "target": target, "refregime": refregime, "fallback": fallback}


def test_injected_gate_ref_name_colliding_with_a_target_node_is_rejected():
    """Round-5 F3: a gate-ref key equal to a target function/transition name must
    raise, rather than let the target node shadow the injected reference value.

    `concatenate_functions({**dag_pool, "__gate__": gate})` resolves the gate's
    `outside` argument to the target's `outside` function (0.9) instead of the
    declared gate-ref (~0.6), silently reversing the gate. Fail-pre the model
    builds with the wrong operand; post-fix construction raises on the collision.
    """
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "target": MappingProxyType({}),
            "refregime": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"outside.*collide",
    ):
        _solve_fixture(_make_gate_ref_name_collision_regimes(), flat_params)


# ----------------------------------------------------------------------------------
# Round-6 audit. Two residual namespace defects survived the round-5 fences:
#   F1 - a gate/projection arg naming a STATE-ONLY target node reaches no dynamic
#        leaf, so `_reject_target_function_params` stays silent -- but name-based
#        concatenation still rebinds the arg to the node and drops a same-named
#        source parameter (a silent gate reversal / wrong projected fallback state).
#   F2 - a gate-ref KEY spelled `V_target` / `D_target` aliases a built-in injected
#        operand; the `injected_names` SET collapses the duplicate and the built-in
#        wins, silently discarding the computed reference value.
# The fixtures give the two candidate bindings different meanings so each is a
# genuine repro, and each fence rejects the topology at construction.
# ----------------------------------------------------------------------------------


def _target_threshold(x: ContinuousState) -> FloatND:
    """A STATE-ONLY target node (reaches no dynamic param) whose name collides with
    a parameter the SOURCE gate declares (`threshold`)."""
    return 0.9 + 0.0 * x


def _gate_reads_shadowed_threshold(V_target: FloatND, threshold: FloatND) -> BoolND:
    """The source MEANS `threshold` as its own param (0.1); the target declares a
    state-only node `threshold(x)=0.9`, which name-based concatenation binds instead,
    silently reversing the gate (round-6 F1)."""
    return V_target > threshold


def _make_threshold_shadow_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_gate_reads_shadowed_threshold,
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
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity, "threshold": _target_threshold},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    return {"src": src, "target": target, "fallback": fallback}


def test_gate_arg_shadowed_by_state_only_target_node_is_rejected():
    """Round-6 F1: a gate arg naming a STATE-ONLY target node must be rejected.

    `_reject_target_function_params` sees no dynamic leaf (the node reads only the
    target state `x`), so it stays silent -- but `concatenate_functions` still binds
    the gate's `threshold` to `_target_threshold`, dropping the source's `threshold`
    parameter and evaluating `V_target > 0.9` where the source meant `> 0.1`. The
    build must raise rather than silently misbind.
    """
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {
                    "H__discount_factor": jnp.asarray(_BETA),
                    "threshold": jnp.asarray(0.1),
                }
            ),
            "target": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"threshold.*TARGET regime's own function",
    ):
        _solve_fixture(_make_threshold_shadow_regimes(), flat_params)


def _gate_uses_v_target(V_target: FloatND) -> BoolND:
    return V_target > 0.5


def _make_gate_ref_v_target_alias_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_gate_uses_v_target,
                gate_refs={
                    # Aliases the built-in target-value operand `V_target`.
                    "V_target": SamePeriodRef(
                        regime="refregime", projection={"x": _identity_x}
                    )
                },
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
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    refregime = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    return {"src": src, "target": target, "refregime": refregime, "fallback": fallback}


def test_gate_ref_key_aliasing_v_target_is_rejected():
    """Round-6 F2: a gate-ref key that aliases a built-in injected operand.

    `injected_names` is a SET, so a `gate_refs` key spelled `V_target` collapses onto
    the target value component; `_assemble_gate_kwargs` resolves the target component
    first and silently discards the computed reference. The build must raise.
    """
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "target": MappingProxyType({}),
            "refregime": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"V_target.*alias a built-in injected gate operand",
    ):
        _solve_fixture(_make_gate_ref_v_target_alias_regimes(), flat_params)


# ----------------------------------------------------------------------------------
# Round-7 F2 - the gate-operand namespace is still not disjoint from target STATE
# names. `_assemble_gate_kwargs` resolves target value component(s) and gate-ref
# values BEFORE the target state mesh, so a target state named `V_target` (or a
# gate-ref key equal to a target state name) is silently preempted -- the gate reads
# the injected value / reference instead of the realized state, reversing routing
# with no error. `_reject_gate_operand_state_name_collision` closes both.
# ----------------------------------------------------------------------------------


def test_reject_gate_operand_state_name_collision_flags_a_value_operand_alias():
    """Unit: a target state named like a built-in value/D operand is rejected."""
    with pytest.raises(
        ModelInitializationError,
        match=r"alias a built-in injected value/D operand",
    ):
        _reject_gate_operand_state_name_collision(
            state_names=("V_target", "x"),
            reserved_operand_names=frozenset({"V_target", "D_target"}),
            gate_ref_names=(),
            edge_target="target",
            context="unit",
        )


def test_reject_gate_operand_state_name_collision_flags_a_gate_ref_alias():
    """Unit: a gate-ref key equal to a target state name is rejected."""
    with pytest.raises(
        ModelInitializationError,
        match=r"alias a gate-ref key",
    ):
        _reject_gate_operand_state_name_collision(
            state_names=("x", "z"),
            reserved_operand_names=frozenset({"V_target", "D_target"}),
            gate_ref_names=("x",),
            edge_target="target",
            context="unit",
        )


def test_reject_gate_operand_state_name_collision_passes_when_disjoint():
    """Unit: disjoint operand/gate-ref/state namespaces raise nothing."""
    _reject_gate_operand_state_name_collision(
        state_names=("x", "z"),
        reserved_operand_names=frozenset({"V_target", "D_target"}),
        gate_ref_names=("ref_v",),
        edge_target="target",
        context="unit",
    )


def _gate_reads_x_operand(V_target: FloatND, x: FloatND) -> BoolND:
    """The source MEANS `x` as the target's realized STATE. A gate-ref keyed `x`
    injects the projected reference value under the SAME name; `_assemble_gate_kwargs`
    resolves the gate ref BEFORE the state mesh, so the gate silently reads the
    reference value instead of the state (round-7 F2)."""
    return V_target > x


def _make_gate_ref_key_aliases_target_state_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _next_x_offgrid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
        gated_edges={
            "target": GatedEdge(
                gate=_gate_reads_x_operand,
                gate_refs={
                    # Aliases the TARGET STATE `x` (not a value/D operand, so the
                    # round-6 gate-ref alias fence stays silent).
                    "x": SamePeriodRef(
                        regime="refregime", projection={"x": _identity_x}
                    )
                },
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
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    refregime = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    return {"src": src, "target": target, "refregime": refregime, "fallback": fallback}


def test_gate_ref_key_aliasing_a_target_state_is_rejected():
    """Round-7 F2 (integration): the builder must wire the state-collision fence.

    A gate-ref key `x` equals the target state `x`. `_reject_gate_ref_operand_alias`
    only checks gate-ref keys against the value/D built-ins, so it stays silent; but
    `_assemble_gate_kwargs` resolves the gate-ref value before the state mesh, so
    `gate(x)` would read the projected reference instead of the realized state and
    silently change routing. Pre-fix the model built; the build must now raise.
    """
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "target": MappingProxyType({}),
            "refregime": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"alias a gate-ref key",
    ):
        _solve_fixture(_make_gate_ref_key_aliases_target_state_regimes(), flat_params)


# ----------------------------------------------------------------------------------
# simulate-round8 F1: a gate/projection arg that is BOTH a TARGET STATE and a
# SOURCE PARAM binds one fold leaf two ways -- solve reads the param
# (`_evaluate_edge_fold` overwrites the state grid), simulate reads the state
# (`_expose` classifies it as a state before recording a source param). The two
# sides then evaluate different gates. `regime_to_flat_param_names[source]` cannot
# catch it at construction: a gate/projection param is bound from a BARE key the
# user adds to `flat_params[source]`, never from the (function-qualified) template.
# The fence therefore runs at solve, where `flat_params` is in hand.
# ----------------------------------------------------------------------------------


def _next_y_identity(y: ContinuousState) -> ContinuousState:
    return y


def _u_src_reads_x_param(
    y: ContinuousState, work: DiscreteAction, x: FloatND
) -> FloatND:
    """Source utility reads param `x` -> `x` is a genuine source param the user
    supplies (bare) in `flat_params['src']`. It ALSO names the target's state."""
    return jnp.zeros_like(y) * work + 0.0 * x


def _u_src_no_param(y: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.zeros_like(y) * work


def _gate_reads_x(V_target: FloatND, x: FloatND) -> BoolND:
    return V_target > x


def _make_gate_param_aliases_target_state_regimes(
    *, source_supplies_x_param: bool
) -> dict[str, Regime]:
    """Source `y`-regime; target state is `x`; the gate reads `x`.

    With `source_supplies_x_param=True` the source utility also reads param `x`,
    so the user supplies a bare `x` in `flat_params['src']` -- the collision. With
    `False` the source never supplies `x`, so `gate(x)` is an unambiguous direct
    read of the target state (the legitimate case that must still solve).
    """
    src = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"y": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"y": _next_y_identity},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": _u_src_reads_x_param
            if source_supplies_x_param
            else _u_src_no_param
        },
        gated_edges={
            "target": GatedEdge(
                gate=_gate_reads_x,
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
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": _u_identity},
    )
    return {"src": src, "target": target, "fallback": fallback}


def test_gate_param_aliasing_a_target_state_and_source_param_is_rejected():
    """simulate-round8 F1: the solve-time fence rejects the double-bound leaf.

    Pre-fix the model solved silently, with the solve-side `Wbar` reading the
    source param `x=0.9` (`_evaluate_edge_fold` overwrites the state grid) and the
    simulate router reading the realized target state instead -- two different
    gates for one edge. `x` is a genuine source param (the source utility reads
    it), supplied bare in `flat_params['src']`, and simultaneously the target
    state name.
    """
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {
                    "H__discount_factor": jnp.asarray(_BETA),
                    "utility__x": jnp.asarray(0.0),
                    "x": jnp.asarray(0.9),
                }
            ),
            "target": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"simultaneously a TARGET state.*and a source parameter",
    ):
        _solve_fixture(
            _make_gate_param_aliases_target_state_regimes(
                source_supplies_x_param=True
            ),
            flat_params,
        )


def test_gate_reading_a_target_state_that_is_not_a_source_param_still_solves():
    """Negative control: a gate reading a target state the source never supplies
    as a param is a legitimate direct state read and must still solve. The fence
    keys on membership in `flat_params[source]`, not on the state name alone."""
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "target": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    # Must not raise.
    _solve_fixture(
        _make_gate_param_aliases_target_state_regimes(source_supplies_x_param=False),
        flat_params,
    )
