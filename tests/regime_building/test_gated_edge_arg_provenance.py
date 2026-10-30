"""Argument PROVENANCE across the regimes a gated edge relates.

One defect class, three symptoms: an argument of an edge-side callable was
bound from a regime that does not own it, because the callable did not record
— and the caller could not tell — which regime does.

The three provenances an edge-side callable mixes, all in one signature:

1. The TARGET's candidate states.
2. The SOURCE's edge params: the gate predicate's own free params, and the free
   params of the source-declared fallback projections. They live in the
   source's edge namespace, `flat_params["edges"][source]` (read through
   `edge_params`), apart from the params of the source regime's own functions in
   `flat_params[source]`. `backward_induction._evaluate_edge_fold` binds every
   param the SOLVE-side fold needs from that edge namespace, so this is not a
   preference between two merges — it is what makes simulate evaluate the same
   object solve folded.
3. Each REFERENCE regime's own interpolation grid, for a leg fallback's,
   same-period reference's or participation read's read of that regime's V.

Covered here:

- **The fallback projector's params.** `route_gated_edges` must project a
  coordinate with the same `flat_params["edges"][source]` the fold projected it
  with.
  Calling the projector with `{**candidate_target_states, **flat_params[target]}`
  instead puts the row in the right fallback REGIME at a STATE the solved policy
  never priced, and carries it into the next period.
- **The reference regime's interpolation params.** A read of another regime's V
  needs that regime's own grid helpers. Exposing them under the PREFIXED
  coordinate name (`__same_period_ref__x__points`) — which no params template
  emits and no caller supplies — makes every reference to a runtime irregular
  grid raise a missing-argument error. Coordinate variables stay prefixed; the
  PARAMS resolve against the reference regime's own namespace
  (`SAME_PERIOD_PARAMS_ARG`).

Value-reading gates and their references are retired in favour of
`ParticipationCheck`; their parameter namespaces are pinned in
`tests/params/test_gate_reads_on_the_target_domain.py`.

Fixture discipline: every one of these is only a repro if the two candidate
bindings genuinely disagree, so each fixture gives the two regimes DIFFERENT
values for the contested name (`_PROJ_SRC_SHIFT` != `_PROJ_TARGET_SHIFT`,
`_REF_POINTS` != `_SRC_POINTS`) and asserts the disagreement itself.
"""

import re
from collections.abc import Mapping
from dataclasses import replace
from inspect import signature
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.params.edges import regime_kernel_params
from _lcm.regime_building.collective import NO_ROLE
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.gated_edges import (
    SOURCE_PARAMS,
    ResolvedProjectedRegimeValue,
    _fence_edge_consumer,
    _with_qualified_params,
)
from _lcm.regime_building.processing import process_regimes
from _lcm.regime_building.Q_and_F import (
    _REF_STATE_PREFIX,
    EDGE_CHANNELS_ARG,
    EDGE_REF_PARAMS_ARG,
    EDGE_REF_V_ARG,
    SAME_PERIOD_PARAMS_ARG,
    SAME_PERIOD_V_ARG,
    _build_same_period_ref_reader,
    evaluate_projected_readers,
)
from _lcm.regime_building.V import create_v_interpolation_info
from _lcm.regime_law import RegimeLawDeclaration
from _lcm.simulation.gated_routing import (
    bind_provenance_params,
    route_gated_edges,
    substitute_gated_edge_continuations,
)
from _lcm.solution.backward_induction import solve
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import get_union_of_args
from _lcm.utils.logging import get_logger
from lcm import (
    AgeRange,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    ParticipationCheck,
    ParticipationRead,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    ValueDependentConstraint,
    categorical,
)
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.koopmans_aggregation import LinearAggregator
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ReferenceName,
    RegimeName,
    ScalarInt,
    UserFunction,
)
from tests.conftest import bind_laws, build_prepared_structure

_BETA = 0.95
_AGES = AgeGrid(start=0, inclusive_stop=2, step="Y")

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


def _u_src(*, x: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.zeros_like(x) * work


def _u_identity(x: ContinuousState) -> FloatND:
    """V(x) = x — linear, hence interpolated EXACTLY on any grid containing the
    read point's bracketing nodes, so every number below is a clean equality
    rather than an interpolation tolerance."""
    return x


# A topology's regimes and the law `Model(edges=...)` binds for each.
type _Spec = tuple[
    dict[str, Regime], dict[RegimeName, RegimeLawDeclaration | Transition]
]


def _split(
    bound: Mapping[str, tuple[Regime, RegimeLawDeclaration | Transition]],
) -> _Spec:
    """Separate each regime from its law between regimes (`None` if terminal)."""
    return (
        {name: regime for name, (regime, _) in bound.items()},
        {name: law for name, (_, law) in bound.items()},
    )


def _process(
    *,
    regimes_dict: Mapping[str, Regime],
    laws: Mapping[RegimeName, RegimeLawDeclaration | Transition],
    ages: AgeGrid,
):
    """Finalize and process the regimes under their bound laws."""
    regime_names_to_ids = MappingProxyType(
        {name: jnp.int32(i) for i, name in enumerate(regimes_dict)}
    )
    bound = bind_laws(laws)
    finalized = finalize_regimes(
        user_regimes=regimes_dict,
        laws=bound,
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )
    regimes = process_regimes(
        prepared_structure=build_prepared_structure(
            user_regimes=finalized, laws=bound, ages=ages
        ),
        user_regimes=finalized,
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=False,
    )
    return regimes, regime_names_to_ids


def _solve_fixture(*, spec: _Spec, flat_params):
    regimes_dict, laws = spec
    regimes, regime_names_to_ids = _process(
        regimes_dict=regimes_dict, laws=laws, ages=_AGES
    )
    _bi_result = solve(
        program_fingerprint="test_gated_edge_arg_provenance",
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    return regimes, regime_names_to_ids, solution


def _same_period_mappings(*, regimes, flat_params, solution):
    base_state_action_spaces = {
        name: regime.solution.state_action_space(regime_params=flat_params[name])
        for name, regime in regimes.items()
    }
    _substituted, mappings = substitute_gated_edge_continuations(
        regime=regimes["src"],
        regime_name="src",
        regimes=regimes,
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=solution,
        flat_params=flat_params,
    )
    return mappings


# The fallback projector must project the coordinate the FOLD projected.

# The route fallback's projection parameter, keyed by its declaration path below
# `params["edges"]["src"]` (target, `routes`, route, `fallback`, projected state,
# parameter), the name both sides of the solve/simulate seam read. The target's
# competing entry is spelled identically so the two namespaces still contest one
# qname.
_PROJ_SHIFT_QNAME = "target__routes__only__fallback__z__shift"
_PROJ_SRC_SHIFT = 1.0
_PROJ_TARGET_SHIFT = 9.0


def _target_bound_call(*, func, kwargs):
    """A projector call bound from the TARGET namespace, for contrast.

    A local copy of a `_call_with_accepted_kwargs` helper of the form
    `_call_with_accepted_kwargs(projector, {**candidate_target_states,
    **flat_params[target_name]})`), kept here rather than in production: with the
    router now binding by provenance, that helper has no caller left, and dead
    engine code retained only to serve a test is worse than four lines here.
    Mirrors `_old_select_own_leg` in
    `test_simulate_gate_param_and_leg_selection.py`.
    """
    accepted = set(signature(func).parameters)
    return func(**{name: value for name, value in kwargs.items() if name in accepted})


def _project_x_plus_shift(*, x: ContinuousState, shift: FloatND) -> FloatND:
    """The leg's fallback projection: `z = x + shift`, `shift` the SOURCE's."""
    return x + shift


def _always_closed_gate(x: ContinuousState) -> BoolND:
    """A gate that is CLOSED on the whole target grid `x in [0, 1]`.

    `Wbar` then equals the CLOSED branch everywhere — `V_fallback(pi(x))` — and
    with `V_fallback(z) = z` (linear, so interpolated exactly) the fold's own
    `Wbar` array literally IS the coordinate the fold projected. That is what
    makes the solve-vs-simulate equality below an observation rather than a
    reimplementation: nothing here recomputes `pi`.
    """
    return x < -1e10


def _u_fallback_identity(z: ContinuousState) -> FloatND:
    return z


def _make_projector_regimes() -> _Spec:
    src = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            state_transitions={"x": _next_x_offgrid},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_src},
        ),
        Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "target": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "target": Gate(
                    predicate=_always_closed_gate,
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="fallback",
                                projection={"z": _project_x_plus_shift},
                            )
                        )
                    },
                )
            },
        ),
    )
    target = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity},
        ),
        None,
    )
    fallback = (
        Regime(
            states={"z": LinSpacedGrid(start=0.0, stop=10.0, n_points=11)},
            functions={"utility": _u_fallback_identity},
        ),
        None,
    )
    return _split({"src": src, "target": target, "fallback": fallback})


def _projector_flat_params():
    return MappingProxyType(
        {
            "src": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "edges": MappingProxyType(
                {
                    "src": MappingProxyType(
                        {_PROJ_SHIFT_QNAME: jnp.asarray(_PROJ_SRC_SHIFT)}
                    )
                }
            ),
            "target": MappingProxyType(
                {_PROJ_SHIFT_QNAME: jnp.asarray(_PROJ_TARGET_SHIFT)}
            ),
            "fallback": MappingProxyType({}),
        }
    )


def test_simulate_projector_equals_the_solve_folds_projected_coordinate():
    """The simulate projector lands on the coordinate the solve fold projected.

    The fold's `Wbar` (gate closed everywhere, `V_fallback(z) = z`) IS the
    coordinate the SOLVE side projected, cell by cell, and the simulate-side
    projector must produce the same number at the same target state. Binding
    `shift` from the target while the fold binds it from the source pulls the
    two apart: that call is run below — `{**states, **flat_params[target]}`,
    filtered to the projector's signature — and disagrees.
    """
    flat_params = _projector_flat_params()
    regimes, _ids, solution = _solve_fixture(
        spec=_make_projector_regimes(), flat_params=flat_params
    )
    projector = regimes["src"].gated_edges["target"].legs[0].fallback_state_projector

    # The two namespaces genuinely disagree about `shift`.
    assert float(flat_params["edges"]["src"][_PROJ_SHIFT_QNAME]) != float(
        flat_params["target"][_PROJ_SHIFT_QNAME]
    )

    # What SOLVE projected, read off the fold's own output: Wbar on the target's
    # {0, 1} grid.
    edge_wbar = np.asarray(
        _same_period_wbar(regimes=regimes, flat_params=flat_params, solution=solution)
    )
    np.testing.assert_allclose(
        edge_wbar, [0.0 + _PROJ_SRC_SHIFT, 1.0 + _PROJ_SRC_SHIFT], atol=1e-6
    )

    # What SIMULATE projects at those same target states, through the production
    # router's own binding.
    target_states = {"x": jnp.array([0.0, 1.0])}
    simulated = projector(
        **{name: target_states[name] for name in projector.arg_provenance.states},
        **bind_provenance_params(
            provenance=projector.arg_provenance,
            flat_params=flat_params,
            source_name="src",
            target_name="target",
        ),
    )
    # THE CONTRACT: solve's projected coordinate == simulate's, at every cell.
    np.testing.assert_allclose(np.asarray(simulated["z"]), edge_wbar, atol=1e-6)

    # The target-bound recipe, on the production projector:
    # `target_kwargs = {**candidate_target_states, **flat_params[target_name]}`,
    # filtered to the projector's signature. It disagrees with the fold by
    # (9.0 - 1.0).
    old_style = _target_bound_call(
        func=projector, kwargs={**target_states, **flat_params["target"]}
    )
    np.testing.assert_allclose(
        np.asarray(old_style["z"]),
        [0.0 + _PROJ_TARGET_SHIFT, 1.0 + _PROJ_TARGET_SHIFT],
        atol=1e-6,
    )
    assert not np.allclose(np.asarray(old_style["z"]), edge_wbar), (
        "The target binding must genuinely disagree with the fold, or this "
        "fixture does not discriminate."
    )


def _same_period_wbar(*, regimes, flat_params, solution):
    """The edge's `Wbar` at the target's own nodes, from the production fold.

    The substituted leaf is the edge's stacked operand channels, which the
    source's continuation reads at the point it lands on and gates there. These
    fixtures land exactly on the target's nodes, so evaluating the edge's own
    combiner at those nodes is that same read, and the result is the value the
    source sees.
    """
    channels = _same_period_channels(
        regimes=regimes, flat_params=flat_params, solution=solution
    )
    edge = regimes["src"].gated_edges["target"]
    target_nodes = regimes["target"].solution.state_action_space(
        regime_params=flat_params["target"]
    )
    supplied = {
        **{name: jnp.asarray(grid) for name, grid in target_nodes.states.items()},
        **regime_kernel_params(flat_params, regime_name="src"),
        "period": jnp.int32(1),
        "age": jnp.asarray(_AGES.period_to_age(1)),
    }
    engine_values = {
        **supplied,
        EDGE_REF_V_ARG: MappingProxyType(dict(solution[1])),
        EDGE_REF_PARAMS_ARG: flat_params,
    }
    edge_fold = edge.combine_at(period=1)
    combine = edge_fold.combine
    # A projected reference is read AT one landing point, and a projection is
    # written for one household — its state is that household's own scalar.
    # The source's kernel supplies the point because it already runs per cell;
    # reproducing the fold at every target node therefore means mapping each
    # reader over the target's grid, which is what production's own dispatcher
    # does around it.
    mapped_readers = tuple(
        replace(
            reader,
            reader=productmap(
                func=reader.reader,
                variables=reader.state_args,
                batch_sizes=dict.fromkeys(reader.state_args, 0),
            ),
        )
        for reader in edge_fold.projected_readers
    )
    projected = evaluate_projected_readers(
        readers=mapped_readers,
        landing_states={
            arg: target_nodes.states[arg]
            for reader in edge_fold.projected_readers
            for arg in reader.state_args
        },
        other_values={
            arg: engine_values[arg]
            for reader in edge_fold.projected_readers
            for arg in reader.other_args
        },
    )
    return combine(
        **{EDGE_CHANNELS_ARG: channels},
        **projected,
        **{
            name: supplied[name]
            for name in get_union_of_args([combine])
            if name != EDGE_CHANNELS_ARG and name not in projected
        },
    )


def _same_period_channels(*, regimes, flat_params, solution):
    """The edge's stacked operand channels on the target grid."""
    substituted, _mappings = substitute_gated_edge_continuations(
        regime=regimes["src"],
        regime_name="src",
        regimes=regimes,
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces={
            name: regime.solution.state_action_space(regime_params=flat_params[name])
            for name, regime in regimes.items()
        },
        period_to_regime_to_V_arr=solution,
        flat_params=flat_params,
    )
    return substituted["target"]


def test_router_writes_the_fold_consistent_fallback_state():
    """The router writes the fold-consistent fallback state.

    The end of the causal chain the equality above pins: the row enters the
    fallback regime, and the state it enters with must be the one the solved
    policy priced (`x + 1.0`), not the target-bound `x + 9.0` that would be
    stored and carried into the next period. That other number is asserted to be
    a DIFFERENT one, so this is not a pin on an arbitrary value.
    """
    flat_params = _projector_flat_params()
    regimes, regime_names_to_ids, solution = _solve_fixture(
        spec=_make_projector_regimes(), flat_params=flat_params
    )
    src = regimes["src"]
    mappings = _same_period_mappings(
        regimes=regimes, flat_params=flat_params, solution=solution
    )

    next_states = MappingProxyType(
        {
            "target": MappingProxyType({"x": jnp.array([_REALIZED_X])}),
            "fallback": MappingProxyType({"z": jnp.array([-999.0])}),
        }
    )
    states, routed_ids, _routed_roles, _closed_masks = route_gated_edges(
        # The source is simulated at period 0, so the gate is decided on
        # the value it would enter at period 1.
        fold_period=1,
        regime=src,
        same_period_mappings=mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.array(
            [regime_names_to_ids["target"]], dtype=jnp.int32
        ),
        subjects_in_regime=jnp.array([True]),
        flat_params=flat_params,
        own_stakeholder=jnp.full_like(jnp.array([True]), NO_ROLE, dtype=jnp.int32),
        new_own_stakeholder=jnp.full_like(jnp.array([True]), NO_ROLE, dtype=jnp.int32),
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


# A REFERENCE regime's runtime irregular grid.

# The reference regime's own points, and the SOURCE's identically named ones.
# Both regimes declare a runtime-points `IrregSpacedGrid`, so both flat-param
# mappings carry a key literally named `x__points` / `z__points`.
_REF_POINTS = (0.0, 1.0)
_SRC_POINTS = (0.0, 10.0)


def _gate_ref_only(ref_v: FloatND) -> BoolND:
    """`V_ref` read at the projected coordinate 0.6, on the REFERENCE's grid.

    `V_ref(x) = x`, so on the reference's own points (0, 1) the read is 0.6 and
    the decision OPENS; on the source's identically named (0, 10) the coordinate
    collapses to 0.06 and it CLOSES.
    """
    return ref_v > 0.5


def _project_realized(x: ContinuousState) -> FloatND:
    return jnp.full_like(jnp.asarray(x, dtype=float), _REALIZED_X)


@categorical(ordered=False)
class _RefGridRegimeId:
    src: ScalarInt
    target: ScalarInt
    refregime: ScalarInt
    fallback: ScalarInt


def _u_zero_x(x: ContinuousState) -> FloatND:
    return 0.0 * x


def test_participation_read_uses_the_reference_regimes_own_runtime_grid() -> None:
    """A participation read interpolates on the REFERENCE regime's own runtime points.

    The source declares an identically named runtime grid on (0, 10). Read on the
    reference's own (0, 1), `V_ref(0.6) = 0.6` opens the decision and the source
    earns `0.95 * V_target(0.6) = 0.57`; read on the source's points the
    coordinate collapses to 0.06, closing it onto the zero-valued fallback.
    """
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    model = Model(
        regimes={
            "src": Regime(
                states={"x": IrregSpacedGrid(n_points=2)},
                state_transitions={"x": _next_x_offgrid},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={"utility": _u_src},
            ),
            "target": Regime(states={"x": grid}, functions={"utility": _u_identity}),
            "refregime": Regime(
                states={"x": IrregSpacedGrid(n_points=2)},
                functions={"utility": _u_identity},
            ),
            "fallback": Regime(states={"x": grid}, functions={"utility": _u_zero_x}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RefGridRegimeId,
        initial_nodes={0: "src"},
        edges={
            "src": Transition(
                law=ByAge(cases={0: {"target": StochasticTransition(func=_prob_one)}}),
                participation={
                    "target": ParticipationCheck(
                        predicate=_gate_ref_only,
                        reads={
                            "ref_v": ParticipationRead(
                                kind="IntegratedFull",
                                regime="refregime",
                                projection={"x": _project_realized},
                            )
                        },
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={"x": _identity_x}
                                )
                            )
                        },
                    )
                },
            )
        },
    )
    solution = model.solve(
        params={
            "discount_factor": _BETA,
            "src": {"x": {"points": jnp.asarray(_SRC_POINTS)}},
            "refregime": {"x": {"points": jnp.asarray(_REF_POINTS)}},
        },
        log_level="off",
    )
    np.testing.assert_allclose(
        np.asarray(solution.values[0]["src"]), [_BETA * _REALIZED_X] * 2, rtol=1e-6
    )


def _u_fallback_z(z: ContinuousState) -> FloatND:
    return z


def _make_fallback_grid_regimes() -> _Spec:
    """The LEG FALLBACK's regime carries a runtime irregular grid — the reader
    `get_edge_fold` builds for the CLOSED branch (a third consumer of
    `_build_same_period_ref_reader`, on the solve side)."""
    src = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            state_transitions={"x": _next_x_offgrid},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_src},
        ),
        Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "target": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "target": Gate(
                    predicate=_always_closed_gate,
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="fallback",
                                projection={"z": _identity_x_to_z},
                            )
                        )
                    },
                )
            },
        ),
    )
    target = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity},
        ),
        None,
    )
    fallback = (
        Regime(
            states={"z": IrregSpacedGrid(n_points=3)},
            functions={"utility": _u_fallback_z},
        ),
        None,
    )
    return _split({"src": src, "target": target, "fallback": fallback})


def _identity_x_to_z(x: ContinuousState) -> FloatND:
    return x


# The fallback regime's own (irregular!) points. `V_fallback(z) = z` is linear,
# but the GRID is not uniform, so reading it at 0.5 on these points and reading
# it at 0.5 on a different regime's points give different coordinates — and the
# value is exact either way (a linear function on any grid), which keeps the
# comparison a clean equality.
_FALLBACK_POINTS = (0.0, 0.25, 1.0)


def test_leg_fallback_reader_reads_the_fallback_regimes_own_runtime_grid():
    """The leg-fallback reader reads the FALLBACK regime's own runtime grid.

    The fourth consumer of `_build_same_period_ref_reader`, and the one that runs
    at SOLVE time inside `get_edge_fold`. With the gate closed everywhere, `Wbar`
    IS this reader's output: `V_fallback(pi(x)) = pi(x) = x`, exactly, whatever
    the fallback's grid spacing — but only if the reader interpolated on the
    FALLBACK's own runtime points. A signature carrying
    `__same_period_ref__z__points`, which nothing supplies, makes the fold
    uncallable.
    """
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "target": MappingProxyType({}),
            "fallback": MappingProxyType({"z__points": jnp.asarray(_FALLBACK_POINTS)}),
        }
    )
    regimes, _ids, solution = _solve_fixture(
        spec=_make_fallback_grid_regimes(), flat_params=flat_params
    )

    prefixed = f"{_REF_STATE_PREFIX}z__points"
    fold = regimes["src"].gated_edges["target"].fold_at(period=1)
    assert prefixed not in get_union_of_args([fold.surfaces])

    wbar = np.asarray(
        _same_period_wbar(regimes=regimes, flat_params=flat_params, solution=solution)
    )
    # Wbar = V_fallback(x) = x on the target's {0, 1} grid — exact only because
    # the read used the fallback's own irregular points.
    np.testing.assert_allclose(wbar, [0.0, 1.0], atol=1e-6)


# The ORDINARY same-period ref (the reader's fourth consumer, and the only one
# that runs inside a regime's own compiled kernel).

# The reference regime `single_f`'s own points, and the READING regime's
# identically named ones. Both declare a state `wage` on a runtime irregular
# grid, so both flat-param mappings carry a key literally named `wage__points`.
_SINGLE_POINTS = (0.0, 10.0)
_MARRIED_POINTS = (0.0, 1.0)


def _u_wage(wage: ContinuousState) -> FloatND:
    return wage


def _u_married_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 5.0 + 0.0 * wage * work


def _u_married_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.0 * wage * work


def _ir_f(*, Q_f: FloatND, V_single_f_ref: FloatND) -> BoolND:
    """`Q_f = 5` against the reference value read at the married node.

    At the married node `wage = 1.0`, projected onto `single_f`'s own points
    (0, 10), the read is coordinate 0.1 -> `V_single_f = 1.0`, so the constraint
    holds and the cell is feasible. Read on the MARRIED regime's identically
    named points (0, 1) instead, the same value lands on coordinate 1.0 ->
    `V_single_f = 10.0`, the mask empties, and the cell publishes `V = -inf`.
    Two very different models, one qname.
    """
    return Q_f >= V_single_f_ref


def _project_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _make_e2_ref_grid_regimes() -> _Spec:
    single_f = (
        Regime(
            states={"wage": IrregSpacedGrid(n_points=2)},
            functions={"utility": _u_wage},
        ),
        None,
    )
    married = (
        Regime(
            states={"wage": IrregSpacedGrid(n_points=2)},
            state_transitions={"wage": _identity_x_wage},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={
                "utility": CollectiveUtility(
                    utilities={"f": _u_married_f, "m": _u_married_m}
                )
            },
            constraints={
                "ir_f": ValueDependentConstraint(
                    predicate=_ir_f,
                    references={
                        "V_single_f_ref": ProjectedRegimeValue(
                            regime="single_f", projection={"wage": _project_wage}
                        )
                    },
                )
            },
        ),
        ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "married_terminal": StochasticTransition(func=_prob_one)
                }
            }
        ),
    )
    married_terminal = (
        Regime(
            states={"wage": IrregSpacedGrid(n_points=2)},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={
                "utility": CollectiveUtility(
                    utilities={"f": _u_married_m, "m": _u_married_m}
                )
            },
        ),
        None,
    )
    return _split(
        {
            "single_f": single_f,
            "married": married,
            "married_terminal": married_terminal,
        }
    )


def _identity_x_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def test_e2_same_period_ref_reads_the_reference_regimes_own_runtime_grid():
    """A same-period ref reads the REFERENCE regime's own runtime grid.

    The one consumer of `_build_same_period_ref_reader` that lives inside a
    regime's own compiled kernel, which is why its reference-grid params could
    not simply be added to that kernel's arguments: they belong to ANOTHER
    regime's namespace, and the kernel is splatted with only its own. They ride
    in `SAME_PERIOD_PARAMS_ARG` beside the same-period V arrays instead.

    A missing-argument error for `__same_period_ref__wage__points` is what an
    unsupplied prefixed param produces. This topology solves instead, and the
    number proves WHICH namespace won: the reference regime's points, not the
    reading regime's
    identically named ones (which would empty the mask — the assertion below is
    the difference between a feasible cell and a dissolved household).
    """
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    regimes_dict, laws = _make_e2_ref_grid_regimes()
    flat_params = MappingProxyType(
        {
            "single_f": MappingProxyType({"wage__points": jnp.asarray(_SINGLE_POINTS)}),
            "married": MappingProxyType(
                {
                    "koopmans_aggregator__discount_factor": jnp.asarray(_BETA),
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

    regimes, _ids = _process(regimes_dict=regimes_dict, laws=laws, ages=ages)
    _bi_result = solve(
        program_fingerprint="test_gated_edge_arg_provenance",
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions

    # V_single_f = wage on ITS OWN points (0, 10).
    np.testing.assert_allclose(np.asarray(solution[0]["single_f"]), [0.0, 10.0])

    # At married's node wage = 1.0 the reference read is 1.0 (coordinate 0.1 on
    # single_f's points) -> `5 >= 1` holds -> feasible, V_f = 5.
    # Had the read used married's own points, it would have been 10.0 -> `5 >=
    # 10` fails -> empty mask -> V_f = -inf.
    V_married = np.asarray(solution[0]["married"])
    np.testing.assert_allclose(V_married[1, 0], 5.0)
    assert np.isfinite(V_married[1, 0])

    # The fixture is only a repro if the contested binding actually decides the
    # outcome, so exercise the production reader itself on the SAME solved
    # `V_single_f` and vary ONLY which regime's points resolve its grid.
    reader = _build_same_period_ref_reader(
        ref=ResolvedProjectedRegimeValue(
            regime="single_f",
            projection={"wage": _project_wage},
            stakeholder_index=None,
        ),
        v_interpolation_info=create_v_interpolation_info(
            user_regime=regimes_dict["single_f"]
        ),
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
    # so the finite solved value above is the reference regime's grid winning and
    # nothing else.
    assert _ref_value(_MARRIED_POINTS) > 5.0
    assert _ref_value(_SINGLE_POINTS) <= 5.0


# A gate or projection that names a TARGET regime function would have that
# function's own parameters bound from the source's namespace. Any argument
# naming a target-DAG node is refused at construction, whatever the node reads.
_HELPER_TARGET_SCALE = 0.9


def _target_scaled_x(*, x: ContinuousState, target_scale: FloatND) -> FloatND:
    """A helper declared in the TARGET regime's functions.

    `target_scale` is a parameter the TARGET regime binds from
    `flat_params[target]` — NOT a parameter the source edge declares.
    """
    return x * target_scale


# Every target-DAG-concatenating consumer is fenced, not only the gate predicate:
# a leg-fallback projection naming a target helper is compiled on a separate path
# and is refused the same way.


def _project_through_target_helper(target_scaled_x: FloatND) -> FloatND:
    """A projection routed through the target regime's own helper.

    The projected coordinate is `target_scaled_x(x, target_scale)`, so
    `target_scale` is a TARGET-owned param reached by a projection, which a
    direct-gate fence does not inspect.
    """
    return target_scaled_x


def _make_leg_fallback_target_helper_regimes() -> _Spec:
    src = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            state_transitions={"x": _next_x_offgrid},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_src},
        ),
        Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "target": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "target": Gate(
                    predicate=_always_closed_gate,
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="fallback",
                                projection={"x": _project_through_target_helper},
                            )
                        )
                    },
                )
            },
        ),
    )
    target = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity, "target_scaled_x": _target_scaled_x},
        ),
        None,
    )
    fallback = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity},
        ),
        None,
    )
    return _split({"src": src, "target": target, "fallback": fallback})


def test_leg_fallback_projection_naming_a_target_helper_is_rejected():
    """A leg-fallback projection naming a target helper is refused at construction.

    The projection would otherwise evaluate the helper's `target_scale` from the
    source's namespace when it writes the routed row's fallback state.
    """
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "target": MappingProxyType(
                {"target_scale": jnp.asarray(_HELPER_TARGET_SCALE)}
            ),
            "fallback": MappingProxyType({}),
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=(
            r"get_edge_fold \(solve-side leg fallback projection\): the edge to "
            r"regime 'target' declares gate/projection argument\(s\) "
            r"\['target_scaled_x'\] that name the TARGET regime's own function"
        ),
    ):
        _solve_fixture(
            spec=_make_leg_fallback_target_helper_regimes(), flat_params=flat_params
        )


# The edge-consumer fence on a hand-checked target DAG: a chain
# `wage -> labor_income -> net_income`, a diamond joining `net_income` and
# `transfers` in `resources`, and a helper no consumer below names.
def _pool_wage(*, human_capital: FloatND, wage_level: float) -> FloatND:
    return human_capital * wage_level


def _pool_labor_income(*, wage: FloatND, hours: float) -> FloatND:
    return wage * hours


def _pool_net_income(*, labor_income: FloatND, tax_rate: float) -> FloatND:
    return labor_income * (1 - tax_rate)


def _pool_transfers(*, wage: FloatND, transfer_rate: float) -> FloatND:
    return wage * transfer_rate


def _pool_resources(
    *, net_income: FloatND, transfers: FloatND, wealth: FloatND
) -> FloatND:
    return net_income + transfers + wealth


def _pool_unrelated_helper(*, wealth: FloatND, shift: float) -> FloatND:
    return wealth * shift


_FENCE_DAG_POOL = MappingProxyType(
    {
        "wage": _pool_wage,
        "labor_income": _pool_labor_income,
        "net_income": _pool_net_income,
        "transfers": _pool_transfers,
        "resources": _pool_resources,
        "unrelated_helper": _pool_unrelated_helper,
    }
)


@pytest.mark.parametrize(
    ("seed_args", "named_nodes"),
    [
        pytest.param(("V_target", "net_income"), "['net_income']", id="chain-tail"),
        pytest.param(("resources", "wealth"), "['resources']", id="diamond-join"),
        pytest.param(("wage",), "['wage']", id="chain-head"),
        pytest.param(
            ("transfers", "net_income", "source_own_param"),
            "['net_income', 'transfers']",
            id="several-nodes-listed-sorted",
        ),
    ],
)
def test_fence_edge_consumer_rejects_every_argument_naming_a_target_node(
    *, seed_args: tuple[ReferenceName, ...], named_nodes: str
) -> None:
    """A consumer argument naming a target-DAG node is refused, and the message
    lists exactly the named nodes, however deep a parameter sits below them."""
    with pytest.raises(
        ModelInitializationError,
        match=(
            r"^ctx: the edge to regime 'target' declares gate/projection "
            rf"argument\(s\) {re.escape(named_nodes)} that name the TARGET "
            r"regime's own function"
        ),
    ):
        _fence_edge_consumer(
            dag_pool=_FENCE_DAG_POOL,
            seed_args=seed_args,
            edge_target="target",
            context="ctx",
        )


@pytest.mark.parametrize(
    "seed_args",
    [
        pytest.param(("V_target", "wealth", "human_capital"), id="target-states"),
        pytest.param(("source_own_param",), id="source-param"),
        pytest.param(("shift",), id="param-named-like-an-unrelated-helper-param"),
        pytest.param((), id="no-arguments"),
    ],
)
def test_fence_edge_consumer_accepts_arguments_naming_no_target_node(
    seed_args: tuple[ReferenceName, ...],
) -> None:
    """Arguments that name no target-DAG node pass, including a source parameter
    spelled like a parameter some target helper reads."""
    assert (
        _fence_edge_consumer(
            dag_pool=_FENCE_DAG_POOL,
            seed_args=seed_args,
            edge_target="target",
            context="ctx",
        )
        is None
    )


def _gate_reads_own_target_scale(
    *, x: ContinuousState, target_scale: FloatND
) -> BoolND:
    """A source gate whose OWN parameter shares its name with the parameter of the
    target's `target_scaled_x` helper."""
    return x * target_scale > 0.5


def _make_own_param_named_like_target_helper_param_regimes() -> _Spec:
    src = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            state_transitions={"x": _next_x_offgrid},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_src},
        ),
        Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "target": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "target": Gate(
                    predicate=_gate_reads_own_target_scale,
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="fallback", projection={"x": _identity_x}
                            )
                        )
                    },
                )
            },
        ),
    )
    target = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity, "target_scaled_x": _target_scaled_x},
        ),
        None,
    )
    fallback = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity},
        ),
        None,
    )
    return _split({"src": src, "target": target, "fallback": fallback})


def test_gate_param_named_like_a_target_helper_param_builds_as_a_source_param():
    """A gate that reads a target state and its own parameter builds, and binds that
    parameter from the source under the gate's declaration path, even though a
    target helper reads a parameter of the same name."""
    regimes_dict, laws = _make_own_param_named_like_target_helper_param_regimes()
    regimes, _ = _process(regimes_dict=regimes_dict, laws=laws, ages=_AGES)
    evaluator = (
        regimes["src"].gated_edges["target"].simulate_gate_evaluator_at(period=1)
    )
    source_qnames = sorted(
        qname
        for namespace, qname in evaluator.arg_provenance.params.values()
        if namespace == SOURCE_PARAMS
    )
    assert source_qnames == ["target__predicate__target_scale"]


# A residual namespace defect survives the fences above: a gate/projection arg
# naming a STATE-ONLY target node, which name-based concatenation rebinds to the
# node, dropping a same-named source parameter (a silent gate reversal / wrong
# projected fallback state). The fixture gives the two candidate bindings
# different meanings so it is a genuine repro, and the fence rejects the topology
# at construction.


def _target_threshold(x: ContinuousState) -> FloatND:
    """A STATE-ONLY target node (reaches no dynamic param) whose name collides with
    a parameter the SOURCE gate declares (`threshold`)."""
    return 0.9 + 0.0 * x


def _gate_reads_shadowed_threshold(*, x: ContinuousState, threshold: FloatND) -> BoolND:
    """The source MEANS `threshold` as its own param (0.1); the target declares a
    state-only node `threshold(x)=0.9`, which name-based concatenation binds instead,
    silently reversing the gate."""
    return x > threshold


def _make_threshold_shadow_regimes() -> _Spec:
    src = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            state_transitions={"x": _next_x_offgrid},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_src},
        ),
        Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "target": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "target": Gate(
                    predicate=_gate_reads_shadowed_threshold,
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="fallback", projection={"x": _identity_x}
                            )
                        )
                    },
                )
            },
        ),
    )
    target = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity, "threshold": _target_threshold},
        ),
        None,
    )
    fallback = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity},
        ),
        None,
    )
    return _split({"src": src, "target": target, "fallback": fallback})


def test_gate_arg_shadowed_by_state_only_target_node_is_rejected():
    """A gate arg naming a STATE-ONLY target node must be rejected.

    The node reads only the target state `x`, yet `concatenate_functions` would
    bind the gate's `threshold` to `_target_threshold`, dropping the source's
    `threshold` parameter and evaluating `x > 0.9` where the source meant
    `> 0.1`. The build must raise rather than silently misbind.
    """
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {
                    "koopmans_aggregator__discount_factor": jnp.asarray(_BETA),
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
        _solve_fixture(spec=_make_threshold_shadow_regimes(), flat_params=flat_params)


# A gate argument named like a target state reads that state. A source function's
# parameter of the same name lives under the source regime's own namespace
# (`utility__x`), and a gate's own parameters under
# `flat_params["edges"][source]` at `<target>__predicate__<param>`, so neither shares
# the fold leaf the target state binds.


def _next_y_identity(y: ContinuousState) -> ContinuousState:
    return y


def _entry_x() -> FloatND:
    """Target-specific ENTRY law for `x`, which `src` does not carry.

    A retained edge must give every target state a next-period value; `x` lives
    only on the target, so `src` supplies it explicitly — there is no implicit
    initialization to fall back on.

    Returns a jax array, not a Python float: the value is fed to
    `LinSpacedGrid.get_coordinate`, whose `FloatND` hint beartype enforces as an
    actual `jax.Array`.
    """
    return jnp.asarray(0.0)


def _u_src_reads_x_param(
    *, y: ContinuousState, work: DiscreteAction, x: FloatND
) -> FloatND:
    """Source utility reads param `x`, which also names the target's state."""
    return jnp.zeros_like(y) * work + 0.0 * x


def _u_src_no_param(*, y: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.zeros_like(y) * work


def _gate_reads_x(x: FloatND) -> BoolND:
    return x > 0.5


def _make_gate_param_aliases_target_state_regimes(
    *, source_supplies_x_param: bool
) -> _Spec:
    """Source `y`-regime; target state is `x`; the gate reads `x`.

    With `source_supplies_x_param=True` the source utility also reads a parameter
    `x`, filed under the source regime's own namespace. Either way `gate(x)` reads
    the target state.
    """
    src = (
        Regime(
            states={"y": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            state_transitions={"y": _next_y_identity, "x": {"target": _entry_x}},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={
                "utility": _u_src_reads_x_param
                if source_supplies_x_param
                else _u_src_no_param
            },
        ),
        Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "target": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "target": Gate(
                    predicate=_gate_reads_x,
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="fallback", projection={"x": _identity_x}
                            )
                        )
                    },
                )
            },
        ),
    )
    target = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity},
        ),
        None,
    )
    fallback = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity},
        ),
        None,
    )
    return _split({"src": src, "target": target, "fallback": fallback})


@pytest.mark.parametrize("source_supplies_x_param", [False, True])
def test_gate_reading_a_target_state_solves_beside_a_source_param_of_its_name(
    source_supplies_x_param,
):
    """A gate reading target state `x` solves, whatever the source's own params.

    With `source_supplies_x_param=True` the source utility reads a parameter
    `x`, supplied at `flat_params['src']['utility__x']`; it does not reach the
    gate, which reads the target's state.
    """
    own_params = {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
    if source_supplies_x_param:
        own_params["utility__x"] = jnp.asarray(0.9)
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(own_params),
            "target": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    _solve_fixture(
        spec=_make_gate_param_aliases_target_state_regimes(
            source_supplies_x_param=source_supplies_x_param
        ),
        flat_params=flat_params,
    )


# The ENGINE argument namespace is reserved against target states.
#
# `_evaluate_edge_fold` binds the internal engine mappings `SAME_PERIOD_V_ARG`
# (always) and `SAME_PERIOD_PARAMS_ARG` (when a ref/gate reads it) and the period
# context under their bare names. A gate's own parameters never meet them: every
# one is renamed to its `<target>__predicate__<name>` slot before the fold is built,
# while the period context stays the engine's. A target STATE spelled like an
# engine argument would share one fold leaf with it, so those names are reserved.
def _gate_reads_params_engine_arg(
    *, x: ContinuousState, same_period_regime_to_params: FloatND
) -> BoolND:
    return x > same_period_regime_to_params


def _gate_reads_period_engine_arg(*, x: ContinuousState, period: ScalarInt) -> BoolND:
    return x > period


def _make_gated_source_regimes(gate: UserFunction) -> _Spec:
    """A source whose edge into `target` is gated by `gate`."""
    src = (
        Regime(
            states={"y": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            state_transitions={"y": _next_y_identity, "x": {"target": _entry_x}},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_src_no_param},
        ),
        Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "target": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "target": Gate(
                    predicate=gate,
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="fallback", projection={"x": _identity_x}
                            )
                        )
                    },
                )
            },
        ),
    )
    target = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity},
        ),
        None,
    )
    fallback = (
        Regime(
            states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_identity},
        ),
        None,
    )
    return _split({"src": src, "target": target, "fallback": fallback})


def _u_id_v_arg_state(same_period_regime_to_V_arr: ContinuousState) -> FloatND:
    return same_period_regime_to_V_arr


def _identity_v_arg_state(
    same_period_regime_to_V_arr: ContinuousState,
) -> ContinuousState:
    return same_period_regime_to_V_arr


def _gate_reads_v_arg_state(same_period_regime_to_V_arr: FloatND) -> BoolND:
    return same_period_regime_to_V_arr > 0.5


def _make_target_state_aliases_engine_v_regimes() -> _Spec:
    """Target STATE named exactly `SAME_PERIOD_V_ARG`; the gate reads it. It shares
    one fold leaf with the engine value mapping."""
    src = (
        Regime(
            states={"y": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            state_transitions={"y": _next_y_identity, "x": {"target": _entry_x}},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_src_no_param},
        ),
        Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "target": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "target": Gate(
                    predicate=_gate_reads_v_arg_state,
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="fallback",
                                projection={SAME_PERIOD_V_ARG: _identity_v_arg_state},
                            )
                        )
                    },
                )
            },
        ),
    )
    target = (
        Regime(
            states={SAME_PERIOD_V_ARG: LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_id_v_arg_state},
        ),
        None,
    )
    fallback = (
        Regime(
            states={SAME_PERIOD_V_ARG: LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            functions={"utility": _u_id_v_arg_state},
        ),
        None,
    )
    return _split({"src": src, "target": target, "fallback": fallback})


@pytest.mark.parametrize(
    ("gate", "expected"),
    [
        pytest.param(
            _gate_reads_params_engine_arg,
            {"x", f"target__predicate__{SAME_PERIOD_PARAMS_ARG}"},
            id="engine-params-spelling-is-a-gate-slot",
        ),
        pytest.param(
            _gate_reads_period_engine_arg,
            {"x", "period"},
            id="period-stays-the-engine-context",
        ),
    ],
)
def test_gate_arguments_spelled_like_engine_names_are_bound_apart_from_them(
    *, gate: UserFunction, expected: set[str]
) -> None:
    """A gate parameter takes its slot name; the period context keeps its own."""
    qualified = _with_qualified_params(
        func=gate, path=("target", "predicate"), wired_names=frozenset({"x"})
    )
    assert set(signature(qualified).parameters) == expected


def test_target_state_aliasing_the_engine_v_arg_is_rejected():
    """A target STATE named `SAME_PERIOD_V_ARG` is rejected: it would share one
    fold leaf with the engine value mapping."""
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "target": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"(?i)engine|same_period_regime_to_V_arr|reserved",
    ):
        _solve_fixture(
            spec=_make_target_state_aliases_engine_v_regimes(),
            flat_params=flat_params,
        )


def _gate_near_engine_name(
    *, x: ContinuousState, same_period_regime_to_params_user: FloatND
) -> BoolND:
    return x > same_period_regime_to_params_user


@pytest.mark.parametrize(
    ("gate", "name"),
    [
        pytest.param(
            _gate_near_engine_name,
            "same_period_regime_to_params_user",
            id="name-resembling-an-engine-name",
        ),
        pytest.param(
            _gate_reads_params_engine_arg,
            SAME_PERIOD_PARAMS_ARG,
            id="name-spelled-like-an-engine-name",
        ),
    ],
)
def test_gate_param_named_like_an_engine_argument_solves_from_its_slot(
    *, gate: UserFunction, name: str
) -> None:
    """A gate parameter is supplied at its `target__predicate__<name>` edge slot."""
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "target": MappingProxyType({}),
            "fallback": MappingProxyType({}),
            "edges": MappingProxyType(
                {
                    "src": MappingProxyType(
                        {f"target__predicate__{name}": jnp.asarray(0.1)}
                    )
                }
            ),
        }
    )
    # Must not raise.
    _solve_fixture(spec=_make_gated_source_regimes(gate), flat_params=flat_params)
