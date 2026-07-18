"""Gated edge objects (E3'): mutual-consent marriage / dissolution routing.

The design-doc §2 E3' construct that unlocks MIXED singleton/collective regime
topologies. A source regime declares, per target regime, a `GatedEdge`. At the
END of each period ``t``'s solve — after every period-``t`` regime is solved, so
their value arrays and dissolution flags are still live — the engine folds, for each
declared edge and each source stakeholder ``s``, a gated continuation object on
the TARGET regime's period-``t`` grid::

    Wbar^s(x) = jnp.where( gate(x), V_target^{leg_s}(x), V_fallback^s(pi_s(x)) )

with ``gate`` a boolean user function on the target grid (EKL consent eq. 27 /
no-dissolution eqs. 9/12) and ``V_fallback^s`` a same-period reference regime's value
at a projection (EKL: the source stakeholder's own single regime). The source's
period ``t-1`` continuation then reads ``Wbar`` in place of the raw target V,
threaded through the ordinary transition machinery (`next_regime_to_V_arr`).

**Numerics (non-negotiable).** The mixture is the strict
``jnp.where(gate, V_target, V_fallback)``, never a linear
``gate*V_target + (1-gate)*V_fallback``: the target value carries the slice-3
``-inf`` sentinel in dissolution cells, and ``0 * -inf = NaN``. Every read that
lands the target value — including the gate's own ``V_target_<s>`` reads — is an
on-grid identity-projection interpolation, so it is exact.

The whole fold reuses the slice-3 same-period reference-reader machinery
(`_build_same_period_ref_reader`): the target's own value components and dissolution
flag are read as identity-projection references of the target regime, and the
gate refs / leg fallbacks as ordinary projected references. The per-cell fold is
product-mapped over the target regime's state grid.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
from dags import concatenate_functions, with_signature

from _lcm.regime_building.Q_and_F import (
    SAME_PERIOD_PARAMS_ARG,
    SAME_PERIOD_V_ARG,
    ResolvedSamePeriodRef,
    _build_same_period_ref_reader,
)
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from _lcm.typing import (
    ConstraintFunction,
    EconFunctionsMapping,
    RegimeName,
    TransitionFunction,
    TransitionFunctionName,
    _ParamsLeaf,
)
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import get_union_of_args
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, ContinuousState, DiscreteState, FloatND

# Suffix under which a target regime's dissolution flag `D` (cast to float) is passed
# in the same-period value mapping the fold consumes. Never a real regime name.
D_KEY_SUFFIX = "__gated_edge_D__"

# The float dissolution flag is 0.0 / 1.0 at grid points; threshold back to boolean.
_D_THRESHOLD = 0.5

# COLLECTIVE-REGIMES (E4, simulate F1 fix). `V_arr_name`s under which the
# target's own (per-component) value array and its float dissolution flag are
# bound inside `get_edge_simulate_gate_evaluator`'s two interpolators. Never
# real regime or gate-ref names.
_SIMULATE_TARGET_V_ARR_NAME = "__simulate_target_component_v_arr__"
_SIMULATE_D_ARR_NAME = "__simulate_target_D_arr__"

# COLLECTIVE-REGIMES (E4, F2/F3 fix). The two parameter namespaces a
# simulate-side edge callable resolves its arguments against. A REGIME, not a
# role, is what a params mapping is keyed by; these are the two roles an edge
# relates, and `_lcm.simulation.gated_routing` maps them to the actual regime
# names it has in hand (the source regime that declares the edge, the target
# regime the edge lands on).
SOURCE_PARAMS = "source"
TARGET_PARAMS = "target"

# Prefixes under which an edge callable EXPOSES a parameter of each namespace.
# The qualification is load-bearing, not cosmetic: a runtime irregular grid's
# helper param is named after the STATE alone and carries no regime
# qualification (`_lcm.regime_building.V._get_coordinate_finder` ->
# `qname_from_tree_path((state_name, "points"))` -> `x__points`), so a source
# and a target that both declare a state `x` contribute a param with the very
# same qname. One keyword argument cannot carry two regimes' arrays, so no
# merge ORDER of two same-named entries can be right; the exposed leaves must be
# distinct in the first place.
_TARGET_PARAM_PREFIX = "__target_param__"
_SOURCE_PARAM_PREFIX = "__source_param__"


@dataclass(frozen=True, kw_only=True)
class EdgeArgProvenance:
    """Where each exposed argument of an edge-side simulate callable comes from.

    COLLECTIVE-REGIMES (E4, F2/F3 fix). The simulate-side router
    (`_lcm.simulation.gated_routing`) holds EVERY regime's flat params and the
    realized candidate target states, and has no other way to tell which of them
    an argument of a gate evaluator / fallback projector wants. Publishing an
    explicit provenance — rather than two name-filtered param dicts merged in
    some order — is what makes the answer well defined:

    - `states` are bound from the realized candidate TARGET states (batched over
      subjects). They mirror the solve side, where the fold is evaluated on the
      target regime's own state grid.
    - `params` maps each exposed argument name to the `(namespace, qname)` pair
      that resolves it, with `namespace` one of `SOURCE_PARAMS` / `TARGET_PARAMS`
      and `qname` the name the parameter carries in THAT regime's own flat
      params. Exposed names are namespace-qualified (see `_TARGET_PARAM_PREFIX`
      / `_SOURCE_PARAM_PREFIX`), so two regimes' identically named params are
      distinct leaves of the callable's signature and neither can clobber the
      other.

    Parameters of a REFERENCE regime (a gate ref's / leg fallback's own
    interpolation grid) are a third provenance, and are NOT represented here:
    they never surface as an outer argument at all, because
    `_build_same_period_ref_reader` resolves them internally against
    `SAME_PERIOD_PARAMS_ARG` (F4). This class covers exactly what remains.
    """

    states: frozenset[str]
    """Exposed args bound from the realized candidate target states."""

    params: MappingProxyType[str, tuple[str, str]]
    """Exposed arg name -> (namespace, qname in that namespace's flat params)."""


class _ProvenanceBuilder:
    """Accumulate an `EdgeArgProvenance` while wiring an edge callable.

    `expose` translates a namespace-owned qname into the qualified name the
    callable exposes it under, recording the provenance on the way; identical
    (namespace, qname) pairs requested twice (e.g. two gate refs that both read
    the same source parameter) collapse onto one leaf, which is correct — they
    genuinely carry the same value.
    """

    def __init__(self, *, states: frozenset[str]) -> None:
        self._states = states
        self._params: dict[str, tuple[str, str]] = {}

    def expose(self, *, qname: str, namespace: str, qualify: bool = True) -> str:
        """Return the exposed name for `qname` in `namespace`, recording it.

        `qualify=False` exposes the parameter under its plain qname. Legal only
        for a callable with a SINGLE params namespace (the fallback projector),
        where no second regime can contribute the same name; the conflict check
        below is what holds that promise to account.
        """
        prefix = (
            _TARGET_PARAM_PREFIX if namespace == TARGET_PARAMS else _SOURCE_PARAM_PREFIX
        )
        exposed = f"{prefix}{qname}" if qualify else qname
        recorded = self._params.get(exposed)
        if recorded is not None and recorded != (namespace, qname):
            # The construction-time guard the qualified scheme is designed to
            # make unreachable, asserted rather than assumed: collapsing two
            # provenances onto one keyword is exactly the defect this mechanism
            # replaced, and neither an unqualified exposure nor a future change
            # to the naming scheme may reintroduce it silently.
            msg = (
                f"Argument '{exposed}' would carry two different values: "
                f"{recorded} and {(namespace, qname)}. One keyword argument "
                "cannot carry two regimes' parameters; expose them under "
                "namespace-qualified names instead."
            )
            raise ValueError(msg)
        self._params[exposed] = (namespace, qname)
        return exposed

    def build(
        self, *, outer_arg_names: tuple[str, ...], engine_args: set[str]
    ) -> EdgeArgProvenance:
        """Finalize, checking the provenance PARTITIONS the outer signature.

        Every outer argument must be classified exactly once: an engine-supplied
        one, a candidate target state, or a parameter of a named namespace. A
        missing classification is the F2 completeness defect (an argument the
        router would have to guess a namespace for); an overlap is the F2
        disjointness defect (a name two provenances both claim).

        Raises:
            ValueError: The provenance does not partition `outer_arg_names`.
        """
        overlap = self._states & set(self._params)
        if overlap:
            msg = (
                f"Arguments {sorted(overlap)} are claimed both as candidate "
                "target states and as parameters."
            )
            raise ValueError(msg)
        unclassified = (
            set(outer_arg_names) - engine_args - self._states - set(self._params)
        )
        if unclassified:
            msg = (
                f"Arguments {sorted(unclassified)} of a gated-edge simulate "
                "callable have no recorded provenance, so there is no way to "
                "tell which regime's parameters resolve them."
            )
            raise ValueError(msg)
        return EdgeArgProvenance(
            states=self._states, params=MappingProxyType(dict(self._params))
        )


@dataclass(frozen=True, kw_only=True)
class ResolvedEdgeLeg:
    """Engine-side form of one source-stakeholder leg of a gated edge (E3')."""

    source_stakeholder: str | None
    """Source stakeholder name, or `None` for a singleton source's single leg."""

    target_component_index: int | None
    """Index of the OPEN-branch value on the target V's trailing stakeholder
    axis, or `None` for a singleton target."""

    fallback: ResolvedSamePeriodRef
    """The CLOSED-branch reference value (regime, projection, stakeholder index)."""


@dataclass(frozen=True, kw_only=True)
class ResolvedGatedEdge:
    """Engine-side form of a user `GatedEdge`, resolved at model processing."""

    target: RegimeName
    """Name of the target regime whose grid the fold lands on."""

    gate: ConstraintFunction
    """Boolean gate predicate (params already renamed to qnames)."""

    gate_refs: Mapping[str, ResolvedSamePeriodRef]
    """Extra same-period references the gate reads (projected from the target grid)."""

    legs: tuple[ResolvedEdgeLeg, ...]
    """One leg per source component, in SOURCE stakeholder order."""

    reference_regimes: tuple[RegimeName, ...]
    """Deduplicated real regimes whose same-period V the fold reads (fallbacks +
    gate refs), excluding the target itself."""


def _pad_reader_to_state_names(
    reader: Callable[..., FloatND],
    *,
    state_names: tuple[str, ...],
) -> Callable[..., FloatND]:
    """Widen a reader's exposed signature to every one of ``state_names``.

    ``reader``'s own args are kept (states it genuinely reads, plus any extra
    runtime params, e.g. grid points for an irregular-grid projection); any
    ``state_names`` entry missing from that set is added as an ignored
    keyword-only argument, so `_grid_reader`'s downstream `productmap` (which
    always maps over the FULL `state_names`) sees every axis in the wrapped
    function's own signature and does not drop it.
    """
    own_args = tuple(get_union_of_args([reader]))
    padded_args = tuple(dict.fromkeys((*own_args, *state_names)))

    @with_signature(args=padded_args, return_annotation="FloatND")
    def padded(**kwargs: _ParamsLeaf) -> FloatND:
        return reader(**{name: kwargs[name] for name in own_args})

    return padded


def _target_function_param_leaves(
    dag_pool: Mapping[str, Callable[..., FloatND]], state_names: frozenset[str]
) -> frozenset[str]:
    """Dynamic parameter leaves introduced by the TARGET regime's own functions.

    The gate/projection DAG concatenates the target regime's functions and
    deterministic transitions (``dag_pool``) with the source-declared gate.
    A *leaf* of that DAG is a free argument produced by no node in it; removing
    the state coordinates leaves the genuine parameters the target regime binds
    from ``flat_params[target]`` -- distinct from a parameter the source edge
    declares directly.
    """
    produced = set(dag_pool.keys())
    leaves: set[str] = set()
    for fn in dag_pool.values():
        leaves |= set(get_union_of_args([fn]))
    leaves -= produced
    leaves -= set(state_names)
    return frozenset(leaves)


def _reject_target_function_params(
    *,
    dag_pool: Mapping[str, Callable[..., FloatND]],
    consumer_arg_names: tuple[str, ...],
    state_names: frozenset[str],
    edge_target: str,
    context: str,
) -> None:
    """Fence for round-4 audit F2 (a target helper param mis-owned as source).

    Collective-edge provenance binds every non-injected gate/projection argument
    from ``flat_params[source]`` -- the solve-side fold does the same, so solve
    and simulate stay mutually consistent. But that is NOT consistent with the
    *target* regime's own kernel, which binds a target function's parameter from
    ``flat_params[target]``: a gate (or projection) that reaches a target-regime
    function with a free dynamic parameter would therefore evaluate that
    parameter from the wrong namespace, and would COLLAPSE with a same-named
    source parameter (a gate reversal, reproduced by the round-4 review).

    Origin-preserving edge compilation (carrying the target/source origin through
    the concatenated DAG, and passing target params as a distinct input to the
    solve-side fold) is not yet implemented. Until it is, reject the topology
    rather than silently misbind it. Source-declared projection parameters are
    NOT affected: they enter through the edge's own gate/projection, not through
    ``dag_pool`` (the target's functions), so they are not leaves of the target
    DAG and this fence does not fire on them.
    """
    contested = sorted(
        _target_function_param_leaves(dag_pool, state_names) & set(consumer_arg_names)
    )
    if contested:
        msg = (
            f"{context}: the edge to regime '{edge_target}' reaches parameter(s) "
            f"{contested} that are introduced by the TARGET regime's own functions "
            "/ deterministic transitions. Collective-edge provenance binds every "
            "non-injected argument from flat_params[source], which would evaluate a "
            "target-regime function parameter from the wrong namespace (and collapse "
            "it with any same-named source parameter). Origin-preserving edge "
            "compilation is not yet implemented, so this topology is rejected rather "
            "than silently misbound. Compute the quantity outside the target regime's "
            "functions (e.g. as a source-declared gate-ref projection), or give the "
            "parameter a source-unique name."
        )
        raise ModelInitializationError(msg)


def get_edge_fold(
    *,
    edge: ResolvedGatedEdge,
    target_v_info: VInterpolationInfo,
    target_functions: EconFunctionsMapping,
    target_deterministic_transitions: Mapping[
        TransitionFunctionName, TransitionFunction
    ],
    reference_v_info: Mapping[RegimeName, VInterpolationInfo],
    target_stakeholders: tuple[str, ...] | None,
) -> Callable[..., FloatND]:
    """Build one edge's fold: a jittable ``Wbar`` producer on the target grid.

    Returns a callable whose keyword arguments are the target regime's state
    grids, the same-period value mapping (under `SAME_PERIOD_V_ARG` — the target
    V, its float dissolution flag, and every reference regime's V) and the gate's
    flat params. It returns ``Wbar`` of shape ``(*target_state_axes,
    n_source_components)`` for a collective source, or ``(*target_state_axes,)``
    for a singleton source (a single leg with no trailing axis).

    COLLECTIVE-REGIMES (E4, simulate F1 fix). Earlier revisions also returned
    the fold's raw grid-level boolean ``gate`` array, which the simulate-side
    value router (`_lcm/simulation/gated_routing.py`) used to decide REGIME
    ROUTING for a realized subject by INTERPOLATING that baked boolean array
    and thresholding the result at 0.5. That does not commute with a
    nonlinear gate predicate (e.g. a strict inequality between two
    interpolated values): interpolate-then-threshold can disagree with
    threshold-then-interpolate arbitrarily close to a grid cell boundary,
    silently flipping routing decisions the fold itself never evaluated at
    that off-grid point. Simulate now instead RECOMPUTES the gate from
    interpolated VALUE OPERANDS via `get_edge_simulate_gate_evaluator`
    (this module), so the fold no longer needs to return `gate` at all — it
    is still computed INTERNALLY below (needed for `Wbar`'s own
    ``jnp.where``), just no longer part of the return value.

    **Numerics.** The target regime's OWN value components and dissolution flag are
    read by DIRECT array indexing off the same-period mapping — never by
    interpolation. Linear interpolation of the target's ``-inf``-bearing V would
    compute ``0 * -inf = NaN`` at the grid points ADJACENT to a dissolution cell
    (the zero-weight neighbour), poisoning the OPEN branch before the
    ``jnp.where`` could guard it. Only the gate references and leg fallbacks —
    which read OTHER (finite) regimes at projected coordinates — are
    interpolated, product-mapped over the target grid.

    Args:
        edge: The resolved edge declaration.
        target_v_info: The target regime's V-interpolation info (its grid).
        target_functions: The target regime's processed functions, so the
            projections and the gate resolve target states / helper functions.
        target_deterministic_transitions: The target regime's merged
            deterministic ``next_<state>`` laws (available to projections).
        reference_v_info: V-interpolation info per reference regime.
        target_stakeholders: The target regime's stakeholders, or `None`.

    Returns:
        The fold callable producing ``Wbar`` on the target grid.
    """
    state_names = target_v_info.state_names

    def _grid_reader(reader: Callable[..., FloatND]) -> Callable[..., FloatND]:
        """Product-map an off-grid reference reader over the target grid.

        COLLECTIVE-REGIMES (E3', slice 5): `productmap` derives its OWN
        outward-facing signature from the wrapped function's own parameters
        (`_lcm.utils.dispatchers.productmap` -> `allow_only_kwargs`), and
        silently DROPS any caller-supplied kwarg not in that signature. A
        same-period-ref projection frequently reads only a STRICT SUBSET of
        the target's `state_names` (e.g. a gate ref projected from a single
        newly-drawn state, ignoring a carried-along one) — but `_grid_reader`
        always maps over the FULL `state_names` (every target-grid axis), and
        `batched_vmap`'s internal closure unconditionally needs every one of
        them present in its call kwargs. Left alone, the unused axes get
        dropped by the signature filter before `batched_vmap` ever sees them,
        raising a `KeyError` on the first unused axis. Padding the reader's
        exposed signature to the full `state_names` (ignoring the padding
        args internally) fixes the mismatch without touching `productmap`
        itself, which is shared far beyond gated edges.
        """
        return productmap(
            func=_pad_reader_to_state_names(reader, state_names=state_names),
            variables=state_names,
            batch_sizes=dict.fromkeys(state_names, 0),
        )

    gate_ref_readers = {
        name: _grid_reader(
            _build_same_period_ref_reader(
                ref=ref,
                v_interpolation_info=reference_v_info[ref.regime],
                functions=target_functions,
                deterministic_transitions=target_deterministic_transitions,
            )
        )
        for name, ref in edge.gate_refs.items()
    }
    fallback_readers = [
        _grid_reader(
            _build_same_period_ref_reader(
                ref=leg.fallback,
                v_interpolation_info=reference_v_info[leg.fallback.regime],
                functions=target_functions,
                deterministic_transitions=target_deterministic_transitions,
            )
        )
        for leg in edge.legs
    ]
    # `get_union_of_args` reflects each reader's EXPOSED signature, which —
    # thanks to `_pad_reader_to_state_names` inside `_grid_reader` — already
    # spans the full `state_names` plus any genuine extra params (e.g.
    # runtime grid points for an irregular-grid projection); no separate
    # union with `state_names` is needed here.
    gate_ref_args = {
        name: tuple(get_union_of_args([reader]))
        for name, reader in gate_ref_readers.items()
    }
    fallback_args = [tuple(get_union_of_args([reader])) for reader in fallback_readers]

    # The gate evaluator: the predicate concatenated with the target DAG, so it
    # may read target states / helper functions; its injected leaves (the
    # `V_target_<s>` components, `D_target`, and the gate-ref names) are bound
    # from the elementwise grid arrays below.
    dag_pool = {
        **dict(target_deterministic_transitions),
        **{k: v for k, v in target_functions.items() if k != "H"},
    }
    gate_evaluator = concatenate_functions(
        functions={**dag_pool, "__gate__": edge.gate},
        targets="__gate__",
        enforce_signature=False,
        set_annotations=True,
    )
    gate_arg_names = tuple(get_union_of_args([gate_evaluator]))
    _reject_target_function_params(
        dag_pool=dag_pool,
        consumer_arg_names=gate_arg_names,
        state_names=frozenset(state_names),
        edge_target=edge.target,
        context="get_edge_fold (solve-side gate)",
    )

    target_component_names = (
        [f"V_target_{s}" for s in target_stakeholders]
        if target_stakeholders is not None
        else ["V_target"]
    )
    injected_names = frozenset({*target_component_names, "D_target", *gate_ref_readers})

    # Outer signature: the target state grids, the same-period value mapping, and
    # any non-injected params/extras the gate or the reference readers need.
    outer_arg_names = sorted(
        {SAME_PERIOD_V_ARG}
        | set(state_names)
        | {arg for args in gate_ref_args.values() for arg in args}
        | {arg for args in fallback_args for arg in args}
        | (set(gate_arg_names) - injected_names)
    )

    singleton_source = all(leg.source_stakeholder is None for leg in edge.legs)

    @with_signature(args=outer_arg_names, return_annotation="FloatND")
    def fold(**kwargs: _ParamsLeaf) -> FloatND:
        same_period_V = cast("Mapping[RegimeName, FloatND]", kwargs[SAME_PERIOD_V_ARG])
        # Direct (un-interpolated) reads of the target's own value and flag, so a
        # `-inf` dissolution cell never poisons a neighbour through interpolation.
        target_V = same_period_V[edge.target]
        target_components: dict[str, FloatND] = {}
        for index, name in enumerate(target_component_names):
            target_components[name] = (
                target_V[..., index] if target_stakeholders is not None else target_V
            )
        d_value = same_period_V.get(f"{edge.target}{D_KEY_SUFFIX}")

        gate_ref_values = {
            name: reader(
                **{arg: kwargs[arg] for arg in gate_ref_args[name]},
            )
            for name, reader in gate_ref_readers.items()
        }
        # Broadcast the target state grids to the full grid for any gate that
        # reads a state directly (EKL's do not; supported for generality).
        state_mesh = dict(
            zip(
                state_names,
                jnp.meshgrid(
                    *[jnp.asarray(kwargs[s]) for s in state_names], indexing="ij"
                ),
                strict=True,
            )
        )
        gate_kwargs = _assemble_gate_kwargs(
            gate_arg_names=gate_arg_names,
            target_components=target_components,
            d_value=d_value,
            gate_ref_values=gate_ref_values,
            state_mesh=state_mesh,
            cell_kwargs=kwargs,
        )
        gate = jnp.asarray(gate_evaluator(**gate_kwargs))

        component_values: list[FloatND] = []
        for leg, fb_reader, fb_arg_names in zip(
            edge.legs, fallback_readers, fallback_args, strict=True
        ):
            open_name = (
                "V_target"
                if leg.target_component_index is None
                else target_component_names[leg.target_component_index]
            )
            open_branch = target_components[open_name]
            fallback = fb_reader(**{arg: kwargs[arg] for arg in fb_arg_names})
            # STRICT where — never `gate*V + (1-gate)*fallback` (`0*-inf = NaN`).
            component_values.append(jnp.where(gate, open_branch, fallback))

        return (
            component_values[0]
            if singleton_source
            else jnp.stack(component_values, axis=-1)
        )

    return fold


def get_edge_simulate_gate_evaluator(
    *,
    edge: ResolvedGatedEdge,
    target_v_info: VInterpolationInfo,
    target_functions: EconFunctionsMapping,
    target_deterministic_transitions: Mapping[
        TransitionFunctionName, TransitionFunction
    ],
    reference_v_info: Mapping[RegimeName, VInterpolationInfo],
    target_stakeholders: tuple[str, ...] | None,
    target_has_process_axis: bool,
) -> Callable[..., BoolND]:
    """Build one edge's SIMULATE-side gate evaluator (E4, simulate F1 fix).

    Companion to `get_edge_fold`, consumed only by forward simulation
    (`_lcm.simulation.gated_routing.route_gated_edges`). RECOMPUTES the gate
    predicate at a realized, generically OFF-GRID candidate target state,
    instead of interpolating the solve-side fold's baked boolean `gate`
    array and thresholding the interpolated float at 0.5 — the old approach
    does not commute with a nonlinear predicate (e.g. a strict inequality
    between two interpolated values: interpolate-then-compare and
    compare-then-interpolate can disagree arbitrarily close to a grid cell
    boundary), silently flipping routing decisions the fold never evaluated
    at that point.

    Mirrors `get_edge_fold`'s own gate-kwargs assembly
    (`_assemble_gate_kwargs` + `gate_evaluator(**kwargs)`, the SAME two
    calls, so solve and simulate apply the identical predicate); only how
    the OPERANDS feeding it are obtained differs:

    - VALUE operands (`V_target_<s>`, every `gate_refs` entry) are
      INTERPOLATED at the realized point, not recomputed: the target's own
      value array (sliced per stakeholder for a collective target) via a
      fresh `get_V_interpolator` over the target's own grid, and each
      declared gate ref via the SAME `_build_same_period_ref_reader` reader
      `get_edge_fold` uses, just called directly at one point instead of
      product-mapped over the whole target grid.

      This is a KNOWN, NON-CONVERGENT residual: the gate is an
      APPROXIMATE router, not the exact E4 recomputation the extension spec
      states. Two earlier versions of this docstring were WRONG about it and
      both errors are recorded here, because each was a confident claim that
      no test could contradict:

      1. It first called these "faithful recomputes". They are not.
         `V_target` is an ALREADY-MAXIMIZED object and interpolation does not
         commute with a `max`: with target actions `u=x` and `u=1-x` on the
         grid `{0,1}`, both nodes give `V=1`, so the interpolant reads 1
         everywhere while the true `max_a Q` at `x=0.5` is 0.5.
      2. It then called the residual "SECOND-ORDER", citing an O(h^2)
         crossing-location rate. That rate was measured against a SMOOTH
         target (`u=sqrt(x)`) — and smoothness is exactly what fails here.
         **At an action-envelope kink the error is O(h), not O(h^2).**

      Worse, and the reason this is not merely a rate question: **value
      convergence does not imply ROUTING convergence.** The consent
      predicate is discontinuous, so refinement does not cure it when the
      candidate distribution has an atom on the equality surface. Take
      `V(x) = max(x, 1-x)`, a deterministic candidate atom at `x=0.5`, and
      the strict gate `V(x) > 0.5`. On every EVEN-cardinality uniform grid
      the two nodes flanking 0.5 both carry `V = 0.5 + h/2`, so
      `interp(V)(0.5) = 0.5 + h/2 > 0.5` and the gate OPENS, while the
      faithful gate is CLOSED. The value error `h/2` vanishes; the routing
      error stays at probability ONE for every such grid. One ordinary
      envelope kink plus a deterministic draw is enough — no pathological
      density is needed.

      Turning value convergence into routing convergence would need a MARGIN
      condition that is neither stated nor checked here, e.g.
      `P(|V_target_<s> - ref| <= eps) -> 0` as `eps -> 0` (no atom at zero,
      with mass control near it), plus a uniform interpolation bound. Callers
      relying on a value gate should treat routing as approximate and check
      grid-convergence of the reported route frequencies themselves.

      A fully faithful evaluator would recompute the target's realized
      `max_a Q` (household argmax + own-component read, for a collective
      target) instead of interpolating its stored V. That means threading the
      target's full state-action space, compiled Q/F, params, `solution[t+2]`
      and `collective_argmax_and_readout` through `route_gated_edges`,
      `Regime`'s compiled artifacts, and the solve/simulate plumbing — and it
      would still leave the LAST interpolation level in place (a non-terminal
      target's recomputed `max_a Q` reads `interp(V_{t+2})`). Not taken here;
      tracked as the known gap between this router and the E4 spec.
    - The BOOLEAN `D_target` operand (a no-dissolution gate, e.g. EKL eqs.
      9/12) is a DOCUMENTED RESIDUAL: linearly interpolating the float-cast
      flag and thresholding at 0.5 (`_assemble_gate_kwargs`'s existing
      `D_target` branch, reused unchanged) is kept, rather than recomputed
      from `D`'s own underlying per-action value comparison at the realized
      point. That would mean re-deriving `D` from IR internals the fold
      itself never exposes here — a deeper, separate slice; a gate reading
      ONLY `D_target` (a pure divorce/no-dissolution gate) is therefore
      NOT yet fully faithful off-grid, only nearest-node-equivalent via
      linear interpolation + threshold, same as before this fix.

    **Numerics.** Unlike `get_edge_fold`'s target-V read (exact grid-point
    indexing, to dodge `0 * -inf = nan` poisoning a dissolution cell's
    neighbour), a realized simulate-side point is generically off-grid, so
    interpolating the target's own V is unavoidable here. The interpolation
    kernel (`_lcm.regime_building.ndimage.map_coordinates`) is already
    zero-weight-`-inf`-safe for the on-grid degenerate case
    (`zero_safe_weighted_term`); a genuinely off-grid point straddling a
    dissolution boundary interpolates TOWARDS `-inf` rather than producing a
    `nan` (a finite corner weight times `-inf` is `-inf`, not `0 * -inf`), so
    a value-operand comparison like `V_target_<s> > ref` degrades to
    `False` there — the same qualitative answer a strict `-inf`-aware
    predicate would give.

    Args:
        edge: The resolved edge declaration (identical to `get_edge_fold`).
        target_v_info: The target regime's V-interpolation info (its grid).
        target_functions: The target regime's processed functions, so the
            gate resolves target states / helper functions.
        target_deterministic_transitions: The target regime's merged
            deterministic ``next_<state>`` laws.
        reference_v_info: V-interpolation info per reference regime.
        target_stakeholders: The target regime's stakeholders, or `None`.
        target_has_process_axis: Whether the target carries a non-folded
            `_ContinuousStochasticProcess` state axis — selects the
            process-aware interpolator (`get_V_interpolator`'s
            `interpolate_process_axes`) for the target's OWN value / `D`
            reads, mirroring `_build_same_period_ref_reader`'s identical
            auto-select for each gate ref (independently, off its OWN
            reference regime's grid).

    Returns:
        A callable returning the recomputed boolean gate at one realized
        candidate point (scalar per subject once `vmap`-ped by the caller),
        keyed by:

        - the target's state names — the realized candidate point;
        - `SAME_PERIOD_V_ARG` — target V, `D`-as-float, and every reference
          regime's V (`build_same_period_mapping_for_fold`'s output);
        - `SAME_PERIOD_PARAMS_ARG` — `{regime: its flat params}`, against
          which each reference reader resolves its OWN regime's runtime grid
          helpers internally (F4);
        - the params named by the returned `EdgeArgProvenance`, exposed under
          NAMESPACE-QUALIFIED leaves (`__target_param__x__points` vs
          `__source_param__x__points`).

        The qualification is the F2 fix and is load-bearing: a runtime grid
        helper is named after the STATE alone (`x__points`), so a source and
        a target that both declare a state `x` contribute the same qname. One
        keyword cannot carry two regimes' arrays, so no merge ORDER is
        correct — the leaves must be distinct. Read the provenance off the
        callable (`.provenance`) rather than name-filtering two param dicts;
        `_ProvenanceBuilder.build` asserts at construction that it PARTITIONS
        the signature (disjoint and complete).
    """
    state_names = target_v_info.state_names

    # Gate-ref readers: the IDENTICAL per-cell construction `get_edge_fold`
    # uses for its own `gate_ref_readers`, but WITHOUT that function's
    # `_grid_reader` product-map wrap — `get_edge_fold` maps these over the
    # full target GRID (solve time, one evaluation per grid cell); here each
    # reader is called directly at ONE realized point (vmapped by the
    # caller over subjects), exactly the off-grid idiom
    # `_build_same_period_ref_reader` is built for everywhere else.
    gate_ref_readers = {
        name: _build_same_period_ref_reader(
            ref=ref,
            v_interpolation_info=reference_v_info[ref.regime],
            functions=target_functions,
            deterministic_transitions=target_deterministic_transitions,
        )
        for name, ref in edge.gate_refs.items()
    }
    gate_ref_args = {
        name: tuple(get_union_of_args([reader]))
        for name, reader in gate_ref_readers.items()
    }

    target_component_interpolator = get_V_interpolator(
        v_interpolation_info=target_v_info,
        state_prefix="",
        V_arr_name=_SIMULATE_TARGET_V_ARR_NAME,
        interpolate_process_axes=target_has_process_axis,
    )
    target_component_args = tuple(
        arg
        for arg in get_union_of_args([target_component_interpolator])
        if arg != _SIMULATE_TARGET_V_ARR_NAME
    )

    d_interpolator = get_V_interpolator(
        v_interpolation_info=target_v_info,
        state_prefix="",
        V_arr_name=_SIMULATE_D_ARR_NAME,
        interpolate_process_axes=target_has_process_axis,
    )
    d_interpolator_args = tuple(
        arg
        for arg in get_union_of_args([d_interpolator])
        if arg != _SIMULATE_D_ARR_NAME
    )

    # The SAME gate predicate `get_edge_fold` builds (identical DAG pool,
    # identical `concatenate_functions` call) — solve and simulate must
    # apply the exact same function, only the operands feeding it differ.
    dag_pool = {
        **dict(target_deterministic_transitions),
        **{k: v for k, v in target_functions.items() if k != "H"},
    }
    gate_evaluator = concatenate_functions(
        functions={**dag_pool, "__gate__": edge.gate},
        targets="__gate__",
        enforce_signature=False,
        set_annotations=True,
    )
    gate_arg_names = tuple(get_union_of_args([gate_evaluator]))
    _reject_target_function_params(
        dag_pool=dag_pool,
        consumer_arg_names=gate_arg_names,
        state_names=frozenset(state_names),
        edge_target=edge.target,
        context="get_edge_simulate_gate_evaluator (simulate-side gate)",
    )
    reads_d_target = "D_target" in gate_arg_names

    target_component_names = (
        [f"V_target_{s}" for s in target_stakeholders]
        if target_stakeholders is not None
        else ["V_target"]
    )
    injected_names = frozenset({*target_component_names, "D_target", *gate_ref_readers})

    # PROVENANCE (F2 fix). Every non-engine argument is attributed to exactly
    # one namespace, and exposed under a namespace-QUALIFIED name, because the
    # namespaces are NOT distinguishable by the qname alone: `get_V_interpolator`
    # names its runtime grid helpers after the STATE (`V.py`'s
    # `qname_from_tree_path((state_name, "points"))` -> `x__points`), with no
    # regime qualification, so a source and a target that both declare a state
    # `x` contribute the identical qname. Two frozensets of UNqualified names
    # cannot express that: the sets intersect, one keyword can only carry one of
    # the two arrays, and whichever merge order the router picks, some argument
    # is bound from the wrong regime.
    #
    # The three provenances (this is the completeness half of F2 — the previous
    # revision knew only two, and assigned every gate-ref reader argument
    # wholesale to the target):
    #
    # 1. TARGET_PARAMS — the target's OWN V / `D` interpolation helpers. These
    #    are simulate-only objects: the solve-side fold reads the target's value
    #    by exact grid indexing and never interpolates it, so these args have no
    #    solve-side counterpart. They interpolate over the TARGET's grid, hence
    #    the target's params.
    # 2. SOURCE_PARAMS — the gate predicate's own free params, and the free
    #    params of the SOURCE-declared gate-ref projections. This is not a
    #    choice: `backward_induction._evaluate_edge_fold` binds every one of the
    #    fold's params from `flat_params[source]`, so binding them anywhere else
    #    here would make simulate evaluate a different predicate than the Wbar
    #    the source's own solved policy was optimized against.
    #    ROUND-4 AUDIT F2: a param introduced by the TARGET regime's OWN
    #    functions/transitions is NOT source-owned — the target regime binds it
    #    from `flat_params[target]` in its own kernel, so attributing it to source
    #    here (and to source in the fold) evaluates it from the wrong namespace and
    #    collapses it with any same-named source param. Origin-preserving edge
    #    compilation is deferred; until then `_reject_target_function_params`
    #    (called above, after `gate_arg_names`) rejects that topology rather than
    #    silently misbinding it, so no target-function param ever reaches this
    #    SOURCE bucket.
    # 3. The REFERENCE regimes' own interpolation grids — resolved inside
    #    `_build_same_period_ref_reader` against `SAME_PERIOD_PARAMS_ARG` (F4),
    #    so they never reach this signature.
    engine_args = {SAME_PERIOD_V_ARG}
    if gate_ref_readers:
        engine_args.add(SAME_PERIOD_PARAMS_ARG)
    provenance_builder = _ProvenanceBuilder(states=frozenset(state_names))

    def _expose(arg: str, namespace: str) -> str:
        if arg in state_names or arg in engine_args:
            return arg
        return provenance_builder.expose(qname=arg, namespace=namespace)

    target_component_exposed = {
        arg: _expose(arg, TARGET_PARAMS) for arg in target_component_args
    }
    d_interpolator_exposed = {
        arg: _expose(arg, TARGET_PARAMS) for arg in d_interpolator_args
    }
    gate_ref_exposed = {
        name: {arg: _expose(arg, SOURCE_PARAMS) for arg in args}
        for name, args in gate_ref_args.items()
    }
    gate_extra_exposed = {
        arg: _expose(arg, SOURCE_PARAMS)
        for arg in sorted(set(gate_arg_names) - injected_names)
    }

    outer_arg_names = tuple(
        sorted(
            engine_args
            | set(state_names)
            | set(target_component_exposed.values())
            | set(d_interpolator_exposed.values())
            | {arg for exposed in gate_ref_exposed.values() for arg in exposed.values()}
            | set(gate_extra_exposed.values())
        )
    )
    arg_provenance = provenance_builder.build(
        outer_arg_names=outer_arg_names, engine_args=engine_args
    )

    @with_signature(args=list(outer_arg_names), return_annotation="BoolND")
    def evaluate_simulate_gate(**kwargs: _ParamsLeaf) -> BoolND:
        same_period_V = cast("Mapping[RegimeName, FloatND]", kwargs[SAME_PERIOD_V_ARG])
        target_V = same_period_V[edge.target]

        # VALUE-operand read: interpolate the target's own (per-component)
        # value array at the realized point, instead of reading the
        # solve-side fold's baked boolean gate off-grid. Exact on nodes,
        # O(h^2) between them -- NOT a recompute of `max_a Q`; see this
        # function's docstring for the measured residual.
        target_components: dict[str, FloatND] = {}
        for index, name in enumerate(target_component_names):
            component_arr = (
                target_V[..., index] if target_stakeholders is not None else target_V
            )
            target_components[name] = target_component_interpolator(
                **{
                    arg: kwargs[exposed]
                    for arg, exposed in target_component_exposed.items()
                },
                **{_SIMULATE_TARGET_V_ARR_NAME: component_arr},
            )

        # DOCUMENTED RESIDUAL: `D_target` is linearly interpolated and
        # thresholded (same recipe as every other simulate-side value read),
        # never recomputed from its own per-action IR comparison — see this
        # function's docstring. Only built/interpolated when the gate
        # actually reads it (`reads_d_target`, a Python-level bool at trace
        # time), so a pure value-operand gate (consent) pays no cost for a
        # `D` array it never uses.
        d_value: FloatND | None = None
        if reads_d_target:
            d_flag = same_period_V.get(f"{edge.target}{D_KEY_SUFFIX}")
            if d_flag is not None:
                d_value = d_interpolator(
                    **{
                        arg: kwargs[exposed]
                        for arg, exposed in d_interpolator_exposed.items()
                    },
                    **{_SIMULATE_D_ARR_NAME: d_flag},
                )

        gate_ref_values = {
            name: reader(
                **{
                    arg: kwargs[exposed]
                    for arg, exposed in gate_ref_exposed[name].items()
                }
            )
            for name, reader in gate_ref_readers.items()
        }

        # Shared with `get_edge_fold`: identical kwargs assembly, then the
        # identical predicate call.
        gate_kwargs = _assemble_gate_kwargs(
            gate_arg_names=gate_arg_names,
            target_components=target_components,
            d_value=d_value,
            gate_ref_values=gate_ref_values,
            state_mesh={name: jnp.asarray(kwargs[name]) for name in state_names},
            # The predicate declares its params under their OWN qnames; map the
            # qualified leaves back before handing them over, so `edge.gate` and
            # `_assemble_gate_kwargs` are byte-identical to the solve side.
            cell_kwargs={
                arg: kwargs[exposed] for arg, exposed in gate_extra_exposed.items()
            },
        )
        return jnp.asarray(gate_evaluator(**gate_kwargs))

    # Published for `route_gated_edges`, which has EVERY regime's flat params in
    # hand and no other way to tell which one an arg belongs to.
    evaluate_simulate_gate.arg_provenance = arg_provenance  # type: ignore[attr-defined]

    return evaluate_simulate_gate


_FALLBACK_PROJECTION_TARGET_PREFIX = "__fallback_state__"


def build_fallback_state_projector(
    *,
    ref: ResolvedSamePeriodRef,
    fallback_v_info: VInterpolationInfo,
    target_state_names: tuple[str, ...],
    target_functions: EconFunctionsMapping,
    target_deterministic_transitions: Mapping[
        TransitionFunctionName, TransitionFunction
    ],
) -> Callable[..., Mapping[str, FloatND]]:
    """Project a target-grid point onto one edge leg's FALLBACK state coordinates.

    COLLECTIVE-REGIMES (E4). Companion to `_build_same_period_ref_reader`
    (which reads the fallback regime's V at these same projected
    coordinates, for the solve-side fold): the simulate-side value router
    does not need the fallback's VALUE (Wbar already folds that in) but does
    need the fallback's own STATE coordinates, to write the dissolutiond/rejected
    stakeholder's next-period row into `states[fallback.regime]`. Reuses the
    identical projection-function construction (same `dag_pool`, same
    `concatenate_functions` targets) so the coordinates are guaranteed
    consistent with whatever the fold read.

    **Provenance (F3 fix).** Consistency with the fold is a claim about the
    projection's INPUTS, not only about how it is built, and the two had drifted
    apart: `backward_induction._evaluate_edge_fold` binds every free parameter
    the fold's fallback reader needs from `flat_params[SOURCE]`, while the router
    called this projector with `{**candidate_target_states, **flat_params[
    TARGET]}`. A projection `z = x + shift` declared on a source with
    `shift = 1.0` against a target with `shift = 9.0` therefore read the
    fallback's V at `x + 1.0` when the solve-side fold folded `Wbar`, but wrote
    the simulated row into the fallback regime at state `x + 9.0` — the right
    regime with a state the solved policy never priced, carried on into the next
    period. (If the target simply lacked the source's parameter, it crashed.)
    The published `arg_provenance` therefore attributes every argument
    explicitly: the target's own STATES (the realized candidate point, the
    simulate counterpart of the target grid the fold maps over) and, for
    everything else, the SOURCE's params — which is not a preference between two
    merges but the only choice that makes the simulated coordinate equal the one
    the fold projected.

    ROUND-4 AUDIT F2 corrects the earlier reasoning here. A projection DAG CAN
    route through a target helper FUNCTION whose own free params the target regime
    binds from `flat_params[target]`; binding them from the source at the fold (as
    the old code did on both sides) evaluates them from the wrong namespace and
    collapses them with any same-named source param. That is not a valid provenance
    for a target-owned parameter, so `_reject_target_function_params` (called after
    `arg_names` below) FENCES a projection that reaches such a param rather than
    misbinding it. Source-declared projection params (a `shift` the edge itself
    names) are unaffected: they are not leaves of the target DAG. Origin-preserving
    edge compilation (carrying target/source origin through the DAG and passing
    target params to the solve-side fold) would lift the fence; it is deferred.

    Unlike `get_edge_simulate_gate_evaluator`, this callable exposes both kinds
    of argument under their PLAIN names: it holds no interpolator of its own, so
    the target-params namespace (the source of the identically-named-leaf
    problem there) is empty here and nothing can collide.

    Args:
        ref: The leg's resolved fallback reference (`ResolvedEdgeLeg.fallback`).
        fallback_v_info: V-interpolation info of the FALLBACK regime (`ref.regime`),
            fixing which state names to project.
        target_state_names: The TARGET regime's own state names, i.e. exactly
            those arguments the router binds from the realized candidate target
            states rather than from a params namespace.
        target_functions: The target regime's processed functions (projections
            are expressed in terms of the target's own states/helpers).
        target_deterministic_transitions: The target regime's merged
            deterministic `next_<state>` laws.

    Returns:
        A callable, keyed by (a subset of) the target's state names plus any
        extra params the projections need, returning a dict of the fallback
        regime's own state-coordinate arrays. It carries an `arg_provenance`
        attribute (`EdgeArgProvenance`) saying which namespace resolves each.
    """
    dag_pool = {
        **dict(target_deterministic_transitions),
        **{k: v for k, v in target_functions.items() if k != "H"},
    }
    projection_funcs: dict[str, Callable[..., FloatND]] = {}
    projection_args: dict[str, tuple[str, ...]] = {}
    for state_name in fallback_v_info.state_names:
        target = f"{_FALLBACK_PROJECTION_TARGET_PREFIX}{state_name}"
        projection_funcs[state_name] = concatenate_functions(
            functions={**dag_pool, target: ref.projection[state_name]},
            targets=target,
            enforce_signature=False,
            set_annotations=True,
        )
        projection_args[state_name] = tuple(
            get_union_of_args([projection_funcs[state_name]])
        )
    arg_names = tuple(
        sorted({arg for args in projection_args.values() for arg in args})
    )
    _reject_target_function_params(
        dag_pool=dag_pool,
        consumer_arg_names=arg_names,
        state_names=frozenset(target_state_names),
        edge_target=ref.regime,
        context="build_fallback_state_projector",
    )

    provenance_builder = _ProvenanceBuilder(
        states=frozenset(arg for arg in arg_names if arg in target_state_names)
    )
    for arg in arg_names:
        if arg not in target_state_names:
            provenance_builder.expose(qname=arg, namespace=SOURCE_PARAMS, qualify=False)
    arg_provenance = provenance_builder.build(
        outer_arg_names=arg_names, engine_args=set()
    )

    @with_signature(args=list(arg_names), return_annotation="Mapping[str, FloatND]")
    def project(**kwargs: _ParamsLeaf) -> Mapping[str, FloatND]:
        return {
            state_name: projection_funcs[state_name](
                **{arg: kwargs[arg] for arg in projection_args[state_name]}
            )
            for state_name in fallback_v_info.state_names
        }

    # Published for `route_gated_edges` (F3), exactly like the simulate gate
    # evaluator's: the router holds every regime's params and cannot otherwise
    # tell a source-declared projection parameter from a target one.
    project.arg_provenance = arg_provenance  # type: ignore[attr-defined]

    return project


def _assemble_gate_kwargs(
    *,
    gate_arg_names: tuple[str, ...],
    target_components: Mapping[str, FloatND],
    d_value: FloatND | None,
    gate_ref_values: Mapping[str, FloatND],
    state_mesh: Mapping[str, ContinuousState | DiscreteState],
    cell_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Bind each gate argument to its grid array (E3').

    Resolves the gate's declared arguments against the target's own value
    components (``V_target_<s>``), its boolean dissolution flag (``D_target``), the
    gate references, the broadcast target-state grids, and remaining cell kwargs.

    COLLECTIVE-REGIMES (E3', slice 5): ``state_mesh`` carries the target
    regime's own state grids broadcast to the full mesh (`jnp.meshgrid`),
    which may include DISCRETE (int-typed) axes — e.g. EKL's encoded
    spouse-type categorical — not just continuous ones; narrowing this to
    `FloatND` was a stale type hint (see the identical fix on
    `_evaluate_edge_fold`'s ``target_states`` in `backward_induction.py`).
    """
    gate_kwargs: dict[str, object] = {}
    for arg in gate_arg_names:
        if arg in target_components:
            gate_kwargs[arg] = target_components[arg]
        elif arg == "D_target":
            if d_value is None:
                # COLLECTIVE-REGIMES (E4). Solve always publishes a dissolution
                # flag for every active collective regime, so this branch is
                # unreachable there; it fires only when a SIMULATE caller
                # invoked the internal `simulate()` without threading
                # `period_to_regime_to_dissolution_flags` (e.g. the public
                # `Model.simulate()`, which does not yet surface it — see
                # `pylcm-extension-collective-regimes.md` v2.1, slice 6). Fail
                # clearly instead of `None > 0.5`.
                msg = (
                    "This gate reads 'D_target', but no dissolution-flag array "
                    "was supplied for the target regime at this period. "
                    "Forward simulation needs `period_to_regime_to_dissolution_flags` "
                    "(the third element `backward_induction.solve` returns) "
                    "threaded through to `simulate()`; the public "
                    "`Model.simulate()` does not yet surface it."
                )
                raise NotImplementedError(msg)
            gate_kwargs[arg] = d_value > _D_THRESHOLD
        elif arg in gate_ref_values:
            gate_kwargs[arg] = gate_ref_values[arg]
        elif arg in state_mesh:
            gate_kwargs[arg] = state_mesh[arg]
        else:
            gate_kwargs[arg] = cell_kwargs[arg]
    return gate_kwargs


def build_reference_params_mapping_for_fold(
    *,
    edge: ResolvedGatedEdge,
    flat_params: Mapping[RegimeName, Mapping[str, _ParamsLeaf]],
) -> MappingProxyType[RegimeName, Mapping[str, _ParamsLeaf]]:
    """Assemble `SAME_PERIOD_PARAMS_ARG` for one edge's reference readers (F4).

    The params counterpart of `build_same_period_mapping_for_fold`, over the
    same key set: every regime whose same-period V an edge reads — the target
    itself, plus each gate ref's and each leg fallback's reference regime —
    mapped to that regime's OWN flat params, so a reference reader interpolates
    over the REFERENCE regime's grid with the REFERENCE regime's own runtime grid
    points, rather than with an identically named param of whichever regime
    happened to supply the reader's kwargs (see `Q_and_F.SAME_PERIOD_PARAMS_ARG`).
    """
    return MappingProxyType(
        {
            regime_name: flat_params[regime_name]
            for regime_name in dict.fromkeys((edge.target, *edge.reference_regimes))
        }
    )


def build_same_period_mapping_for_fold(
    *,
    edge: ResolvedGatedEdge,
    period_solution: Mapping[RegimeName, FloatND],
    period_dissolution_flags: Mapping[RegimeName, BoolND],
) -> MappingProxyType[RegimeName, FloatND]:
    """Assemble the same-period value mapping the fold reads for one edge.

    Carries the target regime's V, its dissolution flag cast to float (under the
    reserved `D_KEY_SUFFIX` key), and every reference regime's V — all
    period-``t`` arrays, still live at the fold site.
    """
    mapping: dict[RegimeName, FloatND] = {edge.target: period_solution[edge.target]}
    d_flag = period_dissolution_flags.get(edge.target)
    if d_flag is not None:
        mapping[f"{edge.target}{D_KEY_SUFFIX}"] = jnp.asarray(d_flag, dtype=float)
    for regime_name in edge.reference_regimes:
        mapping[regime_name] = period_solution[regime_name]
    return MappingProxyType(mapping)
