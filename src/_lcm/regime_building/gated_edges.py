"""Gated edge objects: mutual-consent marriage / dissolution routing.

Gated edges unlock MIXED singleton/collective regime topologies. A source
regime declares, per target regime, a `GatedEdge`. At the END of each
period `t`'s solve — after every period-`t` regime is solved, so their value
arrays and dissolution flags are still live — the engine folds, for each
declared edge and each source stakeholder `s`, a gated continuation object on
the TARGET regime's period-`t` grid:

    Wbar^s(x) = jnp.where( gate(x), V_target^{leg_s}(x), V_fallback^s(pi_s(x)) )

with `gate` a boolean user function on the target grid — a mutual-consent
predicate, or a no-dissolution one — and `V_fallback^s` a same-period reference
regime's value at a projection, typically the source stakeholder's own singleton
regime. The source's period `t-1` continuation then reads `Wbar` in place of the
raw target V, threaded through the ordinary transition machinery
(`next_regime_to_V_arr`).

**Numerics (non-negotiable).** The mixture is the strict
`jnp.where(gate, V_target, V_fallback)`, never a linear
`gate*V_target + (1-gate)*V_fallback`: the target value carries the `-inf`
sentinel in dissolution cells, and `0 * -inf = NaN`. Every read that
lands the target value — including the gate's own `V_target_<s>` reads — is an
on-grid identity-projection interpolation, so it is exact.

The whole fold reuses the same-period reference-reader machinery
(`_build_same_period_ref_reader`): the target's own value components and dissolution
flag are read as identity-projection references of the target regime, and the
gate refs / leg fallbacks as ordinary projected references. The per-cell fold is
product-mapped over the target regime's state grid.
"""

from collections.abc import Callable, Container, Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import NoReturn, cast

import jax.numpy as jnp
from dags import (
    concatenate_functions,
    get_ancestors,
    rename_arguments,
    with_signature,
)
from dags.tree import qname_from_tree_path

from _lcm.regime_building.Q_and_F import (
    SAME_PERIOD_PARAMS_ARG,
    SAME_PERIOD_V_ARG,
    ResolvedSamePeriodRef,
    _build_same_period_ref_reader,
    projection_func_or_fail,
)
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from _lcm.typing import (
    ConstraintFunction,
    EconFunctionsMapping,
    FunctionName,
    RegimeName,
    StateName,
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

# `V_arr_name`s under which the
# target's own (per-component) value array and its float dissolution flag are
# bound inside `get_edge_simulate_gate_evaluator`'s two interpolators. Never
# real regime or gate-ref names.
_SIMULATE_TARGET_V_ARR_NAME = "__simulate_target_component_v_arr__"
_SIMULATE_D_ARR_NAME = "__simulate_target_D_arr__"

# The two parameter namespaces a
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

# Template-entry name of an edge's gate predicate. An edge callable's free
# scalars are ordinary model parameters of the SOURCE regime, and they carry the
# flat name `<target>__<entry>__<param>` there — the qualification is what keeps
# them apart from the source's own parameters, which share that one flat
# namespace: a runtime irregular grid's helper is named after the STATE alone
# (`x__points`), and two edges of one source would otherwise collide on any
# parameter name they happen to share.
EDGE_GATE_ENTRY: FunctionName = "gate"


def edge_gate_ref_entry(*, ref_name: str, state_name: StateName) -> FunctionName:
    """Return the template-entry name of one gate-reference projection.

    Args:
        ref_name: Key of the reference in the edge's `gate_refs`.
        state_name: State of the reference regime this projection supplies.

    Returns:
        The entry name the projection's parameters are collected under.

    """
    return f"gate_ref_{ref_name}_{state_name}"


def edge_leg_fallback_entry(
    *, fallback_regime: RegimeName, state_name: StateName
) -> FunctionName:
    """Return the template-entry name of one leg-fallback projection.

    The leg is named by the regime it falls back to rather than by its key in
    the edge's `legs`: the simulate-side projector
    (`build_fallback_state_projector`) is handed the leg's resolved fallback
    reference and nothing else, so the fallback regime is the one leg identity
    both sides of the solve/simulate seam can spell. Two legs of one edge
    falling back to the same regime therefore share one parameter namespace,
    which is what a single flat source namespace gives them anyway.

    Args:
        fallback_regime: Regime the leg falls back to.
        state_name: State of the fallback regime this projection supplies.

    Returns:
        The entry name the projection's parameters are collected under.

    """
    return f"leg_fallback_{fallback_regime}_{state_name}"


def edge_param_qname(*, target: RegimeName, entry: FunctionName, param: str) -> str:
    """Return the flat name one edge callable's parameter carries in the source.

    Args:
        target: Regime the edge lands on.
        entry: Entry name of the callable within the edge.
        param: Name the callable declares the parameter under.

    Returns:
        The parameter's qname in the source regime's flat params.

    """
    return qname_from_tree_path((target, entry, param))


def is_target_value_operand(arg_name: str) -> bool:
    """Return whether an edge callable's argument names a target value component.

    A gate reads the target regime's value as `V_target` for a singleton target
    and as `V_target_<stakeholder>` for a collective one. The whole `V_target`
    vocabulary is reserved to the engine, so a name in it is an operand the fold
    binds rather than a parameter the user supplies.

    Args:
        arg_name: Argument name of a gate or projection.

    Returns:
        Whether the name belongs to the reserved `V_target` vocabulary.

    """
    return arg_name == "V_target" or arg_name.startswith("V_target_")


def gate_reads_dissolution_flag(*, edge: ResolvedGatedEdge) -> bool:
    """Return whether an edge's gate predicate reads the target's flag `D`.

    The gate's own declared arguments are the complete answer: a gate argument
    may not name a node of the target's DAG
    (`_reject_gate_projection_target_node_read`), so concatenating the predicate
    with the target functions adds no argument to it and `D_target` reaches the
    gate only by being declared on it.

    Args:
        edge: The resolved edge declaration.

    Returns:
        Whether the gate declares the `D_target` operand.

    """
    return "D_target" in get_union_of_args([edge.gate])


@dataclass(frozen=True, kw_only=True)
class EdgeArgProvenance:
    """Where each exposed argument of an edge-side simulate callable comes from.

    The simulate-side router
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
    `SAME_PERIOD_PARAMS_ARG`. This class covers exactly what remains.
    """

    states: frozenset[StateName]
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

    def __init__(self, *, states: frozenset[StateName]) -> None:
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
            # provenances onto one keyword binds an argument from the wrong
            # regime, and neither an unqualified exposure nor a future change
            # to the naming scheme may reintroduce that silently.
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
        missing classification leaves an argument the router would have to guess
        a namespace for; an overlap leaves a name two provenances both claim.

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


def _uncompiled_edge_callable(*_args: object, **_kwargs: object) -> NoReturn:
    """Stand in for a gated-edge callable model processing has not built yet.

    A gated edge is resolved in two stages: the resolution the build-time
    fences read is available before any regime's grid is known, and the folds,
    the simulate gate evaluators, and each leg's fallback projector can only be
    compiled once they are. Only an edge reached through `Regime.gated_edges`
    carries the compiled callables, so calling one on any other edge is a
    staging mistake rather than a bad model, and says so.
    """
    msg = (
        "This gated-edge callable was never compiled. Only edges reached "
        "through `Regime.gated_edges` carry their folds, their simulate gate "
        "evaluators, and their legs' fallback state projectors."
    )
    raise RuntimeError(msg)


@dataclass(frozen=True, kw_only=True)
class ResolvedEdgeLeg:
    """Engine-side form of one source-stakeholder leg of a gated edge."""

    source_stakeholder: str | None
    """Source stakeholder name, or `None` for a singleton source's single leg."""

    target_component_index: int | None
    """Index of the OPEN-branch value on the target V's trailing stakeholder
    axis, or `None` for a singleton target."""

    fallback: ResolvedSamePeriodRef
    """The CLOSED-branch reference value (regime, projection, stakeholder index)."""

    fallback_state_projector: Callable = _uncompiled_edge_callable
    """This leg's FALLBACK state projector.

    Maps a target-grid-coordinate point to the fallback regime's own state
    coordinates (`build_fallback_state_projector`). Compiled at model
    processing and read only by forward simulation's value router, which needs
    the states a routed-away stakeholder carries into its fallback regime; the
    solve-side fold needs the fallback's value, never its raw coordinates.

    Carried per leg rather than alongside the edge because the router consumes
    it once per leg: a separate per-edge sequence would have to be re-paired
    with `ResolvedGatedEdge.legs` positionally at every call.
    """


@dataclass(frozen=True, kw_only=True)
class ResolvedGatedEdge:
    """Engine-side form of a user `GatedEdge`, resolved at model processing."""

    target: RegimeName
    """Name of the target regime whose grid the fold lands on."""

    gate: ConstraintFunction
    """Boolean gate predicate, exactly as the user declared it. The builders
    below rename its parameters to their flat source-regime names
    (`_with_qualified_params`) after the fences have read it as declared."""

    gate_refs: Mapping[str, ResolvedSamePeriodRef]
    """Extra same-period references the gate reads (projected from the target grid)."""

    legs: tuple[ResolvedEdgeLeg, ...]
    """One leg per source component, in SOURCE stakeholder order."""

    reference_regimes: tuple[RegimeName, ...]
    """Deduplicated real regimes whose same-period V the fold reads (fallbacks +
    gate refs), excluding the target itself.

    Excluding the target is right for the consumers that ask "which OTHER
    regimes must be solved and supplied before this edge can fold" — the target
    is already theirs by definition. It is the wrong set for asking which grids
    the fold closes over, because a reference may name the target: use
    `interpolated_regimes`."""

    folds_by_period: MappingProxyType[int, Callable] = MappingProxyType({})
    """Compiled `Wbar` producers, keyed by the FOLD PERIOD — the period whose
    value arrays the fold reads. Read through `fold_at`.

    Built at model processing, in a second pass over the regimes once every
    regime's grid and functions are known. Backward induction evaluates the
    entry for the period the target was solved in, storing `Wbar` in the rolled
    edge-continuation mapping the source's kernel reads at `t - 1`; forward
    simulation, standing at the source's period `t`, evaluates the `t + 1`
    entry from the solved solution and substitutes `Wbar` into the source's own
    continuation. Simulated regime ROUTING reads nothing off these — that is
    `simulate_gate_evaluators_by_period`'s job.

    The key is a period rather than a single callable because a gate reference
    or leg fallback carrying an `AgeSpecializedGrid` is read on that regime's
    own grid, whose nodes move with age while its shape does not: the reference
    reader's coordinate finder closes over those nodes, so one compiled fold
    can only be right at one period, and would silently place the reference
    read on another period's nodes everywhere else. Periods whose read grids
    resolve to identical nodes share one compiled object, so a model with no
    age-specialized read grid carries exactly one.
    """

    simulate_gate_evaluators_by_period: MappingProxyType[int, Callable] = (
        MappingProxyType({})
    )
    """SIMULATE-side gate evaluators, keyed by fold period. Read through
    `simulate_gate_evaluator_at`.

    Built by `get_edge_simulate_gate_evaluator`: recomputes the gate PREDICATE
    at a realized (off-grid or on-grid) candidate target-state point by
    interpolating its VALUE operands — the target's own value components and
    every declared `gate_refs` entry — and re-applying the SAME predicate the
    fold uses. Interpolating the fold's baked boolean `gate` array and
    thresholding the result does not commute with a nonlinear predicate and can
    flip routing decisions near a grid-cell boundary the fold never evaluated.
    `_lcm.simulation.gated_routing.route_gated_edges` calls the entry for the
    period whose arrays it routes against — the source's `period + 1`, the same
    period `substitute_gated_edge_continuations` folded; a `D_target`-reading
    gate still linearly interpolates the dissolution flag and thresholds it (a
    documented residual — see `get_edge_simulate_gate_evaluator`'s docstring).

    Keyed by period for the same reason the folds are, plus one this side owns
    alone: it interpolates the TARGET's own value array too, so an
    age-specialized grid on the target moves its nodes as well.
    """

    @property
    def interpolated_regimes(self) -> tuple[RegimeName, ...]:
        """Every regime whose value the FOLD interpolates, the target included.

        `get_edge_fold` builds one same-period reader per gate reference and per
        leg fallback, each closing over the grid of the regime that reference
        NAMES. A reference may name the gated target — a self-referencing edge —
        and then the fold closes over the target's grid too, which
        `reference_regimes` omits by design.

        This is the set to fingerprint when deciding which fold periods may
        share a compiled object: it is exactly the grids the compiled object
        depends on, so a target no reference names stays out of it and costs
        nothing.
        """
        return tuple(
            dict.fromkeys(
                [leg.fallback.regime for leg in self.legs]
                + [ref.regime for ref in self.gate_refs.values()]
            )
        )

    def fold_at(self, *, period: int) -> Callable:
        """Return the `Wbar` producer for the period whose arrays it folds."""
        return _select_period_callable(
            by_period=self.folds_by_period,
            period=period,
            what="fold",
            target=self.target,
        )

    def simulate_gate_evaluator_at(self, *, period: int) -> Callable:
        """Return the simulate gate evaluator for the period it routes against."""
        return _select_period_callable(
            by_period=self.simulate_gate_evaluators_by_period,
            period=period,
            what="simulate gate evaluator",
            target=self.target,
        )


def _select_period_callable(
    *,
    by_period: Mapping[int, Callable],
    period: int,
    what: str,
    target: RegimeName,
) -> Callable:
    """Pick one period's compiled edge callable, or say why there is none.

    Two distinct absences, and conflating them hides a staging mistake behind
    what looks like a model error:

    - the mapping is empty ⇒ nothing was ever compiled for this edge, i.e. it
      was not reached through `Regime.gated_edges`;
    - the mapping is non-empty but lacks `period` ⇒ the target regime is not
      active there, so no value of it exists to fold.
    """
    if not by_period:
        msg = (
            f"This gated edge's {what} was never compiled. Only edges reached "
            "through `Regime.gated_edges` carry their folds, their simulate "
            "gate evaluators, and their legs' fallback state projectors."
        )
        raise RuntimeError(msg)
    if period not in by_period:
        msg = (
            f"The gated edge to regime '{target}' has no {what} for period "
            f"{period}: '{target}' is not active there, so it holds no value "
            f"to fold. Compiled periods are {sorted(by_period)}."
        )
        raise KeyError(msg)
    return by_period[period]


def _pad_reader_to_state_names(
    reader: Callable[..., FloatND],
    *,
    state_names: tuple[StateName, ...],
) -> Callable[..., FloatND]:
    """Widen a reader's exposed signature to every one of `state_names`.

    `reader`'s own args are kept (states it genuinely reads, plus any extra
    runtime params, e.g. grid points for an irregular-grid projection); any
    `state_names` entry missing from that set is added as an ignored
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


def _reached_target_param_leaves(
    dag_pool: Mapping[FunctionName | TransitionFunctionName, Callable[..., FloatND]],
    seed_args: Iterable[str],
    state_names: frozenset[StateName],
) -> frozenset[str]:
    """Dynamic target-DAG parameter leaves REACHED by one specific edge consumer.

    `seed_args` are a single consumer's OWN declared arguments BEFORE the
    concatenation with the target DAG -- the gate predicate's parameters, or one
    projection's parameters. A seed that names a `dag_pool` node (a target
    regime function or deterministic transition) enters the target's own function
    graph; walking that graph to its leaves and dropping the produced-node names
    and the state coordinates leaves exactly the dynamic parameters the TARGET
    regime binds from `flat_params[target]` and that THIS consumer reaches.

    Restricting to the consumer's actual ancestor closure -- rather than unioning
    the free args of every function in `dag_pool` -- is load-bearing. A
    parameter a source edge declares directly, or an unrelated target
    helper the consumer never calls, is not a leaf reached HERE and must not trip
    the fence; unioning the whole pool would reject those valid topologies purely
    on a name collision.

    The closure is `dags.get_ancestors` over `dag_pool`, the same walk the
    `concatenate_functions` call that actually compiles the consumer performs, so
    the fence sees exactly the arguments that compilation would bind. Two of its
    conventions the seeds have to respect:

    - A target with no function in the pool raises, and `targets=None` would walk
      EVERY node -- the whole-pool union this fence must not take. So the seeds
      are filtered to pool nodes and passed as a list, empty when the consumer
      enters the target graph nowhere.
    - Its ancestor set spans free-parameter nodes as well as function nodes, so
      subtracting the pool's own node names leaves the free leaves.
    """
    seeds = [name for name in seed_args if name in dag_pool]
    ancestors = get_ancestors(dag_pool, targets=seeds, include_targets=False)
    return frozenset(ancestors - set(dag_pool) - set(state_names))


def _projection_seed_args(ref: ResolvedSamePeriodRef) -> frozenset[str]:
    """The OWN declared arguments of every projection function of a same-period ref.

    Seeds the ancestry-aware target-parameter fence for a gate-ref or leg-fallback
    reader: exactly the vocabulary the reader's projections are written in, before
    they are concatenated with the target DAG.
    """
    leaves: set[str] = set()
    for projection in ref.projection.values():
        leaves |= set(get_union_of_args([projection]))
    return frozenset(leaves)


def _with_qualified_params(
    *,
    func: Callable[..., FloatND],
    target: RegimeName,
    entry: FunctionName,
    wired_names: Container[str],
) -> Callable[..., FloatND]:
    """Return `func` with every parameter it declares renamed to its flat name.

    An edge callable is written in the target regime's vocabulary, so what it
    declares is a mix of engine-wired names — the target's states, and for a gate
    the injected value operands — and its own free parameters. Only the latter
    are renamed, to `edge_param_qname`'s `<target>__<entry>__<param>`, which is
    the name the params template gives them in the source regime's flat params.
    Renaming BEFORE the callable is concatenated with the target DAG is what
    keeps the two sides one name: the fold, the simulate gate evaluator, and the
    leg projector each build their signature out of the renamed callable, so
    `backward_induction._evaluate_edge_fold`'s name match and the router's
    provenance lookup both hit the template's own spelling.

    Args:
        func: A gate predicate or one projection of a reference.
        target: Regime the edge lands on.
        entry: Entry name of this callable within the edge.
        wired_names: Names the engine binds itself, which are left alone.

    Returns:
        The callable with its parameters renamed, or `func` when it declares
        none.

    """
    mapper = {
        arg: edge_param_qname(target=target, entry=entry, param=arg)
        for arg in get_union_of_args([func])
        if arg not in wired_names and not is_target_value_operand(arg)
    }
    return rename_arguments(func, mapper=mapper) if mapper else func


def _gate_ref_with_qualified_params(
    *,
    ref: ResolvedSamePeriodRef,
    ref_name: str,
    target: RegimeName,
    state_names: Container[StateName],
) -> ResolvedSamePeriodRef:
    """Return a gate reference whose projections declare their flat param names."""
    return _ref_with_qualified_params(
        ref=ref,
        target=target,
        entry_by_state={
            state_name: edge_gate_ref_entry(ref_name=ref_name, state_name=state_name)
            for state_name in ref.projection
        },
        state_names=state_names,
    )


def _leg_fallback_with_qualified_params(
    *,
    ref: ResolvedSamePeriodRef,
    target: RegimeName,
    state_names: Container[StateName],
) -> ResolvedSamePeriodRef:
    """Return a leg fallback whose projections declare their flat param names."""
    return _ref_with_qualified_params(
        ref=ref,
        target=target,
        entry_by_state={
            state_name: edge_leg_fallback_entry(
                fallback_regime=ref.regime, state_name=state_name
            )
            for state_name in ref.projection
        },
        state_names=state_names,
    )


def _ref_with_qualified_params(
    *,
    ref: ResolvedSamePeriodRef,
    target: RegimeName,
    entry_by_state: Mapping[StateName, FunctionName],
    state_names: Container[StateName],
) -> ResolvedSamePeriodRef:
    """Return `ref` with each projection's parameters renamed to their flat names.

    A projection is evaluated on the TARGET regime's grid, so the target's own
    state names are the engine-wired half of its signature and everything else
    is a parameter of the source that declared the edge.

    Args:
        ref: The resolved reference whose projections are rewritten.
        target: Regime the edge lands on.
        entry_by_state: Entry name to collect each projection's parameters under,
            keyed by the reference-regime state the projection supplies.
        state_names: The target regime's state names.

    Returns:
        A copy of `ref` carrying the renamed projections.

    """
    return replace(
        ref,
        projection=MappingProxyType(
            {
                state_name: _with_qualified_params(
                    func=projection,
                    target=target,
                    entry=entry_by_state[state_name],
                    wired_names=state_names,
                )
                for state_name, projection in ref.projection.items()
            }
        ),
    )


def _reject_target_function_params(
    *,
    dag_pool: Mapping[FunctionName | TransitionFunctionName, Callable[..., FloatND]],
    seed_args: Iterable[str],
    state_names: frozenset[StateName],
    edge_target: RegimeName,
    context: str,
) -> None:
    """Fence a target helper param mis-owned as source.

    Collective-edge provenance binds every non-injected gate/projection argument
    from `flat_params[source]` -- the solve-side fold does the same, so solve
    and simulate stay mutually consistent. But that is NOT consistent with the
    *target* regime's own kernel, which binds a target function's parameter from
    `flat_params[target]`: a consumer that reaches a target-regime function with
    a free dynamic parameter would therefore evaluate that parameter from the
    wrong namespace, and would COLLAPSE with a same-named source parameter,
    reversing the gate.

    `seed_args` are the consumer's OWN declared arguments; the fence walks the
    target DAG's ancestor closure from them (`_reached_target_param_leaves`) so it
    fires exactly when THIS consumer genuinely reaches a target-owned parameter.
    It must be called on EVERY target-DAG-concatenating consumer of an edge -- the
    gate predicate AND each gate-ref / fallback projection reader, which are
    compiled on a separate path and so are easy to leave unchecked -- and
    it is ancestry-aware, so an unrelated same-named target helper does not reject
    a valid direct source parameter.

    Origin-preserving edge compilation (carrying the target/source origin through
    the concatenated DAG, and passing target params as a distinct input to the
    solve-side fold) is not yet implemented. Until it is, reject the topology
    rather than silently misbind it. Source-declared projection parameters are
    NOT affected: they are not leaves of the target DAG reached from the consumer.
    """
    contested = sorted(_reached_target_param_leaves(dag_pool, seed_args, state_names))
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


def _reject_injected_name_collision(
    *,
    injected_names: frozenset[str],
    dag_pool: Mapping[FunctionName | TransitionFunctionName, Callable[..., FloatND]],
    edge_target: RegimeName,
    context: str,
) -> None:
    """Fence a gate operand name that collides with a target-DAG node.

    The gate predicate is compiled as
    `concatenate_functions({**dag_pool, "__gate__": gate})`. Its injected value
    operands -- `V_target` / `V_target_<s>`, `D_target`, and the gate-ref
    keys -- are meant to be free leaves the fold/evaluator fills with the realized
    arrays. If one of those names also names a target function or deterministic
    transition in `dag_pool`, the DAG compiler resolves the gate's argument to
    that TARGET NODE instead, silently substituting an unrelated target value for
    the intended operand (a gate reversal). Reject the collision at construction.
    """
    collisions = sorted(injected_names & set(dag_pool))
    if collisions:
        msg = (
            f"{context}: the edge to regime '{edge_target}' declares injected gate "
            f"operand name(s) {collisions} that collide with the target regime's own "
            "function / deterministic-transition node(s) of the same name. The gate "
            "predicate would then read the target node instead of the injected value "
            "operand (V_target / D_target / a gate-ref), silently substituting an "
            "unrelated quantity. Rename the colliding gate-ref key(s), or the "
            "colliding target function / transition."
        )
        raise ModelInitializationError(msg)


def _reject_gate_projection_target_node_read(
    *,
    dag_pool: Mapping[FunctionName | TransitionFunctionName, Callable[..., FloatND]],
    seed_args: Iterable[str],
    edge_target: RegimeName,
    context: str,
) -> None:
    """Fence a gate/projection arg that DIRECTLY names a target DAG node.

    `_reject_target_function_params` fences a consumer that REACHES a target-owned
    *dynamic parameter*. But a target function / deterministic-transition node that
    depends only on target STATES contributes no dynamic leaf, so that fence stays
    silent -- while `concatenate_functions({**dag_pool, "__consumer__": ...})` still
    resolves a same-named consumer argument to the target NODE. If that name was meant
    as a source parameter (bound from `flat_params[source]`), the source value is
    silently dropped from the compiled signature and replaced by the node's output: a
    gate reversal, a changed solve-side `Wbar`, or a wrong projected fallback
    state.

    Whether the author meant the source value or the target node is NOT decidable at
    construction (the source's edge-param set is not carried in the params template),
    so this fence enforces the only build-time-checkable contract: a gate/projection
    argument must not name a target function / deterministic-transition node at all.
    Compute a target-derived quantity as a source-declared gate-ref PROJECTION (read
    through `_build_same_period_ref_reader`, whose params ARE bound from the source)
    instead of naming the target node directly. This is STRICTER than only rejecting a
    proven source/target collision -- the structural repair, namespace-qualified
    source/target leaves before concatenation, is deferred. Injected operands are
    excluded upstream: `_reject_injected_name_collision` runs first and guarantees no
    injected name is in `dag_pool`, so those never trip this fence.
    """
    entered = sorted(set(seed_args) & set(dag_pool))
    if entered:
        msg = (
            f"{context}: the edge to regime '{edge_target}' declares gate/projection "
            f"argument(s) {entered} that name the TARGET regime's own function / "
            "deterministic-transition node(s). Name-based DAG concatenation would bind "
            "the argument to the target NODE, silently dropping a same-named source "
            "parameter and reversing the gate / changing Wbar / writing the wrong "
            "projected fallback state. This fence is deliberately STRICTER than a "
            "proven source/target collision: whether the name was meant as a source "
            "parameter or the target node is not decidable at build time (the source's "
            "edge-param set is not carried in the params template), so a direct "
            "target-node read is rejected outright. Workaround depends on what you "
            "need: for a target-derived VALUE, add a source-declared gate-ref "
            "projection (its params bind from the source) -- but note a gate ref "
            "returns a REFERENCE REGIME's V at the projected coordinates, not an "
            "arbitrary target helper's output; for a state-only quantity (or a "
            "fallback STATE coordinate, which a gate ref cannot supply) inline the "
            "helper/transition formula directly in the gate/projection using the "
            "target STATES as arguments. General direct target-node reads need either "
            "origin-preserving edge compilation or an explicit per-argument "
            "target-node declaration; neither is implemented yet."
        )
        raise ModelInitializationError(msg)


def _reject_d_target_read_on_singleton_target(
    *,
    gate_arg_names: Iterable[str],
    target_stakeholders: tuple[str, ...] | None,
    edge_target: RegimeName,
    context: str,
) -> None:
    """Fence a gate reading `D_target` on a target that publishes no flag.

    The dissolution flag `D` marks the cells where a COLLECTIVE regime's
    household argmax was taken over an empty feasible set; a singleton regime's
    kernel publishes none. A gate that reads `D_target` on such a target names
    an operand no solved model can supply, and no argument to `solve` or
    `simulate` can repair it — so the declaration is rejected while the model is
    built, where the user can still change it.
    """
    if target_stakeholders is not None or "D_target" not in set(gate_arg_names):
        return
    msg = (
        f"{context}: the edge to regime '{edge_target}' declares a gate reading "
        f"'D_target', but '{edge_target}' is a singleton regime and publishes no "
        "dissolution flag. Only a collective regime — one declaring "
        "`stakeholders` — has a flag `D`, so no argument to `solve` or "
        "`simulate` can supply one for this target. Drop the 'D_target' operand "
        f"from the gate, or declare `stakeholders` on '{edge_target}'."
    )
    raise ModelInitializationError(msg)


def _reject_gate_ref_operand_alias(
    *,
    gate_ref_names: Iterable[str],
    reserved_operand_names: frozenset[str],
    edge_target: RegimeName,
    context: str,
) -> None:
    """Fence a gate-ref key that aliases a built-in injected operand.

    The injected gate operands are assembled into ONE kwargs namespace
    (`_assemble_gate_kwargs`): the target value component(s) `V_target` /
    `V_target_<s>`, the regime-level `D_target`, and every gate-ref value. That
    assembly resolves a name to the target component / `D_target` BEFORE the gate
    refs, and the `injected_names` SET silently collapses a duplicate -- so a public
    `gate_refs` key spelled `V_target` (or `D_target`, or a collective
    `V_target_<s>`) is computed but then discarded, and the built-in operand wins (a
    silent gate reversal). `_reject_injected_name_collision` only checks the injected
    names against `dag_pool`, never the injected categories against EACH OTHER, so it
    misses this. The categories must be disjoint; reject the alias.
    """
    collisions = sorted(set(gate_ref_names) & reserved_operand_names)
    if collisions:
        msg = (
            f"{context}: the edge to regime '{edge_target}' declares gate-ref key(s) "
            f"{collisions} that alias a built-in injected gate operand "
            "(V_target / V_target_<stakeholder> / D_target). The gate assembly would "
            "read the built-in operand and silently discard the computed reference "
            "value. Rename the colliding gate-ref key(s)."
        )
        raise ModelInitializationError(msg)


def _reject_gate_operand_state_name_collision(
    *,
    state_names: Iterable[StateName],
    reserved_operand_names: frozenset[str],
    gate_ref_names: Iterable[str],
    edge_target: RegimeName,
    context: str,
) -> None:
    """Fence a target STATE name that aliases a built-in operand / gate-ref.

    `_assemble_gate_kwargs` resolves each gate argument in a fixed PRECEDENCE order:
    target value component(s) `V_target` / `V_target_<s>` first, then `D_target`,
    then the gate-ref values, and only THEN the target `state_mesh`. So a target
    regime whose own STATE is named `V_target` / `D_target` / `V_target_<s>` has
    that state silently preempted by the injected VALUE operand -- `gate(V_target)`
    reads the target's continuation VALUE, not its realized STATE -- and a gate-ref key
    equal to a target state name silently preempts the state with the projected
    reference value. Either reverses target-vs-fallback routing with no error.
    `_reject_gate_ref_operand_alias` makes the built-ins and gate-ref keys disjoint
    but never checks EITHER category against the target state names, which enter the
    same namespace. Close the gap: the value/D operands, the gate-ref keys, and the
    target state names must be pairwise disjoint.
    """
    state_set = set(state_names)
    reserved_collisions = sorted(state_set & reserved_operand_names)
    gate_ref_collisions = sorted(state_set & set(gate_ref_names))
    if reserved_collisions or gate_ref_collisions:
        parts = [
            (
                f"{context}: the edge to regime '{edge_target}' has target state "
                "name(s) that alias a higher-precedence gate operand in "
                "`_assemble_gate_kwargs`, so the gate would silently read the "
                "operand instead of the state and reverse routing."
            )
        ]
        if reserved_collisions:
            parts.append(
                f" State(s) {reserved_collisions} alias a built-in injected value/D "
                "operand (V_target / V_target_<stakeholder> / D_target)."
            )
        if gate_ref_collisions:
            parts.append(f" State(s) {gate_ref_collisions} alias a gate-ref key.")
        parts.append(" Rename the colliding target state(s) or gate-ref key(s).")
        raise ModelInitializationError("".join(parts))


def _build_target_dag_pool(
    *,
    target_functions: EconFunctionsMapping,
    target_deterministic_transitions: Mapping[
        TransitionFunctionName, TransitionFunction
    ],
) -> dict[FunctionName | TransitionFunctionName, Callable[..., FloatND]]:
    """Return the target-regime nodes a gate or projection resolves against.

    A gated-edge callable is written in the target regime's vocabulary and is
    compiled by concatenating it with these nodes, so the pool fixes which names
    the DAG can bind for it: the target's merged deterministic `next_<state>`
    laws (an edge projects INTO the target's state space, a transition role) and
    the target's own processed functions. The collective Koopmans aggregator
    `H` is left out: it consumes engine-injected continuation values
    (`Q^s = H(u^s, E[V'^s])`), not target-grid quantities, so it is not a node an
    edge callable may read.

    Args:
        target_functions: The target regime's processed functions.
        target_deterministic_transitions: The target regime's merged
            deterministic `next_<state>` laws.

    Returns:
        Dict of node name to callable, the DAG pool every edge-side consumer of
        this target is concatenated with. Keys are the target's function names
        and its deterministic `next_<state>` transition names.

    """
    return {
        **dict(target_deterministic_transitions),
        **{k: v for k, v in target_functions.items() if k != "H"},
    }


def _fence_edge_consumer(
    *,
    dag_pool: Mapping[FunctionName | TransitionFunctionName, Callable[..., FloatND]],
    seed_args: Iterable[str],
    state_names: frozenset[StateName],
    edge_target: RegimeName,
    context: str,
) -> None:
    """Reject both ways a target DAG node can capture one edge consumer's argument.

    Runs on EVERY target-DAG-concatenating consumer — the gate predicate and each
    gate-ref / leg-fallback projection — ancestry-aware from that consumer's own
    declared args:

    - `_reject_target_function_params` rejects reaching a target-owned DYNAMIC
      param, which the edge would bind from the source's namespace.
    - `_reject_gate_projection_target_node_read` closes what the first leaves
      open: a consumer arg naming a STATE-ONLY target node reaches no dynamic
      leaf, so the first fence stays silent while concatenation still rebinds the
      arg to the node and drops a same-named source parameter.

    Args:
        dag_pool: The target regime's DAG nodes (`_build_target_dag_pool`).
        seed_args: The consumer's OWN declared arguments, before concatenation.
        state_names: The target regime's state names.
        edge_target: Regime the edge lands on, named in the diagnostic.
        context: Label of the builder and consumer the fence fired on.

    """
    _reject_target_function_params(
        dag_pool=dag_pool,
        seed_args=seed_args,
        state_names=state_names,
        edge_target=edge_target,
        context=context,
    )
    _reject_gate_projection_target_node_read(
        dag_pool=dag_pool,
        seed_args=seed_args,
        edge_target=edge_target,
        context=context,
    )


@dataclass(frozen=True, kw_only=True)
class _EdgeGateFenceContexts:
    """Diagnostic labels naming which builder and consumer a gate fence fired on.

    Every fence takes a `context` string that prefixes its message. The two gate
    builders compile the identical predicate but are reached from different call
    paths, so they label their fences differently while sharing every check.
    """

    gate: str
    """Label for the gate predicate itself and for the operand-name fences."""

    gate_ref: str
    """`str.format` template for one gate-ref projection, taking `ref_name`."""

    leg_fallback: str | None
    """Label for the leg-fallback projections, or `None` to fence none of them.

    `None` says this builder constructs no fallback reader, so it has no consumer
    to fence — not that the topology is unchecked. `build_fallback_state_projector`
    runs the identical `_fence_edge_consumer` over the identical projections, for
    every leg of every edge.
    """


@dataclass(frozen=True, kw_only=True)
class _CompiledEdgeGate:
    """One edge's compiled gate predicate and the names its operands carry.

    Produced by `_compile_edge_gate` and shared verbatim by the solve-side fold
    and the simulate-side gate evaluator, so the two apply the same predicate to
    the same operand vocabulary and reject the same topologies. Only how the
    operands are OBTAINED differs between them.
    """

    target_component_names: tuple[str, ...]
    """The injected target-value operand names, in stakeholder order."""

    injected_names: frozenset[str]
    """Every engine-bound gate operand: the value components, `D_target`, the
    gate-ref keys."""

    qualified_gate_refs: Mapping[str, ResolvedSamePeriodRef]
    """The gate references with their projections' free params renamed to the
    flat spelling the source's params template gives them."""

    gate_evaluator: Callable[..., BoolND]
    """The gate predicate concatenated with the target regime's DAG nodes."""

    gate_arg_names: tuple[str, ...]
    """`gate_evaluator`'s arguments: the injected operands it reads, the target
    states it reads, and its own free params under their qualified names."""


def _compile_edge_gate(
    *,
    edge: ResolvedGatedEdge,
    state_names: tuple[StateName, ...],
    target_functions: EconFunctionsMapping,
    target_deterministic_transitions: Mapping[
        TransitionFunctionName, TransitionFunction
    ],
    target_stakeholders: tuple[str, ...] | None,
    fence_contexts: _EdgeGateFenceContexts,
) -> _CompiledEdgeGate:
    """Compile and fence one edge's gate predicate against the target DAG.

    The single source of the gate both phases evaluate. Solve
    (`get_edge_fold`) and simulate (`get_edge_simulate_gate_evaluator`) call this
    with the same edge and target, so they compile the identical predicate over
    the identical DAG pool from the identical qualified gate references.

    The fences run over each consumer the CALLING builder constructs, which is
    where the two differ: solve builds a reader per leg fallback and fences them
    here, simulate builds none and fences none. Every leg-fallback projection is
    fenced regardless, by the same `_fence_edge_consumer` inside
    `build_fallback_state_projector`.

    Args:
        edge: The resolved edge declaration.
        state_names: The target regime's state names.
        target_functions: The target regime's processed functions.
        target_deterministic_transitions: The target regime's merged
            deterministic `next_<state>` laws.
        target_stakeholders: The target regime's stakeholders, or `None` for a
            singleton target, which fixes the injected value-operand names.
        fence_contexts: Labels the fences report themselves under, and whether
            the leg-fallback projections are fenced here.

    Returns:
        The compiled gate and its operand vocabulary.

    Raises:
        ModelInitializationError: On any rejected gate / projection topology.

    """
    dag_pool = _build_target_dag_pool(
        target_functions=target_functions,
        target_deterministic_transitions=target_deterministic_transitions,
    )
    target_component_names = (
        tuple(f"V_target_{s}" for s in target_stakeholders)
        if target_stakeholders is not None
        else ("V_target",)
    )
    injected_names = frozenset({*target_component_names, "D_target", *edge.gate_refs})
    reserved_operand_names = frozenset({*target_component_names, "D_target"})
    # An injected operand name that also names a target-DAG node would
    # be captured by that node in the concatenation below.
    _reject_injected_name_collision(
        injected_names=injected_names,
        dag_pool=dag_pool,
        edge_target=edge.target,
        context=fence_contexts.gate,
    )
    # A gate-ref KEY aliasing a built-in injected operand (V_target /
    # D_target) is silently preempted by that operand -- the injected categories
    # must be disjoint, which `_reject_injected_name_collision` does not check.
    _reject_gate_ref_operand_alias(
        gate_ref_names=edge.gate_refs,
        reserved_operand_names=reserved_operand_names,
        edge_target=edge.target,
        context=fence_contexts.gate,
    )
    # A target STATE name aliasing a built-in value/D operand or a gate-ref
    # key is silently preempted by `_assemble_gate_kwargs` precedence (value/D and
    # gate-ref both resolve before the state mesh) -- the gate reads the wrong operand.
    _reject_gate_operand_state_name_collision(
        state_names=state_names,
        reserved_operand_names=reserved_operand_names,
        gate_ref_names=edge.gate_refs,
        edge_target=edge.target,
        context=fence_contexts.gate,
    )

    _fence_edge_consumer(
        dag_pool=dag_pool,
        seed_args=get_union_of_args([edge.gate]),
        state_names=frozenset(state_names),
        edge_target=edge.target,
        context=fence_contexts.gate,
    )
    for ref_name, ref in edge.gate_refs.items():
        _fence_edge_consumer(
            dag_pool=dag_pool,
            seed_args=_projection_seed_args(ref),
            state_names=frozenset(state_names),
            edge_target=edge.target,
            context=fence_contexts.gate_ref.format(ref_name=ref_name),
        )
    if fence_contexts.leg_fallback is not None:
        for leg in edge.legs:
            _fence_edge_consumer(
                dag_pool=dag_pool,
                seed_args=_projection_seed_args(leg.fallback),
                state_names=frozenset(state_names),
                edge_target=edge.target,
                context=fence_contexts.leg_fallback,
            )

    # Every fence above reads the callables AS DECLARED, so the qualification
    # below runs after them: it renames a free parameter to a name no target-DAG
    # node can carry, which is exactly the collision the fences exist to detect.
    qualified_gate = _with_qualified_params(
        func=edge.gate,
        target=edge.target,
        entry=EDGE_GATE_ENTRY,
        wired_names=injected_names | set(state_names),
    )
    qualified_gate_refs = {
        ref_name: _gate_ref_with_qualified_params(
            ref=ref, ref_name=ref_name, target=edge.target, state_names=state_names
        )
        for ref_name, ref in edge.gate_refs.items()
    }

    gate_evaluator = concatenate_functions(
        functions={**dag_pool, "__gate__": qualified_gate},
        targets="__gate__",
        enforce_signature=False,
        set_annotations=True,
    )
    gate_arg_names = tuple(get_union_of_args([gate_evaluator]))
    _reject_d_target_read_on_singleton_target(
        gate_arg_names=gate_arg_names,
        target_stakeholders=target_stakeholders,
        edge_target=edge.target,
        context=fence_contexts.gate,
    )

    return _CompiledEdgeGate(
        target_component_names=target_component_names,
        injected_names=injected_names,
        qualified_gate_refs=MappingProxyType(qualified_gate_refs),
        gate_evaluator=gate_evaluator,
        gate_arg_names=gate_arg_names,
    )


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
    """Build one edge's fold: a jittable `Wbar` producer on the target grid.

    Returns a callable whose keyword arguments are the target regime's state
    grids, the same-period value mapping (under `SAME_PERIOD_V_ARG` — the target
    V, its float dissolution flag, and every reference regime's V) and the gate's
    and projections' flat params. Those params carry the qualified spelling
    `<target>__<entry>__<param>` (`edge_param_qname`), which is the name the
    source regime's params template gives them, so
    `backward_induction._evaluate_edge_fold` binds them by a plain name match
    against `flat_params[source]`. It returns `Wbar` of shape
    `(*target_state_axes, n_source_components)` for a collective source, or
    `(*target_state_axes,)` for a singleton source (a single leg with no
    trailing axis).

    The fold does NOT return its raw grid-level boolean `gate` array; it is
    computed INTERNALLY below, for `Wbar`'s own `jnp.where`, and stays
    there. Handing it to the simulate-side value router
    (`_lcm/simulation/gated_routing.py`) would invite deciding a realized
    subject's REGIME ROUTING by INTERPOLATING that baked boolean array and
    thresholding at 0.5, which does not commute with a nonlinear gate
    predicate (e.g. a strict inequality between two interpolated values):
    interpolate-then-threshold can disagree with threshold-then-interpolate
    arbitrarily close to a grid cell boundary, silently flipping routing
    decisions the fold itself never evaluated at that off-grid point. Simulate
    instead RECOMPUTES the gate from interpolated VALUE OPERANDS via
    `get_edge_simulate_gate_evaluator` (this module).

    **Numerics.** The target regime's OWN value components and dissolution flag are
    read by DIRECT array indexing off the same-period mapping — never by
    interpolation. Linear interpolation of the target's `-inf`-bearing V would
    compute `0 * -inf = NaN` at the grid points ADJACENT to a dissolution cell
    (the zero-weight neighbour), poisoning the OPEN branch before the
    `jnp.where` could guard it. Only the gate references and leg fallbacks —
    which read OTHER (finite) regimes at projected coordinates — are
    interpolated, product-mapped over the target grid.

    Args:
        edge: The resolved edge declaration.
        target_v_info: The target regime's V-interpolation info (its grid).
        target_functions: The target regime's processed functions, used to build
            the target DAG the gate/projections resolve against. Gates and
            projections read target STATES directly; a target helper or
            deterministic-transition NODE may NOT be named directly as a gate/
            projection argument (`_reject_gate_projection_target_node_read` --
            build-time undecidable source/target name collision), so inline its
            formula over the target states or read a VALUE via a gate-ref projection.
        target_deterministic_transitions: The target regime's merged
            deterministic `next_<state>` laws (used to build the target DAG; not
            directly nameable as a projection argument -- see `target_functions`).
        reference_v_info: V-interpolation info per reference regime.
        target_stakeholders: The target regime's stakeholders, or `None`.

    Returns:
        The fold callable producing `Wbar` on the target grid.
    """
    state_names = target_v_info.state_names

    # The gate evaluator: the predicate concatenated with the target DAG, so it
    # may read target states / helper functions; its injected leaves (the
    # `V_target_<s>` components, `D_target`, and the gate-ref names) are bound
    # from the elementwise grid arrays below.
    compiled = _compile_edge_gate(
        edge=edge,
        state_names=state_names,
        target_functions=target_functions,
        target_deterministic_transitions=target_deterministic_transitions,
        target_stakeholders=target_stakeholders,
        # This builder constructs the fallback readers, so it fences them; the
        # simulate-side gate evaluator has none of its own.
        fence_contexts=_EdgeGateFenceContexts(
            gate="get_edge_fold (solve-side gate)",
            gate_ref="get_edge_fold (solve-side gate-ref '{ref_name}' projection)",
            leg_fallback="get_edge_fold (solve-side leg fallback projection)",
        ),
    )
    target_component_names = compiled.target_component_names
    injected_names = compiled.injected_names
    gate_evaluator = compiled.gate_evaluator
    gate_arg_names = compiled.gate_arg_names

    qualified_fallbacks = [
        _leg_fallback_with_qualified_params(
            ref=leg.fallback, target=edge.target, state_names=state_names
        )
        for leg in edge.legs
    ]

    def _grid_reader(reader: Callable[..., FloatND]) -> Callable[..., FloatND]:
        """Product-map an off-grid reference reader over the target grid.

        `productmap` derives its OWN
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
        ref_name: _grid_reader(
            _build_same_period_ref_reader(
                ref=ref,
                v_interpolation_info=reference_v_info[ref.regime],
                functions=target_functions,
                deterministic_transitions=target_deterministic_transitions,
            )
        )
        for ref_name, ref in compiled.qualified_gate_refs.items()
    }
    fallback_readers = [
        _grid_reader(
            _build_same_period_ref_reader(
                ref=ref,
                v_interpolation_info=reference_v_info[ref.regime],
                functions=target_functions,
                deterministic_transitions=target_deterministic_transitions,
            )
        )
        for ref in qualified_fallbacks
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

    # Whether the state mesh below is read at all. `_assemble_gate_kwargs` is its
    # only consumer -- the gate-ref and fallback readers take the raw 1-D grids
    # straight from `kwargs` -- and there a gate argument reaches the state branch
    # exactly by naming a target state: the branches ahead of it (the value
    # components, `D_target`, the gate refs) cannot carry a state name, since
    # `_reject_gate_operand_state_name_collision` rejects that aliasing when the
    # gate is compiled.
    gate_reads_a_state = bool(set(gate_arg_names) & set(state_names))

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
        # reads a state directly (supported for generality; the usual gate
        # reads only value operands, and then nothing looks the mesh up, so the
        # outer product is never formed).
        state_mesh: dict[StateName, ContinuousState | DiscreteState] = {}
        if gate_reads_a_state:
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
    """Build one edge's SIMULATE-side gate evaluator.

    Companion to `get_edge_fold`, consumed only by forward simulation
    (`_lcm.simulation.gated_routing.route_gated_edges`). RECOMPUTES the gate
    predicate at a realized, generically OFF-GRID candidate target state,
    rather than interpolating the solve-side fold's baked boolean `gate`
    array and thresholding the interpolated float at 0.5. Thresholding an
    interpolant does not commute with a nonlinear predicate (e.g. a strict
    inequality between two interpolated values: interpolate-then-compare and
    compare-then-interpolate can disagree arbitrarily close to a grid cell
    boundary), which would silently flip routing decisions the fold never
    evaluated at that point.

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

      This is a KNOWN, NON-CONVERGENT residual: the gate is an APPROXIMATE
      router, not an exact recomputation of the target's realized optimum.
      Two things are wrong with the interpolated read, and neither is a rate
      that refinement cures:

      1. Interpolating `V_target` is not a faithful recompute. `V_target` is
         an ALREADY-MAXIMIZED object and interpolation does not commute with
         a `max`: with target actions `u=x` and `u=1-x` on the grid `{0,1}`,
         both nodes give `V=1`, so the interpolant reads 1 everywhere while
         the true `max_a Q` at `x=0.5` is 0.5. **At an action-envelope kink
         the value error is $O(h)$**; the $O(h^2)$ rate holds only against a
         SMOOTH target, and smoothness is exactly what fails here.
      2. **Value convergence does not imply ROUTING convergence.** The
         consent predicate is discontinuous, so refinement does not cure it
         when the candidate distribution has an atom on the equality surface.
         Take `V(x) = max(x, 1-x)`, a deterministic candidate atom at
         `x=0.5`, and the strict gate `V(x) > 0.5`. On every
         EVEN-cardinality uniform grid the two nodes flanking 0.5 both carry
         `V = 0.5 + h/2`, so `interp(V)(0.5) = 0.5 + h/2 > 0.5` and the gate
         OPENS, while the faithful gate is CLOSED. The value error `h/2`
         vanishes; the routing error stays at probability ONE for every such
         grid. One ordinary envelope kink plus a deterministic draw is
         enough — no pathological density is needed.

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
      target's recomputed `max_a Q` reads `interp(V_{t+2})`). That is not what
      this evaluator does: the value gate is approximate by construction, and
      the two failure modes above are what a caller has to plan around.
    - The BOOLEAN `D_target` operand (a no-dissolution gate) is a DOCUMENTED
      RESIDUAL: the float-cast flag is linearly interpolated and thresholded
      at 0.5 (`_assemble_gate_kwargs`'s `D_target` branch), rather than
      recomputed from `D`'s own underlying per-action value comparison at the
      realized point. Recomputing it would mean re-deriving `D` from
      internals the fold never exposes here, so a gate reading ONLY
      `D_target` (a pure dissolution gate) is only nearest-node-equivalent
      off-grid — linear interpolation plus a threshold — not exact.

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
            deterministic `next_<state>` laws.
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
          helpers internally;
        - the params named by the returned `EdgeArgProvenance`, exposed under
          NAMESPACE-QUALIFIED leaves (`__target_param__x__points` vs
          `__source_param__x__points`). A source param's own qname is the
          edge-qualified `<target>__<entry>__<param>` the params template gives
          it (`edge_param_qname`), so the router finds it in
          `flat_params[source]` under exactly that name.

        The namespace qualification is load-bearing: a runtime grid
        helper is named after the STATE alone (`x__points`), so a source and
        a target that both declare a state `x` contribute the same qname. One
        keyword cannot carry two regimes' arrays, so no merge ORDER is
        correct — the leaves must be distinct. Read the provenance off the
        callable (`.provenance`) rather than name-filtering two param dicts;
        `_ProvenanceBuilder.build` asserts at construction that it PARTITIONS
        the signature (disjoint and complete).
    """
    state_names = target_v_info.state_names

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

    # The SAME gate predicate `get_edge_fold` builds, from the same builder, so
    # solve and simulate apply the exact same function and reject the exact same
    # topologies; only the operands feeding it differ.
    compiled = _compile_edge_gate(
        edge=edge,
        state_names=state_names,
        target_functions=target_functions,
        target_deterministic_transitions=target_deterministic_transitions,
        target_stakeholders=target_stakeholders,
        # No leg-fallback context: this builder constructs no fallback reader.
        # Those projections are fenced by the identical `_fence_edge_consumer`
        # inside `build_fallback_state_projector`, which processing builds for
        # every leg of every edge alongside this evaluator.
        fence_contexts=_EdgeGateFenceContexts(
            gate="get_edge_simulate_gate_evaluator (simulate-side gate)",
            gate_ref=(
                "get_edge_simulate_gate_evaluator (gate-ref '{ref_name}' projection)"
            ),
            leg_fallback=None,
        ),
    )
    target_component_names = compiled.target_component_names
    injected_names = compiled.injected_names
    gate_evaluator = compiled.gate_evaluator
    gate_arg_names = compiled.gate_arg_names
    reads_d_target = "D_target" in gate_arg_names

    # Gate-ref readers: the IDENTICAL per-cell construction `get_edge_fold`
    # uses for its own `gate_ref_readers` (the same qualified refs, off the
    # same builder), but WITHOUT that function's `_grid_reader` product-map
    # wrap — `get_edge_fold` maps these over the full target GRID (solve time,
    # one evaluation per grid cell); here each reader is called directly at ONE
    # realized point (vmapped by the caller over subjects), exactly the
    # off-grid idiom `_build_same_period_ref_reader` is built for everywhere else.
    gate_ref_readers = {
        ref_name: _build_same_period_ref_reader(
            ref=ref,
            v_interpolation_info=reference_v_info[ref.regime],
            functions=target_functions,
            deterministic_transitions=target_deterministic_transitions,
        )
        for ref_name, ref in compiled.qualified_gate_refs.items()
    }
    gate_ref_args = {
        name: tuple(get_union_of_args([reader]))
        for name, reader in gate_ref_readers.items()
    }

    # PROVENANCE. Every non-engine argument is attributed to exactly
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
    # The three provenances — all three are needed; a gate-ref reader's
    # arguments in particular do not all belong to the target:
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
    #    A param introduced by the TARGET regime's OWN
    #    functions/transitions is NOT source-owned — the target regime binds it
    #    from `flat_params[target]` in its own kernel, so attributing it to source
    #    here (and to source in the fold) evaluates it from the wrong namespace and
    #    collapses it with any same-named source param. Origin-preserving edge
    #    compilation is deferred; until then `_reject_target_function_params`
    #    (called above, after `gate_arg_names`) rejects that topology rather than
    #    silently misbinding it, so no target-function param ever reaches this
    #    SOURCE bucket.
    # 3. The REFERENCE regimes' own interpolation grids — resolved inside
    #    `_build_same_period_ref_reader` against `SAME_PERIOD_PARAMS_ARG`,
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
        # solve-side fold's baked boolean gate off-grid. Exact on nodes and
        # interpolated between them -- NOT a recompute of `max_a Q`; see this
        # function's docstring for the residual that leaves.
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
            # `_assemble_gate_kwargs` see the names the solve side passes them.
            cell_kwargs={
                arg: kwargs[exposed] for arg, exposed in gate_extra_exposed.items()
            },
        )
        return jnp.asarray(gate_evaluator(**gate_kwargs))

    # Published for `route_gated_edges`, which has EVERY regime's flat params in
    # hand and no other way to tell which one an arg belongs to.
    evaluate_simulate_gate.arg_provenance = arg_provenance  # ty: ignore[unresolved-attribute]

    return evaluate_simulate_gate


_FALLBACK_PROJECTION_TARGET_PREFIX = "__fallback_state__"


def build_fallback_state_projector(
    *,
    ref: ResolvedSamePeriodRef,
    fallback_simulate_state_names: tuple[StateName, ...],
    target_regime_name: RegimeName,
    target_state_names: tuple[StateName, ...],
    target_functions: EconFunctionsMapping,
    target_deterministic_transitions: Mapping[
        TransitionFunctionName, TransitionFunction
    ],
) -> Callable[..., Mapping[StateName, FloatND]]:
    """Project a target-grid point onto one edge leg's FALLBACK state coordinates.

    Companion to `_build_same_period_ref_reader`
    (which reads the fallback regime's V at these same projected
    coordinates, for the solve-side fold): the simulate-side value router
    does not need the fallback's VALUE (Wbar already folds that in) but does
    need the fallback's own STATE coordinates, to write the dissolved/rejected
    stakeholder's next-period row into `states[fallback.regime]`. Reuses the
    identical projection-function construction (same `dag_pool`, same
    `concatenate_functions` targets) so the coordinates are guaranteed
    consistent with whatever the fold read.

    **Which states it projects.** A written row is a SIMULATE-phase object, so
    the projector covers the fallback regime's simulate states — its solve
    states plus the states it carries only in simulation
    (`Phased(solve=..., simulate=Grid)`). A carried state is no axis of the
    fallback's solved V, so the fold's reader has nothing to read on it and
    stays on the solve states; but forward simulation carries it per subject,
    and a slot the projector skips keeps whatever the row held before the edge
    routed it there. The two consumers therefore read the same projection
    mapping over different, nested name sets, each picking its own by name.

    **Provenance.** Consistency with the fold is a claim about the projection's
    INPUTS, not only about how it is built.
    `backward_induction._evaluate_edge_fold` binds every free parameter the
    fold's fallback reader needs from `flat_params[SOURCE]`, so the router must
    too. Calling this projector with `{**candidate_target_states,
    **flat_params[TARGET]}` instead would, for a projection `z = x + shift`
    declared on a source with `shift = 1.0` against a target with
    `shift = 9.0`, read the fallback's V at `x + 1.0` when the solve-side fold
    folded `Wbar` but write the simulated row into the fallback regime at state
    `x + 9.0` — the right regime with a state the solved policy never priced,
    carried on into the next period (and a crash outright where the target
    lacks the source's parameter). The published `arg_provenance` therefore
    attributes every argument explicitly: the target's own STATES (the realized
    candidate point, the simulate counterpart of the target grid the fold maps
    over) and, for everything else, the SOURCE's params — which is not a
    preference between two merges but the only choice that makes the simulated
    coordinate equal the one the fold projected.

    A projection DAG CAN route through a target helper FUNCTION whose own free
    params the target regime binds from `flat_params[target]`; binding those
    from the source at the fold would evaluate them from the wrong namespace and
    collapse them with any same-named source param. That is not a valid provenance
    for a target-owned parameter, so `_reject_target_function_params` (called after
    `arg_names` below) FENCES a projection that reaches such a param rather than
    misbinding it. Source-declared projection params (a `shift` the edge itself
    names) are unaffected: they are not leaves of the target DAG. Origin-preserving
    edge compilation (carrying target/source origin through the DAG and passing
    target params to the solve-side fold) would lift the fence; it is deferred.

    Unlike `get_edge_simulate_gate_evaluator`, this callable exposes both kinds
    of argument without a namespace prefix: it holds no interpolator of its own,
    so the target-params namespace (the source of the identically-named-leaf
    problem there) is empty here and nothing can collide. A parameter still
    carries the edge-qualified qname `<target>__<entry>__<param>`
    (`edge_leg_fallback_entry`), which is both what the params template emits and
    what the solve-side fold's reader for this leg declares.

    Args:
        ref: The leg's resolved fallback reference (`ResolvedEdgeLeg.fallback`).
        fallback_simulate_state_names: Simulate-phase state names of the FALLBACK
            regime (`ref.regime`) — its solve states plus its carried-only ones,
            i.e. exactly the slots a routed row occupies there.
        target_regime_name: The regime the gated edge lands on, which qualifies
            the projections' parameter names and names the fenced target.
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
    dag_pool = _build_target_dag_pool(
        target_functions=target_functions,
        target_deterministic_transitions=target_deterministic_transitions,
    )
    # The same fences the solve-side fold runs over this very leg's fallback
    # projections, from the same helper. `dag_pool` is the GATED TARGET's nodes, so
    # the fence's `edge_target` names the gated target whose node would capture a
    # projection arg -- NOT `ref.regime`, which is the FALLBACK regime the projection
    # maps INTO; naming that one would mislabel the diagnostic.
    _fence_edge_consumer(
        dag_pool=dag_pool,
        seed_args=_projection_seed_args(ref),
        state_names=frozenset(target_state_names),
        edge_target=target_regime_name,
        context="build_fallback_state_projector",
    )

    # The same qualification the solve-side fold applies to this very leg's
    # fallback reader, from the same helper, so the coordinate simulate projects
    # is read off the parameter the fold projected it with.
    qualified_ref = _leg_fallback_with_qualified_params(
        ref=ref, target=target_regime_name, state_names=target_state_names
    )
    projection_funcs: dict[StateName, Callable[..., FloatND]] = {}
    projection_args: dict[StateName, tuple[str, ...]] = {}
    for state_name in fallback_simulate_state_names:
        target = f"{_FALLBACK_PROJECTION_TARGET_PREFIX}{state_name}"
        projection_funcs[state_name] = concatenate_functions(
            functions={
                **dag_pool,
                target: projection_func_or_fail(
                    ref=qualified_ref, state_name=state_name
                ),
            },
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

    provenance_builder = _ProvenanceBuilder(
        states=frozenset(arg for arg in arg_names if arg in target_state_names)
    )
    for arg in arg_names:
        if arg not in target_state_names:
            provenance_builder.expose(qname=arg, namespace=SOURCE_PARAMS, qualify=False)
    arg_provenance = provenance_builder.build(
        outer_arg_names=arg_names, engine_args=set()
    )

    @with_signature(
        args=list(arg_names), return_annotation="Mapping[StateName, FloatND]"
    )
    def project(**kwargs: _ParamsLeaf) -> Mapping[StateName, FloatND]:
        return {
            state_name: projection_funcs[state_name](
                **{arg: kwargs[arg] for arg in projection_args[state_name]}
            )
            for state_name in fallback_simulate_state_names
        }

    # Published for `route_gated_edges`, exactly like the simulate gate
    # evaluator's: the router holds every regime's params and cannot otherwise
    # tell a source-declared projection parameter from a target one.
    project.arg_provenance = arg_provenance  # ty: ignore[unresolved-attribute]

    return project


def _assemble_gate_kwargs(
    *,
    gate_arg_names: tuple[str, ...],
    target_components: Mapping[str, FloatND],
    d_value: FloatND | None,
    gate_ref_values: Mapping[str, FloatND],
    state_mesh: Mapping[StateName, ContinuousState | DiscreteState],
    cell_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Bind each gate argument to its grid array.

    Resolves the gate's declared arguments against the target's own value
    components (`V_target_<s>`), its boolean dissolution flag (`D_target`), the
    gate references, the broadcast target-state grids, and remaining cell kwargs.

    `state_mesh` carries the target regime's own state grids broadcast to the
    full mesh (`jnp.meshgrid`), which may include DISCRETE (int-typed) axes —
    an encoded categorical, say — not just continuous ones, so it must not be
    narrowed to `FloatND` (the same holds for `_evaluate_edge_fold`'s
    `target_states` in `backward_induction.py`).
    """
    gate_kwargs: dict[str, object] = {}
    for arg in gate_arg_names:
        if arg in target_components:
            gate_kwargs[arg] = target_components[arg]
        elif arg == "D_target":
            if d_value is None:
                # A mapping from `build_same_period_mapping_for_fold` always
                # carries a flag entry for the target, and that builder refuses
                # a gate reading `D_target` with no flag to read, so this
                # guards a hand-assembled mapping only. Fail clearly instead of
                # `None > 0.5`.
                msg = (
                    "This gate reads 'D_target', but the same-period value "
                    "mapping carries no dissolution-flag array for the target "
                    "regime. Build it with `build_same_period_mapping_for_fold`, "
                    "passing the `dissolution_flags` field of `solve`'s result."
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


def source_reads_folded_wbar(
    *,
    source_active_periods: Container[int],
    fold_period: int,
) -> bool:
    """Say whether the `Wbar` folded at `fold_period` is read by anyone.

    A gated edge's `Wbar` lands on the TARGET regime's grid at the period the
    target's value function was tabulated on, and the SOURCE consumes it one
    period earlier, as the continuation of its own decision. So the fold at
    period `t` is read exactly when the source is active at `t - 1`. A
    target-active period with no source one period earlier — a self-loop edge
    at the target's earliest active period, say — produces a `Wbar` no
    decision ever reads.

    That distinction is what separates a boundary no-op from a misconfigured
    edge: at an unread period a reference regime may legitimately be unsolved,
    at a read period it may not (`edge_may_fold_at_period`).
    """
    return (fold_period - 1) in source_active_periods


def edge_may_fold_at_period(
    *,
    edge: ResolvedGatedEdge,
    source_name: RegimeName,
    fold_period: int,
    solved_regimes: Container[RegimeName],
    source_reads_wbar: bool,
) -> bool:
    """Answer whether one gated edge's `Wbar` may be folded on this period.

    The single answer both phases consult — backward induction rolling an
    edge forward through the periods, and forward simulation substituting
    `Wbar` for the raw target V. A fold needs every array it reads to be
    solved at `fold_period`: the target's own V, and each reference regime's
    (`ResolvedGatedEdge.reference_regimes` — leg fallbacks and gate refs).
    Three outcomes:

    - the target is unsolved ⇒ `False`. The edge does not exist at this period
      (a repeating edge past its target's activity boundary, e.g.), so the
      caller keeps whatever value it already holds.
    - the target and every reference are solved ⇒ `True`.
    - the target is solved and a reference is not ⇒ `False` when no source
      reads this period's `Wbar`, and a rejection when one does. An
      unread `Wbar` may be left at its previous value; a read one cannot,
      because the source would silently consume a stale later-period value
      instead of the reference the edge declares.

    `source_reads_wbar` is the caller's answer to `source_reads_folded_wbar`
    — the one fact the two phases do not share. Backward induction folds
    period `t` for a consumer at `t - 1` that may not exist and has to
    ask; forward simulation folds `period + 1` on behalf of the very source
    it is simulating at `period`, so it is the consumer and passes `True`.

    Args:
        edge: The resolved edge whose `Wbar` would be folded.
        source_name: Name of the regime declaring the edge, for the message.
        fold_period: The period whose value arrays the fold reads.
        solved_regimes: Names of the regimes solved at `fold_period`.
        source_reads_wbar: Whether the source consumes this period's `Wbar`.

    Raises:
        ModelInitializationError: A reference regime is unsolved at a period
            whose `Wbar` the source reads.
    """
    if edge.target not in solved_regimes:
        return False
    missing = tuple(
        regime_name
        for regime_name in edge.reference_regimes
        if regime_name not in solved_regimes
    )
    if not missing:
        return True
    if not source_reads_wbar:
        return False
    msg = (
        f"Regime '{source_name}', gated_edges['{edge.target}']: the target "
        f"regime '{edge.target}' is solved at period {fold_period}, but the "
        f"edge's reference regime(s) {missing} are not — a malformed ACTIVE "
        "edge (a fallback or gate reference regime must be solved at the same "
        "period as the target whenever the target itself is, and this period's "
        f"Wbar is read by '{source_name}' at period {fold_period - 1}). "
        f"Declare the missing reference regime active at period {fold_period}, "
        "or drop the reference."
    )
    raise ModelInitializationError(msg)


def build_reference_params_mapping_for_fold(
    *,
    edge: ResolvedGatedEdge,
    flat_params: Mapping[RegimeName, Mapping[str, _ParamsLeaf]],
) -> MappingProxyType[RegimeName, Mapping[str, _ParamsLeaf]]:
    """Assemble `SAME_PERIOD_PARAMS_ARG` for one edge's reference readers.

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
    period-`t` arrays, still live at the fold site.

    The key set does not depend on whether a flag was supplied: the mapping is
    an argument of one jitted fold that both backward induction and forward
    simulation call, and a key present on one side and absent on the other
    traces that fold twice. A target with no flag this period gets an unread
    stand-in of the shape a flag would have — unread because a gate that
    consumes `D_target` without a flag to consume is refused below.

    Raises:
        NotImplementedError: The edge's gate reads `D_target` but no dissolution
            flag was supplied for the target regime at this period.
    """
    target_V = period_solution[edge.target]
    d_flag = period_dissolution_flags.get(edge.target)
    if d_flag is None and gate_reads_dissolution_flag(edge=edge):
        msg = (
            f"The gated edge into '{edge.target}' has a gate reading 'D_target', "
            "but no dissolution-flag array was supplied for that regime at this "
            "period. Forward simulation needs "
            "`period_to_regime_to_dissolution_flags` (the `dissolution_flags` "
            "field of `solve`'s result). Either let `Model.simulate` solve first "
            "(auto-solve threads them), or pass "
            "`Model.solve(return_dissolution_flags=True)`'s flags to "
            "`Model.simulate(period_to_regime_to_dissolution_flags=...)`."
        )
        raise NotImplementedError(msg)
    mapping: dict[RegimeName, FloatND] = {
        edge.target: target_V,
        f"{edge.target}{D_KEY_SUFFIX}": (
            jnp.asarray(d_flag, dtype=float)
            if d_flag is not None
            else _unsupplied_dissolution_flag(edge=edge, target_V=target_V)
        ),
    }
    for regime_name in edge.reference_regimes:
        mapping[regime_name] = period_solution[regime_name]
    return MappingProxyType(mapping)


def _unsupplied_dissolution_flag(
    *, edge: ResolvedGatedEdge, target_V: FloatND
) -> FloatND:
    """Return the stand-in flag array for a target that supplied none.

    Shaped like the flag the target would publish — the target's state axes,
    which for a collective target is its V without the trailing stakeholder
    axis — so the fold is traced once whether or not a flag was supplied.
    """
    collective_target = any(leg.target_component_index is not None for leg in edge.legs)
    shape = target_V.shape[:-1] if collective_target else target_V.shape
    return jnp.zeros(shape)
