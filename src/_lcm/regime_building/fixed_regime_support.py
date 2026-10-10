"""Remove ordinary regime edges proved to have constant zero probability.

Only construction-time fixed leaves may feed the probability or its ordinary DAG
ancestors. States, actions, time, transition outputs and free parameters make an
edge conditional; no state probes or runtime values narrow the graph.

A removed edge takes every declaration toward its target with it, so the pruned
model, including its validation, is the one an author would write without that
edge:

- per-target state laws and joint kernels toward the target leave unchecked;
  target-cell ownership is validated on the effective graph alone, so every
  live edge keeps its checks;
- a source state whose only authored law was such a kernel gets the empty
  per-target law `{}`: the state is covered and no target cell is produced;
- the states and actions those removed declarations read are recorded only to
  explain an error: pruning removes as much as it can up front, so a variable
  read only across a removed edge is unused, exactly as without that edge.
"""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
import pandas as pd
from dags.tree import qname_from_tree_path
from jax import Array

from _lcm.params.edges import EDGES
from _lcm.params.processing import (
    cast_params_to_canonical_dtypes,
    find_param_candidates,
)
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.regime_building.schedules import ProbabilityCell
from _lcm.regime_law import RegimeLaw, RegimeLaws, bind_regime_law
from _lcm.typing import (
    EconFunctionArg,
    FlatParams,
    FlatRegimeParams,
    ParamsLeaf,
    QualifiedName,
    RegimeName,
    StateName,
)
from _lcm.utils.functools import is_user_function
from _lcm.utils.namespace import flatten_regime_namespace
from lcm.exceptions import InvalidNameError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.regime import StateTransitionEntry
from lcm.temporal import TimeVarying
from lcm.transition import ByAge, JointTransition, StochasticTransition
from lcm.typing import ReferenceName, UserFunction, UserParams, UserParamsLeaf

type Side = Literal["solve", "simulate"]
# A joint kernel as a regime declares it toward one target: one kernel for both
# phases, or a `Phased` pair of kernels.
type _JointDeclaration = JointTransition | Phased
type _JointDeclarations = Mapping[RegimeName, Mapping[str, _JointDeclaration]]


@dataclass(frozen=True, kw_only=True)
class FixedRegimeSupport:
    """Keep the reduced declarations and exact consumed fixed-key provenance."""

    user_regimes: MappingProxyType[RegimeName, UserRegime]
    """Regimes without the declarations toward removed edges."""

    laws: RegimeLaws
    """Laws with constant-zero ordinary transition cells removed."""

    consumed_param_keys: frozenset[str]
    """Supplied flat keys used to prove a removed cell constant and zero."""

    removed_edge_reads: MappingProxyType[
        RegimeName, MappingProxyType[str, tuple[RegimeName, ...]]
    ]
    """Per regime, each state or action read by a declaration removed with its
    zero edge, and the targets of those edges; used only in error messages."""


def prune_fixed_regime_support(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    fixed_params: UserParams,
) -> FixedRegimeSupport:
    """Remove exactly zero cells whose entire dependency graph is fixed.

    Per-target state laws toward a target removed from their phase are omitted
    too, and so are joint kernels toward a target removed from both phases. Bare
    laws and source variables remain declared. Fixed keys consumed by removed
    cells retain their provenance for later unknown-key validation.
    """
    fixed_flat = flatten_regime_namespace(fixed_params)
    consumed: set[str] = set()
    result: dict[RegimeName, UserRegime] = {}
    pruned_laws: dict[RegimeName, RegimeLaw] = {}
    removed_edge_reads: dict[
        RegimeName, MappingProxyType[str, tuple[RegimeName, ...]]
    ] = {}
    for name, regime in user_regimes.items():
        transition, removed, law_keys = _prune_regime_transition(
            regime_name=name, regime=regime, law=laws[name], fixed_flat=fixed_flat
        )
        consumed.update(law_keys)
        removed_in_both = removed["solve"] & removed["simulate"]
        reads: dict[RegimeName, set[str]] = {}
        joint_transitions = _trim_joint_transitions(
            removed=removed_in_both,
            regime_name=name,
            regime=regime,
            fixed_flat=fixed_flat,
            consumed=consumed,
            reads=reads,
        )
        state_transitions = {
            state: _trim_state_law(
                law=law,
                removed=removed,
                regime_name=name,
                state=state,
                regime=regime,
                fixed_flat=fixed_flat,
                consumed=consumed,
                reads=reads,
            )
            for state, law in regime.state_transitions.items()
        }
        # The pruned model equals the edge-free model, in which the author
        # declares these states' laws as empty per-target mappings.
        state_transitions |= dict.fromkeys(
            _states_covered_only_by_removed_joints(
                regime=regime, joint_transitions=joint_transitions
            ),
            MappingProxyType({}),
        )
        variables = set(regime.states) | set(regime.actions)
        removed_edge_reads[name] = MappingProxyType(
            {
                variable: tuple(
                    sorted(target for target, read in reads.items() if variable in read)
                )
                for variable in sorted(variables & set().union(*reads.values()))
            }
        )
        result[name] = regime.replace(
            state_transitions=state_transitions,
            joint_transitions=joint_transitions,
        )
        pruned_laws[name] = bind_regime_law(
            transition, gated_edges=laws[name].gated_edges
        )
    return FixedRegimeSupport(
        user_regimes=MappingProxyType(result),
        laws=MappingProxyType(pruned_laws),
        consumed_param_keys=frozenset(consumed),
        removed_edge_reads=MappingProxyType(removed_edge_reads),
    )


def _prune_regime_transition(
    *,
    regime_name: RegimeName,
    regime: UserRegime,
    law: RegimeLaw,
    fixed_flat: Mapping[QualifiedName, UserParamsLeaf],
) -> tuple[object, dict[Side, frozenset[str]], frozenset[str]]:
    """Remove zero cells, keeping a joint-lottery edge in both phases or neither.

    A joint kernel is declared once for both phases, so its edge can leave the
    effective graph only where it is removed in both. An edge removed in one
    phase only keeps its zero cell there, which prices it at exactly zero.

    Returns:
        The pruned law, each phase's targets removed at every age, and the fixed
        keys consumed by removed cells.
    """
    protected: frozenset[str] = frozenset()
    while True:
        consumed: set[str] = set()
        transition = _prune_law(
            law=law.transition,
            side=None,
            regime_name=regime_name,
            regime=regime,
            fixed_flat=fixed_flat,
            consumed=consumed,
            protected=protected,
        )
        removed: dict[Side, frozenset[str]] = {
            side: _targets(law=law.transition, side=side)
            - _targets(law=transition, side=side)
            for side in ("solve", "simulate")
        }
        one_phase_joint = frozenset(regime.joint_transitions) & (
            removed["solve"] ^ removed["simulate"]
        )
        if not one_phase_joint - protected:
            return transition, removed, frozenset(consumed)
        protected |= one_phase_joint


def _joint_kernels(raw: _JointDeclaration) -> tuple[tuple[Side, JointTransition], ...]:
    """Pair each phase with the joint kernel a declaration uses there."""
    return tuple(
        (side, cast("JointTransition", getattr(raw, side)))
        if isinstance(raw, Phased)
        else (side, raw)
        for side in ("solve", "simulate")
    )


def _states_covered_only_by_removed_joints(
    *, regime: UserRegime, joint_transitions: _JointDeclarations
) -> tuple[StateName, ...]:
    """Source states whose only authored law is a removed joint kernel's output.

    Without the removed edge, an author would declare each of these states with
    the empty per-target law `{}`; pruning gives them exactly that.
    """
    return tuple(
        sorted(
            state
            for state in _joint_outputs(regime.joint_transitions)
            - _joint_outputs(joint_transitions)
            if state in regime.states
            and state not in regime.state_transitions
            and not isinstance(regime.states[state], _ContinuousStochasticProcess)
        )
    )


def _joint_outputs(joints: _JointDeclarations) -> set[str]:
    """Every state any kernel of `joints` produces, in either phase."""
    return {
        output
        for kernels in joints.values()
        for raw in kernels.values()
        for _, kernel in _joint_kernels(raw)
        for output in kernel.outputs
    }


def _trim_joint_transitions(
    *,
    removed: frozenset[str],
    regime_name: RegimeName,
    regime: UserRegime,
    fixed_flat: Mapping[QualifiedName, UserParamsLeaf],
    consumed: set[str],
    reads: dict[RegimeName, set[str]],
) -> MappingProxyType[RegimeName, Mapping[str, _JointDeclaration]]:
    """Omit joint kernels toward targets removed in both phases at every age."""
    node_names = frozenset(
        kernel_name
        for kernels in regime.joint_transitions.values()
        for kernel_name in kernels
    )
    for target in removed & regime.joint_transitions.keys():
        for kernel_name, raw in regime.joint_transitions[target].items():
            for side, kernel in _joint_kernels(raw):
                roles = [
                    ((target, kernel_name, "probabilities"), kernel.probabilities),
                    *(
                        ((target, f"next_{output}"), func)
                        for output, func in kernel.outputs.items()
                    ),
                ]
                if is_user_function(kernel.support):
                    roles.append(((target, kernel_name, "support"), kernel.support))
                for path, func in roles:
                    consumed.update(
                        _get_declared_fixed_keys(
                            func=func,
                            path=(regime_name, *path),
                            side=side,
                            regime=regime,
                            fixed_flat=fixed_flat,
                            ancestors=(),
                            non_params=node_names,
                            reads=reads.setdefault(target, set()),
                        )
                    )
    return MappingProxyType(
        {
            target: kernels
            for target, kernels in regime.joint_transitions.items()
            if target not in removed
        }
    )


def _prune_law(
    *,
    law: object,
    side: Side | None,
    regime_name: RegimeName,
    regime: UserRegime,
    fixed_flat: Mapping[QualifiedName, UserParamsLeaf],
    consumed: set[str],
    protected: frozenset[str],
) -> object:
    if isinstance(law, ByAge):
        return law.with_mapped_laws(
            func=lambda case: _prune_law(
                law=case,
                side=side,
                regime_name=regime_name,
                regime=regime,
                fixed_flat=fixed_flat,
                consumed=consumed,
                protected=protected,
            )
        )
    if isinstance(law, Phased):
        return Phased(
            solve=_prune_law(
                law=law.solve,
                side="solve",
                regime_name=regime_name,
                regime=regime,
                fixed_flat=fixed_flat,
                consumed=consumed,
                protected=protected,
            ),
            simulate=_prune_law(
                law=law.simulate,
                side="simulate",
                regime_name=regime_name,
                regime=regime,
                fixed_flat=fixed_flat,
                consumed=consumed,
                protected=protected,
            ),
        )
    if not isinstance(law, Mapping):
        return law
    if side is None:
        # A shared law can read phase-specific helpers, so its structural zeros
        # must be proved independently in each phase.
        solve = _prune_law(
            law=law,
            side="solve",
            regime_name=regime_name,
            regime=regime,
            fixed_flat=fixed_flat,
            consumed=consumed,
            protected=protected,
        )
        simulate = _prune_law(
            law=law,
            side="simulate",
            regime_name=regime_name,
            regime=regime,
            fixed_flat=fixed_flat,
            consumed=consumed,
            protected=protected,
        )
        return (
            solve
            if set(cast("Mapping[RegimeName, ProbabilityCell]", solve))
            == set(cast("Mapping[RegimeName, ProbabilityCell]", simulate))
            else Phased(solve=solve, simulate=simulate)
        )
    retained: dict[RegimeName, ProbabilityCell] = {}
    removed_keys: set[str] = set()
    for target, cell in law.items():
        evaluated = (
            _evaluate_fixed_function(
                func=cell.func,
                path=(EDGES, regime_name, target),
                side=side,
                regime=regime,
                fixed_flat=fixed_flat,
                ancestors=(),
            )
            if isinstance(cell, StochasticTransition) and target not in protected
            else None
        )
        if evaluated is None:
            retained[target] = cell
            continue
        value, keys = evaluated
        scalar = np.asarray(value)
        if (
            scalar.shape != ()
            or scalar.dtype.kind not in "fiu"
            or not np.isfinite(scalar)
        ):
            retained[target] = cell
            continue
        # Exact zero is a structural predicate, never a tolerance comparison.
        if np.array_equal(scalar, np.zeros_like(scalar)):
            removed_keys.update(keys)
        else:
            retained[target] = cell
    if not retained:
        # Keep an invalid zero-mass row available to the ordinary probability
        # validator; it must not become an empty law or a terminal regime.
        return law
    consumed.update(removed_keys)
    return MappingProxyType(retained)


def _evaluate_fixed_function(
    *,
    func: UserFunction,
    path: tuple[str, ...],
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[QualifiedName, UserParamsLeaf],
    ancestors: tuple[str, ...],
) -> tuple[EconFunctionArg, frozenset[str]] | None:
    """Evaluate a complete fixed DAG, or retain an unresolved conditional edge."""
    arguments: dict[ReferenceName, EconFunctionArg] = {}
    keys: set[str] = set()
    for arg_name, parameter in inspect.signature(func).parameters.items():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return None
        if _is_runtime_argument(arg_name=arg_name, regime=regime):
            return None
        evaluated = _resolve_fixed_argument(
            arg_name=arg_name,
            path=path,
            side=side,
            regime=regime,
            fixed_flat=fixed_flat,
            ancestors=ancestors,
        )
        if evaluated is None:
            return None
        arguments[arg_name], argument_keys = evaluated
        keys.update(argument_keys)
    try:
        value = func(**arguments)
    except Exception:  # noqa: BLE001 -- failure proves no structural zero
        # The ordinary compilation/evaluation route owns errors in a user law.
        # Failed evaluation proves no structural zero.
        return None
    if not isinstance(value, Array | float | int):
        # Only a numeric value can prove a structural zero.
        return None
    return value, frozenset(keys)


def _resolve_fixed_argument(
    *,
    arg_name: ReferenceName,
    path: tuple[str, ...],
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[QualifiedName, UserParamsLeaf],
    ancestors: tuple[str, ...],
) -> tuple[EconFunctionArg, frozenset[str]] | None:
    """Resolve one argument through its helper DAG or a fixed parameter leaf.

    `path` is the params path of the callable reading the argument: a regime
    function's `(regime, function)`, or a law cell's slot below
    `("edges", regime, target)`.
    """
    regime_name = path[1] if path[0] == EDGES else path[0]
    helper = regime.functions.get(arg_name)
    if isinstance(helper, Phased):
        helper = helper.solve if side == "solve" else helper.simulate
    if helper is not None:
        if not is_user_function(helper) or arg_name in ancestors:
            return None
        return _evaluate_fixed_function(
            func=helper,
            path=(regime_name, arg_name),
            side=side,
            regime=regime,
            fixed_flat=fixed_flat,
            ancestors=(*ancestors, arg_name),
        )
    qname = qname_from_tree_path((*path, arg_name))
    candidates = find_param_candidates(qname=qname, params_flat=fixed_flat)
    if len(candidates) > 1:
        raise InvalidNameError(
            f"Ambiguous parameter specification for {qname!r}. Found values "
            f"at: {candidates}"
        )
    if not candidates:
        return None
    key = candidates[0]
    # A Series declares coordinate-indexed values; fixed parameters may
    # therefore still vary by age or model state.
    if isinstance(fixed_flat[key], pd.Series | TimeVarying):
        return None
    return _canonicalize_fixed_leaf(
        regime_name=regime_name, qname=qname, value=fixed_flat[key]
    ), frozenset((key,))


def _is_runtime_argument(*, arg_name: ReferenceName, regime: UserRegime) -> bool:
    """Identify leaves whose value depends on a model problem or realization."""
    return (
        arg_name in regime.states
        or arg_name in regime.actions
        or arg_name in regime.derived_categoricals
        or arg_name in {"age", "period", "CE", "utility"}
        or arg_name.startswith("next_")
    )


def _canonicalize_fixed_leaf(
    *, regime_name: RegimeName, qname: QualifiedName, value: UserParamsLeaf
) -> ParamsLeaf:
    """Apply the parameter dtype boundary before executing a fixed callable."""
    canonical = cast_params_to_canonical_dtypes(
        cast(
            "FlatParams",
            MappingProxyType({regime_name: MappingProxyType({qname: value})}),
        )
    )
    # The cast input holds one regime level only, so no edge level comes back.
    return cast("FlatRegimeParams", canonical[regime_name])[qname]


def _targets(*, law: object, side: Side) -> frozenset[str]:
    """Collect one phase's target names across every age case."""
    if isinstance(law, ByAge):
        return frozenset(
            target for case in law.laws for target in _targets(law=case, side=side)
        )
    if isinstance(law, Phased):
        return _targets(law=law.solve if side == "solve" else law.simulate, side=side)
    if isinstance(law, Mapping):
        return frozenset(law)
    if isinstance(law, str):
        return frozenset((law,))
    return frozenset(getattr(law, "targets", ()) or ())


def _trim_state_law(
    *,
    law: StateTransitionEntry,
    removed: Mapping[Side, frozenset[str]],
    regime_name: RegimeName,
    state: StateName,
    regime: UserRegime,
    fixed_flat: Mapping[QualifiedName, UserParamsLeaf],
    consumed: set[str],
    reads: dict[RegimeName, set[str]],
) -> StateTransitionEntry:
    """Omit explicit laws toward targets removed from the relevant phase."""
    if isinstance(law, Phased):
        for side, variant in (("solve", law.solve), ("simulate", law.simulate)):
            _record_removed_state_keys(
                law=variant,
                removed=removed[side],
                regime_name=regime_name,
                state=state,
                side=side,
                regime=regime,
                fixed_flat=fixed_flat,
                consumed=consumed,
                reads=reads,
            )
        return Phased(
            solve=_trim_state_side(law=law.solve, removed=removed["solve"]),
            simulate=_trim_state_side(law=law.simulate, removed=removed["simulate"]),
        )
    for side in ("solve", "simulate"):
        _record_removed_state_keys(
            law=law,
            removed=removed[side],
            regime_name=regime_name,
            state=state,
            side=side,
            regime=regime,
            fixed_flat=fixed_flat,
            consumed=consumed,
            reads=reads,
        )
    solve = _trim_state_side(law=law, removed=removed["solve"])
    simulate = _trim_state_side(law=law, removed=removed["simulate"])
    return (
        solve
        if removed["solve"] == removed["simulate"]
        else Phased(solve=solve, simulate=simulate)
    )


def _trim_state_side(
    *, law: StateTransitionEntry, removed: frozenset[str]
) -> StateTransitionEntry:
    """Keep a bare law, or the explicit target cells still needed in this phase."""
    if isinstance(law, Mapping):
        return MappingProxyType(
            {target: cell for target, cell in law.items() if target not in removed}
        )
    return law


def _record_removed_state_keys(
    *,
    law: StateTransitionEntry,
    removed: frozenset[str],
    regime_name: RegimeName,
    state: StateName,
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[QualifiedName, UserParamsLeaf],
    consumed: set[str],
    reads: dict[RegimeName, set[str]],
) -> None:
    """Keep fixed keys and reads of handoff cells omitted with their zero edges."""
    if not isinstance(law, Mapping):
        return
    for target in removed & law.keys():
        cell = law[target]
        if is_user_function(cell):
            consumed.update(
                _get_declared_fixed_keys(
                    func=cell,
                    path=(regime_name, target, f"next_{state}"),
                    side=side,
                    regime=regime,
                    fixed_flat=fixed_flat,
                    ancestors=(),
                    reads=reads.setdefault(target, set()),
                )
            )


def _get_declared_fixed_keys(
    *,
    func: UserFunction,
    path: tuple[str, ...],
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[QualifiedName, UserParamsLeaf],
    ancestors: tuple[str, ...],
    non_params: frozenset[str] = frozenset(),
    reads: set[str],
) -> frozenset[str]:
    """Resolve supplied fixed leaves without evaluating an omitted handoff law.

    Runtime arguments the law or its helpers read are added to `reads`.
    """
    result: set[str] = set()
    for arg_name in inspect.signature(func).parameters:
        if arg_name in non_params:
            continue
        if _is_runtime_argument(arg_name=arg_name, regime=regime):
            reads.add(arg_name)
            continue
        helper = regime.functions.get(arg_name)
        if isinstance(helper, Phased):
            helper = helper.solve if side == "solve" else helper.simulate
        if is_user_function(helper) and arg_name not in ancestors:
            result.update(
                _get_declared_fixed_keys(
                    func=helper,
                    path=(path[0], arg_name),
                    side=side,
                    regime=regime,
                    fixed_flat=fixed_flat,
                    ancestors=(*ancestors, arg_name),
                    non_params=non_params,
                    reads=reads,
                )
            )
        elif helper is None:
            qname = qname_from_tree_path((*path, arg_name))
            candidates = find_param_candidates(qname=qname, params_flat=fixed_flat)
            if len(candidates) > 1:
                raise InvalidNameError(
                    f"Ambiguous parameter specification for {qname!r}. "
                    f"Found values at: {candidates}"
                )
            for candidate in candidates:
                if not isinstance(fixed_flat[candidate], pd.Series):
                    _canonicalize_fixed_leaf(
                        regime_name=path[0], qname=qname, value=fixed_flat[candidate]
                    )
            result.update(candidates)
    return frozenset(result)
