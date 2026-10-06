"""Remove ordinary regime edges proved to have constant zero probability.

Only construction-time fixed leaves may feed the probability or its ordinary DAG
ancestors. States, actions, time, transition outputs and free parameters make an
edge conditional; no state probes or runtime values narrow the graph.

Removing an edge changes the effective graph, never the authored model's validity:

- joint kernels leaving with an edge are checked for output ownership first;
- a source state whose only authored law was such a kernel gets the empty
  per-target law `{}`, so the pruned regime is the one an author would write
  without the removed edge: the state is covered and no target cell is produced;
- the states and actions those removed declarations read are recorded only to
  explain an error: pruning removes as much as it can up front, so a variable
  read only across a removed edge is unused, exactly as without that edge.
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from dags.tree import qname_from_tree_path

from _lcm.params.processing import (
    cast_params_to_canonical_dtypes,
    find_param_candidates,
)
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.typing import FlatParams, RegimeName, StateName
from _lcm.utils.error_messages import format_messages
from _lcm.utils.namespace import flatten_regime_namespace
from lcm.exceptions import InvalidNameError, ModelInitializationError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import ByAge, JointTransition, StochasticTransition
from lcm.typing import UserParams

type Side = Literal["solve", "simulate"]


@dataclass(frozen=True, kw_only=True)
class FixedRegimeSupport:
    """Keep the reduced declarations and exact consumed fixed-key provenance."""

    user_regimes: MappingProxyType[RegimeName, UserRegime]
    """Regimes with constant-zero ordinary transition cells removed."""

    consumed_param_keys: frozenset[str]
    """Supplied flat keys used to prove a removed cell constant and zero."""

    removed_edge_reads: MappingProxyType[
        RegimeName, MappingProxyType[str, tuple[RegimeName, ...]]
    ]
    """Per regime, each state or action read by a declaration removed with its
    zero edge, and the targets of those edges; used only in error messages."""


def prune_fixed_regime_support(
    *, user_regimes: Mapping[RegimeName, UserRegime], fixed_params: UserParams
) -> FixedRegimeSupport:
    """Remove exactly zero cells whose entire dependency graph is fixed.

    Per-target state laws toward a target removed from their phase are omitted
    too, and so are joint kernels toward a target removed from both phases. Bare
    laws and source variables remain declared. Fixed keys consumed by removed
    cells retain their provenance for later unknown-key validation.

    Raises:
        ModelInitializationError: If a removed joint kernel claims a target-state
            cell that the target lacks or that another producer also claims.
    """
    fixed_flat = flatten_regime_namespace(fixed_params)
    consumed: set[str] = set()
    errors: list[str] = []
    result: dict[RegimeName, UserRegime] = {}
    removed_edge_reads: dict[
        RegimeName, MappingProxyType[str, tuple[RegimeName, ...]]
    ] = {}
    for name, regime in user_regimes.items():
        transition, removed, law_keys = _prune_regime_transition(
            regime_name=name, regime=regime, fixed_flat=fixed_flat
        )
        consumed.update(law_keys)
        removed_in_both = removed["solve"] & removed["simulate"]
        errors += _removed_joint_ownership_errors(
            removed=removed_in_both, regime_name=name, user_regimes=user_regimes
        )
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
            regime_transitions=transition,
            state_transitions=state_transitions,
            joint_transitions=joint_transitions,
        )
    if errors:
        raise ModelInitializationError(format_messages(errors))
    return FixedRegimeSupport(
        user_regimes=MappingProxyType(result),
        consumed_param_keys=frozenset(consumed),
        removed_edge_reads=MappingProxyType(removed_edge_reads),
    )


def _prune_regime_transition(
    *,
    regime_name: RegimeName,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
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
            law=regime.regime_transitions,
            side=None,
            regime_name=regime_name,
            regime=regime,
            fixed_flat=fixed_flat,
            consumed=consumed,
            protected=protected,
        )
        removed: dict[Side, frozenset[str]] = {
            side: _targets(law=regime.regime_transitions, side=side)
            - _targets(law=transition, side=side)
            for side in ("solve", "simulate")
        }
        one_phase_joint = frozenset(regime.joint_transitions) & (
            removed["solve"] ^ removed["simulate"]
        )
        if not one_phase_joint - protected:
            return transition, removed, frozenset(consumed)
        protected |= one_phase_joint


def _joint_kernels(raw: object) -> tuple[tuple[Side, JointTransition], ...]:
    """Pair each phase with the joint kernel a declaration uses there."""
    return tuple(
        (side, cast("JointTransition", getattr(raw, side)))
        if isinstance(raw, Phased)
        else (side, cast("JointTransition", raw))
        for side in ("solve", "simulate")
    )


def _removed_joint_ownership_errors(
    *,
    removed: frozenset[str],
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> list[str]:
    """Check the target-state cells claimed by joint kernels about to be removed.

    The checks on retained kernels run on the effective graph; these give a
    removed kernel the same output and unique-producer contract.
    """
    regime = user_regimes[regime_name]
    errors: list[str] = []
    for target in sorted(removed & regime.joint_transitions.keys()):
        target_states = user_regimes[target].states if target in user_regimes else {}
        for side in ("solve", "simulate"):
            owners: dict[StateName, str] = {}
            for kernel_name, raw in regime.joint_transitions[target].items():
                kernel = dict(_joint_kernels(raw))[side]
                for output in kernel.outputs:
                    if output not in target_states:
                        errors.append(
                            f"regime '{regime_name}' ({side}): joint-transition "
                            f"output '{output}' of kernel '{kernel_name}' is not a "
                            f"target state of regime '{target}'."
                        )
                    if output in owners:
                        errors.append(
                            f"regime '{regime_name}' ({side}): multiple producers "
                            f"claim target-state cell ('{target}', '{output}'): "
                            f"joint kernels '{owners[output]}' and '{kernel_name}'."
                        )
                    owners.setdefault(output, kernel_name)
            for output in owners:
                law = regime.state_transitions.get(output)
                if isinstance(law, Phased):
                    law = getattr(law, side)
                if isinstance(law, Mapping) and target in law:
                    errors.append(
                        f"regime '{regime_name}' ({side}): multiple producers "
                        f"claim target-state cell ('{target}', '{output}'); an "
                        "explicit per-target ordinary law and a joint-transition "
                        "output cannot own the same state."
                    )
    return sorted(set(errors))


def _states_covered_only_by_removed_joints(
    *, regime: UserRegime, joint_transitions: Mapping[str, object]
) -> tuple[StateName, ...]:
    """Source states whose only authored law is a removed joint kernel's output.

    Without the removed edge, an author would declare each of these states with
    the empty per-target law `{}`; pruning gives them exactly that.
    """

    def outputs(joints: Mapping[str, object]) -> set[str]:
        return {
            output
            for kernels in joints.values()
            for raw in cast("Mapping[str, object]", kernels).values()
            for _, kernel in _joint_kernels(raw)
            for output in kernel.outputs
        }

    return tuple(
        sorted(
            state
            for state in outputs(regime.joint_transitions) - outputs(joint_transitions)
            if state in regime.states
            and state not in regime.state_transitions
            and not isinstance(regime.states[state], _ContinuousStochasticProcess)
        )
    )


def _trim_joint_transitions(
    *,
    removed: frozenset[str],
    regime_name: RegimeName,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    consumed: set[str],
    reads: dict[RegimeName, set[str]],
) -> MappingProxyType[str, object]:
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
                if callable(kernel.support):
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
    fixed_flat: Mapping[str, object],
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
            if set(cast("Mapping[str, object]", solve))
            == set(cast("Mapping[str, object]", simulate))
            else Phased(solve=solve, simulate=simulate)
        )
    retained: dict[str, object] = {}
    removed_keys: set[str] = set()
    for target, cell in law.items():
        evaluated = (
            _evaluate_fixed_function(
                func=cell.func,
                path=(regime_name, target, "next_regime"),
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
    func: Callable[..., Any],
    path: tuple[str, ...],
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    ancestors: tuple[str, ...],
) -> tuple[Any, frozenset[str]] | None:
    """Evaluate a complete fixed DAG, or retain an unresolved conditional edge."""
    arguments: dict[str, object] = {}
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
    return value, frozenset(keys)


def _resolve_fixed_argument(
    *,
    arg_name: str,
    path: tuple[str, ...],
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    ancestors: tuple[str, ...],
) -> tuple[Any, frozenset[str]] | None:
    """Resolve one argument through its helper DAG or a fixed parameter leaf."""
    helper = regime.functions.get(arg_name)
    if isinstance(helper, Phased):
        helper = helper.solve if side == "solve" else helper.simulate
    if helper is not None:
        if not callable(helper) or arg_name in ancestors:
            return None
        return _evaluate_fixed_function(
            func=helper,
            path=(path[0], arg_name),
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
    if isinstance(fixed_flat[key], pd.Series):
        return None
    return _canonicalize_fixed_leaf(
        regime_name=path[0], qname=qname, value=fixed_flat[key]
    ), frozenset((key,))


def _is_runtime_argument(*, arg_name: str, regime: UserRegime) -> bool:
    """Identify leaves whose value depends on a model problem or realization."""
    return (
        arg_name in regime.states
        or arg_name in regime.actions
        or arg_name in regime.derived_categoricals
        or arg_name in {"age", "period", "CE", "utility"}
        or arg_name.startswith("next_")
    )


def _canonicalize_fixed_leaf(
    *, regime_name: RegimeName, qname: str, value: object
) -> object:
    """Apply the parameter dtype boundary before executing a fixed callable."""
    canonical = cast_params_to_canonical_dtypes(
        cast(
            "FlatParams",
            MappingProxyType({regime_name: MappingProxyType({qname: value})}),
        )
    )
    return canonical[regime_name][qname]


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
    law: object,
    removed: Mapping[Side, frozenset[str]],
    regime_name: RegimeName,
    state: str,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    consumed: set[str],
    reads: dict[RegimeName, set[str]],
) -> object:
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


def _trim_state_side(*, law: object, removed: frozenset[str]) -> object:
    """Keep a bare law, or the explicit target cells still needed in this phase."""
    if isinstance(law, Mapping):
        return MappingProxyType(
            {target: cell for target, cell in law.items() if target not in removed}
        )
    return law


def _record_removed_state_keys(
    *,
    law: object,
    removed: frozenset[str],
    regime_name: RegimeName,
    state: str,
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
    consumed: set[str],
    reads: dict[RegimeName, set[str]],
) -> None:
    """Keep fixed keys and reads of handoff cells omitted with their zero edges."""
    if not isinstance(law, Mapping):
        return
    for target in removed & law.keys():
        cell = law[target]
        if callable(cell):
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
    func: Callable[..., Any],
    path: tuple[str, ...],
    side: Side,
    regime: UserRegime,
    fixed_flat: Mapping[str, object],
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
        if callable(helper) and arg_name not in ancestors:
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
