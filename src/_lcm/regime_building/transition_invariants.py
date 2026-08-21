"""Construction-time checks that the two transition namespaces stayed separate.

A next-period state reaches the engine as at most two objects: the public
`next_<state>` name, which always produces the physical value, and — where the
target holds its value function on discrete nodes — a private support axis the
interpolator is indexed along. Everything downstream depends on that separation
holding, and a violation of it is silent: a consumer that reads node indices
where it expected a value still computes, and still returns a number.

These checks run once per phase while the model builds, over Python-level
descriptors and function signatures, so they cost nothing on the traced path.
"""

from collections.abc import Mapping
from types import MappingProxyType

from dags import get_annotations
from dags.tree.tree_utils import QNAME_DELIMITER

from _lcm.grids import Grid
from _lcm.transition_plans import (
    InterpolationBasisInfo,
    LotteryLifetime,
    TargetTransitionPlan,
    TargetTransitionPlans,
)
from _lcm.typing import (
    RegimeName,
    StateOrActionName,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import UserFunction


def fail_if_transition_namespaces_are_mixed(
    *,
    source_regime_name: RegimeName,
    transitions: TransitionFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    processed_functions: MappingProxyType[str, UserFunction],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
) -> None:
    """Check one phase's transition laws against the two-namespace contract.

    Three things are asserted, per target regime:

    - A law whose weights are an interpolation basis carries the full set — a
      physical producer under its public name, a private support axis, and one
      weight function — and does not also claim to emit node indices.
    - Every `next_<state>` a transition or weight law reads has a producer in the
      same target's bundle.

    Args:
        source_regime_name: Regime whose laws are being checked, named in
            messages.
        transitions: Immutable mapping of target regime names to their bundles of
            unqualified `next_<state>` transition functions.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.
        processed_functions: Immutable mapping of qualified function names to
            functions, carrying the synthesized weight laws.
        all_grids: Immutable mapping of regime names to Grid spec objects.

    Raises:
        ModelInitializationError: If any of the three conditions fails.

    """
    _fail_if_joint_node_scopes_are_crossed(
        source_regime_name=source_regime_name,
        transitions=transitions,
        transition_plans=transition_plans,
        processed_functions=processed_functions,
    )
    for target, bundle in transitions.items():
        plan = transition_plans[target]
        _fail_if_a_stochastic_law_carries_a_basis(
            source_regime_name=source_regime_name, target=target, plan=plan
        )
        _fail_if_a_basis_law_is_incomplete(
            source_regime_name=source_regime_name,
            target=target,
            bundle=bundle,
            plan=plan,
        )
        _fail_if_a_read_next_state_has_no_value(
            source_regime_name=source_regime_name,
            target=target,
            bundle=bundle,
            plan=plan,
            processed_functions=processed_functions,
            target_state_names=set(all_grids.get(target, MappingProxyType({}))),
        )


def _functions_feeding(
    *,
    roots: Mapping[str, UserFunction],
    candidates: Mapping[str, UserFunction],
) -> dict[str, UserFunction]:
    """Return the unqualified functions the roots read, directly or through others.

    Args:
        roots: Functions whose arguments start the walk.
        candidates: Immutable mapping of function names to functions, searched for
            each argument read. Target-qualified names are skipped: they belong to
            another target's bundle, not to this regime's plain namespace.

    Returns:
        Dictionary of the reachable candidate functions, excluding the roots.

    """
    found: dict[str, UserFunction] = {}
    frontier = list(roots.values())
    while frontier:
        func = frontier.pop()
        for arg in get_annotations(func):
            if arg == "return" or arg in roots or arg in found:
                continue
            if QNAME_DELIMITER in arg or arg not in candidates:
                continue
            found[arg] = candidates[arg]
            frontier.append(candidates[arg])
    return found


def _fail_if_joint_node_scopes_are_crossed(
    *,
    source_regime_name: RegimeName,
    transitions: TransitionFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    processed_functions: MappingProxyType[str, UserFunction],
) -> None:
    """Reject a target output DAG that reads another target's local node."""
    all_local_nodes = frozenset(
        lottery.name
        for plan in transition_plans.values()
        for lottery in plan.lotteries.values()
        if lottery.lifetime is LotteryLifetime.TRANSITION_LOCAL
    )
    if not all_local_nodes:
        return

    for target, bundle in transitions.items():
        plan = transition_plans[target]
        target_local_nodes = frozenset(
            lottery.name
            for lottery in plan.lotteries.values()
            if lottery.lifetime is LotteryLifetime.TRANSITION_LOCAL
        )
        consumers: dict[str, UserFunction] = dict(bundle)
        weight_names = {lottery.weight_name for lottery in plan.lotteries.values()} | {
            output.continuation_coordinate.weight_name
            for output in plan.outputs.values()
            if isinstance(output.continuation_coordinate, InterpolationBasisInfo)
        }
        consumers |= {
            name: processed_functions[name]
            for name in weight_names
            if name in processed_functions
        }
        consumers |= _functions_feeding(roots=consumers, candidates=processed_functions)

        for consumer_name, consumer in consumers.items():
            foreign = sorted(
                arg
                for arg in get_annotations(consumer)
                if arg in all_local_nodes and arg not in target_local_nodes
            )
            if not foreign:
                continue
            msg = (
                f"'{consumer_name}' of regime '{source_regime_name}' on the way "
                f"into target '{target}' reads transition-local joint node(s) "
                f"{foreign}, but those nodes are scoped to another target edge. "
                "A JointTransition node is available only to outputs and helpers "
                "of the target that declares it."
            )
            raise ModelInitializationError(msg)


def _fail_if_a_stochastic_law_carries_a_basis(
    *,
    source_regime_name: RegimeName,
    target: RegimeName,
    plan: TargetTransitionPlan,
) -> None:
    """Check that no law is both a draw and a declared entry.

    The two place a value on the target's nodes for opposite reasons. A declared
    entry names one value, and its coefficients express that value in the node
    basis; they sum to one but are not probabilities. A draw names a distribution,
    and its weights are probabilities. Reading one as the other prices a different
    object and would otherwise do so silently — the coefficients of an entry, run
    through a certainty equivalent, give a mean over nodes the entry never took.
    """
    for law in plan.outputs.values():
        next_state_name = law.next_state_name
        if not isinstance(law.continuation_coordinate, InterpolationBasisInfo):
            continue
        if not law.lottery_dependencies:
            continue
        msg = (
            f"The law '{next_state_name}' of regime '{source_regime_name}' into "
            f"regime '{target}' is marked as realizing a draw, but it also "
            f"carries the node basis of a declared entry. A draw's weights are "
            f"probabilities over the target's nodes; an entry's are the "
            f"coefficients expressing one value in them. Nothing is both, and "
            f"reading either as the other prices a different object."
        )
        raise ModelInitializationError(msg)


def _fail_if_a_basis_law_is_incomplete(
    *,
    source_regime_name: RegimeName,
    target: RegimeName,
    bundle: MappingProxyType[TransitionFunctionName, UserFunction],
    plan: TargetTransitionPlan,
) -> None:
    """Check that each declared entry has all three of its parts, and only those."""
    for law in plan.outputs.values():
        next_state_name = law.next_state_name
        if not isinstance(law.continuation_coordinate, InterpolationBasisInfo):
            continue
        basis = law.continuation_coordinate
        missing = (
            ("a physical producer under its public name", next_state_name in bundle),
            ("a private support axis", basis.axis_name is not None),
            ("a node-basis weight function", basis.weight_name is not None),
            (
                (
                    "a public name free of node indices — it is marked as "
                    "emitting them, which would shadow the physical value"
                ),
                not law.emits_support_index,
            ),
        )
        for description, holds in missing:
            if holds:
                continue
            msg = (
                f"The declared entry '{next_state_name}' of regime "
                f"'{source_regime_name}' into regime '{target}' names a physical "
                f"value that is interpolated over the target's nodes, but it "
                f"lacks {description}. A declared entry needs its physical value, "
                f"its private node axis, and the weights relating the two; "
                f"without all three the value function would be indexed by "
                f"something other than what the law names."
            )
            raise ModelInitializationError(msg)


def _fail_if_a_read_next_state_has_no_value(
    *,
    source_regime_name: RegimeName,
    target: RegimeName,
    bundle: MappingProxyType[TransitionFunctionName, UserFunction],
    plan: TargetTransitionPlan,
    processed_functions: MappingProxyType[str, UserFunction],
    target_state_names: set[StateOrActionName],
) -> None:
    """Check that every `next_<state>` a law reads is produced and not a draw."""
    consumers: dict[str, UserFunction] = dict(bundle)
    weight_names = {lottery.weight_name for lottery in plan.lotteries.values()} | {
        output.continuation_coordinate.weight_name
        for output in plan.outputs.values()
        if isinstance(output.continuation_coordinate, InterpolationBasisInfo)
    }
    consumers |= {
        name: processed_functions[name]
        for name in weight_names
        if name in processed_functions
    }
    # A helper reads the draw just as a law does, and reaches the same place by
    # feeding one -- but only if it feeds one. `next_<state>` on a function no
    # law consumes, `utility` above all, names an ordinary parameter: utility is
    # evaluated at this period's states, where no next-period value exists to
    # confuse it with.
    consumers |= _functions_feeding(roots=consumers, candidates=processed_functions)

    for consumer_name, consumer in consumers.items():
        for arg in get_annotations(consumer):
            if arg == "return" or not arg.startswith("next_"):
                continue
            state_name = arg.removeprefix("next_")
            if state_name not in target_state_names:
                continue
            if arg not in bundle:
                msg = (
                    f"'{consumer_name}' of regime '{source_regime_name}' reads "
                    f"'{arg}' on the way into regime '{target}', but no law of that "
                    f"transition produces it. Declare a state transition for "
                    f"'{state_name}' toward '{target}', or read a value the "
                    f"transition does produce."
                )
                raise ModelInitializationError(msg)
