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

from types import MappingProxyType

from dags import get_annotations
from dags.tree.tree_utils import QNAME_DELIMITER

from _lcm.grids import Grid
from _lcm.transition_laws import TransitionLawInfo, TransitionLaws
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
    transition_laws: TransitionLaws,
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
    - No such read lands on a genuinely stochastic law. A realized draw has no
      value while the expectation over it is still being built, so the
      composition is rejected here rather than resolved into something else.

    Args:
        source_regime_name: Regime whose laws are being checked, named in
            messages.
        transitions: Immutable mapping of target regime names to their bundles of
            unqualified `next_<state>` transition functions.
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        processed_functions: Immutable mapping of qualified function names to
            functions, carrying the synthesized weight laws.
        all_grids: Immutable mapping of regime names to Grid spec objects.

    Raises:
        ModelInitializationError: If any of the three conditions fails.

    """
    for target, bundle in transitions.items():
        laws = transition_laws.get(target, MappingProxyType({}))
        _fail_if_a_basis_law_is_incomplete(
            source_regime_name=source_regime_name,
            target=target,
            bundle=bundle,
            laws=laws,
        )
        _fail_if_a_read_next_state_has_no_value(
            source_regime_name=source_regime_name,
            target=target,
            bundle=bundle,
            laws=laws,
            processed_functions=processed_functions,
            target_state_names=set(all_grids.get(target, MappingProxyType({}))),
        )


def _fail_if_a_basis_law_is_incomplete(
    *,
    source_regime_name: RegimeName,
    target: RegimeName,
    bundle: MappingProxyType[TransitionFunctionName, UserFunction],
    laws: MappingProxyType[TransitionFunctionName, TransitionLawInfo],
) -> None:
    """Check that each declared entry has all three of its parts, and only those."""
    for next_state_name, law in laws.items():
        if not law.interpolation_basis:
            continue
        missing = (
            ("a physical producer under its public name", next_state_name in bundle),
            ("a private support axis", law.support_axis_name is not None),
            ("a node-basis weight function", law.weight_name is not None),
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
    laws: MappingProxyType[TransitionFunctionName, TransitionLawInfo],
    processed_functions: MappingProxyType[str, UserFunction],
    target_state_names: set[StateOrActionName],
) -> None:
    """Check that every `next_<state>` a law reads is produced and not a draw."""
    consumers: dict[str, UserFunction] = dict(bundle)
    consumers |= {
        law.weight_name: processed_functions[law.weight_name]
        for law in laws.values()
        if law.weight_name is not None and law.weight_name in processed_functions
    }
    # A helper reads the draw just as a law does, and reaches the same place by
    # feeding one. Helpers carry no target qualification, so they are checked
    # against every target; a read naming a state the target does not carry is
    # skipped below and the check stays target-local.
    consumers |= {
        name: func
        for name, func in processed_functions.items()
        if QNAME_DELIMITER not in name and name not in consumers
    }

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
            if laws[arg].stochastic:
                msg = (
                    f"'{consumer_name}' of regime '{source_regime_name}' reads "
                    f"'{arg}' on the way into regime '{target}', but that law "
                    f"realizes a draw. A draw has no value while the expectation "
                    f"over it is being built, so it cannot feed another law; "
                    f"representing the dependence needs a joint kernel over both "
                    f"states rather than one marginal each. Declare '{state_name}' "
                    f"as a deterministic transition, or move the dependence into "
                    f"the law of '{arg}' itself."
                )
                raise ModelInitializationError(msg)
