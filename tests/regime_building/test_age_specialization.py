"""Behavior of the `AgeSpecializedFunction` period-specialization wiring.

Two contracts, promoted from the Pro-review counterexample RT1:

- the signature/resolution helpers key on every policy-consuming node class and
  recurse into pylcm's nested transition tree, so ages with different policy
  closures never false-share a compiled program; and
- the `Regime` validator rejects — loudly, before any program is built — the two
  specialized-transition compositions v1 does not support.
"""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.grids import DiscreteGrid
from _lcm.regime_building.age_normalization import periodized_tree_signature
from _lcm.regime_building.age_specialization import (
    INVARIANT,
    _tree_signature,
    node_signature,
    tree_signature,
)
from _lcm.regime_building.processing import _fail_if_phase_state_nodes_disagree
from _lcm.user_regime_validation import _validate_logical_consistency
from lcm.exceptions import RegimeInitializationError
from lcm.transition import AgeSpecializedFunction, MarkovTransition
from lcm.typing import Float1D
from tests.mock_regime import MockRegime


def _feasible_below_age(age: float):
    """Return a feasibility constraint whose threshold is the calendar-policy age."""
    return lambda action: action <= age


def test_node_signature_of_a_plain_callable_is_invariant():
    """A bare callable carries no age-varying closure, so its signature is constant."""
    assert node_signature(node=lambda x: x, age=60.0) is INVARIANT


def test_node_signature_of_age_specialized_varies_with_age():
    """An `AgeSpecializedFunction` node's signature separates ages by closure."""
    node = AgeSpecializedFunction(
        build=_feasible_below_age, signature=lambda age: ("limit", age)
    )

    assert node_signature(node=node, age=60.0) != node_signature(node=node, age=61.0)


def test_tree_signature_separates_a_constraint_that_changes_by_age():
    """Constraints feed feasibility (F), so a per-age constraint must split the key.

    Two ages whose only difference is a policy-dependent constraint must not
    false-share a compiled `Q_and_F`.
    """
    constraints = {
        "eligibility": AgeSpecializedFunction(
            build=_feasible_below_age, signature=lambda age: ("limit", age)
        )
    }

    assert tree_signature(tree=constraints, age=60.0) != tree_signature(
        tree=constraints, age=61.0
    )


def test_tree_signature_recurses_into_the_nested_transition_mapping():
    """Processed transitions are nested `{target_regime: {name: fn}}`.

    A signature taken over top-level values sees the inner mapping and misses the
    `AgeSpecializedFunction` node inside; the tree signature must descend and
    separate ages.
    """
    transitions = {
        "retired": {
            "next_points": AgeSpecializedFunction(
                build=lambda age: lambda points: points + age,
                signature=lambda age: ("policy", age),
            )
        }
    }

    assert tree_signature(tree=transitions, age=60.0) != tree_signature(
        tree=transitions, age=61.0
    )


def test_tree_signature_helper_is_invariant_to_key_insertion_order():
    """`_tree_signature` sorts keys, so insertion order cannot affect the result."""
    tree_a = {"b": "second", "a": "first"}
    tree_b = {"a": "first", "b": "second"}

    assert _tree_signature(tree=tree_a, leaf_signature=str) == _tree_signature(
        tree=tree_b, leaf_signature=str
    )


def test_tree_signature_helper_recurses_regardless_of_leaf_signature():
    """A different `leaf_signature` still descends into nested mappings."""
    nested = {"outer": {"inner": "value"}}

    assert _tree_signature(tree=nested, leaf_signature=lambda leaf: len(str(leaf))) == (
        ("outer", (("inner", len("value")),)),
    )


def test_tree_signature_wrappers_share_recursive_semantics():
    """`tree_signature` (age) and `periodized_tree_signature` (period) agree.

    Both are thin wrappers over the same recursive helper, so an
    age-invariant tree and a period-invariant tree with identical structure
    (only the leaf-signature source differs) recurse and sort keys the same
    way.
    """
    tree = {"b": lambda x: x, "a": {"nested": lambda x: x}}

    assert tree_signature(tree=tree, age=60.0) == periodized_tree_signature(
        tree=tree, period=3
    )


def test_age_specialized_regime_transition_is_rejected(binary_category_class):
    """A policy-specialized regime `transition` is rejected for v1."""
    regime = MockRegime(
        actions={"a": DiscreteGrid(category_class=binary_category_class)},
        states={"b": DiscreteGrid(category_class=binary_category_class)},
        state_transitions={"b": lambda b: b},
        transition=AgeSpecializedFunction(
            build=lambda age: lambda b: b,  # noqa: ARG005
            signature=lambda age: ("regime", age),
        ),
        functions={"utility": lambda a, b: None},  # noqa: ARG005
    )

    with pytest.raises(RegimeInitializationError):
        _validate_logical_consistency(regime)


def test_markov_transition_wrapping_age_specialized_is_rejected(binary_category_class):
    """A stochastic transition whose probability law is policy-specialized.

    `MarkovTransition(AgeSpecializedFunction(...))` is out of scope for v1 and
    must raise.
    """
    regime = MockRegime(
        actions={"a": DiscreteGrid(category_class=binary_category_class)},
        states={"b": DiscreteGrid(category_class=binary_category_class)},
        state_transitions={
            "b": MarkovTransition(
                AgeSpecializedFunction(
                    build=lambda age: lambda b: b,  # noqa: ARG005
                    signature=lambda age: ("stochastic", age),
                )
            )
        },
        transition=lambda: 0,
        functions={"utility": lambda a, b: None},  # noqa: ARG005
    )

    with pytest.raises(RegimeInitializationError):
        _validate_logical_consistency(regime)


def test_age_specialized_deterministic_state_transition_is_rejected(
    binary_category_class,
):
    """A deterministic state transition cannot itself be `AgeSpecializedFunction`.

    Policy-dependent laws of motion are expressed as a plain transition reading an
    `AgeSpecializedFunction` helper function; a direct marker in
    `state_transitions` must raise before any program is built.
    """
    regime = MockRegime(
        actions={"a": DiscreteGrid(category_class=binary_category_class)},
        states={"b": DiscreteGrid(category_class=binary_category_class)},
        state_transitions={
            "b": AgeSpecializedFunction(
                build=lambda age: lambda b: b,  # noqa: ARG005
                signature=lambda age: ("deterministic", age),
            )
        },
        transition=lambda: 0,
        functions={"utility": lambda a, b: None},  # noqa: ARG005
    )

    with pytest.raises(RegimeInitializationError):
        _validate_logical_consistency(regime)


def test_age_specialized_in_terminal_regime_is_rejected(binary_category_class):
    """A terminal regime cannot contain `AgeSpecializedFunction` functions/constraints.

    The terminal value program is built once and shared across all periods, so a
    policy-specialized terminal function must raise instead of silently using one
    age's closure.
    """
    regime = MockRegime(
        states={"b": DiscreteGrid(category_class=binary_category_class)},
        transition=None,
        functions={
            "utility": AgeSpecializedFunction(
                build=lambda age: lambda b: b,  # noqa: ARG005
                signature=lambda age: ("terminal", age),
            )
        },
    )

    with pytest.raises(RegimeInitializationError):
        _validate_logical_consistency(regime)


def test_regime_transition_reading_age_specialized_helper_is_rejected(
    binary_category_class,
):
    """A plain regime transition cannot read an `AgeSpecializedFunction` helper.

    Regime-transition probabilities are built once, not per period, so a
    policy-specialized value flowing into `next_regime` would silently reuse one
    age's policy closure across all periods. It must raise instead.
    """

    def next_regime(policy_threshold):
        return policy_threshold

    regime = MockRegime(
        actions={"a": DiscreteGrid(category_class=binary_category_class)},
        states={"b": DiscreteGrid(category_class=binary_category_class)},
        transition=MarkovTransition(next_regime),
        functions={
            "utility": lambda a, b: None,  # noqa: ARG005
            "policy_threshold": AgeSpecializedFunction(
                build=lambda age: lambda b: age,  # noqa: ARG005
                signature=lambda age: ("threshold", age),
            ),
        },
    )

    with pytest.raises(RegimeInitializationError):
        _validate_logical_consistency(regime)


def test_regime_transition_reading_age_specialized_constraint_is_rejected(
    binary_category_class,
):
    """A plain regime transition cannot read an `AgeSpecializedFunction` in
    `constraints`.

    `constraints` is an equally supported home for `AgeSpecializedFunction` as
    `functions` (the sibling error message for a bare-marker transition explicitly
    names both), so the ancestor walk that detects a transition reading a
    specialized value must scan `constraints` too, not just `functions`.
    """

    def next_regime(policy_threshold):
        return policy_threshold

    regime = MockRegime(
        actions={"a": DiscreteGrid(category_class=binary_category_class)},
        states={"b": DiscreteGrid(category_class=binary_category_class)},
        transition=MarkovTransition(next_regime),
        functions={"utility": lambda a, b: None},  # noqa: ARG005
        constraints={
            "policy_threshold": AgeSpecializedFunction(
                build=lambda age: lambda b: age,  # noqa: ARG005
                signature=lambda age: ("threshold", age),
            ),
        },
    )

    with pytest.raises(RegimeInitializationError):
        _validate_logical_consistency(regime)


def test_regime_transition_with_transitive_age_specialized_ancestor_is_rejected(
    binary_category_class,
):
    """The regime-transition guard follows dependencies through plain functions.

    `next_regime` reads a plain function which itself reads an `AgeSpecializedFunction`
    helper; the policy dependency is transitive but just as unsound, so it must
    raise.
    """

    def eligibility(policy_threshold):
        return policy_threshold

    def next_regime(eligibility):
        return eligibility

    regime = MockRegime(
        actions={"a": DiscreteGrid(category_class=binary_category_class)},
        states={"b": DiscreteGrid(category_class=binary_category_class)},
        transition=MarkovTransition(next_regime),
        functions={
            "utility": lambda a, b: None,  # noqa: ARG005
            "eligibility": eligibility,
            "policy_threshold": AgeSpecializedFunction(
                build=lambda age: lambda b: age,  # noqa: ARG005
                signature=lambda age: ("threshold", age),
            ),
        },
    )

    with pytest.raises(RegimeInitializationError):
        _validate_logical_consistency(regime)


def test_regime_transition_markov_wrapping_age_specialized_is_rejected(
    binary_category_class,
):
    """A `MarkovTransition(AgeSpecializedFunction(...))` regime transition must raise.

    Regime-transition probabilities are built once, not per period, so a
    policy-specialized probability law as the regime transition is just as
    unsound as a bare `AgeSpecializedFunction` transition and must be rejected at
    `Regime` construction.
    """
    regime = MockRegime(
        actions={"a": DiscreteGrid(category_class=binary_category_class)},
        states={"b": DiscreteGrid(category_class=binary_category_class)},
        transition=MarkovTransition(
            AgeSpecializedFunction(
                build=lambda age: lambda b: b,  # noqa: ARG005
                signature=lambda age: ("regime", age),
            )
        ),
        functions={"utility": lambda a, b: None},  # noqa: ARG005
    )

    with pytest.raises(RegimeInitializationError):
        _validate_logical_consistency(regime)


def _nodes(*values: float) -> Float1D:
    """Return a node array shaped like a resolved age-specialized grid axis.

    The dtype is the suite's active precision, not a pinned one: the guard
    compares node arrays exactly, and that has to hold in whichever float
    format the model is actually built in.
    """
    return jnp.asarray(values)


def _table(
    mapping: dict[int, dict[str, Float1D]],
) -> MappingProxyType[int, MappingProxyType[str, Float1D]]:
    """Return a per-period node table in the immutable form the phases carry."""
    return MappingProxyType(
        {period: MappingProxyType(states) for period, states in mapping.items()}
    )


def test_agreeing_phase_state_nodes_are_accepted():
    """A regime whose two phases resolve the same nodes per period builds."""
    table = _table({0: {"illiquid": _nodes(0.0, 1.0, 2.0)}})

    _fail_if_phase_state_nodes_disagree(
        regime_name="working",
        solve_nodes=table,
        simulate_axes=_table({0: {"illiquid": _nodes(0.0, 1.0, 2.0)}}),
    )


def test_an_age_invariant_regime_carries_no_node_tables_and_is_accepted():
    """A regime with no age-specialized state resolves `None` in both phases."""
    _fail_if_phase_state_nodes_disagree(
        regime_name="working", solve_nodes=None, simulate_axes=None
    )


def test_phase_state_nodes_differing_in_value_are_rejected():
    """Two phases resolving different nodes for one period's state is refused."""
    with pytest.raises(RegimeInitializationError, match="illiquid"):
        _fail_if_phase_state_nodes_disagree(
            regime_name="working",
            solve_nodes=_table({0: {"illiquid": _nodes(0.0, 1.0, 2.0)}}),
            simulate_axes=_table({0: {"illiquid": _nodes(0.0, 1.0, 9.0)}}),
        )


def test_a_node_table_present_in_only_one_phase_is_rejected():
    """A regime whose solve resolves per-period nodes and whose simulation does
    not is refused, rather than letting simulation fall back to the published
    grid and admit against different endpoints than the solve certified."""
    with pytest.raises(RegimeInitializationError, match="working"):
        _fail_if_phase_state_nodes_disagree(
            regime_name="working",
            solve_nodes=_table({0: {"illiquid": _nodes(0.0, 1.0, 2.0)}}),
            simulate_axes=None,
        )


def test_phase_state_nodes_differing_in_length_are_rejected():
    """Two phases resolving a different number of nodes for one state is refused."""
    with pytest.raises(RegimeInitializationError, match="illiquid"):
        _fail_if_phase_state_nodes_disagree(
            regime_name="working",
            solve_nodes=_table({0: {"illiquid": _nodes(0.0, 1.0, 2.0)}}),
            simulate_axes=_table({0: {"illiquid": _nodes(0.0, 2.0)}}),
        )


def test_a_node_disagreement_names_the_offending_values():
    """The refusal reports where two equal-length axes part and what they hold.

    Two axes of the same length differ somewhere in their values, so a message
    quoting only their shapes names the same shape twice and says nothing about
    the disagreement it is diagnosing.
    """
    with pytest.raises(RegimeInitializationError, match="index 2"):
        _fail_if_phase_state_nodes_disagree(
            regime_name="working",
            solve_nodes=_table({0: {"illiquid": _nodes(0.0, 1.0, 2.0)}}),
            simulate_axes=_table({0: {"illiquid": _nodes(0.0, 1.0, 9.0)}}),
        )
