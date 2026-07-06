"""Behavior of the `AgeSpecialized` period-specialization wiring.

Two contracts, promoted from the Pro-review counterexample RT1:

- the signature/resolution helpers key on every policy-consuming node class and
  recurse into pylcm's nested transition tree, so ages with different policy
  closures never false-share a compiled program; and
- the `Regime` validator rejects — loudly, before any program is built — the two
  specialized-transition compositions v1 does not support.
"""

import pytest

from _lcm.grids import DiscreteGrid
from _lcm.regime_building.age_specialization import (
    INVARIANT,
    node_signature,
    tree_signature,
)
from _lcm.user_regime_validation import _validate_logical_consistency
from lcm.exceptions import RegimeInitializationError
from lcm.transition import AgeSpecialized, MarkovTransition
from tests.mock_regime import MockRegime


def _feasible_below_age(age: float):
    """Return a feasibility constraint whose threshold is the calendar-policy age."""
    return lambda action: action <= age


def test_node_signature_of_a_plain_callable_is_invariant():
    """A bare callable carries no age-varying closure, so its signature is constant."""
    assert node_signature(lambda x: x, age=60.0) is INVARIANT


def test_node_signature_of_age_specialized_varies_with_age():
    """An `AgeSpecialized` node's signature separates ages with different closures."""
    node = AgeSpecialized(
        build=_feasible_below_age, signature=lambda age: ("limit", age)
    )

    assert node_signature(node, age=60.0) != node_signature(node, age=61.0)


def test_tree_signature_separates_a_constraint_that_changes_by_age():
    """Constraints feed feasibility (F), so a per-age constraint must split the key.

    Two ages whose only difference is a policy-dependent constraint must not
    false-share a compiled `Q_and_F`.
    """
    constraints = {
        "eligibility": AgeSpecialized(
            build=_feasible_below_age, signature=lambda age: ("limit", age)
        )
    }

    assert tree_signature(constraints, age=60.0) != tree_signature(
        constraints, age=61.0
    )


def test_tree_signature_recurses_into_the_nested_transition_mapping():
    """Processed transitions are nested `{target_regime: {name: fn}}`.

    A signature taken over top-level values sees the inner mapping and misses the
    `AgeSpecialized` node inside; the tree signature must descend and separate ages.
    """
    transitions = {
        "retired": {
            "next_points": AgeSpecialized(
                build=lambda age: lambda points: points + age,
                signature=lambda age: ("policy", age),
            )
        }
    }

    assert tree_signature(transitions, age=60.0) != tree_signature(
        transitions, age=61.0
    )


def test_age_specialized_regime_transition_is_rejected(binary_category_class):
    """A policy-specialized regime `transition` is rejected for v1."""
    regime = MockRegime(
        actions={"a": DiscreteGrid(binary_category_class)},
        states={"b": DiscreteGrid(binary_category_class)},
        state_transitions={"b": lambda b: b},
        transition=AgeSpecialized(
            build=lambda age: lambda b: b,  # noqa: ARG005
            signature=lambda age: ("regime", age),
        ),
        functions={"utility": lambda a, b: None},  # noqa: ARG005
    )

    with pytest.raises(RegimeInitializationError):
        _validate_logical_consistency(regime)


def test_markov_transition_wrapping_age_specialized_is_rejected(binary_category_class):
    """A stochastic transition whose probability law is policy-specialized.

    `MarkovTransition(AgeSpecialized(...))` is out of scope for v1 and must raise.
    """
    regime = MockRegime(
        actions={"a": DiscreteGrid(binary_category_class)},
        states={"b": DiscreteGrid(binary_category_class)},
        state_transitions={
            "b": MarkovTransition(
                AgeSpecialized(
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
    """A deterministic state transition cannot itself be `AgeSpecialized`.

    Policy-dependent laws of motion are expressed as a plain transition reading an
    `AgeSpecialized` helper function; a direct marker in `state_transitions` must
    raise before any program is built.
    """
    regime = MockRegime(
        actions={"a": DiscreteGrid(binary_category_class)},
        states={"b": DiscreteGrid(binary_category_class)},
        state_transitions={
            "b": AgeSpecialized(
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
    """A terminal regime cannot contain `AgeSpecialized` functions or constraints.

    The terminal value program is built once and shared across all periods, so a
    policy-specialized terminal function must raise instead of silently using one
    age's closure.
    """
    regime = MockRegime(
        states={"b": DiscreteGrid(binary_category_class)},
        transition=None,
        functions={
            "utility": AgeSpecialized(
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
    """A plain regime transition cannot read an `AgeSpecialized` helper function.

    Regime-transition probabilities are built once, not per period, so a
    policy-specialized value flowing into `next_regime` would silently reuse one
    age's policy closure across all periods. It must raise instead.
    """

    def next_regime(policy_threshold):
        return policy_threshold

    regime = MockRegime(
        actions={"a": DiscreteGrid(binary_category_class)},
        states={"b": DiscreteGrid(binary_category_class)},
        transition=MarkovTransition(next_regime),
        functions={
            "utility": lambda a, b: None,  # noqa: ARG005
            "policy_threshold": AgeSpecialized(
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

    `next_regime` reads a plain function which itself reads an `AgeSpecialized`
    helper; the policy dependency is transitive but just as unsound, so it must
    raise.
    """

    def eligibility(policy_threshold):
        return policy_threshold

    def next_regime(eligibility):
        return eligibility

    regime = MockRegime(
        actions={"a": DiscreteGrid(binary_category_class)},
        states={"b": DiscreteGrid(binary_category_class)},
        transition=MarkovTransition(next_regime),
        functions={
            "utility": lambda a, b: None,  # noqa: ARG005
            "eligibility": eligibility,
            "policy_threshold": AgeSpecialized(
                build=lambda age: lambda b: age,  # noqa: ARG005
                signature=lambda age: ("threshold", age),
            ),
        },
    )

    with pytest.raises(RegimeInitializationError):
        _validate_logical_consistency(regime)
