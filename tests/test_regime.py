"""Test Regime class validation."""

import inspect

import jax.numpy as jnp
import pytest
from dags.tree import QNAME_DELIMITER

from lcm import DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import _IdentityTransition
from lcm.typing import ContinuousState, DiscreteState


def utility(consumption):
    return consumption


def next_wealth(wealth, consumption):
    return wealth - consumption


WEALTH_GRID = LinSpacedGrid(start=1, stop=10, n_points=5)
CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=5, n_points=5)


def test_regime_name_does_not_contain_separator():
    """Regime name validation happens at Model level, not Regime level."""

    @categorical
    class RegimeId:
        work__test: int  # Contains separator - but RegimeId class has matching field
        dead: int

    working = Regime(
        functions={"utility": utility},
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        transition=lambda: 0,
        active=lambda age: age < 5,
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0},
        active=lambda age: age >= 5,
    )
    ages = AgeGrid(start=0, stop=5, step="Y")

    # Regime name containing separator should raise at Model creation
    with pytest.raises(ModelInitializationError, match=QNAME_DELIMITER):
        Model(
            regimes={f"work{QNAME_DELIMITER}test": working, "dead": dead},
            ages=ages,
            regime_id_class=RegimeId,
        )


def test_function_name_does_not_contain_separator():
    with pytest.raises(RegimeInitializationError, match=QNAME_DELIMITER):
        Regime(
            states={"wealth": WEALTH_GRID},
            actions={f"consumption{QNAME_DELIMITER}action": CONSUMPTION_GRID},
            transition=next_wealth,
            functions={"utility": utility, f"helper{QNAME_DELIMITER}func": lambda: 1},
            active=lambda age: age < 5,
        )


def test_state_name_does_not_contain_separator():
    with pytest.raises(RegimeInitializationError, match=QNAME_DELIMITER):
        Regime(
            functions={"utility": utility},
            states={f"my{QNAME_DELIMITER}wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            transition=next_wealth,
            active=lambda age: age < 5,
        )


# ======================================================================================
# Terminal Regime Tests
# ======================================================================================


def test_terminal_regime_creation():
    """Terminal regime (transition=None) can be created with states and utility."""
    regime = Regime(
        transition=None,
        functions={"utility": lambda wealth: wealth * 0.5},
        states={"wealth": WEALTH_GRID},
        active=lambda age: age >= 5,
    )
    assert regime.terminal is True


def test_terminal_regime_with_actions():
    """Terminal regime can have actions for final decisions."""
    regime = Regime(
        transition=None,
        functions={"utility": lambda wealth, bequest_share: wealth * bequest_share},
        states={"wealth": WEALTH_GRID},
        actions={"bequest_share": LinSpacedGrid(start=0, stop=1, n_points=11)},
        active=lambda age: age >= 5,
    )
    assert regime.terminal is True
    assert "bequest_share" in regime.actions


def test_non_terminal_regime_has_transition():
    """A regime with a transition function is non-terminal."""
    regime = Regime(
        functions={"utility": utility},
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        transition=next_wealth,
        active=lambda age: age < 5,
    )
    assert regime.terminal is False


def test_terminal_regime_can_be_created_without_states():
    """Terminal regime can be created without states (e.g., death state)."""
    regime = Regime(
        transition=None,
        functions={"utility": lambda: 0},
        states={},
        active=lambda age: age >= 5,
    )
    assert regime.terminal is True
    assert regime.states == {}


# ======================================================================================
# Active Attribute Tests
# ======================================================================================


def test_regime_with_active_callable():
    """Regime can specify active periods with a callable."""
    regime = Regime(
        transition=next_wealth,
        functions={"utility": utility},
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        active=lambda age: age < 5,
    )
    assert callable(regime.active)
    assert regime.active(3) is True
    assert regime.active(5) is False


def test_regime_requires_utility_in_functions():
    """Regime must have 'utility' in the functions dict."""
    with pytest.raises(RegimeInitializationError, match=r"utility.*must be provided"):
        Regime(
            transition=None,
            functions={"helper": lambda: 1},
            states={"wealth": WEALTH_GRID},
        )


def test_active_validation_rejects_non_callable():
    """Active attribute must be a callable."""
    with pytest.raises(RegimeInitializationError, match="must be a callable"):
        Regime(
            transition=next_wealth,
            functions={"utility": utility},
            states={"wealth": WEALTH_GRID},
            actions={"consumption": CONSUMPTION_GRID},
            active=[0, 1, 2],  # ty: ignore[invalid-argument-type]  # Not a callable
        )


# ======================================================================================
# _IdentityTransition Tests
# ======================================================================================


def test_identity_transition_call():
    """Identity transition returns the state value unchanged."""
    identity = _IdentityTransition("wealth", annotation=ContinuousState)
    result = identity(wealth=jnp.array(42.0))
    assert result == jnp.array(42.0)


def test_identity_transition_discrete():
    """Identity transition works for discrete states."""
    identity = _IdentityTransition("education", annotation=DiscreteState)
    result = identity(education=jnp.array(1))
    assert result == jnp.array(1)


def test_identity_transition_name():
    """Identity transition has the correct __name__."""
    identity = _IdentityTransition("wealth", annotation=ContinuousState)
    assert identity.__name__ == "next_wealth"


def test_identity_transition_signature():
    """Identity transition has a proper signature with annotation."""
    identity = _IdentityTransition("wealth", annotation=ContinuousState)
    sig = inspect.signature(identity)
    assert list(sig.parameters) == ["wealth"]
    assert sig.parameters["wealth"].annotation is ContinuousState
    assert sig.return_annotation is ContinuousState


def test_identity_transition_annotations():
    """Identity transition exposes __annotations__ for dags discovery."""
    identity = _IdentityTransition("education", annotation=DiscreteState)
    assert identity.__annotations__ == {
        "education": DiscreteState,
        "return": DiscreteState,
    }


def test_identity_transition_is_auto_identity():
    """Identity transition is flagged as auto-generated."""
    identity = _IdentityTransition("x", annotation=ContinuousState)
    assert identity._is_auto_identity is True  # noqa: SLF001


def test_get_all_functions_includes_identity_for_fixed_discrete_state():
    """Fixed discrete states get identity transitions with DiscreteState annotation."""

    @categorical
    class Edu:
        low: int
        high: int

    regime = Regime(
        transition=lambda: 0,
        functions={"utility": lambda education: education},
        states={"education": DiscreteGrid(Edu)},
    )
    all_fns = regime.get_all_functions()
    identity_fn = all_fns["next_education"]
    assert isinstance(identity_fn, _IdentityTransition)
    assert identity_fn.__annotations__["education"] is DiscreteState
    assert identity_fn.__annotations__["return"] is DiscreteState


def test_get_all_functions_includes_identity_for_fixed_continuous_state():
    """Fixed continuous states get identity transitions with correct annotation."""
    regime = Regime(
        transition=lambda: 0,
        functions={"utility": lambda wealth: wealth},
        states={"wealth": LinSpacedGrid(start=0, stop=10, n_points=5)},
    )
    all_fns = regime.get_all_functions()
    identity_fn = all_fns["next_wealth"]
    assert isinstance(identity_fn, _IdentityTransition)
    assert identity_fn.__annotations__["wealth"] is ContinuousState
    assert identity_fn.__annotations__["return"] is ContinuousState
