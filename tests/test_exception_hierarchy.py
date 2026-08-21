"""Model construction raises one catchable family."""

from lcm.exceptions import ModelInitializationError, RegimeInitializationError


def test_regime_initialization_error_is_a_model_initialization_error() -> None:
    """A regime-level construction failure is catchable as a model failure."""
    assert issubclass(RegimeInitializationError, ModelInitializationError)
