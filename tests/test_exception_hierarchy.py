"""Model construction raises one catchable family."""

from lcm.exceptions import (
    ExecutionPlanningError,
    ModelInitializationError,
    PyLCMError,
    RegimeInitializationError,
)


def test_regime_initialization_error_is_a_model_initialization_error() -> None:
    """A regime-level construction failure is catchable as a model failure."""
    assert issubclass(RegimeInitializationError, ModelInitializationError)


def test_execution_planning_error_is_a_pylcm_error() -> None:
    assert issubclass(ExecutionPlanningError, PyLCMError)
