"""Model construction raises one catchable family."""

import lcm.exceptions as exceptions_module
from lcm.exceptions import (
    ExecutionPlanningError,
    ModelInitializationError,
    ModelSealError,
    PyLCMError,
    RegimeInitializationError,
)


def test_regime_initialization_error_is_a_model_initialization_error() -> None:
    """A regime-level construction failure is catchable as a model failure."""
    assert issubclass(RegimeInitializationError, ModelInitializationError)


def test_execution_planning_error_is_a_pylcm_error() -> None:
    assert issubclass(ExecutionPlanningError, PyLCMError)


def test_model_identity_errors_belong_to_the_project_family() -> None:
    """Runtime model identity failures have a public common catch."""
    identity_error = getattr(exceptions_module, "ModelIdentityError", None)
    assert identity_error is not None
    assert issubclass(identity_error, PyLCMError)


def test_model_seal_errors_belong_to_the_model_identity_family() -> None:
    """Catching model identity errors also catches durable binding movement."""
    identity_error = getattr(exceptions_module, "ModelIdentityError", None)
    assert identity_error is not None
    assert issubclass(ModelSealError, identity_error)
