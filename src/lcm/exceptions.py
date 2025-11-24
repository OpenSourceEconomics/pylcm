from __future__ import annotations


class PyLCMError(Exception):
    """Base class for all PyLCM exceptions."""


class InvalidValueFunctionError(PyLCMError):
    """Raised when the value function array is invalid."""

class RegimeInitializationError(PyLCMError):
    """Raised when there is an error in the regime initialization."""


class ModelInitializationError(PyLCMError):
    """Raised when there is an error in the model initialization."""

class ShockInitializationError(PyLCMError):
    """Raised when there is an error in the shock initialization."""

class GridInitializationError(PyLCMError):
    """Raised when there is an error in the grid initialization."""


def format_messages(errors: str | list[str]) -> str:
    """Convert message or list of messages into a single string."""
    if isinstance(errors, str):
        formatted = errors
    elif len(errors) == 1:
        formatted = errors[0]
    else:
        enumerated = "\n\n".join([f"{i}. {error}" for i, error in enumerate(errors, 1)])
        formatted = f"The following errors occurred:\n\n{enumerated}"
    return formatted
