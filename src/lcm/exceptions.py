class PyLCMError(Exception):
    """Base class for all PyLCM exceptions."""


class InvalidValueFunctionError(PyLCMError):
    """Raised when the value function array is invalid.

    Attributes:
        partial_solution: Value function arrays for periods that completed
            before the error. Attached by `validate_V` so callers can save
            debug snapshots.
        diagnostics: Per-intermediate NaN fraction summary, attached by
            `validate_V` when diagnostic functions are available.

    """

    partial_solution: object = None
    diagnostics: object = None


class InvalidRegimeTransitionProbabilitiesError(PyLCMError):
    """Raised when the regime transition probabilities are invalid."""


class InvalidStateTransitionProbabilitiesError(PyLCMError):
    """Raised when a stochastic state transition produces invalid probabilities.

    Surfaces shape mismatches, values outside [0, 1], and rows that don't sum
    to 1 in a state's `MarkovTransition` function output. Raised by the
    pre-solve sweep `validate_state_transitions_all_periods` in
    `regime_building/runtime_checks.py` and by the process-time AST
    subscript-order check in `regime_building/static_checks.py`.
    """


class InvalidInitialConditionsError(PyLCMError):
    """Raised when the initial conditions (states or regimes) are invalid."""


class InvalidParamsError(PyLCMError):
    """Raised when the params structure does not match the params template."""


class InvalidNameError(PyLCMError):
    """Raised when names are invalid (e.g., contain separator or are not disjoint)."""


class InvalidAdditionalTargetsError(PyLCMError):
    """Raised when the additional targets are invalid."""


class RegimeInitializationError(PyLCMError):
    """Raised when there is an error in the regime initialization."""


class ModelInitializationError(PyLCMError):
    """Raised when there is an error in the model initialization."""


class GridInitializationError(PyLCMError):
    """Raised when there is an error in the grid initialization."""


class CategoricalDefinitionError(PyLCMError):
    """Raised when an `@categorical`-decorated class fails the contract.

    `@categorical` requires every field to be annotated as `ScalarInt`
    (the 0-d `jnp.int32` scalar pylcm produces for category codes).
    Violations are caught at decoration time, before any grid, regime,
    or derived-categorical mapping is built.
    """


class FunctionDispatchError(PyLCMError):
    """Raised when there is an error during the function dispatch."""


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
