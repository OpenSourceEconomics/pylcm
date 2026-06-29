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

    Covers a `MarkovTransition` function whose output has the wrong outcome-axis
    size, values outside [0, 1], rows that don't sum to 1, or `probs_array[…]`
    subscripts that don't match the signature parameter order.
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


class BQSEGMCaseError(PyLCMError):
    """Raised when a BQSEGM case-boundary or formula-piece declaration is invalid.

    Covers an un-inferable boundary specification (a bare `(variable, threshold)`
    tuple that does not declare equality ownership), incomplete piece coverage
    (a missing `when` or `otherwise` side for an output), and a case boundary
    that does not return a Boolean.
    """
