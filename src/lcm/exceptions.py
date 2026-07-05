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


class NBEGMCaseError(PyLCMError):
    """Raised when a NBEGM case-boundary or formula-piece declaration is invalid.

    Covers three families of checks:

    - Invalid boundary/piece declarations: a bare `(variable, threshold)` tuple
      that does not declare equality ownership, a case boundary with no
      `lcm.boundary(...)` surface, a piece referencing an undeclared predicate,
      or a duplicate/missing `when`/`otherwise` side for an output.
    - The AST/JAXPR smoothness gate: hidden branching (a Python `if`, a bare
      comparison, a piecewise primitive inside a helper) in a case's economic
      nodes.
    - The v1 scope gate: a non-`'subsidy'` split output, a state-dependent
      piece, a `'when'`-owned equality, a non-`'jump'` boundary kind, or a
      boundary on a variable other than the liquid state.
    """
