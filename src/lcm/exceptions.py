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


class InvalidSimulationInputError(PyLCMError):
    """Raised when caller-supplied solve artifacts cannot drive simulation.

    Covers:

    - A missing value-function array for a solution-graph continuation target.
    - A missing or mismatched replay policy where the solver's decision cannot
      be reconstructed from value functions alone.
    """


class UnsupportedOperationError(PyLCMError):
    """Raised when a valid model requests an unsupported runtime operation."""


class InvalidParamsError(PyLCMError):
    """Raised when the params structure does not match the params template."""


class InvalidNameError(PyLCMError):
    """Raised when names are invalid (e.g., contain separator or are not disjoint)."""


class InvalidAdditionalTargetsError(PyLCMError):
    """Raised when the additional targets are invalid."""


class ModelInitializationError(PyLCMError):
    """Raised when there is an error in the model initialization."""


class RegimeInitializationError(ModelInitializationError):
    """Raised when there is an error in the regime initialization.

    A regime is a component of a model and is validated both at its own
    construction and again when a model finalizes it, so the same defect
    surfaces from either call. Catching `ModelInitializationError` catches
    both.
    """


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


class OuterSearchConvergenceError(PyLCMError):
    """Raised when an adaptive outer mesh cannot resolve within its budget.

    Inference-grade continuous-outer solves must fail closed: reaching the
    node or round budget while validation-marked intervals remain raises
    instead of silently degrading the solution.
    """


class NBEGMCaseError(PyLCMError):
    """Raised when a NBEGM case-boundary or formula-piece declaration is invalid.

    Covers three families of checks:

    - Invalid boundary/piece declarations: a case split that is not one ordered
      structured comparison, a piece referencing a boundary absent from the
      function mapping, or a duplicate/missing `when`/`otherwise` side.
    - The AST/JAXPR smoothness gate: hidden branching (a Python `if`, a bare
      comparison, a piecewise primitive inside a helper) in a case's economic
      nodes.
    - The case-piece scope gate: a non-`'subsidy'` split output, a
      state-dependent piece, a non-`'jump'` boundary kind, or a boundary that
      does not compare the liquid state with a state-independent threshold.
    """


class ExactAffineKernelUnavailableError(PyLCMError):
    """Raised when a certified exact-affine operation cannot load its kernel.

    DCEGM's `ExactEnvelope` and NBEGM's `"certified"` ownership mode use exact
    arithmetic to determine which candidate delivers the highest value. The compiled
    exact-affine kernel ships as part of pylcm. If the compatible CPU or CUDA payload
    is missing or cannot be loaded for the active JAX backend, pylcm raises this error
    instead of silently using approximate floating-point comparisons.
    """


class ScaledLotteryDifferentiationError(PyLCMError):
    """Raised when differentiating a lottery with extremely small probabilities.

    `aggregate_scaled` preserves probabilities that are too small to represent as
    ordinary floating-point numbers. Derivatives with respect to such probabilities
    are not supported. pylcm therefore raises this error instead of returning zero,
    which could incorrectly suggest to an optimizer that the objective is locally
    flat.
    """
