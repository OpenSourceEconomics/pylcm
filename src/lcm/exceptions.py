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
    """Raised when a caller-supplied `period_to_regime_to_V_arr` is incomplete.

    Every solution-graph continuation target for the current
    `(period, source regime)` decision must have a value-function array
    present; a missing one is a simulation-input error, not a solver defect.
    """


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


class ExactAffineKernelUnavailableError(PyLCMError):
    """Raised when the certified upper envelope is used without its kernel.

    The certified path decides candidate ownership in exact integer arithmetic
    over the stored operand bits, which a compiled shared object provides. Where
    that object is absent or cannot be loaded, the path states so at the moment
    it is asked for a verdict rather than falling back to floating arithmetic
    that cannot make the guarantee.
    """


class ScaledLotteryDifferentiationError(PyLCMError):
    """Raised when a scaled certainty-equivalent reduction is differentiated.

    A lottery reaches `aggregate_scaled` with its weights split into a
    coefficient and a base-two shift precisely because no ordinary float states
    the probability. The same holds of a derivative with respect to such a
    weight, so the scaled reduction states no derivative rather than reporting
    a zero that a gradient-based caller would read as a flat objective.
    """
