"""Per-exception `BeartypeConf` instances used at the pylcm perimeter.

Decorators at user-facing entry points configure beartype to raise
the existing project exception class on parameter-type violations,
preserving the documented exception hierarchy.

"""

from beartype import BeartypeConf, BeartypeStrategy

from lcm.exceptions import (
    CategoricalDefinitionError,
    GridInitializationError,
    InvalidParamsError,
    ModelInitializationError,
    RegimeInitializationError,
)


def _conf(exc: type[Exception]) -> BeartypeConf:
    # Full O(n) container validation so every bad entry in a mapping/sequence
    # gets reported, not just one sampled element. The decorated entry points
    # are called rarely (construction, solve, simulate), so per-call cost is
    # invisible.
    return BeartypeConf(violation_param_type=exc, strategy=BeartypeStrategy.On)


# Used on `Regime` and `MarkovTransition`.
REGIME_CONF = _conf(RegimeInitializationError)

# Used on `Model`.
MODEL_CONF = _conf(ModelInitializationError)

# Used on all grid and shock-grid constructors.
GRID_CONF = _conf(GridInitializationError)

# Used on the `categorical` decorator factory.
CATEGORICAL_CONF = _conf(CategoricalDefinitionError)

# Used on `Model.solve` and `Model.simulate`.
PARAMS_CONF = _conf(InvalidParamsError)
