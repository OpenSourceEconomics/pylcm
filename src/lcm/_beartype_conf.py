"""Per-exception `BeartypeConf` instances used at the pylcm perimeter.

Decorators at user-facing entry points configure beartype to raise
the existing project exception class on parameter-type violations,
preserving the documented exception hierarchy.

"""

from beartype import BeartypeConf

from lcm.exceptions import (
    CategoricalDefinitionError,
    GridInitializationError,
    InvalidParamsError,
    ModelInitializationError,
    RegimeInitializationError,
)


def _conf(exc: type[Exception]) -> BeartypeConf:
    return BeartypeConf(violation_param_type=exc)


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
