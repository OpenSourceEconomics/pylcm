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
    # `On` strategy: full O(n) container validation so every bad entry in a
    # mapping/sequence is reported, not just one sampled element. The
    # decorated entry points are called rarely (construction, solve,
    # simulate), so per-call cost is invisible.
    # `is_pep484_tower=True`: respect the PEP-484 numeric tower so `int`
    # satisfies `float`-typed parameters (matches the implicit numeric
    # conversion that Python and ruff's PYI041 both assume).
    return BeartypeConf(
        violation_param_type=exc,
        strategy=BeartypeStrategy.On,
        is_pep484_tower=True,
    )


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

# Used by the claw on `lcm.regime_building` (regime compilation pipeline,
# part of model construction).
REGIME_BUILDING_CONF = _conf(ModelInitializationError)
