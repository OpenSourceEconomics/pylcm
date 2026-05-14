"""`BeartypeConf` instances for pylcm's perimeter and internal claws.

Perimeter confs map type violations to the existing project exception
class, preserving the documented exception hierarchy at user-facing
entry points. `INTERNAL_CONF` covers packages that run behind the
perimeter, where a violation is an internal bug rather than user error
and so surfaces as beartype's own `BeartypeCallHintViolation`.

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

# Used by the claw on `lcm.solution` and `lcm.simulation`. These packages run
# behind the construction perimeter — their inputs are already validated by
# `Model.solve` / `Model.simulate` and `validate_initial_conditions` — so a
# type violation there is an internal pylcm bug, not user error. It surfaces
# as beartype's own `BeartypeCallHintViolation` rather than a project
# exception, which would mislabel it as a user-facing error.
INTERNAL_CONF = BeartypeConf(
    strategy=BeartypeStrategy.On,
    is_pep484_tower=True,
)
