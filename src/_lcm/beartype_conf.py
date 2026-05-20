"""`BeartypeConf` instances for pylcm's perimeter and internal claws.

`INTERNAL_CONF` is the default conf for the package-wide claw on `lcm`
and `_lcm`, registered in `lcm/__init__.py`. Violations under that claw
surface as beartype's own `BeartypeCallHintViolation`, marking them as
internal pylcm bugs rather than user error.

The remaining confs (`MODEL_CONF`, `REGIME_CONF`, `GRID_CONF`,
`PARAMS_CONF`, `CATEGORICAL_CONF`) are used by explicit
`@beartype(conf=...)` decorators on user-facing constructors and entry
points. They map type violations to the existing project exception
class, preserving the documented exception hierarchy at the user
boundary. The decorators stack on top of the package claw and take
precedence at the call sites they cover.

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

# Used on all grid and process-grid constructors.
GRID_CONF = _conf(GridInitializationError)

# Used on the `categorical` decorator factory.
CATEGORICAL_CONF = _conf(CategoricalDefinitionError)

# Used on `Model.solve`, `Model.simulate`, and the `as_leaf` factory.
PARAMS_CONF = _conf(InvalidParamsError)

# Default conf for the package-wide claw on `lcm` registered in
# `lcm/__init__.py`. A type violation in any internal helper surfaces as
# beartype's own `BeartypeCallHintViolation` rather than a project
# exception. User-facing constructors layer their own
# `@beartype(conf=...)` decorators on top to map violations to project
# exceptions; those decorators take precedence at the call sites they
# cover.
INTERNAL_CONF = BeartypeConf(
    strategy=BeartypeStrategy.On,
    is_pep484_tower=True,
)
