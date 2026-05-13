"""Per-exception `BeartypeConf` instances used at the pylcm perimeter.

Decorators at user-facing entry points configure beartype to raise
the existing project exception class on parameter-type violations,
preserving the documented exception hierarchy.

"""

from collections.abc import Callable

from beartype import BeartypeConf, BeartypeStrategy, beartype

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


def beartype_init[C](conf: BeartypeConf) -> Callable[[type[C]], type[C]]:
    """Class decorator that beartype-checks `__init__` only.

    Bare `@beartype` on a class wraps every method, which surfaces
    annotation drift in helpers like `compute_gridpoints(**kwargs: float)`
    where runtime kwargs are actually JAX arrays. Restricting decoration
    to `__init__` keeps the perimeter check (parameter types at
    construction) without policing every method's runtime types.

    """

    def deco(cls: type[C]) -> type[C]:
        cls.__init__ = beartype(conf=conf)(cls.__init__)  # ty: ignore[invalid-assignment]
        return cls

    return deco


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
