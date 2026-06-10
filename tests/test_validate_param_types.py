"""Tests for params accepted at the boundary by `_validate_param_types`.

After `process_params` casts typed numeric arrays to canonical pylcm
dtypes, every supported user input form (numpy arrays, JAX arrays,
Python scalars) reaches the validator as a JAX array or Python scalar
and is accepted.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from _lcm.dtypes import canonical_float_dtype
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import ScalarInt


@categorical(ordered=True)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _next_regime() -> ScalarInt:
    return RegimeId.dead


working = UserRegime(
    transition=_next_regime,
    active=lambda age: age < 30,
    states={
        "health": DiscreteGrid(Health),
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
    },
    state_transitions={
        "health": fixed_transition("health"),
        "wealth": lambda wealth: wealth,
    },
    functions={"utility": lambda wealth, health, bonus: wealth + health + bonus},
)

dead = UserRegime(
    transition=None,
    functions={"utility": lambda: 0.0},
)


def _make_model() -> Model:
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=25, stop=30, step="Y"),
        regime_id_class=RegimeId,
    )


def test_numpy_array_param_normalised_to_canonical_jax_array() -> None:
    """A numpy array param is cast to a JAX array at `canonical_float_dtype()`."""
    model = _make_model()
    internal = model._process_params(
        params={"bonus": np.asarray(1.0, dtype=np.float64), "discount_factor": 0.95}
    )
    bonus = internal["working"]["utility__bonus"]
    assert isinstance(bonus, Array)
    assert bonus.dtype == canonical_float_dtype()


def test_jax_array_param_kept_at_canonical_dtype() -> None:
    """A typed JAX array param is kept (or cast) at `canonical_float_dtype()`."""
    model = _make_model()
    internal = model._process_params(
        params={"bonus": jnp.asarray(1.0), "discount_factor": 0.95}
    )
    bonus = internal["working"]["utility__bonus"]
    assert bonus.dtype == canonical_float_dtype()  # ty: ignore[unresolved-attribute]


def test_python_float_param_cast_to_canonical_dtype() -> None:
    """A Python `float` param is cast to `canonical_float_dtype()`."""
    model = _make_model()
    internal = model._process_params(params={"bonus": 1.0, "discount_factor": 0.95})
    bonus = internal["working"]["utility__bonus"]
    assert float(bonus) == 1.0  # ty: ignore[invalid-argument-type]
    assert bonus.dtype == canonical_float_dtype()  # ty: ignore[unresolved-attribute]
