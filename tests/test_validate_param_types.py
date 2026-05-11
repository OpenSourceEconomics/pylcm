"""Tests for params accepted at the boundary by `_validate_param_types`.

After `process_params` casts typed numeric arrays to canonical pylcm
dtypes, every supported user input form (numpy arrays, JAX arrays,
Python scalars) reaches the validator as a JAX array or Python scalar
and is accepted.
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.dtypes import canonical_float_dtype


@categorical(ordered=True)
class Health:
    bad: int
    good: int


@categorical(ordered=False)
class RegimeId:
    working: int
    dead: int


def _next_regime() -> int:
    return RegimeId.dead


working = Regime(
    transition=_next_regime,
    active=lambda age: age < 30,
    states={
        "health": DiscreteGrid(Health),
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
    },
    state_transitions={"health": None, "wealth": lambda wealth: wealth},
    functions={"utility": lambda wealth, health, bonus: wealth + health + bonus},
)

dead = Regime(
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
        params={"bonus": np.asarray(1.0, dtype=np.float64), "discount_factor": 0.95}  # ty: ignore[invalid-argument-type]
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
    assert bonus.dtype == canonical_float_dtype()


def test_python_float_param_cast_to_canonical_dtype() -> None:
    """A Python `float` param is cast to `canonical_float_dtype()`."""
    model = _make_model()
    internal = model._process_params(params={"bonus": 1.0, "discount_factor": 0.95})
    bonus = internal["working"]["utility__bonus"]
    assert float(bonus) == 1.0
    assert bonus.dtype == canonical_float_dtype()
