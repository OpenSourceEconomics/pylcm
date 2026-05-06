"""Test that numpy arrays in params are rejected after processing."""

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical


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


def test_numpy_array_param_accepted_and_normalised() -> None:
    """Numpy arrays are auto-converted to JAX at the params boundary."""
    model = _make_model()
    # Should solve cleanly; the boundary cast normalises numpy -> JAX.
    model.solve(params={"bonus": np.array(1.0), "discount_factor": 0.95})  # ty: ignore[invalid-argument-type]


def test_jax_array_param_accepted() -> None:
    """JAX arrays should be accepted."""
    model = _make_model()
    model.solve(params={"bonus": jnp.array(1.0), "discount_factor": 0.95})


def test_python_scalar_param_accepted() -> None:
    """Python scalars should be accepted."""
    model = _make_model()
    model.solve(params={"bonus": 1.0, "discount_factor": 0.95})
