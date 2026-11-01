"""Lifetime of the simulation runtimes a model binds per canonical generation."""

import jax.numpy as jnp
import pytest

from lcm import IrregSpacedGrid
from lcm.model import _DECLARED_AUTHORITY_CACHE_SIZE
from tests.simulation.initial_conditions._models import make_constraint_model

_N_CALLS = 2 * _DECLARED_AUTHORITY_CACHE_SIZE


def _simulate_at_wealth_points(*, model, lowest_point: float) -> None:
    model.simulate(
        params={
            "discount_factor": 0.95,
            "working_life": {
                "wealth": {"points": jnp.linspace(lowest_point, 10.0, 15)}
            },
            "final_age_alive": 1,
        },
        initial_conditions={
            "age": jnp.array([0.0]),
            "wealth": jnp.array([5.0]),
            "regime_id": jnp.array([model.regime_names_to_ids["working_life"]]),
        },
        log_level="off",
    )


@pytest.mark.parametrize(
    ("shift", "expected_runtimes"),
    [(0.0, 1), (0.01, _DECLARED_AUTHORITY_CACHE_SIZE)],
    ids=["repeated_points", "distinct_points"],
)
def test_simulate_keeps_runtimes_only_for_cached_generations(
    *, shift: float, expected_runtimes: int
) -> None:
    """Calls at ever-new runtime grid points keep at most one runtime per cached
    generation, so a long run of distinct points holds a bounded set of runtimes."""
    model = make_constraint_model(wealth_grid=IrregSpacedGrid(n_points=15))
    for call in range(_N_CALLS):
        _simulate_at_wealth_points(model=model, lowest_point=2.0 + shift * call)
    assert len(model._simulate_runtime_regimes) == expected_runtimes
