"""Simulation programs are shared across parameters that only move grid points."""

import jax
import jax.numpy as jnp
import pandas as pd
import pytest
from jax._src import monitoring

from lcm import ExecutionConfig, IrregSpacedGrid
from lcm.model import Model
from lcm.typing import UserParams
from tests.simulation.initial_conditions._models import make_constraint_model

_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


def _model() -> Model:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("Select this test with at least two visible devices.")
    return make_constraint_model(
        wealth_grid=IrregSpacedGrid(n_points=15),
        execution_config=ExecutionConfig(
            devices=tuple(device.id for device in devices[:2]),
            sharded_states=(),
            simulation_sharding="subjects",
        ),
    )


def _params(*, lowest_point: float) -> UserParams:
    return {
        "discount_factor": 0.95,
        "working_life": {"wealth": {"points": jnp.linspace(lowest_point, 10.0, 15)}},
        "final_age_alive": 1,
    }


def _simulate(*, model: Model, lowest_point: float) -> pd.DataFrame:
    params = _params(lowest_point=lowest_point)
    return model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(4),
            "wealth": jnp.array([5.0, 3.0, 8.0, 9.5]),
            "regime_id": jnp.full(4, model.regime_names_to_ids["working_life"]),
        },
        solution=model.solve(params=params, log_level="off"),
        log_level="off",
        seed=11,
    ).to_dataframe()


def _count_compiles(*, model: Model, lowest_point: float) -> int:
    """Count backend compilations, persistent-cache loads included, of one call."""
    events: list[str] = []

    # keyword-only-exempt: library-callback=jax.monitoring
    def listener(event: str, duration_secs: float, **kwargs: str | int) -> None:
        del duration_secs, kwargs
        if event == _COMPILE_EVENT:
            events.append(event)

    monitoring.register_event_duration_secs_listener(listener)
    try:
        _simulate(model=model, lowest_point=lowest_point)
    finally:
        monitoring.unregister_event_duration_listener(listener)
    return len(events)


@pytest.mark.parametrize(
    ("second_lowest_point", "expected_compiles"),
    [(2.0, 0), (2.5, 0)],
    ids=["repeated_points", "new_points"],
)
def test_simulate_at_new_grid_points_compiles_nothing(
    *, second_lowest_point: float, expected_compiles: int
) -> None:
    """A second simulate whose parameters only move runtime grid points reuses
    every program the first call compiled, as a repeat of the same points does."""
    model = _model()
    _simulate(model=model, lowest_point=2.0)
    assert (
        _count_compiles(model=model, lowest_point=second_lowest_point)
        == expected_compiles
    )


def test_first_simulate_compiles_its_programs() -> None:
    """The compile count reads the first call's programs, so a zero above is real."""
    assert _count_compiles(model=_model(), lowest_point=3.0) > 0


def test_simulate_with_reused_programs_matches_a_fresh_model() -> None:
    """Reusing programs across grid points yields exactly a fresh model's panel."""
    model = _model()
    _simulate(model=model, lowest_point=2.0)
    reused = _simulate(model=model, lowest_point=2.5)
    jax.clear_caches()
    fresh = _simulate(model=_model(), lowest_point=2.5)
    pd.testing.assert_frame_equal(reused, fresh, check_exact=True)
