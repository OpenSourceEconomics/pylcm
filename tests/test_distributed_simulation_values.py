"""Simulation reads placed values without changing the owned solution.

Run this module alone so its four-CPU-device topology precedes JAX initialization.
"""

from functools import cache
from typing import cast

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from _lcm.solution.artifacts import OwnedSolutionView
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime
from lcm.result import SimulationResult
from lcm.solver_api import SolutionResult
from lcm.typing import ContinuousState, ScalarFloat, ScalarInt
from tests.conftest import assert_agrees_to_ulp

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _TOPOLOGY_AVAILABLE = True
except RuntimeError:
    _TOPOLOGY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TOPOLOGY_AVAILABLE, reason="Requires a fresh four-CPU-device process"
)


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt


@categorical(ordered=True)
class _ThreeTypes:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


@categorical(ordered=True)
class _FourTypes:
    low: ScalarInt
    lower_middle: ScalarInt
    upper_middle: ScalarInt
    high: ScalarInt


def _utility(
    *, wealth: ContinuousState, consumption: ScalarFloat, preference: ScalarInt
) -> ScalarFloat:
    return jnp.log(consumption) + (preference + 1) * wealth * 0.001


def _next_wealth(*, wealth: ContinuousState, consumption: ScalarFloat) -> ScalarFloat:
    return wealth - consumption


def _transition(age: ScalarFloat) -> ScalarInt:
    return jnp.where(age >= 1, _RegimeId.retired, _RegimeId.working)


def _terminal_utility(wealth: ContinuousState) -> ScalarFloat:
    return wealth * 0.5


def _build_model(
    *, n_types: int, devices: tuple[int, ...], sharded: bool, prewarm: bool = False
) -> Model:
    """Build a sharded preference axis beside an unsharded terminal regime."""
    wealth = LinSpacedGrid(start=1, stop=20, n_points=6)
    return Model(
        regimes={
            "working": Regime(
                transition=_transition,
                active=lambda age: age < 3,
                states={"wealth": wealth},
                state_transitions={"wealth": _next_wealth},
                actions={"consumption": LinSpacedGrid(start=1, stop=5, n_points=5)},
                functions={"utility": _utility},
            ),
            "retired": Regime(
                transition=None,
                states={"wealth": wealth},
                functions={"utility": _terminal_utility},
            ),
        },
        states={
            "preference": DiscreteGrid(_ThreeTypes if n_types == 3 else _FourTypes)
        },
        state_transitions={"preference": fixed_transition("preference")},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(
            sharded_states=("preference",) if sharded else (), devices=devices
        ),
        n_subjects=7 if prewarm else None,
    )


@cache
def _simulate(
    *, n_types: int, devices: tuple[int, ...], sharded: bool, prewarm: bool = False
) -> tuple[SolutionResult, SimulationResult, SimulationResult]:
    """Replay the same owned solution twice with a population requiring padding."""
    model = _build_model(
        n_types=n_types, devices=devices, sharded=sharded, prewarm=prewarm
    )
    params = {"discount_factor": 0.95}
    solution = model.solve(params=params, log_level="off")
    initial = {
        "wealth": jnp.full(7, 12.0),
        "age": jnp.zeros(7),
        "preference": jnp.arange(7, dtype=jnp.int32) % n_types,
        "regime_id": jnp.full(7, _RegimeId.working),
    }
    first = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=0,
        log_level="off",
    )
    second = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=0,
        log_level="off",
    )
    return solution, first, second


_PLACEMENTS = [(3, (0, 1, 2, 3)), (4, (0, 1, 2, 3)), (3, (1, 2, 3))]


@pytest.mark.parametrize(("n_types", "devices"), _PLACEMENTS)
@pytest.mark.parametrize("prewarm", [False, True])
def test_placed_solution_simulates_to_the_single_device_result(
    *, n_types: int, devices: tuple[int, ...], prewarm: bool
) -> None:
    """Full and proper submeshes preserve all seven subjects' simulated paths."""
    _, actual, _ = _simulate(
        n_types=n_types, devices=devices, sharded=True, prewarm=prewarm
    )
    _, expected, _ = _simulate(n_types=n_types, devices=(0,), sharded=False)
    got = actual.to_dataframe(use_labels=False)
    want = expected.to_dataframe(use_labels=False)
    pd.testing.assert_index_equal(got.index, want.index)
    pd.testing.assert_index_equal(got.columns, want.columns)
    for column in want:
        if column == "value":
            assert_agrees_to_ulp(
                got=got[column].to_numpy(),
                expected=want[column].to_numpy(),
                n_ulp=8,
                err_msg=f"{column=}, {n_types=}, {devices=}",
            )
        else:
            pd.testing.assert_series_equal(got[column], want[column], check_exact=True)


@pytest.mark.parametrize(("n_types", "devices"), _PLACEMENTS)
def test_simulation_retains_original_value_arrays_on_their_solve_placements(
    *, n_types: int, devices: tuple[int, ...]
) -> None:
    """Published value mappings keep the original arrays alive after repeated replay."""
    solution, first, second = _simulate(n_types=n_types, devices=devices, sharded=True)
    # Public ValueStore reads deliberately return detached copies. The engine's
    # immutable owned view is the original placement/lifetime contract being tested.
    owned = cast("OwnedSolutionView", solution._engine_view)
    for period, values in owned.values.items():
        for regime, value in values.items():
            assert not value.is_deleted()
            assert first.period_to_regime_to_V_arr[period][regime] is value
            assert second.period_to_regime_to_V_arr[period][regime] is value


@pytest.mark.parametrize(("n_types", "devices"), _PLACEMENTS)
def test_repeated_simulation_of_the_same_placed_solution_is_identical(
    *, n_types: int, devices: tuple[int, ...]
) -> None:
    """Temporary transfers from one replay never invalidate the next replay."""
    _, first, second = _simulate(n_types=n_types, devices=devices, sharded=True)
    pd.testing.assert_frame_equal(
        first.to_dataframe(), second.to_dataframe(), check_exact=True
    )
