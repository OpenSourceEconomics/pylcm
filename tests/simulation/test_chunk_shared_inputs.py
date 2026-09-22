"""Shared spaces are completed once; stochastic ID staging stays admitted."""

from types import MappingProxyType
from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.engine import SolutionPhase
from _lcm.simulation.transitions import draw_key_from_dict
from lcm import ExecutionConfig, LinSpacedGrid
from tests.simulation.test_population_allocation_budget import (
    _memory,
    _UnadmittedAllocationError,
)
from tests.test_models.deterministic.regression import RegimeId, get_model, get_params


def test_complete_spaces_are_call_local_across_subject_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = get_model(
        n_periods=2,
        wealth_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        consumption_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        execution_config=ExecutionConfig(axis_widths={"subject": 3}),
    )
    params = get_params(n_periods=2)
    solution = model.solve(params=params, log_level="off")
    counts: dict[int, int] = {}
    original = SolutionPhase.state_action_space

    def observe(self: SolutionPhase, **kwargs: Any) -> object:
        counts[id(self)] = counts.get(id(self), 0) + 1
        return original(self, **kwargs)

    monkeypatch.setattr(SolutionPhase, "state_action_space", observe)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.linspace(1, 3, 7),
            "age": jnp.full(7, 18.0),
            "regime_id": jnp.full(7, RegimeId.working_life, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
    )
    assert result.n_subjects == 7
    assert len(counts) == len(model._regimes)
    assert tuple(counts.values()) == (1,) * len(model._regimes)


@pytest.mark.parametrize("scalar", [False, True])
def test_draw_ids_are_constructed_inside_the_admitted_numerical_body(
    *, monkeypatch: pytest.MonkeyPatch, scalar: bool
) -> None:
    keys = jax.random.split(jax.random.key(17), 3)
    ids = MappingProxyType({"high": jnp.int32(9), "low": jnp.int32(2)})
    probabilities = MappingProxyType(
        {
            "high": jnp.asarray(0.8 if scalar else [0.8, 0.2, 0.5]),
            "low": jnp.asarray(0.2 if scalar else [0.2, 0.8, 0.5]),
        }
    )
    rows = (
        np.asarray([0.2, 0.8])
        if scalar
        else np.asarray([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]])
    )
    expected = np.asarray(
        [
            jax.random.choice(
                key,
                np.asarray([2, 9], dtype=np.int32),
                p=rows if scalar else rows[position],
            )
            for position, key in enumerate(keys)
        ]
    )
    memory = _memory(inputs=(keys, ids, probabilities), budget=1_000_000)
    original = jnp.asarray

    def guard(value: object, *args: Any, **kwargs: Any) -> object:
        if isinstance(value, list | tuple) and any(
            isinstance(leaf, jax.Array) and not isinstance(leaf, jax.core.Tracer)
            for leaf in jax.tree.leaves(value)
        ):
            raise _UnadmittedAllocationError(
                "Regime ID vector staged outside its admitted body"
            )
        return original(value, *args, **kwargs)

    with monkeypatch.context() as capture:
        capture.setattr(jnp, "asarray", guard)
        with pytest.raises(_UnadmittedAllocationError):
            jnp.asarray(list(ids.values()))
        actual = draw_key_from_dict(
            d=probabilities, regime_names_to_ids=ids, keys=keys, memory=memory
        )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == jnp.int32
