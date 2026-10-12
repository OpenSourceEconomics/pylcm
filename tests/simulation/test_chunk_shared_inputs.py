"""Shared spaces are completed once; stochastic ID staging stays admitted."""

from types import MappingProxyType

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from jax.typing import DTypeLike

from _lcm.engine import SolutionPhase, StateActionSpace
from _lcm.processes.grid_resolution import ProcessGridResolver
from _lcm.simulation.transitions import draw_key_from_dict
from _lcm.typing import FlatRegimeParams
from lcm import ExecutionConfig, LinSpacedGrid
from tests.simulation.test_population_allocation_budget import (
    _memory,
    _UnadmittedAllocationError,
)
from tests.test_models.deterministic.regression import RegimeId, get_model, get_params

_N_SUBJECTS = 7


@pytest.mark.parametrize("subject_width", [1, 3, 7])
def test_complete_spaces_are_call_local_across_subject_chunks(
    *, monkeypatch: pytest.MonkeyPatch, subject_width: int
) -> None:
    """Each regime's space is completed a fixed number of times per simulate call.

    The count does not depend on the subject-chunk width: each regime is
    completed once for the simulation inputs. The regime-selection check
    already passed for these params in `solve`, so simulate reuses it.
    """
    model = get_model(
        n_periods=2,
        wealth_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        consumption_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        execution_config=ExecutionConfig(axis_widths={"subject": subject_width}),
    )
    params = get_params(n_periods=2)
    solution = model.solve(params=params, log_level="off")
    names = {id(regime.solution): name for name, regime in model._regimes.items()}
    counts: dict[str, int] = {}
    original = SolutionPhase.state_action_space

    # keyword-only-exempt: library-callback=SolutionPhase.state_action_space
    def observe(
        self: SolutionPhase,
        *,
        regime_params: FlatRegimeParams,
        process_grid_resolver: ProcessGridResolver | None = None,
    ) -> StateActionSpace:
        counts[names[id(self)]] = counts.get(names[id(self)], 0) + 1
        return original(
            self,
            regime_params=regime_params,
            process_grid_resolver=process_grid_resolver,
        )

    with monkeypatch.context() as patch:
        patch.setattr(SolutionPhase, "state_action_space", observe)
        result = model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "wealth": jnp.linspace(1, 3, _N_SUBJECTS),
                "age": jnp.full(_N_SUBJECTS, 18.0),
                "regime_id": jnp.full(
                    _N_SUBJECTS, RegimeId.working_life, dtype=jnp.int32
                ),
            },
            seed=17,
            log_level="off",
        )
    assert (result.n_subjects, counts) == (
        _N_SUBJECTS,
        {"working_life": 1, "dead": 1},
    )


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

    # keyword-only-exempt: library-callback=jax.numpy.asarray
    def guard(
        value: npt.ArrayLike,
        dtype: DTypeLike | None = None,
        order: str | None = None,
        *,
        copy: bool | None = None,
        device: jax.Device | jax.sharding.Sharding | None = None,
        out_sharding: jax.NamedSharding | jax.P | None = None,
    ) -> jax.Array:
        if isinstance(value, list | tuple) and any(
            isinstance(leaf, jax.Array) and not isinstance(leaf, jax.core.Tracer)
            for leaf in jax.tree.leaves(value)
        ):
            raise _UnadmittedAllocationError(
                "Regime ID vector staged outside its admitted body"
            )
        return original(
            value, dtype, order, copy=copy, device=device, out_sharding=out_sharding
        )

    with monkeypatch.context() as capture:
        capture.setattr(jnp, "asarray", guard)
        with pytest.raises(_UnadmittedAllocationError):
            jnp.asarray(list(ids.values()))
        actual = draw_key_from_dict(
            d=probabilities, regime_names_to_ids=ids, keys=keys, memory=memory
        )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == jnp.int32
