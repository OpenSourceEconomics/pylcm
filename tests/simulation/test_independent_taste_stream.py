"""A taste seed couples counterfactual draws without fixing their state stream."""

import importlib
from collections.abc import Mapping
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.simulation.program_types import SimulationPrograms
from lcm import ExecutionConfig
from lcm.exceptions import InvalidSimulationInputError
from tests.test_models import taste_shocks_toy

simulation = importlib.import_module("_lcm.simulation.simulate")


def _initial_conditions(*, count: int = 24) -> dict[str, jax.Array]:
    return {
        "age": jnp.full(count, 40.0),
        "wealth": jnp.full(count, 4.6),
        "regime_id": jnp.zeros(count, dtype=jnp.int32),
    }


def _capture_decision_keys(*, monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    captured: list[np.ndarray] = []
    original = simulation.execute_simulation_program

    def record(
        *,
        programs: SimulationPrograms,
        family: str,
        period: int,
        arguments: Mapping[str, object],
        n_subjects: int,
    ) -> object:
        if family == "decision" and "taste_shock_key" in arguments:
            captured.append(
                np.array(
                    jax.random.key_data(
                        cast("jax.Array", arguments["taste_shock_key"])
                    ),
                    copy=True,
                )
            )
        return original(
            programs=programs,
            family=family,
            period=period,
            arguments=arguments,
            n_subjects=n_subjects,
        )

    monkeypatch.setattr(simulation, "execute_simulation_program", record)
    return captured


def test_policy_and_ordinary_seed_changes_preserve_independent_taste_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatched subject keys agree while policy changes actual choices."""
    captured = _capture_decision_keys(monkeypatch=monkeypatch)
    choices = []
    for ordinary_seed, kappa in ((11, 0.0), (92, 5.0)):
        model = taste_shocks_toy.get_model()
        params = taste_shocks_toy.get_params(scale=0.2)
        params["alive"]["utility"]["kappa"] = kappa
        result = model.simulate(
            params=params,
            initial_conditions=_initial_conditions(),
            seed=ordinary_seed,
            taste_shock_seed=721,
            log_level="debug",
        )
        choices.append(
            result.to_dataframe(use_labels=False)
            .query("period == 0")["work"]
            .to_numpy()
        )
    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0], captured[1])
    assert np.any(choices[0] != choices[1])


def test_independent_seed_is_effective_and_subject_window_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunking preserves real-row keys and changing the taste seed changes them."""
    captured = _capture_decision_keys(monkeypatch=monkeypatch)
    params = taste_shocks_toy.get_params(scale=0.2)
    for taste_seed, count, batch in ((12, 24, 0), (12, 27, 8), (13, 24, 0)):
        model = taste_shocks_toy.get_model(
            execution_config=ExecutionConfig(
                axis_widths={} if batch == 0 else {"subject": batch}
            )
        )
        solution = model.solve(params=params, log_level="debug")
        model.simulate(
            params=params,
            initial_conditions=_initial_conditions(count=count),
            solution=solution,
            seed=11,
            taste_shock_seed=taste_seed,
            log_level="debug",
        )
    assert len(captured) == 6
    np.testing.assert_array_equal(captured[0], np.concatenate(captured[1:5])[:24])
    assert np.any(captured[0] != captured[5])


def test_explicit_none_preserves_default_seeded_result() -> None:
    """Omitting the optional stream and supplying None produce identical arrays."""
    model = taste_shocks_toy.get_model()
    params = taste_shocks_toy.get_params(scale=0.2)
    solution = model.solve(params=params, log_level="debug")
    initial = _initial_conditions()
    ordinary = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=11,
        log_level="debug",
    ).to_dataframe(use_labels=False)
    explicit = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=11,
        log_level="debug",
        taste_shock_seed=None,
    ).to_dataframe(use_labels=False)
    pd.testing.assert_frame_equal(ordinary, explicit, check_exact=True)


def test_boolean_taste_seed_is_rejected_before_solving() -> None:
    """Boolean configuration is refused before incomplete parameters are processed."""
    model = taste_shocks_toy.get_model()
    with pytest.raises(InvalidSimulationInputError, match="taste_shock_seed"):
        model.simulate(
            params={},
            initial_conditions=_initial_conditions(),
            taste_shock_seed=True,
            log_level="debug",
        )


@pytest.mark.parametrize("impl", ["threefry2x32", "rbg"])
@pytest.mark.parametrize("partitionable", [False, True])
def test_budgeted_padded_chunks_preserve_independent_taste_keys_and_choices(
    *, monkeypatch: pytest.MonkeyPatch, impl: str, partitionable: bool
) -> None:
    """Admitted chunks keep each original row's addressed key and actual action."""
    captured = _capture_decision_keys(monkeypatch=monkeypatch)
    frames = []
    params = taste_shocks_toy.get_params(scale=0.2)
    with jax.default_prng_impl(impl), jax.threefry_partitionable(partitionable):
        for width in (0, 3):
            model = taste_shocks_toy.get_model(
                execution_config=ExecutionConfig(
                    axis_widths={} if width == 0 else {"subject": width},
                    device_memory_bytes=None if width == 0 else 2**30,
                )
            )
            solution = model.solve(params=params, log_level="off")
            result = model.simulate(
                params=params,
                initial_conditions=_initial_conditions(count=7),
                solution=solution,
                seed=11,
                taste_shock_seed=721,
                log_level="off",
            )
            frames.append(result.to_dataframe(use_labels=False))
    assert len(captured) == 4
    np.testing.assert_array_equal(captured[0], np.concatenate(captured[1:])[:7])
    # This control owns exact RNG, state and selected-action semantics. Value
    # arithmetic remains covered by the separate width-comparison contracts.
    for column in (
        "subject_id",
        "period",
        "regime_name",
        "wealth",
        "work",
        "consumption",
    ):
        pd.testing.assert_series_equal(
            frames[0][column], frames[1][column], check_exact=True
        )
