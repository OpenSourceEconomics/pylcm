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
from _lcm.simulation.random_address import RandomAddress
from _lcm.typing import PytreeValue, ShapeDtypePytree
from lcm import (
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    Model,
    NormalIIDProcess,
    Periods,
    Regime,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import ReferenceName, UserSeed
from tests.test_models import taste_shocks_toy

simulation = importlib.import_module("_lcm.simulation.simulate")


def test_taste_toggle_keeps_scheduled_named_shock_draws() -> None:
    """Enabling decision noise cannot advance the scheduled innovation's root."""
    process = NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)
    panels = []
    for enabled in (False, True):
        model = Model(
            regimes={
                "alive": Regime(
                    states={"shock": process},
                    actions={"work": DiscreteGrid(taste_shocks_toy.WorkChoice)},
                    functions={"utility": lambda shock, work: shock + work},
                    taste_shocks=ExtremeValueTasteShocks() if enabled else None,
                ),
                "done": Regime(
                    states={"shock": process},
                    functions={"utility": lambda shock: shock},
                ),
            },
            n_periods=2,
            regime_id_class=taste_shocks_toy.ToyRegimeId,
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={"alive": {"done": Periods(values=(0,))}},
            fixed_params={"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}},
        )
        panels.append(
            model.simulate(
                params={"alive": {"taste_shocks": {"scale": 1.0}}} if enabled else {},
                initial_conditions={
                    "shock": jnp.zeros(7),
                    "period": jnp.zeros(7, dtype=jnp.int32),
                    "regime_id": jnp.zeros(7, dtype=jnp.int32),
                },
                seed=31,
                log_level="off",
            ).to_dataframe()
        )
    pd.testing.assert_frame_equal(
        panels[0][["period", "shock"]], panels[1][["period", "shock"]], check_exact=True
    )


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
        arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
        n_subjects: int,
    ) -> PytreeValue:
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
        alive = params["alive"]
        assert isinstance(alive, dict)
        utility = alive["utility"]
        assert isinstance(utility, dict)
        utility["kappa"] = kappa
        result = model.simulate(
            params=params,
            initial_conditions=_initial_conditions(),
            seed=ordinary_seed,
            taste_shock_seed=721,
            log_level="debug",
        )
        choices.append(
            result.to_dataframe(use_categorical_codes=True)
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
        # The pinned chunks are the windows under test; a budget may merge them.
        model = taste_shocks_toy.get_model(
            execution_config=ExecutionConfig(
                axis_widths={} if batch == 0 else {"subject": batch},
                device_memory_bytes=None,
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
    ).to_dataframe(use_categorical_codes=True)
    explicit = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=11,
        log_level="debug",
        taste_shock_seed=None,
    ).to_dataframe(use_categorical_codes=True)
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
            frames.append(result.to_dataframe(use_categorical_codes=True))
    assert len(captured) == 2
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


@pytest.mark.parametrize("role", ["seed", "taste_shock_seed"])
@pytest.mark.parametrize("budgeted", [False, True])
def test_typed_seed_matches_integer_draws_and_choices(
    *, monkeypatch: pytest.MonkeyPatch, role: str, budgeted: bool
) -> None:
    """A typed root preserves integer draws, including admitted padded chunks."""
    captured = _capture_decision_keys(monkeypatch=monkeypatch)
    model = taste_shocks_toy.get_model(
        execution_config=ExecutionConfig(
            axis_widths={"subject": 3} if budgeted else {},
            device_memory_bytes=2**30 if budgeted else None,
        )
    )
    params = taste_shocks_toy.get_params(scale=0.2)
    solution = model.solve(params=params, log_level="off")
    frames = []
    for typed in (False, True):
        seeds: dict[str, UserSeed] = {"seed": 11, "taste_shock_seed": 721}
        if typed:
            seeds[role] = jax.random.key(seeds[role])
        frames.append(
            model.simulate(
                params=params,
                initial_conditions=_initial_conditions(count=7),
                solution=solution,
                log_level="off",
                seed=seeds["seed"],
                taste_shock_seed=seeds["taste_shock_seed"],
            ).to_dataframe(use_categorical_codes=True)
        )
    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0], captured[1])
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


@pytest.mark.parametrize("role", ["seed", "taste_shock_seed"])
def test_folded_seed_children_change_dispatched_draws(
    *, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    """Independent children change draws while a repeated child repeats them."""
    captured = _capture_decision_keys(monkeypatch=monkeypatch)
    model = taste_shocks_toy.get_model()
    params = taste_shocks_toy.get_params(scale=0.2)
    solution = model.solve(params=params, log_level="off")
    root = jax.random.key(17)
    for child in (0, 1, 0):
        seeds: dict[str, UserSeed | None] = {"seed": 11, "taste_shock_seed": None}
        seeds[role] = jax.random.fold_in(root, child)
        model.simulate(
            params=params,
            initial_conditions=_initial_conditions(),
            solution=solution,
            log_level="off",
            seed=seeds["seed"],
            taste_shock_seed=seeds["taste_shock_seed"],
        )
    assert len(captured) == 3
    assert np.any(captured[0] != captured[1])
    np.testing.assert_array_equal(captured[0], captured[2])


@pytest.mark.parametrize("role", ["seed", "taste_shock_seed"])
def test_legacy_seed_names_typed_key_constructor_before_solving(role: str) -> None:
    """Raw uint32 keys are refused with the typed constructor in the message."""
    model = taste_shocks_toy.get_model()
    with pytest.raises(InvalidSimulationInputError, match=r"jax.random.key"):
        model.simulate(
            params={},
            initial_conditions=_initial_conditions(),
            log_level="off",
            seed=jax.random.PRNGKey(17) if role == "seed" else None,
            taste_shock_seed=jax.random.PRNGKey(17)
            if role == "taste_shock_seed"
            else None,
        )


@pytest.mark.parametrize("impl", ["rbg", "unsafe_rbg"])
def test_simulate_refuses_a_non_threefry_ordinary_seed(impl: str) -> None:
    """Only Threefry keeps ordinary draws independent of the subject chunk width."""
    model = taste_shocks_toy.get_model()
    with pytest.raises(InvalidSimulationInputError, match="threefry2x32"):
        model.simulate(
            params=taste_shocks_toy.get_params(scale=0.2),
            initial_conditions=_initial_conditions(count=7),
            seed=jax.random.key(11, impl=impl),
            log_level="off",
        )


@pytest.mark.parametrize("budgeted", [False, True])
def test_integer_seed_draws_threefry_under_an_rbg_default(*, budgeted: bool) -> None:
    """An integer `seed` gives the Threefry population whatever the default PRNG."""
    model = taste_shocks_toy.get_model(
        execution_config=ExecutionConfig(
            axis_widths={"subject": 3} if budgeted else {},
            device_memory_bytes=2**30 if budgeted else None,
        )
    )
    params = taste_shocks_toy.get_params(scale=0.2)
    solution = model.solve(params=params, log_level="off")
    with jax.default_prng_impl("threefry2x32"):
        expected = model.simulate(
            params=params,
            initial_conditions=_initial_conditions(count=7),
            solution=solution,
            seed=11,
            log_level="off",
        ).to_dataframe(use_categorical_codes=True)
    with jax.default_prng_impl("rbg"):
        actual = model.simulate(
            params=params,
            initial_conditions=_initial_conditions(count=7),
            solution=solution,
            seed=11,
            log_level="off",
        ).to_dataframe(use_categorical_codes=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("width", [1, 4])
def test_integer_seed_draws_ignore_subject_width_under_an_rbg_default(
    width: int,
) -> None:
    """Ordinary-stream choices do not move with the subject chunk width."""
    params = taste_shocks_toy.get_params(scale=0.2)
    frames = []
    for subject_width in (width, 32):
        model = taste_shocks_toy.get_model(
            execution_config=ExecutionConfig(
                axis_widths={"subject": subject_width},
                device_memory_bytes=2**30,
            )
        )
        with jax.default_prng_impl("rbg"):
            frames.append(
                model.simulate(
                    params=params,
                    initial_conditions=_initial_conditions(count=32),
                    seed=11,
                    log_level="off",
                ).to_dataframe(use_categorical_codes=True)
            )
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


@pytest.mark.parametrize("impl", ["rbg", "unsafe_rbg"])
@pytest.mark.parametrize("budgeted", [False, True])
def test_independent_taste_key_keeps_its_implementation(
    *, budgeted: bool, impl: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Addressed taste draws preserve the supplied root's implementation and words."""

    captured = _capture_decision_keys(monkeypatch=monkeypatch)
    model = taste_shocks_toy.get_model(
        execution_config=ExecutionConfig(
            axis_widths={"subject": 3} if budgeted else {},
            device_memory_bytes=2**30 if budgeted else None,
        )
    )
    params = taste_shocks_toy.get_params(scale=0.2)
    solution = model.solve(params=params, log_level="off")
    root = jax.random.key(17, impl=impl)
    words = RandomAddress(kind="taste", name="alive").words()
    addressed = root
    for word in words:
        addressed = jax.random.fold_in(addressed, np.uint32(word))
    expected = np.stack(
        [
            np.asarray(
                jax.random.key_data(
                    jax.random.fold_in(
                        jax.random.fold_in(
                            jax.random.fold_in(addressed, np.uint32(0)), np.uint32(row)
                        ),
                        np.uint32(0),
                    )
                )
            )
            for row in range(7)
        ]
    )
    with jax.default_prng_impl("threefry2x32"):
        model.simulate(
            params=params,
            initial_conditions=_initial_conditions(count=7),
            solution=solution,
            taste_shock_seed=root,
            log_level="off",
        )
    # A non-Threefry row key carries a trailing length-one axis for its scalar draw.
    actual = np.concatenate(captured)[:7, 0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("role", ["seed", "taste_shock_seed"])
def test_batched_typed_seed_is_refused_before_solving(role: str) -> None:
    """Only a scalar typed root crosses the public simulation boundary."""
    model = taste_shocks_toy.get_model()
    roots = jax.random.split(jax.random.key(17), 2)
    with pytest.raises(InvalidSimulationInputError, match=r"scalar jax.random.key"):
        model.simulate(
            params={},
            initial_conditions=_initial_conditions(),
            log_level="off",
            seed=roots if role == "seed" else None,
            taste_shock_seed=roots if role == "taste_shock_seed" else None,
        )
