"""Finite scalar oracle for a synthetic unfolded two-coordinate sharded model.

The oracle enumerates actions, quadrature nodes and bilinear interpolation in
Python, without using engine interpolation or reduction functions. Its value
bound is 32 working-format epsilons relative to the scalar reference, separately
in each precision. Discrete policies, feasibility and seeded streams are exact.
This is structural acceptance of a tiny fixture, not production ACA equality.

Run alone in a fresh eight-CPU-device process; this module never sets device
topology, so its eight-device witnesses skip in an ordinary battery.
"""

from bisect import bisect_right
from collections.abc import Mapping
from itertools import product
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution import output_layout, value_transfer
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from lcm.exceptions import ExecutionPlanningError
from tests.test_continuous_assets_aca_vocabulary import _model


def _reference(coordinates: Mapping[str, Any]) -> dict:
    """Enumerate the finite Bellman equations using explicit scalar interpolation."""
    assets, aime, shocks, persistent = (
        tuple(float(point) for point in np.asarray(coordinates[name]))
        for name in ("assets", "aime", "shock", "persistent")
    )
    weights = (1 / 6, 2 / 3, 1 / 6)
    transition = (
        (9 / 16, 6 / 16, 1 / 16),
        (3 / 16, 10 / 16, 3 / 16),
        (1 / 16, 6 / 16, 9 / 16),
    )
    tables: dict = {
        (2, "dead"): {
            (pref, asset): -40 - (asset - 5) ** 2 / 8 + pref
            for pref, asset in product(range(2), assets)
        }
    }
    for period in (1, 0):
        table = {}
        for pref, shock_index, persistent_index, asset, income in product(
            range(2), range(3), range(3), assets, aime
        ):
            candidates = []
            for decision in range(3):
                next_asset = 15 - asset + (decision - 1) / 4
                lower = min(
                    max(bisect_right(assets, next_asset) - 1, 0), len(assets) - 2
                )
                asset_weight = (next_asset - assets[lower]) / (
                    assets[lower + 1] - assets[lower]
                )
                next_income = 3 - income / 2
                income_lower = min(max(bisect_right(aime, next_income) - 1, 0), 2)
                income_weight = (next_income - aime[income_lower]) / (
                    aime[income_lower + 1] - aime[income_lower]
                )
                if period == 1:
                    target = tables[2, "dead"]
                    continuation = (1 - asset_weight) * target[
                        pref, assets[lower]
                    ] + asset_weight * target[pref, assets[lower + 1]]
                else:
                    target = tables[1, "working"]
                    continuation = 0.0
                    for next_shock, next_persistent in product(range(3), range(3)):
                        interpolated = 0.0
                        for asset_offset, income_offset in product(range(2), range(2)):
                            weight = (
                                asset_weight if asset_offset else 1 - asset_weight
                            ) * (income_weight if income_offset else 1 - income_weight)
                            interpolated += (
                                weight
                                * target[
                                    pref,
                                    next_shock,
                                    next_persistent,
                                    assets[lower + asset_offset],
                                    aime[income_lower + income_offset],
                                ]
                            )
                        continuation += (
                            weights[next_shock]
                            * transition[persistent_index][next_persistent]
                            * interpolated
                        )
                utility = (
                    -100
                    - (asset - 5) ** 2 / 32
                    - income**2 / 8
                    - 20 * (decision - pref) ** 2
                    + shocks[shock_index] / 8
                    + persistent[persistent_index] / 16
                    + income / 8
                )
                candidates.append(utility + continuation / 2)
            table[pref, shock_index, persistent_index, asset, income] = max(candidates)
        tables[period, "working"] = table
    return tables


def _assert_shards(value: Any) -> None:
    """Assert disjoint middle-axis assets shards, including the terminal rank."""
    expected_shape = (2, 3, 3, 24, 4) if value.ndim == 5 else (2, 24)
    assert value.shape == expected_shape
    axis = 3 if value.ndim == 5 else 1
    expected_spec = (
        jax.P(None, None, None, "assets", None)
        if value.ndim == 5
        else jax.P(None, "assets")
    )
    assert value.sharding.spec == expected_spec
    assert len(value.addressable_shards) == 8
    intervals = []
    for shard in value.addressable_shards:
        index = shard.index[axis]
        intervals.append((index.start, index.stop))
        assert shard.data.shape[axis] == 3
    assert sorted(intervals) == [(start, start + 3) for start in range(0, 24, 3)]


def _require_eight() -> None:
    if jax.default_backend() != "cpu" or jax.device_count() != 8:
        pytest.skip("Requires a fresh eight-device CPU process")


@pytest.mark.parametrize("widths", [(1, 1), (3, 24)])
def test_two_coordinate_values_and_carried_simulation_match_reference(  # noqa: C901
    *,
    widths: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
) -> None:
    """Eight assets shards preserve two-coordinate values and carried subject paths."""
    _require_eight()

    model = _model(widths=widths)
    control_model = _model(sharded=False, widths=widths)
    params = {"discount_factor": 0.5}
    control = control_model.solve(
        params=params, log_level="off", max_compilation_workers=1
    )
    birth_shapes = []
    original = output_layout.execute_with_pending_work

    def observe_birth(**kwargs: Any) -> Any:
        result = original(**kwargs)
        for value in jax.tree.leaves(result):
            if isinstance(value, jax.Array) and value.shape in (
                (2, 3, 3, 24, 4),
                (2, 24),
            ):
                _assert_shards(value)
                birth_shapes.append(value.shape)
        return result

    with monkeypatch.context() as probe:
        probe.setattr(output_layout, "execute_with_pending_work", observe_birth)
        solution = model.solve(
            params=params, log_level="off", max_compilation_workers=1
        )
    assert len(birth_shapes) == 3
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    reference = _reference(
        model._regimes["working"]
        .solution.state_action_space(regime_params=MappingProxyType({}))
        .states
    )
    snapshots = []
    for period, values in solution.values.items():
        for name, value in values.items():
            _assert_shards(value)
            observed = np.asarray(value)
            phase = model._regimes[name].solution
            coordinates = phase.state_action_space(
                regime_params=MappingProxyType({})
            ).states
            labelled = _labelled_values(
                value=observed, state_names=phase.state_names, coordinates=coordinates
            )
            expected = _labelled_reference(
                table=reference[period, name], regime_name=name, coordinates=coordinates
            )
            control_phase = control_model._regimes[name].solution
            control_labelled = _labelled_values(
                value=np.asarray(control.values[period][name]),
                state_names=control_phase.state_names,
                coordinates=control_phase.state_action_space(
                    regime_params=MappingProxyType({})
                ).states,
            )
            assert labelled.keys() == expected.keys() == control_labelled.keys()
            keys = sorted(expected)
            tolerance = float(32 * np.finfo(dtype).eps)
            np.testing.assert_allclose(
                [labelled[key] for key in keys],
                [expected[key] for key in keys],
                rtol=tolerance,
                atol=0,
            )
            np.testing.assert_allclose(
                [labelled[key] for key in keys],
                [control_labelled[key] for key in keys],
                rtol=tolerance,
                atol=0,
            )
            snapshots.append((value, observed.copy()))
    record_property(
        "oracle_value_count", sum(len(values) for values in reference.values())
    )
    initial = {
        "assets": jnp.full(17, 5.0),
        "aime": jnp.full(17, 0.5),
        "pref_type": jnp.arange(17, dtype=jnp.int32) % 2,
        "shock": jnp.zeros(17),
        "persistent": jnp.zeros(17),
        "pension": jnp.arange(17, dtype=dtype) + 4,
        "age": jnp.zeros(17),
        "regime_id": jnp.zeros(17, dtype=jnp.int32),
    }
    frames = []
    for active_model, active_solution in (
        (control_model, control),
        (model, solution),
        (model, solution),
    ):
        result = active_model.simulate(
            params=params,
            solution=active_solution,
            initial_conditions=initial,
            seed=42,
            log_level="off",
            max_compilation_workers=1,
        )
        frames.append(
            result.to_dataframe(use_labels=False)
            .sort_values(["subject_id", "period"])
            .reset_index(drop=True)
        )
    for frame in frames:
        working = frame.query('regime_name == "working"')
        assert len(working) == 34
        np.testing.assert_array_equal(
            working["pension"], 4 + working["subject_id"] + 2 * working["period"]
        )
        np.testing.assert_array_equal(
            working["decision"].astype(int), working["subject_id"] % 2
        )
    for frame in frames[1:]:
        for name in (
            "subject_id",
            "period",
            "shock",
            "persistent",
            "pension",
            "decision",
        ):
            np.testing.assert_array_equal(frame[name], frames[0][name])
    for value, snapshot in snapshots:
        assert not value.is_deleted()
        np.testing.assert_array_equal(value, snapshot)


def test_multidimensional_replica_budget_refuses_before_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full continuation replica remains a lower bound at small action widths."""
    _require_eight()

    model = _model(budget=1)

    def forbidden(**kwargs: Any) -> Any:
        del kwargs
        pytest.fail("Budget refusal must precede materializing a continuation copy")

    monkeypatch.setattr(value_transfer, "apply_value_transfer", forbidden)
    with pytest.raises(ExecutionPlanningError):
        model.solve(
            params={"discount_factor": 0.5}, log_level="off", max_compilation_workers=1
        )


def _labelled_values(
    *, value: np.ndarray, state_names: tuple[str, ...], coordinates: Mapping[str, Any]
) -> dict[tuple[tuple[str, float], ...], float]:
    """Join values to named physical coordinates independently of array axis order."""
    coordinates = {name: np.asarray(points) for name, points in coordinates.items()}
    assert value.shape == tuple(len(coordinates[name]) for name in state_names)
    return {
        tuple(
            sorted(
                (name, float(coordinates[name][index]))
                for name, index in zip(state_names, indices, strict=True)
            )
        ): float(value[indices])
        for indices in np.ndindex(value.shape)
    }


def _labelled_reference(
    *, table: dict, regime_name: str, coordinates: Mapping[str, Any]
) -> dict[tuple[tuple[str, float], ...], float]:
    """Label the independent oracle, translating its process-node indices only."""
    coordinates = {name: np.asarray(points) for name, points in coordinates.items()}
    names = (
        ("pref_type", "assets")
        if regime_name == "dead"
        else ("pref_type", "shock", "persistent", "assets", "aime")
    )
    return {
        tuple(
            sorted(
                (
                    name,
                    float(coordinates[name][int(coordinate)])
                    if name in {"shock", "persistent"}
                    else float(coordinate),
                )
                for name, coordinate in zip(names, key, strict=True)
            )
        ): value
        for key, value in table.items()
    }


def test_coordinate_join_survives_preference_axis_relocation() -> None:
    """Relocating preference preserves every named coordinate and value."""
    coordinates = {"pref_type": np.array([0, 1]), "assets": np.array([-4, 5, 19])}
    values = np.array([[-40.0, -50.0, -60.0], [-140.0, -150.0, -160.0]])
    original = _labelled_values(
        value=values, state_names=("pref_type", "assets"), coordinates=coordinates
    )
    relocated = _labelled_values(
        value=values.T, state_names=("assets", "pref_type"), coordinates=coordinates
    )
    assert original == relocated
    assert original[(("assets", 5.0), ("pref_type", 1.0))] == -150.0


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_negative_ties_and_infeasible_padding_preserve_named_max(width: int) -> None:
    """Two-coordinate cells retain negative maxima, lowest-ID ties and empty masks."""
    values = jnp.array(
        [
            [[-9.0, -3.0, -3.0, 0.0], [-5.0, -4.0, -2.0, 0.0]],
            [[-7.0, -1.0, -6.0, 0.0], [-jnp.inf, -2.0, -3.0, 0.0]],
        ]
    )
    feasible = jnp.array(
        [
            [[True, True, True, False], [False, False, False, False]],
            [[True, False, True, False], [True, False, False, False]],
        ]
    )
    action_ids = jnp.array([5, 8, 2, 0], dtype=jnp.int32)
    accumulator = HARD_MAX_REDUCTION.initialize(value_template=jnp.zeros((2, 2)))
    for start in range(0, 4, width):
        accumulator = HARD_MAX_REDUCTION.add(
            accumulator=accumulator,
            values=values[..., start : start + width],
            feasible=feasible[..., start : start + width],
            action_ids=action_ids[start : start + width],
        )
    result = HARD_MAX_REDUCTION.finalize(accumulator=accumulator)
    np.testing.assert_array_equal(result.best_value, [[-3.0, -np.inf], [-6.0, -np.inf]])
    np.testing.assert_array_equal(result.best_global_action_id, [[2, -1], [2, 5]])
    np.testing.assert_array_equal(result.any_feasible, [[True, False], [True, True]])
