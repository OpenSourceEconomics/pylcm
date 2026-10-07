"""Forward simulation work covers exactly the pairs a subject can occupy.

Budgeted simulation profiles forward units before dispatching them. Every model
below solves some `(period, regime)` pairs only for their value. Those pairs must
never be profiled or dispatched as forward units, their solved values must stay
available to the decisions that read them, and budgeted results must equal the
unbudgeted ones.
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation import chunk_profiles, simulate
from lcm import (
    AgeGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.typing import ContinuousState, FloatND, ScalarInt
from tests.test_demand_worklists import GatedId, _gated_model

_DISCOUNT = 0.5


@categorical(ordered=False)
class ControlId:
    source: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt
    end: ScalarInt


@categorical(ordered=False)
class TerminalId:
    source: ScalarInt
    perceived: ScalarInt
    end: ScalarInt


def _utility(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _perceived_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth + 1.0


def _regime(*, terminal: bool = False, perceived: bool = False) -> Regime:
    return Regime(
        states={"wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={} if terminal else {"wealth": fixed_transition("wealth")},
        functions={"utility": _perceived_utility if perceived else _utility},
    )


def _config(*, budgeted: bool) -> ExecutionConfig:
    if not budgeted:
        return ExecutionConfig()
    return ExecutionConfig(
        devices=(0,), axis_widths={"subject": 2}, device_memory_bytes=2**30
    )


def _mixed_model(*, budgeted: bool) -> Model:
    """`perceived` is value-only at period 1 and physically visited at period 2."""
    return Model(
        regimes={
            "source": _regime(),
            "perceived": _regime(perceived=True),
            "realized": _regime(),
            "end": _regime(terminal=True),
        },
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=ControlId,
        initial_nodes={0: "source"},
        execution_config=_config(budgeted=budgeted),
        edges=Phased(
            solve={
                "source": {"perceived": 0},
                "perceived": {"perceived": 1, "end": 2},
                "realized": {"perceived": 1},
            },
            simulate={
                "source": {"realized": 0},
                "perceived": {"perceived": 1, "end": 2},
                "realized": {"perceived": 1},
            },
        ),
    )


def _value_only_terminal_model(*, budgeted: bool) -> Model:
    """The value-only pair is a terminal regime."""
    return Model(
        regimes={
            "source": _regime(),
            "perceived": _regime(terminal=True, perceived=True),
            "end": _regime(terminal=True),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=TerminalId,
        initial_nodes={0: "source"},
        execution_config=_config(budgeted=budgeted),
        edges=Phased(
            solve={"source": {"perceived": 0}}, simulate={"source": {"end": 0}}
        ),
    )


def _multi_root_model(*, budgeted: bool) -> Model:
    """Two cohorts start at different ages; `perceived` stays value-only."""
    return Model(
        regimes={
            "source": _regime(),
            "perceived": _regime(perceived=True),
            "realized": _regime(),
            "end": _regime(terminal=True),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=ControlId,
        initial_nodes={0: "source", 1: "realized"},
        execution_config=_config(budgeted=budgeted),
        edges=Phased(
            solve={
                "source": {"perceived": 0},
                "perceived": {"end": 1},
                "realized": {"end": 1},
            },
            simulate={
                "source": {"realized": 0},
                "perceived": {"end": 1},
                "realized": {"end": 1},
            },
        ),
    )


def _gated_value_only_model(*, budgeted: bool) -> Model:
    """A gate reference and a phased fallback's solve leg are read by value only."""
    model = _gated_model(phased_fallback=True)
    if not budgeted:
        return model
    return Model(
        edges=model.edges,
        regimes=model.user_regimes,
        ages=model.ages,
        regime_id_class=GatedId,
        initial_nodes={40: "source"},
        execution_config=_config(budgeted=True),
    )


def _initial(*, model: Model, starts: tuple[tuple[float, str], ...]) -> dict[str, Any]:
    wealth = jnp.linspace(0.0, 1.0, 3)
    return {
        "wealth": jnp.concatenate([wealth] * len(starts)),
        "age": jnp.concatenate([jnp.full(3, age) for age, _ in starts]),
        "regime_id": jnp.concatenate(
            [jnp.full(3, model.regime_names_to_ids[name]) for _, name in starts]
        ),
    }


_CASES: Mapping[str, dict[str, Any]] = {
    "mixed_value_only_and_visited_periods": {
        "build": _mixed_model,
        "starts": ((0.0, "source"),),
        "params": {"discount_factor": _DISCOUNT},
        "forward": {(0, "source"), (1, "realized"), (2, "perceived"), (3, "end")},
        "value_only": {(1, "perceived")},
        # V_perceived(1, w) = 1 + w + (1 + 1.5 w) / 2 at w = 0 and w = 1.
        "value_only_values": {(1, "perceived"): (1.5, 3.25)},
        # V_source(w) = w + (1.5 + 1.75 w) / 2 at w = 0 and w = 1.
        "source_values": (0.75, 2.625),
        "paths": {"source": 0, "realized": 1, "perceived": 2, "end": 3},
    },
    "wholly_value_only_terminal": {
        "build": _value_only_terminal_model,
        "starts": ((0.0, "source"),),
        "params": {"discount_factor": _DISCOUNT},
        "forward": {(0, "source"), (1, "end")},
        "value_only": {(1, "perceived")},
        "value_only_values": {(1, "perceived"): (1.0, 2.0)},
        # V_source(w) = w + (w + 1) / 2 at w = 0 and w = 1.
        "source_values": (0.5, 2.0),
        "paths": {"source": 0, "end": 1},
    },
    "multi_root_cohorts": {
        "build": _multi_root_model,
        "starts": ((0.0, "source"), (1.0, "realized")),
        "params": {"discount_factor": _DISCOUNT},
        "forward": {(0, "source"), (1, "realized"), (2, "end")},
        "value_only": {(1, "perceived")},
        # V_perceived(1, w) = 1 + w + w / 2 at w = 0 and w = 1.
        "value_only_values": {(1, "perceived"): (1.0, 2.5)},
        # V_source(w) = w + (1 + 1.5 w) / 2 at w = 0 and w = 1.
        "source_values": (0.5, 2.25),
        "paths": {"realized": 1, "end": 2},
    },
    "gated_value_only_references": {
        "build": _gated_value_only_model,
        "starts": ((40.0, "source"),),
        "params": {"discount_factor": 0.9},
        "forward": {(0, "source"), (1, "target"), (1, "fallback")},
        "value_only": {(1, "reference"), (1, "priced")},
        # Both are terminal and read their utility `w`.
        "value_only_values": {(1, "reference"): (0.0, 1.0), (1, "priced"): (0.0, 1.0)},
        "source_values": None,
        # The gate compares equal values, so every subject takes the fallback.
        "paths": {"source": 0, "fallback": 1},
    },
}


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("case", list(_CASES))
def test_budgeted_forward_units_are_exactly_the_visited_pairs(
    *, case: str, workers: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Profiled and dispatched units equal the visited pairs; values stay solved."""
    spec = _CASES[case]
    build: Callable[..., Model] = spec["build"]
    baseline = build(budgeted=False)
    budgeted = build(budgeted=True)
    solution = budgeted.solve(
        params=spec["params"], log_level="off", max_compilation_workers=1
    )
    base_solution = baseline.solve(
        params=spec["params"], log_level="off", max_compilation_workers=1
    )
    assert set(spec["value_only_values"]) == spec["value_only"]
    for (period, name), expected_values in spec["value_only_values"].items():
        # The value-only pair's solved value stays published.
        np.testing.assert_allclose(
            np.asarray(solution.values[period][name]), expected_values, rtol=1e-6
        )
    if spec["source_values"] is not None:
        np.testing.assert_allclose(
            np.asarray(solution.values[0]["source"]),
            spec["source_values"],
            rtol=1e-6,
        )
    initial = _initial(model=budgeted, starts=spec["starts"])
    reference = baseline.simulate(
        params=spec["params"],
        solution=base_solution,
        initial_conditions=initial,
        seed=7,
        log_level="off",
    )

    profiled: list[tuple[int, str]] = []
    dispatched: list[tuple[int, str]] = []
    profile_unit = chunk_profiles.profile_forward_unit
    simulate_unit = simulate._simulate_regime_in_period

    def _record_profile(**kwargs: Any) -> Any:
        profiled.append((kwargs["period"], kwargs["name"]))
        return profile_unit(**kwargs)

    def _record_dispatch(**kwargs: Any) -> Any:
        dispatched.append((kwargs["period"], kwargs["regime_name"]))
        return simulate_unit(**kwargs)

    monkeypatch.setattr(chunk_profiles, "profile_forward_unit", _record_profile)
    monkeypatch.setattr(simulate, "_simulate_regime_in_period", _record_dispatch)
    result = budgeted.simulate(
        params=spec["params"],
        solution=solution,
        initial_conditions=initial,
        seed=7,
        log_level="off",
        max_compilation_workers=workers,
    )

    assert (set(profiled), set(dispatched)) == (spec["forward"], spec["forward"])
    assert not (set(profiled) | set(dispatched)) & spec["value_only"]
    for name, period in spec["paths"].items():
        record = result.raw_results[name][period]
        expected = reference.raw_results[name][period]
        np.testing.assert_array_equal(
            np.asarray(record.in_regime), np.asarray(expected.in_regime)
        )
        in_regime = np.asarray(record.in_regime)
        assert in_regime.any()
        # Wealth is carried unchanged along every realized path.
        np.testing.assert_allclose(
            np.asarray(record.states["wealth"])[in_regime],
            np.asarray(initial["wealth"])[in_regime],
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(record.states["wealth"]),
            np.asarray(expected.states["wealth"]),
        )
        np.testing.assert_array_equal(
            np.asarray(record.V_arr), np.asarray(expected.V_arr)
        )
