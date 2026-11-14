"""Outer callable interfaces own DAG inputs; captured helpers remain evidence."""

import pytest
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
import tempfile
from unittest.mock import patch

from _lcm_regime_finder import find_regimes, verify_production_result
from _lcm_regime_finder.callable_identity import capture_session
from _lcm_regime_finder.cli import load_problem
from _lcm_regime_finder.declarations import Normalizer, _unresolved_input_signature
from _lcm_regime_finder.model_snapshot import normalize
from _lcm_regime_finder.pytask_api import export_model
from _lcm_regime_finder.semantic_check import (
    _CallableRecords,
    _original_inputs_unresolved,
)
from _lcm_regime_finder.structural_store import final_input_check


def variadic_implementation(**actions):
    raise AssertionError("Metadata inspection must not execute implementation helpers")


@dataclass
class FiniteOuter:
    implementation: Callable[..., int]

    def __call__(self, *, hours, labor):
        raise AssertionError("Metadata inspection must not execute outer callbacks")


@dataclass
class VariadicOuter:
    implementation: Callable[..., int]

    def __call__(self, **actions):
        raise AssertionError("Metadata inspection must not execute outer callbacks")


@dataclass
class Composite:
    first: Callable[..., int]
    second: Callable[..., int]


def income(earnings):
    return earnings


def health(health_status):
    return health_status


def utility(*, hours, labor):
    raise AssertionError("Metadata analysis must not execute economic utility")


def terminal():
    return 0.0


def test_finite_outer_signature_owns_inputs_despite_variadic_helper() -> None:
    captured = normalize(FiniteOuter(variadic_implementation))
    assert not (_unresolved_input_signature(captured))
    assert not (_original_inputs_unresolved(captured, _CallableRecords()))
    normalizer = Normalizer({"callback": captured}, {})
    key = normalizer.callback(captured, "/callback")
    assert (normalizer.callbacks[key]["arguments"]) == (["hours", "labor"])
    assert (set(normalizer.callbacks[key]["reads"])) == ({"hours", "labor"})
    assert ("implementation") in (captured["fields"])


def test_genuine_variadic_outer_stays_unresolved() -> None:
    captured = normalize(VariadicOuter(income))
    assert _unresolved_input_signature(captured)
    assert _original_inputs_unresolved(captured, _CallableRecords())


def test_unavailable_or_malformed_outer_metadata_stays_unresolved(
    *, subtests: pytest.Subtests
) -> None:
    original = normalize(VariadicOuter(income))
    for mutation in ("missing_arguments", "malformed_arguments", "missing_callable"):
        with subtests.test(mutation=mutation):
            captured = deepcopy(original)
            if mutation == "missing_arguments":
                captured["callable"].pop("arguments")
            elif mutation == "malformed_arguments":
                captured["callable"]["arguments"] = [None]
            else:
                captured["callable"] = None
            assert _unresolved_input_signature(captured)
            assert _original_inputs_unresolved(captured, _CallableRecords())


def test_raw_bound_implementation_mutation_still_invalidates_capture() -> None:
    outer = FiniteOuter(variadic_implementation)
    with capture_session() as session:
        captured = normalize(outer)
        outer.implementation = income
    assert (session.report()["status"]) == ("INCOMPLETE")
    assert ("implementation") in (captured["fields"])
    assert ("bound_self") in (captured["callable"])


def test_noncallable_composite_keeps_all_economic_children() -> None:
    captured = normalize(Composite(income, health))
    normalizer = Normalizer({"callback": captured}, {})
    key = normalizer.callback(captured, "/callback")
    assert (set(normalizer.callbacks[key]["reads"])) == ({"earnings", "health_status"})


def test_native_conditional_action_exports_and_independently_checks() -> None:
    from lcm import (
        ConditionalAction,
        DiscreteGrid,
        InitialNodes,
        LinSpacedGrid,
        Model,
        Periods,
        Regime,
        categorical,
    )
    from lcm.typing import ScalarInt

    @categorical(ordered=False)
    class Labor:
        no: ScalarInt
        yes: ScalarInt

    @categorical(ordered=False)
    class Regimes:
        alive: ScalarInt
        final: ScalarInt

    model = Model(
        n_periods=2,
        regime_id_class=Regimes,
        regimes={
            "alive": Regime(
                actions={
                    "labor": DiscreteGrid(Labor),
                    "hours": ConditionalAction(
                        grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
                        active_when={"labor": Labor.yes},
                        inactive_value=0.0,
                    ),
                },
                functions={"utility": utility},
            ),
            "final": Regime(functions={"utility": terminal}),
        },
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"final": Periods(values=(0,))}},
        enable_jit=False,
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "model.json"
        with patch.object(Model, "_acquire_generation", side_effect=AssertionError):
            receipt = export_model(
                model,
                path,
                production_metadata={"fixture": True},
                seconds=None,
                host_gib=None,
                source_transaction={
                    "kind": "exclusive-owner",
                    "authority": "test owns Model",
                },
            )
        assert (receipt["status"]) == ("COMPLETE"), receipt
        source = load_problem(path)
        assert not (
            _unresolved_input_signature(
                source["user_regimes"], source["callable_records"]
            )
        )
        assert not (
            _original_inputs_unresolved(
                source["user_regimes"], _CallableRecords(source["callable_records"])
            )
        )
        final_input_check(source)
        result = find_regimes(source, budget={"seconds": 30}, search_mode="exact")
        assert (result["status"]) == ("COMPLETE"), result["analysis"]["obligations"]
        assert (verify_production_result(source, result)["status"]) == ("PASS")

        altered = deepcopy(result)
        callbacks = altered["analysis"]["declarations"]["callbacks"]
        candidate = next(
            value
            for value in callbacks.values()
            if {"hours", "labor"} <= set(value.get("arguments", []))
        )
        candidate["arguments"] = ["invented_input"]
        candidate["reads"] = ["invented_input"]
        assert (verify_production_result(source, altered)["status"]) != ("PASS")
