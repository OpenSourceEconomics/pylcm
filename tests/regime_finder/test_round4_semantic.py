"""Independent semantic regressions without native Model evaluation."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json

from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.records import JSONValue, SemanticCheck
from tests.regime_finder.test_declaration_ingestion import (
    json_list,
    json_member,
    json_rows,
)
from _lcm_regime_finder.semantic_check import (
    check_semantics,
    _closed_native_constant,
)


def callback(
    name: str, arguments: list[str], expression: str, *, status: str = "proved"
) -> dict[str, JSONValue]:
    source = f"def {name}({', '.join(arguments)}):\n    return {expression}\n"
    original = {
        "kind": "callable",
        "arguments": arguments,
        "raw_arguments": arguments,
        "source": source,
        "bindings": {},
        "call_targets": {},
        "variadic": False,
    }
    return {
        "arguments": arguments,
        "reads": arguments,
        "source": source,
        "bindings": {},
        "expression": expression if status == "proved" else None,
        "status": status,
        "original": original,
        "identity": hashlib.sha256(source.encode()).hexdigest(),
    }


def one_cell(
    callbacks: Mapping[str, JSONValue],
    functions: Mapping[str, JSONValue],
    consumers: Sequence[JSONValue],
    *,
    states: Mapping[str, JSONValue] | None = None,
    actions: Mapping[str, JSONValue] | None = None,
    domains: Mapping[str, JSONValue] | None = None,
    period: int = 0,
) -> dict[str, JSONValue]:
    return {
        "schema": "regime-declarations/v1",
        "source_identity": "explicit round4 fixture",
        "domains": domains or {},
        "callbacks": callbacks,
        "cells": [
            {
                "id": "c",
                "regime": "life",
                "age": 20 + period,
                "period": period,
                "states": states or {},
                "actions": actions or {},
                "functions": functions,
                "consumers": consumers,
                "stage": "decision",
                "role": "one-person",
                "terminal": True,
                "admissible_start": True,
            }
        ],
        "relations": [],
        "characteristics": {},
        "obligations": [],
        "allocation": {"status": "known", "kind": "per_block"},
    }


def produce(
    source: dict[str, JSONValue],
) -> tuple[dict[str, JSONValue], dict[str, JSONValue], SemanticCheck]:
    declarations = normalize_declarations(source)
    analysis = MetadataAnalysis(declarations, facts_backend="metadata").run()
    return declarations, analysis, check_semantics(source, declarations, analysis)


def first_option(analysis: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """The first interface option of the single cell `c`."""
    return json_rows(analysis, "interfaces", "c", "options")[0]


def grid(values: list[int]) -> dict[str, JSONValue]:
    return {
        "kind": "object",
        "type": "lcm.DiscreteGrid",
        "fields": {"values": values, "codes": list(range(len(values)))},
    }


def captured_gate_graph(
    *, dormant_admitted: bool = False, simulation_edge: bool = True
) -> dict[str, JSONValue]:
    def raw(name: str, arguments: list[str], body: str) -> JSONValue:
        return callback(name, arguments, body)["original"]

    def regime() -> dict[str, JSONValue]:
        return {
            "kind": "dataclass",
            "type": "lcm.Regime",
            "fields": {
                "states": {"history": grid([0, 1])},
                "actions": {},
                "functions": {"utility": raw("utility", [], "0")},
                "constraints": {},
                "state_transitions": {
                    "history": raw("history", ["history"], "history")
                },
            },
        }

    fallback = {
        "kind": "dataclass",
        "type": "lcm.ProjectedRegimeValue",
        "fields": {
            "regime": "outside",
            "projection": {"history": raw("project", ["history"], "history")},
        },
    }
    gate = {
        "kind": "dataclass",
        "type": "_lcm.gated_edge.GatedEdge",
        "fields": {
            "gate": raw("accept", ["history"], "history > 0"),
            "gate_refs": {},
            "legs": {
                "person": {
                    "kind": "dataclass",
                    "type": "lcm.MemberRoute",
                    "fields": {"fallback": fallback},
                }
            },
            "off_grid": "extrapolate",
        },
    }
    nodes = [[0, "source"], [1, "landing"], [1, "outside"]]
    starts = [[0, "source"]]
    if dormant_admitted:
        nodes.append([1, "dormant"])
        starts.append([1, "dormant"])
    reach = {
        "kind": "dataclass",
        "fields": {
            "nodes": nodes,
            "solution": {
                "kind": "dataclass",
                "fields": {"targets_by_period": [{"source": ["landing"]}]},
            },
            "simulation": {
                "kind": "dataclass",
                "fields": {
                    "targets_by_period": [
                        {"source": ["landing"] if simulation_edge else []}
                    ]
                },
            },
        },
    }
    regimes = {name: regime() for name in ("source", "landing", "outside", "dormant")}
    return {
        "schema": "constructed-pylcm-model-graph-1",
        "n_periods": 2,
        "ages": {"kind": "object", "fields": {"exact_values": [0, 1]}},
        "user_regimes": regimes,
        "graph": {
            "kind": "dataclass",
            "fields": {
                "coordinate_kind": "age",
                "reachability": reach,
                "initial_nodes": starts,
                "laws": {
                    "source": {
                        "kind": "dataclass",
                        "fields": {
                            "transition": "landing",
                            "gated_edges": {"landing": gate, "dormant": deepcopy(gate)},
                        },
                    }
                },
            },
        },
        "semantic_contract": {
            "roles": dict.fromkeys(regimes, "one-person"),
            "allocation": {"status": "known", "kind": "per_block"},
            "production_metadata": {
                "status": "regression",
                "no_production_claim": True,
            },
        },
    }


def native_constant() -> dict[str, JSONValue]:
    return {
        "kind": "dataclass",
        "type": "_lcm.regime_building.schedules._RegimeCodeConstant",
        "fields": {"value": 1},
        "callable": {
            "kind": "callable",
            "arguments": [],
            "raw_arguments": ["self"],
            "variadic": False,
            "capture_contract": "inspection-only-frozen-bindings-1",
            "bindings": {"jnp": {"kind": "module", "name": "jax.numpy"}},
            "bound_self": {
                "type": "_lcm.regime_building.schedules._RegimeCodeConstant",
                "fields": {"value": 1},
            },
            "source": "def __call__(self):\n    return jnp.asarray(self.value)\n",
        },
    }


def test_integral_float_provider_binding_has_same_exact_numeric_image() -> None:
    for period in (0, 1, 20, 99):
        cbs = {
            "experience": callback("experience", ["period"], "period + 1.0"),
            "utility": callback("utility", ["experience"], "experience"),
        }
        source = one_cell(
            cbs,
            {"experience": "experience", "utility": "utility"},
            [{"kind": "reward", "callback": "utility", "phase": "simulation"}],
            period=period,
        )
        _, analysis, check = produce(source)
        assert (check["status"]) == ("PASS"), check
        record = json_member(
            first_option(analysis), "function_interfaces", "simulation", "utility"
        )
        assert (json_member(record, "bound_arguments")["experience"]) == (period + 1)


def test_wrong_numeric_or_boolean_binding_is_rejected_without_tolerance() -> None:
    cbs = {
        "experience": callback("experience", ["period"], "period + 1.0"),
        "utility": callback("utility", ["experience"], "experience"),
    }
    source = one_cell(
        cbs,
        {"experience": "experience", "utility": "utility"},
        [{"kind": "reward", "callback": "utility", "phase": "simulation"}],
    )
    declarations, analysis, _ = produce(source)
    for replacement in (True, "1", 1.0000000000000002):
        bad = deepcopy(analysis)
        json_member(
            first_option(bad),
            "function_interfaces",
            "simulation",
            "utility",
            "bound_arguments",
        )["experience"] = replacement
        check = check_semantics(source, declarations, bad)
        assert any(
            e["kind"] == "unproved bound original argument" for e in check["errors"]
        ), check


def test_consumed_opaque_provider_keeps_proved_singleton_binding() -> None:
    cbs = {
        "report": callback("report", ["alpha"], "opaque(alpha)", status="unresolved")
    }
    source = one_cell(
        cbs,
        {"report": "report"},
        [{"kind": "reporting", "callback": "report", "phase": "simulation"}],
        states={"alpha": "singleton"},
        domains={
            "singleton": {
                "kind": "categorical",
                "size": 1,
                "values": [0.0],
                "codes": [7],
            }
        },
    )
    _, analysis, check = produce(source)
    option = first_option(analysis)
    record = json_member(option, "function_interfaces", "simulation", "report")
    assert (json_member(option, "state_projection", "alpha")["kind"]) == ("drop")
    assert (json_member(record, "implementation")["kind"]) == (
        "source_bound_partial_application"
    )
    assert (record["bound_arguments"]) == ({"alpha": 0.0})
    assert not (check["errors"]), check
    assert (check["status"]) == ("UNRESOLVED")


def test_reporting_root_retains_history_and_rejects_provider_omission() -> None:
    cbs = {
        "report": callback("report", ["history"], "history"),
        "utility": callback("utility", [], "0"),
    }
    source = one_cell(
        cbs,
        {"report": "report", "utility": "utility"},
        [
            {"kind": "reward", "callback": "utility", "phase": "simulation"},
            {"kind": "reporting", "callback": "report", "phase": "simulation"},
        ],
        states={"history": "two"},
        domains={
            "two": {"kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1]}
        },
    )
    declarations, analysis, check = produce(source)
    assert (check["status"]) == ("PASS"), check
    assert (json_member(analysis, "dependency_closure")["c"]) == (["history"])
    bad = deepcopy(analysis)
    record = json_member(
        first_option(bad), "function_interfaces", "simulation", "report"
    )
    record["implementation"] = {"kind": "omit_unconsumed_provider"}
    record["arguments"] = []
    errors = check_semantics(source, declarations, bad)["errors"]
    assert any(e["kind"] == "consumed original provider was omitted" for e in errors)


def test_distinct_represented_feasibility_phases_do_not_intersect_actions() -> None:
    cbs = {
        "utility": callback("utility", ["a"], "a"),
        "solve": callback("solve", ["a"], "True"),
        "simulation": callback("simulation", ["a"], "a == 0"),
    }
    consumers = [
        {"kind": "reward", "callback": "utility", "phase": "both"},
        {"kind": "feasible", "callback": "solve", "phase": "solve"},
        {"kind": "feasible", "callback": "simulation", "phase": "simulation"},
    ]
    source = one_cell(
        cbs,
        {"utility": "utility"},
        consumers,
        actions={"a": "two"},
        domains={
            "two": {"kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1]}
        },
    )
    _, analysis, check = produce(source)
    assert (json_member(first_option(analysis), "action_reconstruction")["a"]) == (
        {"kind": "identity", "source": "a"}
    )
    assert (check["status"]) == ("PASS"), check


def test_single_truth_feasibility_channel_still_eliminates_forced_action() -> None:
    cbs = {
        "utility": callback("utility", ["a"], "a"),
        "truth": callback("truth", ["a"], "a == 0"),
    }
    source = one_cell(
        cbs,
        {"utility": "utility"},
        [
            {"kind": "reward", "callback": "utility", "phase": "simulation"},
            {
                "kind": "feasible",
                "callback": "truth",
                "phase": "simulation",
                "decision_phases": ["solve", "simulation"],
            },
        ],
        actions={"a": "two"},
        domains={
            "two": {"kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1]}
        },
    )
    _, analysis, check = produce(source)
    assert (first_option(analysis)["actions"]) == ({})
    assert (check["status"]) == ("PASS"), check


def test_absent_phase_constraint_does_not_restrict_that_phase() -> None:
    cbs = {
        "utility": callback("utility", ["a"], "a"),
        "solve": callback("solve", ["a"], "a == 0"),
    }
    source = one_cell(
        cbs,
        {"utility": "utility"},
        [
            {"kind": "reward", "callback": "utility", "phase": "both"},
            {"kind": "feasible", "callback": "solve", "phase": "solve"},
        ],
        actions={"a": "two"},
        domains={
            "two": {"kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1]}
        },
    )
    _, analysis, check = produce(source)
    assert (first_option(analysis)["actions"]) == ({"a": "two"})
    assert (check["status"]) == ("PASS"), check


def test_constant_routing_bridge_requires_source_and_receiver_agreement() -> None:
    original = native_constant()
    assert _closed_native_constant(original)
    mutations = []
    bad = deepcopy(original)
    json_member(bad, "fields")["value"] = 2
    mutations.append(bad)
    bad = deepcopy(original)
    json_member(bad, "callable")["arguments"] = ["history"]
    mutations.append(bad)
    bad = deepcopy(original)
    json_member(bad, "callable")["source"] = (
        "def __call__(self):\n    return jnp.asarray(history)\n"
    )
    mutations.append(bad)
    bad = deepcopy(original)
    json_member(bad, "callable", "bindings", "jnp")["name"] = "custom.numpy"
    mutations.append(bad)
    for bad in mutations:
        assert not (_closed_native_constant(bad))


def test_constant_route_does_not_erase_real_future_dependency() -> None:
    cbs = {
        "zero": callback("zero", [], "0"),
        "future": callback("future", ["history"], "history"),
        "route": {
            "arguments": [],
            "reads": [],
            "bindings": {},
            "expression": None,
            "status": "opaque",
            "identity": "original constant route",
            "original": native_constant(),
        },
    }
    source = one_cell(
        cbs,
        {"utility": "zero"},
        [{"kind": "reward", "callback": "zero", "phase": "both"}],
        states={"history": "two"},
        domains={
            "two": {"kind": "categorical", "size": 2, "values": [0, 1], "codes": [0, 1]}
        },
    )
    cells = source["cells"]
    assert isinstance(cells, list)
    first = cells[0]
    assert isinstance(first, dict)
    target = deepcopy(first)
    target.update(id="future", age=21, period=1)
    target["functions"] = {"utility": "future"}
    target["consumers"] = [{"kind": "reward", "callback": "future", "phase": "both"}]
    first["terminal"] = False
    cells.append(target)
    source["relations"] = [
        {
            "id": "edge",
            "kind": "transition",
            "source": "c",
            "targets": ["future"],
            "phase": "both",
            "callbacks": ["route"],
            "state_updates": {"history": "history"},
            "produces": ["history"],
        }
    ]
    _, analysis, check = produce(source)
    assert (json_member(analysis, "dependency_closure")["c"]) == (["history"])
    assert not (check["errors"]), check


def test_native_gate_predicate_and_only_active_owned_routes_are_retained() -> None:
    for dormant in (False, True):
        for simulation in (False, True):
            source = captured_gate_graph(
                dormant_admitted=dormant, simulation_edge=simulation
            )
            declarations, analysis, check = produce(source)
            cells = {c["id"]: c for c in json_rows(declarations, "cells")}
            relations = json_rows(declarations, "relations")
            expected_phases = {"solve", "simulation"} if simulation else {"solve"}
            gates = [r for r in relations if r["kind"] == "gate"]
            assert ({r["phase"] for r in gates}) == (expected_phases)
            assert all(cells[r["source"]]["regime"] == "landing" for r in gates)
            assert all(len(json_list(r, "callbacks")) == 1 for r in gates)
            fallbacks = [r for r in relations if r["kind"] == "fallback"]
            assert ({r["phase"] for r in fallbacks}) == (expected_phases)
            assert all(
                json_member(r, "ownership")["target_regime"] == "landing"
                for r in fallbacks
            )
            assert not (check["errors"]), check


def test_native_gate_callback_or_owned_leg_mutation_is_independently_rejected() -> None:
    source = captured_gate_graph()
    declarations, _, _ = produce(source)
    for mutation in ("callback", "ownership", "phase"):
        bad = deepcopy(declarations)
        if mutation == "callback":
            next(r for r in json_rows(bad, "relations") if r["kind"] == "gate")[
                "callbacks"
            ] = []
        elif mutation == "ownership":
            fallback = next(
                r for r in json_rows(bad, "relations") if r["kind"] == "fallback"
            )
            json_member(fallback, "ownership")["leg"] = "other"
        else:
            bad["relations"] = [
                r
                for r in json_rows(bad, "relations")
                if not (r["kind"] == "fallback" and r["phase"] == "simulation")
            ]
        check = check_semantics(
            source,
            bad,
            MetadataAnalysis(bad, facts_backend="metadata").run(),
        )
        assert (check["status"]) == ("FAIL"), check


def test_native_gate_reward_contrast_retains_the_decisive_history() -> None:
    source = captured_gate_graph()
    json_member(
        source, "graph", "fields", "laws", "source", "fields", "gated_edges"
    ).pop("dormant")
    json_member(source, "user_regimes", "outside", "fields", "functions")["utility"] = (
        callback("outside_reward", [], "10")["original"]
    )
    declarations, analysis, check = produce(source)
    # Literal oracle: history 0 fails the gate and receives outside value
    # 10; history 1 passes and receives landing value 0. This difference
    # cannot factor through a state-free source or landing interface.
    closure = json_member(analysis, "dependency_closure")
    for cell in json_rows(declarations, "cells"):
        expected = ["history"] if cell["regime"] in {"source", "landing"} else []
        assert (closure[str(cell["id"])]) == (expected)
    assert (check["status"]) == ("PASS"), check
