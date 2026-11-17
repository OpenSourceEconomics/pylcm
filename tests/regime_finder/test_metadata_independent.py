"""Independent original-source and projection attack regressions.

Production entry points are the systems under test. Source scalar truth never
calls their translator/evaluator/scorer. All numerical evaluations here are tiny
regressions, not production analysis.
"""

import pytest
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.model_snapshot import source_function
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics
from .metadata_oracle import (
    ScalarSourceOracle,
    independent_joint_transport,
    joint_pushforward,
)
from .tiny_metadata_family import original


def custom_where(condition, left, right):
    del condition, right
    return left


def custom_right(condition, left, right):
    del condition, left
    return right


_alias_of_custom = custom_where


def custom_utility(history, age):
    return 1 + custom_where(age < 22, history, 0)


def aliased_utility(history, age):
    return 1 + _alias_of_custom(age < 22, history, 0)


def closure_utility(helper):
    def utility(history, age):
        return 1 + helper(age < 22, history, 0)

    return utility


def identity_history(history):
    return history


def simple_utility(history):
    return history


def record(source, arguments, **extra):
    return {
        "kind": "callable",
        "source": source,
        "arguments": arguments,
        "bindings": {},
        "source_file": "independent-original-fixture.py",
        "source_sha256": "explicit regression source",
        "source_line": 1,
        **extra,
    }


def captured_snapshot(callback=None):
    """Explicit original captured declaration, no proposed reduction/cuts."""
    grid = {
        "kind": "object",
        "type": "lcm.DiscreteGrid",
        "fields": {"values": [0, 1], "codes": [9, 3], "categories": ["low", "high"]},
    }
    dates = [20, 21, 22, 23, 24]
    phase = {
        "kind": "dataclass",
        "fields": {"targets_by_period": [{"life": ["life"]} for _ in dates[:-1]]},
    }
    reach = {
        "kind": "dataclass",
        "fields": {
            "nodes": [[date, "life"] for date in dates],
            "solution": deepcopy(phase),
            "simulation": deepcopy(phase),
        },
    }
    return {
        "schema": "constructed-pylcm-model-graph-1",
        "n_periods": len(dates),
        "ages": {"kind": "object", "fields": {"exact_values": dates}},
        "reachability": deepcopy(reach),
        "graph": {
            "kind": "dataclass",
            "fields": {
                "coordinate_kind": "age",
                "reachability": reach,
                "initial_nodes": [[20, "life"]],
                "laws": {},
            },
        },
        "user_regimes": {
            "life": {
                "kind": "dataclass",
                "type": "lcm.Regime",
                "fields": {
                    "states": {"history": grid},
                    "actions": {},
                    "functions": {
                        "utility": callback or source_function(simple_utility)
                    },
                    "constraints": {},
                    "state_transitions": {"history": source_function(identity_history)},
                },
            }
        },
        "semantic_contract": {
            "roles": {
                "life": {
                    "actor": "individual",
                    "observation_order": ["state", "choice"],
                    "commitments": [],
                    "discount_boundary": "one year",
                }
            },
            "allocation": {"status": "known", "kind": "per_block", "boundary_unit": 1},
            "production_metadata": {
                "status": "regression",
                "no_production_claim": True,
            },
        },
    }


def _produce(spec=None):
    source = original() if spec is None else spec
    declarations = normalize_declarations(source)
    analysis = MetadataAnalysis(declarations, facts_backend="metadata").run()
    return source, declarations, analysis


def test_trillion_point_axis_is_metadata_not_a_dense_tensor() -> None:
    source, declarations, analysis = _produce()
    assert (analysis["declarations"]["domains"]["assets_large"]["size"]) == (10**12)
    assert (analysis["production_execution"]["callbacks_called"]) == (0)
    assert (analysis["production_execution"]["dense_cartesian_cells_enumerated"]) == (0)
    checked = check_semantics(source, declarations, analysis)
    assert (checked["dense_cartesian_rows_evaluated"]) == (0)


def test_reduced_function_signatures_are_checked() -> None:
    source, declaration, analysis = _produce()
    corrupted = deepcopy(analysis)
    option = corrupted["interfaces"]["c1"]["options"][-1]
    function = option["function_interfaces"]["simulation"]["utility"]
    function["arguments"].remove("assets")
    function["unused_arguments"].append("assets")
    checked = check_semantics(source, declaration, corrupted)
    assert (checked["status"]) == ("FAIL"), checked
    assert any(
        error["kind"] == "live original argument removed from function interface"
        for error in checked["errors"]
    )


def test_source_custom_and_aliased_helpers_are_not_primitives() -> None:
    oracle = ScalarSourceOracle()
    for utility in (custom_utility, aliased_utility, closure_utility(custom_where)):
        metadata = source_function(utility)
        assert metadata["call_targets"]
        assert all(
            target["kind"] != "primitive"
            for target in metadata["call_targets"].values()
        )
        for age in range(20, 25):
            for history in (0, 1):
                assert (oracle.call(metadata, {"history": history, "age": age})) == (
                    utility(history, age)
                )
        snapshot = captured_snapshot(metadata)
        _, declaration, analysis = _produce(snapshot)
        assert all(
            "history" in option["states"]
            for record in analysis["interfaces"].values()
            for option in record["options"]
        )
        assert (check_semantics(snapshot, declaration, analysis)["status"]) != ("FAIL")


def test_captured_helper_binding_changes_identity_and_truth() -> None:
    left, right = closure_utility(custom_where), closure_utility(custom_right)
    left_capture, right_capture = source_function(left), source_function(right)
    assert (left_capture["identity_sha256"]) != (right_capture["identity_sha256"])
    assert (left(1, 24)) == (2)
    assert (right(1, 24)) == (1)
    oracle = ScalarSourceOracle()
    assert (oracle.call(left_capture, {"history": 1, "age": 24})) == (2)
    assert (oracle.call(right_capture, {"history": 1, "age": 24})) == (1)


def test_full_joint_correlations_are_not_marginal_transport() -> None:
    diagonal = [((0, 0), "1/2"), ((1, 1), "1/2")]
    opposite = [((0, 1), "1/2"), ((1, 0), "1/2")]
    for coordinate in (0, 1):
        assert (joint_pushforward(diagonal, lambda state: state[coordinate])) == (
            joint_pushforward(opposite, lambda state: state[coordinate])
        )
    checked = independent_joint_transport(diagonal, opposite, lambda state: state)
    assert (checked["status"]) == ("FAIL")
    assert ("positive joint support") in (checked["errors"])
    reward = lambda state: state[0] == state[1]
    assert (sum(reward(state) * Fraction(mass) for state, mass in diagonal)) == (1)
    assert (sum(reward(state) * Fraction(mass) for state, mass in opposite)) == (0)


def test_joint_projection_aggregates_equal_images_and_rare_support() -> None:
    rows = [((0, "a"), "1/4"), ((0, "b"), "1/4"), ((1, "c"), "1/2")]
    reduced = [(0, "1/2"), (1, "1/2")]
    assert (
        independent_joint_transport(rows, reduced, lambda state: state[0])["status"]
    ) == ("PASS")
    rare = Fraction(1, 10**50)
    checked = independent_joint_transport(
        [(0, 1 - rare), (1, rare)], [(0, 1)], lambda state: state
    )
    assert (checked["status"]) == ("FAIL")
    assert ("positive joint support") in (checked["errors"])


def test_ordinary_probability_roundoff_is_not_a_semantic_failure() -> None:
    checked = independent_joint_transport(
        [(0, 0.1), (1, 0.2), (2, 0.7000000000000001)],
        [(0, "1/10"), (1, "1/5"), (2, "7/10")],
        lambda state: state,
    )
    assert (checked["status"]) == ("PASS"), checked


def test_native_source_normalization_mutations(*, subtests: pytest.Subtests) -> None:
    snapshot = captured_snapshot()
    _, declaration, analysis = _produce(snapshot)
    assert (check_semantics(snapshot, declaration, analysis)["status"]) == ("PASS")
    for change in ("phase", "role", "age", "domain", "consumer", "start"):
        normalized = deepcopy(declaration)
        if change == "phase":
            normalized["relations"] = [
                relation
                for relation in normalized["relations"]
                if relation["phase"] != "simulation"
            ]
        if change == "role":
            normalized["cells"][0]["role"]["actor"] = "different actor"
        if change == "age":
            normalized["cells"][0]["age"] = 19
        if change == "domain":
            next(iter(normalized["domains"].values()))["values"] = [0, 2]
        if change == "consumer":
            normalized["cells"][0]["consumers"] = []
        if change == "start":
            normalized["cells"][0]["admissible_start"] = False
        altered = deepcopy(analysis)
        altered["declarations"] = normalized
        with subtests.test(change=change):
            assert (check_semantics(snapshot, normalized, altered)["status"]) == (
                "FAIL"
            )


def test_original_period_age_provider_forces_correct_decision_dates() -> None:
    snapshot = captured_snapshot()
    dates = list(range(5))
    snapshot["ages"] = None
    snapshot["graph"]["fields"]["coordinate_kind"] = "period"
    for reach in (
        snapshot["reachability"],
        snapshot["graph"]["fields"]["reachability"],
    ):
        reach["fields"]["nodes"] = [[period, "life"] for period in dates]
    snapshot["graph"]["fields"]["initial_nodes"] = [[0, "life"]]
    declaration = snapshot["user_regimes"]["life"]["fields"]
    declaration["actions"]["choice"] = {
        "kind": "object",
        "type": "lcm.DiscreteGrid",
        "fields": {"values": [0, 1], "codes": [7, 9]},
    }
    declaration["functions"]["age"] = record(
        "def age(period):\n    return 20 + period\n", ["period"]
    )
    declaration["constraints"]["choice"] = record(
        "def allowed(choice, age):\n    return (choice == 0) | (age < 22)\n",
        ["choice", "age"],
    )
    source, normalized, analysis = _produce(snapshot)
    for cell in normalized["cells"]:
        option = analysis["interfaces"][cell["id"]]["options"][0]
        assert (cell["age"]) is None
        assert (option["action_reconstruction"]["choice"]["kind"]) == (
            "constant" if cell["period"] >= 2 else "identity"
        )
    assert (check_semantics(source, normalized, analysis)["status"]) == ("PASS")


def test_categorical_action_codes_do_not_replace_decoded_values() -> None:
    snapshot = captured_snapshot()
    declaration = snapshot["user_regimes"]["life"]["fields"]
    declaration["actions"]["choice"] = {
        "kind": "object",
        "type": "lcm.DiscreteGrid",
        "fields": {"values": [10, 20], "codes": [9, 3]},
    }
    declaration["constraints"]["code"] = record(
        "def allowed(choice_code):\n    return choice_code == 3\n", ["choice_code"]
    )
    source, normalized, analysis = _produce(snapshot)
    for interface in analysis["interfaces"].values():
        assert (interface["options"][0]["action_reconstruction"]["choice"]) == (
            {"kind": "constant", "value": 20}
        )
    assert (check_semantics(source, normalized, analysis)["status"]) == ("PASS")


def test_native_original_schedule_joint_gate_and_imputation_replay(
    *, subtests: pytest.Subtests
) -> None:
    directory = Path(__file__).with_name("fixtures_metadata")
    for name in (
        "native-basic",
        "native-schedule",
        "native-gate",
        "native-joint",
        "native-imputation",
    ):
        source = json.loads((directory / (name + ".json")).read_text())
        declaration = normalize_declarations(source)
        analysis = MetadataAnalysis(declaration, facts_backend="metadata").run()
        with subtests.test(fixture=name):
            report = check_semantics(source, declaration, analysis)
            assert (report["status"]) == ("PASS"), report
            assert (report["dense_cartesian_rows_evaluated"]) == (0)


def test_original_relation_payload_and_consumer_wiring_mutations(
    *, subtests: pytest.Subtests
) -> None:
    directory = Path(__file__).with_name("fixtures_metadata")
    attacks = [
        ("native-basic", "transition", "state_updates"),
        ("native-basic", "transition", "produces"),
        ("native-joint", "joint", "callbacks"),
        ("native-joint", "joint", "support"),
        ("native-joint", "joint", "joint_outputs"),
        ("native-gate", "gate", "callbacks"),
        ("native-gate", "fallback", "state_updates"),
        ("native-gate", "fallback", "ownership"),
        ("native-imputation", "imputation", "remove"),
    ]
    for name, kind, field in attacks:
        source = json.loads((directory / (name + ".json")).read_text())
        declaration = normalize_declarations(source)
        analysis = MetadataAnalysis(declaration, facts_backend="metadata").run()
        modified = deepcopy(declaration)
        relation = next(item for item in modified["relations"] if item["kind"] == kind)
        if field == "remove":
            modified["relations"].remove(relation)
        elif field == "support":
            relation[field] = [[0, 1], [1, 0]]
        else:
            relation[field] = {} if isinstance(relation.get(field), dict) else []
        altered = deepcopy(analysis)
        altered["declarations"] = modified
        with subtests.test(fixture=name, field=field):
            assert (check_semantics(source, modified, altered)["status"]) == ("FAIL")


def test_derived_category_codes_retain_original_value_dependencies() -> None:
    snapshot = captured_snapshot()
    declaration = snapshot["user_regimes"]["life"]["fields"]
    declaration["functions"]["class_value"] = record(
        "def class_value(history):\n    return [10, 20][history]\n", ["history"]
    )
    declaration["derived_categoricals"] = {
        "class_value": {
            "kind": "object",
            "type": "lcm.DiscreteGrid",
            "fields": {"values": [10, 20], "codes": [9, 3]},
        }
    }
    declaration["functions"]["utility"] = record(
        "def utility(class_value_code):\n    return class_value_code\n",
        ["class_value_code"],
    )
    source, normalized, analysis = _produce(snapshot)
    assert (check_semantics(source, normalized, analysis)["status"]) == ("PASS")
    for cell in normalized["cells"]:
        assert ("history") in (analysis["dependency_closure"][cell["id"]])
    modified = deepcopy(normalized)
    modified["cells"][0]["derived_code_providers"] = {}
    altered = deepcopy(analysis)
    altered["declarations"] = modified
    assert (check_semantics(source, modified, altered)["status"]) == ("FAIL")


def test_malformed_metadata_never_passes() -> None:
    source = original()
    normalized = normalize_declarations(source)
    analysis = MetadataAnalysis(normalized, facts_backend="metadata").run()
    broken = deepcopy(analysis)
    broken["interfaces"] = None
    assert (check_semantics(source, normalized, broken)["status"]) == ("FAIL")


def test_legacy_decorator_and_added_evaluated_arguments_stay_opaque() -> None:
    for callback in (
        record("def utility(history):\n    return history\n", ["history", "period"]),
        record(
            "@changes_behavior\ndef utility(history):\n    return history\n",
            ["history"],
        ),
    ):
        snapshot = captured_snapshot(callback)
        source, normalized, analysis = _produce(snapshot)
        bound = next(
            item
            for item in normalized["callbacks"].values()
            if item.get("original") == callback
        )
        assert (bound["status"]) == ("opaque")
        assert (set(bound["reads"])) == (set(callback["arguments"]))
        checked = check_semantics(source, normalized, analysis)
        assert (checked["status"]) != ("FAIL"), checked
