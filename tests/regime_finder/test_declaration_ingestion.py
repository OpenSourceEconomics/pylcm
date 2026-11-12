"""Metadata/source identity regressions; no native Model or solver required."""

from tests.regime_finder._callable_records import validate_callable_records

import pytest
from copy import deepcopy
import ast
from dataclasses import dataclass
import functools
import hashlib
import marshal
from pathlib import Path

from _lcm_regime_finder.callable_identity import (
    resolve_callable,
)
from _lcm_regime_finder.model_snapshot import (
    source_function,
    normalize,
    read_declarations,
)
from _lcm_regime_finder.declarations import (
    normalize_declarations,
    ScalarSource,
    DeclarationError,
)
from _lcm_regime_finder.model_adapter import adapt_snapshot
from tests.regime_finder.test_model_adapter import snapshot, function, grid


def where(condition, left, right):
    del condition, right
    return left


alias_where = where


def custom(history, age):
    return alias_where(age < 2, history, 0)


def captured_factory(cutoff):
    def callback(history, age):
        return history if age < cutoff else 0

    return callback


def numeric_bits(history):
    return history & 1


def decorated(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs) + 11

    return wrapper


@decorated
def decorated_callback(history):
    return history


def helper_default(value=9):
    return value


def with_helper_default(history):
    del history
    return helper_default()


@dataclass
class BoundHelper:
    shift: int

    def where(self, condition, left, right):
        del condition, right
        return left + self.shift


def test_custom_alias_where_has_actual_helper_identity() -> None:
    meta = source_function(custom)
    reference = meta["call_targets"]["alias_where"]
    assert (reference["kind"]) == ("callable-reference")
    assert validate_callable_records(meta["callable_records"])
    assert (reference) == (meta["bindings"]["alias_where"])
    target = resolve_callable(reference, meta["callable_records"])
    assert (target["kind"]) == ("callable")
    assert (target["qualname"]) == (where.__qualname__)
    assert (target["module"]) == (where.__module__)
    assert (target["code_sha256"]) == (
        hashlib.sha256(marshal.dumps(where.__code__)).hexdigest()
    )
    assert (target["source_sha256"]) == (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    assert ("return left") in (target["source"])
    assert (ScalarSource(meta).expression()) == ("history")


def test_captured_binding_changes_callable_identity() -> None:
    a, b = source_function(captured_factory(2)), source_function(captured_factory(3))
    assert (a["identity_sha256"]) != (b["identity_sha256"])
    assert (a["binding_sha256"]) != (b["binding_sha256"])
    assert (ScalarSource(a).expression()) != (ScalarSource(b).expression())


def test_bound_receiver_and_partial_arguments_are_captured() -> None:
    a, b = source_function(BoundHelper(2).where), source_function(BoundHelper(3).where)
    assert (a["identity_sha256"]) != (b["identity_sha256"])
    assert (a["bound_self"]["fields"]["shift"]) == (2)
    p = source_function(functools.partial(where, left=7))
    assert (p["partial_bindings"]["left"]) == (7)


def test_actual_numpy_alias_can_be_identified_when_available() -> None:
    try:
        import numpy as np
    except ImportError:
        pytest.skip("Existing NumPy unavailable; no installation attempted")
    choose = np.where

    def callback(history, age):
        return choose(age < 2, history, 0)

    meta = source_function(callback)
    reference = meta["call_targets"]["choose"]
    assert (reference["kind"]) == ("callable-reference")
    assert validate_callable_records(meta["callable_records"])
    assert (reference) == (meta["bindings"]["choose"])
    target = resolve_callable(reference, meta["callable_records"])
    assert (choose) is (np.where)
    assert (target["kind"]) == ("primitive")
    assert (target["canonical"]) == ("numpy.where")
    assert (target["resolution"]) == ("actual_object_identity")
    assert (target["operation"]) == ("select")
    assert (target["library_module_sha256"]) == (
        hashlib.sha256(Path(np.__file__).read_bytes()).hexdigest()
    )


def test_unbound_textual_jnp_where_is_not_a_primitive() -> None:
    x = snapshot()
    x["user_regimes"]["original"]["fields"]["functions"]["utility"] = function(
        "def utility(history,age):\n    return jnp.where(age < 2, history, 0)\n",
        ["history", "age"],
    )
    p = adapt_snapshot(x)
    from _lcm_regime_finder import find_regimes

    result = find_regimes(p)
    assert (result["status"]) == ("OPTIMAL")
    assert all("history" in atom["live_states"] for atom in result["analysis"]["atoms"])


def test_large_metadata_array_is_never_materialized() -> None:
    class Huge:
        shape = (10**8, 10**8)
        dtype = "float64"

        def __array__(self, *args, **kwargs):
            raise AssertionError("Dense conversion forbidden")

    value = normalize(Huge())
    assert (value["shape"]) == ([10**8, 10**8])
    assert (value["identity_status"]) == ("UNRESOLVED_LARGE_BINDING")
    assert ("values") not in (value)


def test_huge_grid_cardinality_is_only_metadata() -> None:
    x = snapshot()
    x["user_regimes"]["original"]["fields"]["states"]["x"] = {
        "kind": "dataclass",
        "type": "lcm.LinSpacedGrid",
        "fields": {"n_points": 10**18, "start": 0, "stop": 1},
    }
    n = normalize_declarations(x)
    d = n["domains"][n["cells"][0]["states"]["x"]]
    assert (d["size"]) == (10**18)
    assert ("values") not in (d)
    assert (n["execution"]["numerical_model_rows"]) == (0)


def test_stateful_phase_imputation_is_a_real_relation() -> None:
    x = snapshot()
    r = x["user_regimes"]["original"]["fields"]
    r["states"]["belief"] = {
        "kind": "dataclass",
        "type": "lcm.Phased",
        "fields": {
            "solve": function(
                "def impute(history):\n    return history\n", ["history"]
            ),
            "simulate": grid([0, 1]),
        },
    }
    r["functions"]["utility"] = function(
        "def utility(belief):\n    return belief\n", ["belief"]
    )
    r["state_transitions"]["belief"] = function(
        "def next_belief(belief):\n    return belief\n", ["belief"]
    )
    n = normalize_declarations(x)
    impute = next(r for r in n["relations"] if r["kind"] == "imputation")
    assert (impute["phase"]) == ("solve")
    assert (impute["callbacks"]) == ([])
    assert (n["callbacks"][impute["state_updates"]["belief"]]["reads"]) == (["history"])
    assert not (n["domains"][n["cells"][0]["states"]["belief"]]["solve_axis"])


def test_joint_kernel_remains_single_law_with_decoded_outputs() -> None:
    x = snapshot()
    r = x["user_regimes"]["original"]["fields"]
    r["state_transitions"] = {}
    r["joint_transitions"] = {
        "original": {
            "joint": {
                "kind": "dataclass",
                "type": "lcm.JointTransition",
                "fields": {
                    "support_size": 2,
                    "support": [[0, 0], [1, 1]],
                    "probabilities": function(
                        "def probabilities(choice):\n    return choice\n", ["choice"]
                    ),
                    "outputs": {
                        "x": function(
                            "def x(joint):\n    return joint[0]\n", ["joint"]
                        ),
                        "history": function(
                            "def history(joint):\n    return joint[1]\n", ["joint"]
                        ),
                    },
                },
            }
        }
    }
    n = normalize_declarations(x)
    j = next(r for r in n["relations"] if r["kind"] == "joint")
    assert (j["support"]) == ([[0, 0], [1, 1]])
    assert (set(j["joint_outputs"])) == ({"x", "history"})
    assert (j["produces"]) == (["history", "x"])
    assert (len(j["callbacks"])) == (1)


def test_spec_export_does_not_call_arbitrary_factories() -> None:
    def never():
        raise AssertionError("Factory executed")

    captured = read_declarations({"regimes": {}, "factory": never})
    assert (captured["execution_contract"]["specialization_factories_executed"]) == (0)
    assert (captured["factory"]["kind"]) == ("callable")


def _create_raw_spec():
    r = deepcopy(snapshot()["user_regimes"]["original"])
    selector = {
        "kind": "dataclass",
        "type": "lcm.AgeRange",
        "fields": {"exclusive_stop": 21},
    }
    schedule = {
        "kind": "object",
        "type": "lcm.ByAge",
        "fields": {
            "cases": [[selector, "working"], [21, "retired"]],
            "default": None,
            "until": None,
        },
    }
    return {
        "regimes": {"working": r, "retired": deepcopy(r)},
        "ages": {"kind": "object", "fields": {"exact_values": [20, 20.5, 21, 21.5]}},
        "initial_nodes": {
            "kind": "dataclass",
            "type": "lcm.InitialNodes",
            "fields": {"by_age": {"kind": "mapping", "items": [[20, ["working"]]]}},
        },
        "edges": {
            "working": {
                "kind": "dataclass",
                "type": "lcm.Transition",
                "fields": {"law": schedule},
            }
        },
    }


def test_preprogram_spec_resolves_original_schedule_and_exact_dates() -> None:
    x = _create_raw_spec()
    n = normalize_declarations(x)
    assert ([(c["age"], c["regime"]) for c in n["cells"]]) == (
        [(20, "working"), (20.5, "working"), (21, "working"), (21.5, "retired")]
    )
    assert (n["graph_derivation"]["kind"]) == ("resolved_original_declarations")
    assert (len(n["relations"])) == (6)
    assert all(c["stage"] is None for c in n["cells"])


def test_until_uses_actual_predecessor_and_later_initial_starts() -> None:
    x = _create_raw_spec()
    x["edges"]["working"]["fields"]["law"]["fields"] = {
        "until": {
            "kind": "dataclass",
            "type": "lcm._Until",
            "fields": {
                "stop_age_exclusive": 21.5,
                "start_age_inclusive": 20,
                "law": "working",
                "then": "retired",
            },
        }
    }
    x["initial_nodes"]["fields"]["by_age"]["items"].append([21.5, ["working"]])
    n = normalize_declarations(x)
    starts = [(c["age"], c["regime"]) for c in n["cells"] if c["admissible_start"]]
    assert (starts) == ([(20, "working"), (21.5, "working")])
    assert ((21.5, "retired")) in ([(c["age"], c["regime"]) for c in n["cells"]])


def test_old_empty_schedule_requires_real_metadata() -> None:
    x = _create_raw_spec()
    x["edges"]["working"]["fields"]["law"]["fields"] = {}
    with pytest.raises(DeclarationError, match="omitted its cases"):
        normalize_declarations(x)


def test_separately_supplied_semantics_contract_is_bound() -> None:
    from typing import TypedDict

    class Contract(TypedDict):
        roles: dict[str, dict[str, str]]
        stages: dict[str, str]
        allocation: dict[str, str]

    contract: Contract = {
        "roles": {"original": {"actor": "one"}},
        "stages": {"original": "decision"},
        "allocation": {"kind": "per_block"},
    }
    n = normalize_declarations(snapshot(), contract)
    assert (n["original_contract"]) == (contract)
    assert all(c["stage"] == "decision" for c in n["cells"])
    changed = deepcopy(contract)
    changed["roles"]["original"]["actor"] = "two"
    assert (
        (n["source_identity"])
        != (normalize_declarations(snapshot(), changed)["source_identity"])
    )


def test_integer_bitwise_operation_cannot_prove_wrong_characteristic() -> None:
    with pytest.raises(DeclarationError, match="bitwise"):
        ScalarSource(source_function(numeric_bits)).expression()


def test_decorator_body_is_not_unwrapped_into_another_function() -> None:
    meta = source_function(decorated_callback)
    assert meta["variadic"]
    wrapper = ast.parse(meta["source"]).body[0]
    assert isinstance(wrapper, ast.FunctionDef)
    assert wrapper.name == "wrapper"
    returned = wrapper.body[0]
    assert isinstance(returned, ast.Return)
    expression = returned.value
    assert isinstance(expression, ast.BinOp)
    assert isinstance(expression.op, ast.Add)
    assert isinstance(expression.left, ast.Call)
    assert isinstance(expression.right, ast.Constant)
    assert expression.right.value == 11


def test_actual_helper_defaults_are_bound() -> None:
    assert (ScalarSource(source_function(with_helper_default)).expression()) == ("9")


def test_old_unwrapped_decorator_signature_cannot_drop_period() -> None:
    x = snapshot()
    r = x["user_regimes"]["original"]["fields"]
    r["functions"]["utility"] = function(
        '@time_varying_params("table")\ndef utility(history,table):\n    return table[history]\n',
        ["history", "table", "period"],
    )
    n = normalize_declarations(x)
    cid = n["cells"][0]["functions"]["utility"]
    record = n["callbacks"][cid]
    assert (record["status"]) == ("opaque")
    assert (record["expression"]) is None
    assert (record["reads"]) == (["history", "period", "table"])


def test_tuple_subscript_is_not_rewritten_as_list_index() -> None:
    meta = function(
        "def lookup(table,x,y):\n    return table[x,y]\n", ["table", "x", "y"]
    )
    assert (ScalarSource(meta).expression()) == ("(table)[(x, y)]")


def test_legacy_decorator_with_unchanged_signature_stays_opaque() -> None:
    meta = function(
        "@custom_decorator\ndef utility(history):\n    return 0\n", ["history"]
    )
    with pytest.raises(DeclarationError, match="decorated source"):
        ScalarSource(meta).expression()


def test_current_payoff_uses_truth_variant_and_next_state_provider() -> None:
    x = snapshot()
    r = x["user_regimes"]["original"]["fields"]
    r["functions"]["utility"] = {
        "kind": "object",
        "type": "lcm.Phased",
        "fields": {
            "solve": function(
                "def utility(history):\n    return history\n", ["history"]
            ),
            "simulate": function(
                "def utility(next_x):\n    return next_x\n", ["next_x"]
            ),
        },
    }
    n = normalize_declarations(x)
    for c in n["cells"]:
        roots = [v for v in c["consumers"] if v["kind"] == "reward"]
        assert ([v["phase"] for v in roots]) == (["simulation"])
        assert (n["callbacks"][roots[0]["callback"]]["reads"]) == (["next_x"])
        assert ("next_x") in (c["phase_functions"]["simulation"])


def _create_gated_spec():
    x = _create_raw_spec()
    x["ages"]["fields"]["exact_values"] = [20, 20.5]
    x["regimes"]["outside"] = deepcopy(x["regimes"]["retired"])
    projection = {
        "x": function("def projected(history):\n    return history\n", ["history"]),
        "history": function(
            "def projected(history):\n    return history\n", ["history"]
        ),
    }
    fallback = {
        "kind": "dataclass",
        "type": "lcm.ProjectedRegimeValue",
        "fields": {"regime": "outside", "projection": projection},
    }
    route = {
        "kind": "dataclass",
        "type": "lcm.StakeholderRoute",
        "fields": {"fallback": fallback},
    }
    gate = {
        "kind": "dataclass",
        "type": "lcm.Gate",
        "fields": {
            "predicate": function(
                "def gate(history):\n    return history > 0\n", ["history"]
            ),
            "routes": {"self": route},
        },
    }
    x["edges"]["working"]["fields"] = {"law": "retired", "gates": {"retired": gate}}
    return x


def test_gate_and_fallback_use_landing_context_and_incoming_owner() -> None:
    n = normalize_declarations(_create_gated_spec())
    cells = {c["id"]: c for c in n["cells"]}
    gate = next(r for r in n["relations"] if r["kind"] == "gate")
    assert ((cells[gate["source"]]["regime"], cells[gate["source"]]["age"])) == (
        ("retired", 20.5)
    )
    assert (cells[gate["incoming_source"]]["regime"]) == ("working")
    for fallback in [r for r in n["relations"] if r["kind"] == "fallback"]:
        assert (fallback["source"]) == (gate["source"])
        assert (fallback["incoming_source"]) == (gate["incoming_source"])
        assert (fallback["ownership"]["leg"]) == ("self")
        assert (cells[fallback["targets"][0]]["regime"]) == ("outside")


def test_array_class_metadata_does_not_access_shape_descriptor() -> None:
    class ArrayType:
        @property
        def shape(self):
            raise AssertionError("A class must not be instantiated")

        @property
        def dtype(self):
            raise AssertionError("A class must not be instantiated")

    value = normalize(ArrayType)
    assert (value["kind"]) == ("type")
    assert value["type"].endswith(".ArrayType")


def test_binding_named_type_is_not_a_declaration_tag() -> None:
    from _lcm_regime_finder.declarations import short

    assert (short({"type": {"kind": "type", "type": "builtins.float"}})) == ("")


def test_class_call_target_is_not_a_bound_instance() -> None:
    meta = source_function(float)
    assert ("bound_self") not in (meta)
    assert meta["source_unavailable"]
    assert (meta["qualname"]) == ("float")


def test_opaque_native_callback_retains_domain_in_independent_family() -> None:
    from _lcm_regime_finder.production import analyze_production

    original = snapshot()
    regime = original["user_regimes"]["original"]["fields"]
    regime["states"]["other_history"] = grid([0, 1])
    regime["functions"]["utility"] = {
        "kind": "dataclass",
        "type": "native.UnknownPayoff",
        "fields": {},
    }
    result = analyze_production(original, limits={"seconds": 15})
    errors = result.get("semantic_check", {}).get("errors", [])
    assert not (
        any("projection policy is outside" in e.get("kind", "") for e in errors)
    ), errors
    assert (result.get("semantic_check", {}).get("status")) != ("PASS")


def test_cached_original_identities_match_canonical_json() -> None:
    from _lcm_regime_finder.declarations import Normalizer, digest

    values = [
        None,
        True,
        5,
        "alpha",
        {"name": "ä/β", "nested": [1, None, {"size": 81}]},
    ]
    normal = Normalizer({}, None)
    for value in values:
        assert (normal.value_digest(value)) == (digest(value))
        for field, setting in [
            ("period", None),
            ("period", 41),
            ("selected_phase", "simulation"),
        ]:
            expected = digest({"declaration": value, field: setting})
            assert (normal.declaration_key(value, field, setting)) == (expected)
            assert (normal.declaration_key(value, field, setting)) == (expected)


def test_native_linear_expectation_does_not_read_unrelated_current_state() -> None:
    from _lcm_regime_finder.production import analyze_production

    original = snapshot()
    regime = original["user_regimes"]["original"]["fields"]
    regime["states"]["unread"] = grid([0, 1])
    regime["certainty_equivalent"] = {
        "kind": "dataclass",
        "type": "_lcm.certainty_equivalent.LinearExpectation",
        "fields": {},
    }
    result = analyze_production(original, limits={"seconds": 15})
    errors = result.get("semantic_check", {}).get("errors", [])
    assert not (
        any(
            "dependency closure omitted" in e.get("kind", "")
            or "projection policy is outside" in e.get("kind", "")
            for e in errors
        )
    ), errors
