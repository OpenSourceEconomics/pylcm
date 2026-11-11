"""Independent source/binding mutation controls; no Model execution."""

import pytest
from copy import deepcopy

from _lcm_regime_finder.semantic_check import SourceProof
from _lcm_regime_finder.model_snapshot import source_function
from _lcm_regime_finder.declarations import ScalarSource, DeclarationError
from _lcm_regime_finder.expressions import partial
from _lcm_regime_finder import find_regimes
from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from tests.regime_finder.test_round5_local_obligations import native_fixture
from tests.regime_finder.round6_source_oracle import (
    literal_transitive,
    varies_in_history,
)


def metadata(source, arguments, **extra):
    return {
        "kind": "callable",
        "source": source,
        "arguments": arguments,
        "raw_arguments": arguments,
        "variadic": False,
        "capture_contract": "inspection-only-frozen-bindings-1",
        **extra,
    }


def wrapper_metadata():
    # This is the original Adda capture shape: an effective x argument, while
    # the inspected wrapper body receives self/*args/**kwargs. The checker
    # must not decide independence from those raw implementation names.
    return metadata(
        "def __call__(self, *args, **kwargs):\n    return self.fun(*args, **kwargs)\n",
        ["x"],
        raw_arguments=["self", "args", "kwargs"],
        variadic=True,
        parameter_kinds={"x": "POSITIONAL_OR_KEYWORD"},
    )


def known_helper(history, shift=2):
    return history + shift


def known_caller(history):
    return known_helper(history)


def table_callback(table):
    def lookup(history):
        return history + table["bias"][0]

    return lookup


def shared_table_caller():
    table = {"bias": [2, 3]}
    left, right = table_callback(table), table_callback(table)

    def caller(history):
        return left(history) + right(history)

    return caller


def tuple_default_helper(history, pair=(2, 3)):
    return history if pair == (2, 3) else 0


def tuple_default_caller(history):
    return tuple_default_helper(history)


def binding_callback(payload):
    def callback(history):
        return history + payload[0]

    return callback


def test_unknown_effective_wrapper_retains_actual_argument() -> None:
    captured = metadata(
        "def consumer(history):\n    return wrapped(history)\n",
        ["history"],
        call_targets={"wrapped": wrapper_metadata()},
    )
    assert varies_in_history(literal_transitive)
    checked = SourceProof({}).callback(captured)
    assert ("history") in (checked.reads)
    assert not (checked.proved)


def test_transitive_effective_arguments_are_not_replaced_by_raw_formals(
    *, subtests: pytest.Subtests
) -> None:
    callbacks = {
        "derived": metadata(
            "def derived(history, shift):\n    return history + shift\n",
            ["history", "shift"],
        ),
        "consumer": metadata(
            "def consumer(derived):\n    return wrapped(derived)\n",
            ["derived"],
            call_targets={"wrapped": wrapper_metadata()},
        ),
    }
    for shift in (-2, 0, 3):
        with subtests.test(shift=shift):
            checked = SourceProof(
                callbacks, {"derived": "derived"}, {"shift": shift}
            ).callback("consumer")
            assert ("history") in (checked.reads)
            assert ("shift") not in (checked.reads)
            assert not (checked.proved)


def test_named_unknown_helper_also_preserves_original_input() -> None:
    for source in ("return wrapped(history)", "return wrapped(x=history)"):
        captured = metadata(
            "def consumer(history):\n    " + source + "\n",
            ["history"],
            call_targets={"wrapped": wrapper_metadata()},
        )
        assert ("history") in (SourceProof({}).callback(captured).reads)


def test_fixed_input_does_not_reappear_as_raw_wrapper_name() -> None:
    captured = metadata(
        "def consumer(history):\n    return wrapped(history)\n",
        ["history"],
        call_targets={"wrapped": wrapper_metadata()},
    )
    checked = SourceProof({}, static={"history": 4}).callback(captured)
    assert ("history") not in (checked.reads)
    assert not (checked.proved)


def test_unknown_effective_default_does_not_read_an_outer_axis() -> None:
    target = wrapper_metadata()
    target["defaults"] = {"x": 4}
    captured = metadata(
        "def consumer():\n    return wrapped()\n", [], call_targets={"wrapped": target}
    )
    checked = SourceProof({}).callback(captured)
    assert ("x") not in (checked.reads)
    assert not (checked.proved)


def test_time_branch_and_transitive_input_mutations() -> None:
    for period in (0, 1, 2, 3):
        for source, live in (
            ("history if period < 2 else 0", period < 2),
            ("0 if period < 2 else history", period >= 2),
        ):
            callbacks = {
                "derived": metadata(
                    "def derived(history, period):\n    return " + source + "\n",
                    ["history", "period"],
                ),
                "consumer": metadata(
                    "def consumer(derived):\n    return wrapped(derived)\n",
                    ["derived"],
                    call_targets={"wrapped": wrapper_metadata()},
                ),
            }
            checked = SourceProof(
                callbacks, {"derived": "derived"}, {"period": period}
            ).callback("consumer")
            assert ("history" in checked.reads) == (live)
            assert ("period") not in (checked.reads)


def test_compact_helper_table_is_consumed_by_both_interpreters() -> None:
    captured = source_function(known_caller)
    expression = ScalarSource(captured).expression()
    for history in (-3, 0, 7):
        answer = SourceProof({}, static={"history": history}).callback(captured)
        assert answer.proved
        assert (answer.value) == (known_caller(history))
        assert (partial(expression, {"history": history}).value) == (answer.value)


def test_compact_graph_identity_mutation_is_not_certified() -> None:
    captured = source_function(known_caller)
    assert ("callable_records") in (captured)
    for field in ("source", "identity_sha256", "capture_status"):
        mutated = deepcopy(captured)
        record = next(
            record
            for record in mutated["callable_records"].values()
            if record.get("qualname", "").endswith("known_helper")
        )
        record[field] = (
            "changed" if field != "capture_status" else "UNRESOLVED_MUTATION"
        )
        answer = SourceProof({}).callback(mutated)
        assert not (answer.proved)
        assert ("history") in (answer.reads)


def test_compact_root_copy_must_match_its_record() -> None:
    captured = source_function(known_caller)
    for field, value in (
        ("source", "def known_caller(history):\n    return 0\n"),
        ("bindings", {"unrecorded": 0}),
    ):
        mutated = deepcopy(captured)
        mutated[field] = value
        checked = SourceProof({}).callback(mutated)
        assert not (checked.proved)
        assert ("history") in (checked.reads)


def test_shared_structured_binding_is_interpreted_once_as_original_data() -> None:
    original = shared_table_caller()
    captured = source_function(original)
    table_refs = [
        row["bindings"]["table"]["capture_ref"]
        for row in captured["callable_records"].values()
        if row.get("qualname", "").endswith("table_callback.<locals>.lookup")
    ]
    assert (len(table_refs)) == (2)
    assert (len(set(table_refs))) == (1)
    expression = ScalarSource(captured).expression()
    for history in (-2, 0, 7):
        checked = SourceProof({}, static={"history": history}).callback(captured)
        assert checked.proved
        assert (checked.value) == (original(history))
        assert (partial(expression, {"history": history}).value) == (checked.value)


def test_default_binding_keeps_tuple_identity() -> None:
    captured = source_function(tuple_default_caller)
    expression = ScalarSource(captured).expression()
    for history in (-2, 0, 7):
        checked = SourceProof({}, static={"history": history}).callback(captured)
        assert checked.proved
        assert (checked.value) == (history)
        assert (partial(expression, {"history": history}).value) == (history)


def test_shared_binding_payload_identity_and_status_mutations_fail_closed() -> None:
    captured = source_function(shared_table_caller())
    for field, value in (
        ("value", [0]),
        ("identity_sha256", "changed"),
        ("capture_status", "UNRESOLVED_MUTATION"),
    ):
        mutated = deepcopy(captured)
        binding = next(
            row
            for row in mutated["callable_records"].values()
            if row.get("kind") == "binding-record"
        )
        binding[field] = value
        checked = SourceProof({}).callback(mutated)
        assert not (checked.proved)
        assert ("history") in (checked.reads)


def test_shared_binding_cycle_is_unresolved_without_expansion() -> None:
    payload = []
    payload.append(payload)
    captured = source_function(binding_callback(payload))
    checked = SourceProof({}).callback(captured)
    assert not (checked.proved)
    assert ("history") in (checked.reads)
    with pytest.raises(DeclarationError):
        ScalarSource(captured).expression()


def test_compact_data_dag_cannot_force_exponential_literal_expansion() -> None:
    payload = [1]
    for _ in range(40):
        payload = [payload, payload]
    captured = source_function(binding_callback(payload))
    assert (len(captured["callable_records"])) < (50)
    checked = SourceProof({}).callback(captured)
    assert not (checked.proved)
    assert ("history") in (checked.reads)
    with pytest.raises(DeclarationError):
        ScalarSource(captured).expression()


def test_callable_binding_is_not_certified_as_an_immutable_value() -> None:
    captured = source_function(binding_callback(known_helper))
    checked = SourceProof({}).callback(captured)
    assert not (checked.proved)
    assert ("history") in (checked.reads)


def test_bounded_interpretation_keeps_actual_input_at_any_helper_depth() -> None:
    for depth in (1, 8, 40):
        inner = metadata("def leaf(x):\n    return x\n", ["x"])
        for _ in range(depth):
            inner = metadata(
                "def helper(x):\n    return child(x)\n",
                ["x"],
                call_targets={"child": inner},
            )
        outer = metadata(
            "def consumer(history):\n    return helper(history)\n",
            ["history"],
            call_targets={"helper": inner},
        )
        answer = SourceProof({}).callback(outer)
        assert ("history") in (answer.reads)
        if depth == 40:
            assert not (answer.proved)


def test_unknown_context_has_concrete_unadopted_merge_question() -> None:
    source = native_fixture()
    for regime in source["user_regimes"].values():
        regime["fields"].pop("semantic_role")
    report = find_regimes(source)
    boundary = report["analysis"]["family_boundary"]
    assert not (boundary["unrestricted_model_representation_optimum_claimed"])
    assert not (boundary["endpoint_preference_is_admissibility_rule"])
    candidates = boundary["conditional_merge_candidates"]
    assert ([row["regimes"] for row in candidates]) == ([["opaque", "safe"]])
    assert (candidates[0]["status"]) == ("CONDITIONAL_NOT_ADMITTED")
    assert not (candidates[0]["automatic_graph_change"])
    assert (len(candidates[0]["source_paths"])) == (2)
    assert candidates[0]["required_obligations"]
    assert any(
        row["disposition"] == "adopt" and row["changed_interfaces"]
        for row in report["proposal_decisions"]
    )


def test_conditional_display_budget_does_not_silently_exhaust_search_family() -> None:
    source = native_fixture()
    for regime in source["user_regimes"].values():
        regime["fields"].pop("semantic_role")
    analysis = MetadataAnalysis(
        normalize_declarations(source), {"conditional_merge_pair_checks": 0}
    ).run()
    assert analysis["discovery_complete"]
    assert (analysis["family_boundary"]["conditional_merge_pairs_unexamined"]) == (1)
    assert (analysis["family_boundary"]["conditional_merge_candidates"]) == ([])
