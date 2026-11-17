"""Known generated type-check wrappers retain a verified finite interface."""

from copy import deepcopy
import functools
import inspect
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Literal, cast

from beartype import beartype, BeartypeConf, BeartypeStrategy
import pytest
from lcm.declarations import FunctionDeclaration
from lcm.typing import UserFunction
from _lcm_regime_finder import find_regimes, verify_production_result
from _lcm_regime_finder.callable_identity import (
    _beartype_interface,
    capture_session,
    resolve_callable,
)
from _lcm_regime_finder.model_snapshot import source_function
from _lcm_regime_finder.records import JSONValue
from _lcm_regime_finder.semantic_check import SourceProof
from tests.regime_finder.test_metadata_independent import captured_snapshot


def typed_history(history: int) -> int:
    raise AssertionError("Export must never evaluate an economic callback")


def typed_history_value(*, history: int) -> int:
    return history + 1


CONF = BeartypeConf(strategy=BeartypeStrategy.On, is_pep484_tower=True)


def _declared_capture(
    function: UserFunction,
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    with capture_session() as session:
        session.declaration_interfaces = MappingProxyType(
            {
                id(function): FunctionDeclaration(
                    function=function, signature=inspect.signature(function)
                )
            }
        )
        record = source_function(function)
    return record, session.records


def test_native_model_export_keeps_beartyped_utility_and_default_aggregator() -> None:
    from lcm import Model, Regime, DiscreteGrid, AgeGrid, categorical
    from lcm.typing import ScalarInt
    from _lcm_regime_finder.pytask_api import export_model
    from _lcm_regime_finder.cli import load_problem

    @categorical(ordered=False)
    class Binary:
        low: ScalarInt
        high: ScalarInt

    @categorical(ordered=False)
    class Regimes:
        alive: ScalarInt
        final: ScalarInt

    def terminal() -> float:
        return 0.0

    def next_history(history: int) -> int:
        return history

    model = Model(
        regimes={
            "alive": Regime(
                states={"history": DiscreteGrid(Binary)},
                functions={"utility": beartype(conf=CONF)(typed_history)},
                state_transitions={"history": next_history},
            ),
            "final": Regime(functions={"utility": beartype(conf=CONF)(terminal)}),
        },
        regime_id_class=Regimes,
        ages=AgeGrid(start=20, inclusive_stop=21, step="Y"),
        edges={"alive": {"final": (20,)}},
        initial_nodes={20: "alive"},
        enable_jit=False,
    )
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "model.json"
        export_model(
            model,
            path,
            production_metadata={"status": "regression"},
            seconds=None,
            host_gib=None,
            source_transaction={
                "kind": "exclusive-owner",
                "authority": "This test owns the newly constructed Model",
            },
        )
        original = load_problem(path)
        result = find_regimes(
            original,
            budget={"seconds": 30},
            facts_backend="metadata",
        )
        assert (result["status"]) == ("COMPLETE"), result["analysis"]["obligations"]
        assert (verify_production_result(original, result)["status"]) == ("PASS")


def test_generated_wrapper_is_reconciled_without_mutating_or_calling_target() -> None:
    wrapper = beartype(conf=CONF)(typed_history)
    before = dict(vars(typed_history))
    actual = source_function(wrapper)
    assert (vars(typed_history)) == (before)
    assert (_beartype_interface(wrapper)) is not None
    assert actual["variadic"]
    assert ("kwargs") in (actual["raw_arguments"])
    assert (actual["arguments"]) == (["history"])
    assert (actual["parameter_kinds"]) == ({"history": "POSITIONAL_OR_KEYWORD"})
    assert (actual["advertised_signature_origin"]) == (
        "regenerated-beartype-wrapper/v1"
    )
    assert ("__beartype_func") in (actual["defaults"])
    source = captured_snapshot(actual)
    result = find_regimes(source, budget={"seconds": 20}, facts_backend="metadata")
    record = result["analysis"]["declarations"]["callbacks"][
        result["analysis"]["declarations"]["cells"][0]["functions"]["utility"]
    ]
    assert not (record.get("unresolved_input_signature", False))
    assert all(
        "history" in option["states"]
        for item in result["analysis"]["interfaces"].values()
        for option in item["options"]
    )
    assert (verify_production_result(source, result)["status"]) == ("PASS")
    declared, records = _declared_capture(wrapper)
    target = resolve_callable(declared.get("wrapped_target", {}), records)
    assert (
        declared["arguments"],
        declared["parameter_kinds"],
        declared["advertised_signature_origin"],
        declared.get("wrapped_target_contract"),
        target.get("qualname"),
    ) == (
        ["history"],
        {"history": "POSITIONAL_OR_KEYWORD"},
        "pylcm-model-declarations/v1",
        "regenerated-beartype-wrapper/v1",
        typed_history.__qualname__,
    )


def test_marker_alone_and_changed_wrapper_cannot_claim_a_finite_interface() -> None:
    @functools.wraps(typed_history)
    def forged(*args: int, **kwargs: int) -> int:
        return kwargs["other_history"]

    setattr(forged, "__beartype_wrapper", True)
    assert (_beartype_interface(forged)) is None
    original = beartype(conf=CONF)(typed_history)
    changed = functools.wraps(original)(lambda *args, **kwargs: kwargs["other_history"])
    changed.__wrapped__ = typed_history
    setattr(changed, "__kwdefaults__", dict(getattr(original, "__kwdefaults__")))
    assert (_beartype_interface(changed)) is None
    altered = beartype(conf=CONF)(typed_history)
    defaults = dict(getattr(altered, "__kwdefaults__") or {})
    defaults["__beartype_func"] = lambda history: 42
    setattr(altered, "__kwdefaults__", defaults)
    assert (_beartype_interface(altered)) is None
    for untrusted in (forged, changed, altered):
        declared, _ = _declared_capture(untrusted)
        assert "wrapped_target_contract" not in declared


def test_declared_generated_wrapper_preserves_verified_body_reads() -> None:
    """Authenticate a finite typed body without evaluating its wrapped function."""
    wrapper = beartype(conf=CONF)(typed_history_value)
    original, records = _declared_capture(wrapper)
    answer = SourceProof(
        {"utility": {"original": original}}, callable_records=records
    ).callback("utility")
    assert (answer.proved, answer.reads) == (True, frozenset({"history"}))


@pytest.mark.parametrize("field", ("contract", "default"))
def test_declared_generated_wrapper_rejects_changed_transport(
    field: Literal["contract", "default"],
) -> None:
    """A changed transport marker or target default cannot prove fewer reads."""
    wrapper = beartype(conf=CONF)(typed_history_value)
    original, records = _declared_capture(wrapper)
    changed = deepcopy(original)
    if field == "contract":
        changed["wrapped_target_contract"] = "unverified"
    else:
        defaults = cast("dict[str, JSONValue]", changed["defaults"])
        defaults["__beartype_func"] = {}
    answer = SourceProof(
        {"utility": {"original": changed}}, callable_records=records
    ).callback("utility")
    assert not answer.proved


def test_original_payload_still_controls_independent_signature_authority() -> None:
    from tests.regime_finder.test_declaration_ingestion import decorated_callback

    source = captured_snapshot(source_function(decorated_callback))
    result = find_regimes(source, budget={"seconds": 20}, facts_backend="metadata")
    forged = deepcopy(result)
    for record in forged["analysis"]["declarations"]["callbacks"].values():
        record["arguments"] = ["history"]
        record["parameter_kinds"] = {"history": "POSITIONAL_OR_KEYWORD"}
        record.pop("unresolved_input_signature", None)
        record["status"] = "structural"
    assert (verify_production_result(source, forged)["status"]) != ("PASS")


def test_a_beartype_wrapper_identity_holds_no_memory_address() -> None:
    """beartype names a parameter and the wrapper's code file by memory address.

    Those differ between processes, so recording them changed every derived id.
    """
    import json
    import re

    from lcm.koopmans_aggregation import LinearAggregator

    wrapper = type(LinearAggregator()).__call__
    addresses = [
        name.removeprefix("__beartype_object_")
        for name in wrapper.__kwdefaults__ or {}
        if name.startswith("__beartype_object_")
    ]
    match = re.search(r" at 0x([0-9a-f]+)>$", wrapper.__code__.co_filename)
    assert addresses and match, "positive control: the wrapper carries addresses"
    with capture_session() as session:
        source_function(wrapper)
    text = json.dumps(session.records, default=str)
    for address in (*addresses, match.group(1)):
        assert address not in text
