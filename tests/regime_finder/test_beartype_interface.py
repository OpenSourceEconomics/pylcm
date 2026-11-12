"""Known generated type-check wrappers retain a verified finite interface."""

from copy import deepcopy
import functools
from pathlib import Path
import tempfile

from beartype import beartype, BeartypeConf, BeartypeStrategy
from _lcm_regime_finder import find_regimes, verify_production_result
from _lcm_regime_finder.callable_identity import _beartype_interface
from _lcm_regime_finder.model_snapshot import source_function
from tests.regime_finder.test_metadata_independent import captured_snapshot


def typed_history(history: int) -> int:
    raise AssertionError("Export must never evaluate an economic callback")


CONF = BeartypeConf(strategy=BeartypeStrategy.On, is_pep484_tower=True)


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

    def next_history(history):
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
        result = find_regimes(original, budget={"seconds": 30}, search_mode="exact")
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
    result = find_regimes(source, budget={"seconds": 20}, search_mode="exact")
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


def test_marker_alone_and_changed_wrapper_cannot_claim_a_finite_interface() -> None:
    @functools.wraps(typed_history)
    def forged(*args, **kwargs):
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


def test_original_payload_still_controls_independent_signature_authority() -> None:
    from tests.regime_finder.test_declaration_ingestion import decorated_callback

    source = captured_snapshot(source_function(decorated_callback))
    result = find_regimes(source, budget={"seconds": 20}, search_mode="exact")
    forged = deepcopy(result)
    for record in forged["analysis"]["declarations"]["callbacks"].values():
        record["arguments"] = ["history"]
        record["parameter_kinds"] = {"history": "POSITIONAL_OR_KEYWORD"}
        record.pop("unresolved_input_signature", None)
        record["status"] = "structural"
    assert (verify_production_result(source, forged)["status"]) != ("PASS")
