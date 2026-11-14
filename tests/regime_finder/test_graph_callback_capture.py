"""The topology checkpoint is partial; the final graph must keep its callbacks."""

from dataclasses import dataclass
from collections.abc import Callable
from types import SimpleNamespace
from _lcm_regime_finder.model_snapshot import read_model


def gate(divorce_shock):
    raise AssertionError("An exporter must never execute the gate")


@dataclass
class Graph:
    gate: Callable[..., None]


def test_final_graph_expands_callbacks_without_changing_topology_checkpoint() -> None:
    model = SimpleNamespace(
        description="capture regression",
        n_periods=1,
        user_regimes={},
        fixed_params={},
        initial_nodes={},
        regime_names_to_ids={},
        graph=Graph(gate),
    )
    checkpoints = []
    source = read_model(model, checkpoint=checkpoints.append)
    partial = checkpoints[0]["snapshot"]["graph"]["fields"]["gate"]
    assert (partial["capture_status"]) == ("INCOMPLETE")
    callback = source["graph"]["fields"]["gate"]
    if callback.get("kind") == "callable-reference":
        callback = source["callable_records"][callback["capture_ref"]]
    assert (callback["arguments"]) == (["divorce_shock"])
    assert (callback["capture_status"]) == ("COMPLETE")
    assert (partial["arguments"]) == ([])
    assert checkpoints[0]["topology_complete"]


def test_old_checkpoint_signatures_stay_unresolved_and_cannot_be_forged_complete() -> (
    None
):
    from copy import deepcopy
    from _lcm_regime_finder import find_regimes, verify_production_result
    from tests.regime_finder.test_metadata_independent import captured_snapshot

    source = captured_snapshot()
    source["user_regimes"]["life"]["fields"]["functions"]["utility"] = {
        "kind": "callable",
        "arguments": [],
        "capture_status": "INCOMPLETE",
        "source_unavailable": True,
        "variadic": True,
        "obligations": ["Original graph checkpoint omits optional callback expansion"],
    }
    result = find_regimes(source, budget={"seconds": 10}, search_mode="exact")
    assert (result["status"]) == ("UNRESOLVED")
    assert any(
        o["kind"] == "graph_callback_signature"
        for o in result["analysis"]["obligations"]
    )
    assert all(
        "history" in row["states"]
        for interface in result["analysis"]["interfaces"].values()
        for row in interface["options"]
    )
    assert (verify_production_result(source, result)["status"]) == ("UNRESOLVED")
    forged = deepcopy(result)
    forged["analysis"]["obligations"] = []
    for callback in forged["analysis"]["declarations"]["callbacks"].values():
        callback.pop("unresolved_graph_signature", None)
        callback["dependency_contract"] = "pylcm-declared-inputs/v1"
    assert (verify_production_result(source, forged)["status"]) != ("PASS")
