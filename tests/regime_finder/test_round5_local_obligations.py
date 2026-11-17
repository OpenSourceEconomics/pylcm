"""Public native normalization must preserve locality of interpretation gaps."""

from collections.abc import Callable, Sequence
from functools import partial

from _lcm_regime_finder import find_regimes as _find_regimes
from _lcm_regime_finder.records import JSONValue

# These regressions exercise the optional stronger semantic audit, not the
# Model-graph contract whose primitives retain their declared dependencies.
find_regimes = partial(_find_regimes, analysis_scope="full_semantics")
from _lcm_regime_finder.model_snapshot import source_function
from .test_declaration_ingestion import json_rows
from .test_model_adapter import grid


def opaque_utility(x: int, history: int) -> int:
    result = x
    for weight in (1, 2):
        result = result + weight * history
    return result


def safe_utility(x: int) -> int:
    return x


def native_fixture(
    *, callback: Callable[..., float] = opaque_utility, values: Sequence[int] = (0, 1)
) -> dict[str, JSONValue]:
    """Two independently admitted original regimes, no reduced answer supplied."""
    regimes = {}
    for name, function in (("opaque", callback), ("safe", safe_utility)):
        regimes[name] = {
            "kind": "dataclass",
            "type": "lcm.Regime",
            "fields": {
                "states": {"x": grid(list(values)), "history": grid(list(values))},
                "actions": {},
                "functions": {"utility": source_function(function)},
                "constraints": {},
                "state_transitions": {},
                "semantic_role": {
                    "actor": name,
                    "information": ["x", "history"],
                    "commitments": [],
                    "discount_owner": name,
                },
            },
        }
    phase = {"kind": "dataclass", "fields": {"targets_by_period": []}}
    nodes = [[0, name] for name in regimes]
    reachability = {
        "kind": "dataclass",
        "fields": {"nodes": nodes, "solution": phase, "simulation": phase},
    }
    return {
        "schema": "constructed-pylcm-model-graph-1",
        "n_periods": 1,
        "ages": None,
        "user_regimes": regimes,
        "reachability": reachability,
        "graph": {
            "kind": "dataclass",
            "fields": {
                "coordinate_kind": "period",
                "reachability": reachability,
                "initial_nodes": nodes,
                "laws": {},
            },
        },
        "semantic_contract": {
            "allocation": {"kind": "per_block", "status": "declared"},
            "production_metadata": {
                "status": "explicit-small-source-regression",
                "native_model_constructed": False,
            },
        },
    }


def _produce(
    source: dict[str, JSONValue] | None = None,
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    source = native_fixture() if source is None else source
    report = find_regimes(source, facts_backend="metadata")
    assert not (report["semantic_check"]["errors"]), report["semantic_check"]
    return source, report


def test_actual_capture_gap_is_local_on_public_native_path() -> None:
    _, report = _produce()
    obligations = json_rows(report, "analysis", "obligations")
    assert ([o["kind"] for o in obligations]) == (["callback_interpretation"])
    assert (report["status"]) == ("UNRESOLVED")
    (obligation,) = obligations
    source_path = obligation["source_path"]
    assert isinstance(source_path, str), obligation
    assert source_path.startswith("/user_regimes/opaque/"), obligation
