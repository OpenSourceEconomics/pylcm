"""Public native normalization must preserve locality of interpretation gaps."""

from _lcm_regime_finder import find_regimes as _find_regimes
from functools import partial

# These regressions exercise the optional stronger semantic audit, not the
# Model-graph contract whose primitives retain their declared dependencies.
find_regimes = partial(_find_regimes, analysis_scope="full_semantics")
from _lcm_regime_finder.model_snapshot import source_function
from .test_model_adapter import grid


def opaque_utility(x, history):
    result = x
    for weight in (1, 2):
        result = result + weight * history
    return result


def safe_utility(x):
    return x


def variadic_utility(x, history, **extra):
    result = x
    for weight in (1, 2):
        result = result + weight * history
    return result


def native_fixture(*, callback=opaque_utility, values=(0, 1)):
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


def _produce(source=None):
    source = native_fixture() if source is None else source
    report = find_regimes(source, facts_backend="metadata")
    assert not (report["semantic_check"]["errors"]), report["semantic_check"]
    return source, report


def test_actual_capture_gap_is_local_on_public_native_path() -> None:
    _, report = _produce()
    obligations = report["analysis"]["obligations"]
    assert ([o["kind"] for o in obligations]) == (["callback_interpretation"])
    assert (report["status"]) == ("UNRESOLVED")
    (obligation,) = obligations
    assert obligation["source_path"].startswith("/user_regimes/opaque/"), obligation
