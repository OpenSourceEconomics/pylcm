"""Native metadata differential tests: delivered, deliberately NOT RUN in audit.

These add a new compressed-versus-literal guard. Keep existing fibre-source,
fibre-facts and source-authentication tests unchanged. No Model is constructed.
"""

from collections.abc import Mapping, Sequence
from _lcm_regime_finder.records import JSONValue
from _lcm_regime_finder.fibre_source import FibreSource, Wire, _mapping
from itertools import product, permutations
import json

import pytest


def fixture_map(value: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(value, dict)
    return value


from _lcm_regime_finder.implicit.implicit_native import CellCompiler
from _lcm_regime_finder.implicit.verify_native import check_graph, native_read, same
from _lcm_regime_finder.fibre_source import _text, _strings
from tests.regime_finder.test_model_adapter import function, grid
from tests.regime_finder.test_fibre_source import (
    _original,
    _fields,
    _stored_category_source,
)


def compare(original: Wire, declared: Mapping[str, Sequence[str]]) -> None:
    producer = FibreSource(original)
    literal = FibreSource(original)
    count = 0
    for cell_id in literal.get_cell_ids():
        cell = literal._cells[cell_id]
        states = _mapping(cell["states"])
        actions_map = _mapping(cell["actions"])
        compiler = CellCompiler(
            source=producer,
            cell_id=cell_id,
            status_axes=declared[_text(cell["regime"])],
        )
        row = compiler.compile(status_axes=declared)
        domains = tuple(
            (
                name,
                tuple(literal._category(literal._domains[_text(states[name])]).by_code),
            )
            for name in compiler.axes
        )
        expected_keys = {"direct"}
        actions = [
            name
            for name, identity in actions_map.items()
            if literal._domains[_text(identity)].get("kind") == "categorical"
        ]
        expected_keys.update("exclude:" + name for name in actions)
        outgoing = [
            (rid, relation)
            for rid, relation in literal._relations.items()
            if relation["source"] == cell_id
            and relation["kind"]
            in {"transition", "joint", "value_read", "fallback", "exit"}
            and relation["targets"]
        ]
        for rid, relation in outgoing:
            target = literal._cells[_strings(relation["targets"])[0]]
            expected_keys.update(
                "read:" + rid + "::" + coordinate
                for coordinate in _mapping(target["states"])
            )
            expected_keys.update(
                "support:" + rid + "::" + coordinate
                for coordinate in declared[_text(target["regime"])]
            )
        lookup = check_graph(_mapping(row["diagram"]), domains, expected_keys)
        for codes in product(*(codes for _, codes in domains)):
            statuses = dict(zip(compiler.axes, codes))
            assert same(
                lookup("direct", codes),
                native_read(
                    literal.get_direct_reads(
                        cell_id=cell_id, statuses=statuses, removal_order=compiler.axes
                    )
                ),
            )
            for action in actions:
                assert same(
                    lookup("exclude:" + action, codes),
                    literal.get_excluded_action_values(
                        cell_id=cell_id,
                        action=action,
                        status_values={
                            name: (code,) for name, code in statuses.items()
                        },
                    ),
                )
            for rid, relation in outgoing:
                target = literal._cells[_strings(relation["targets"])[0]]
                for coordinate in _mapping(target["states"]):
                    assert same(
                        lookup("read:" + rid + "::" + coordinate, codes),
                        native_read(
                            literal.get_update_reads(
                                relation_id=rid,
                                target_coordinate=coordinate,
                                statuses=statuses,
                                removal_order=compiler.axes,
                            )
                        ),
                    )
                for coordinate in declared[_text(target["regime"])]:
                    support = literal.get_target_support(
                        relation_id=rid, target_coordinate=coordinate, statuses=statuses
                    )
                    assert same(
                        lookup("support:" + rid + "::" + coordinate, codes),
                        ("support", support.possible_codes, support.proved),
                    )
            count += 1
    assert count > 0


@pytest.mark.parametrize("order", list(permutations(("a", "b", "c"))))
def test_joint_majority_exact_original_removal_order(order: Sequence[str]) -> None:
    original = _original()
    fields = _fields(original)
    fields["states"] = {name: grid([0, 1]) for name in order}
    fields["functions"] = {
        "utility": function(
            "def utility(a, b, c):\n    return (a and b) or (a and c) or (b and c)\n",
            ["a", "b", "c"],
        )
    }
    fields["state_transitions"] = {
        name: function("def update(" + name + "):\n    return " + name + "\n", [name])
        for name in order
    }
    compare(original, {"original": order})


@pytest.mark.parametrize("stochastic", [False, True])
def test_deterministic_zero_and_stochastic_zero_stay_distinct(stochastic: bool) -> None:
    original = _original()
    callback = function("def next_x():\n    return 0\n", [])
    fixture_map(_fields(original)["state_transitions"])["x"] = (
        {
            "kind": "dataclass",
            "type": "lcm.StochasticTransition",
            "fields": {"func": callback},
        }
        if stochastic
        else callback
    )
    compare(original, {"original": ("x", "history")})


def test_full_hull_exclusions_cover_unkept_fibres() -> None:
    original = _original()
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(choice, x, history):\n    return (choice != 0) | (x != history)\n",
            ["choice", "x", "history"],
        )
    }
    compare(original, {"original": ("x", "history")})


@pytest.mark.parametrize("storage", ["sequence", "native", "strided", "reversed"])
@pytest.mark.parametrize("duplicates", [False, True])
def test_stored_payload_and_duplicate_nominal_codes(
    storage: str, duplicates: bool
) -> None:
    original, reader = _stored_category_source(
        storage=storage, duplicate_values=duplicates
    )
    fixture_map(_fields(original)["functions"])["utility"] = function(
        "def utility(history_code, x):\n    return 0 if history_code == 11 else x\n",
        ["history_code", "x"],
    )
    compare(original, {"original": ("history",)})
    assert reader.doc["status"] == "COMPLETE"


@pytest.mark.parametrize("simulation_values", [[0], [0, 1]])
def test_phased_imputation_keeps_original_context(
    simulation_values: Sequence[int],
) -> None:
    original = _original()
    fields = _fields(original)
    fixture_map(fields["states"])["x"] = {
        "kind": "dataclass",
        "type": "lcm.Phased",
        "fields": {
            "solve": function(
                "def imputed(history):\n    return history\n", ["history"]
            ),
            "simulate": grid(simulation_values),
        },
    }
    fixture_map(fields["functions"])["utility"] = function(
        "def utility(x):\n    return x\n", ["x"]
    )
    # A singleton realised x with a nonconstant solve imputation is not promoted
    # to a target-support status in this control: the baseline correctly rejects
    # a deterministic imputation image outside that singleton target grid.
    selected = ("history", "x") if len(simulation_values) > 1 else ("history",)
    compare(original, {"original": selected})


@pytest.mark.parametrize("registered", [False, True])
def test_registered_conditional_law_inputs_remain_source_bound(
    registered: bool,
) -> None:
    original = _original()
    _fields(original)["shocks"] = ["offers"] if registered else []
    original["shocks"] = {
        "offers": {
            "kind": "dataclass",
            "type": "lcm.shocks.ShockGroup",
            "fields": {
                "shocks": {
                    "offer": {
                        "kind": "dataclass",
                        "type": "lcm.shocks.Discrete",
                        "fields": {
                            "probs": function(
                                "def probs(history):\n    return history\n", ["history"]
                            )
                        },
                    }
                }
            },
        }
    }
    compare(original, {"original": ("history",)})


def test_joint_and_ordinary_actual_producers_not_identity() -> None:
    original = _original()
    fields = _fields(original)
    fixture_map(fields["state_transitions"]).pop("x")
    fields["joint_transitions"] = {
        "original": {
            "lottery": {
                "kind": "dataclass",
                "type": "lcm.JointTransition",
                "fields": {
                    "support_size": 2,
                    "support": [0, 1],
                    "probabilities": function(
                        "def probabilities(history):\n    return [history, 1-history]\n",
                        ["history"],
                    ),
                    "outputs": {
                        "x": function(
                            "def draw(history, lottery):\n    return history\n",
                            ["history", "lottery"],
                        )
                    },
                },
            }
        }
    }
    compare(original, {"original": ("x", "history")})


@pytest.mark.parametrize("slot", ["missing", "opaque"])
def test_original_conservative_missing_and_unsupported_cases(slot: str) -> None:
    original = _original()
    if slot == "missing":
        fixture_map(_fields(original)["state_transitions"]).pop("x")
    else:
        fixture_map(_fields(original)["functions"])["utility"] = function(
            "def utility(x):\n    return unknown_operation(x)\n", ["x"]
        )
    compare(original, {"original": ("x", "history")})


def test_json_roundtrip_preserves_capture_codes_and_order() -> None:
    original = _original()
    fixture_map(_fields(original)["states"])["history"] = grid(
        [2.5, 2.5, 9.5], codes=[7, 11, 19]
    )
    fixture_map(_fields(original)["functions"])["utility"] = function(
        "def utility(history_code, x):\n    return x if history_code == 7 else 0\n",
        ["history_code", "x"],
    )
    compare(original, {"original": ("history",)})
    compare(json.loads(json.dumps(original)), {"original": ("history",)})


def test_original_authentication_rejects_competing_producers() -> None:
    original = _original()
    _fields(original)["joint_transitions"] = {
        "original": {
            "lottery": {
                "kind": "dataclass",
                "type": "lcm.JointTransition",
                "fields": {
                    "support_size": 2,
                    "support": [0, 1],
                    "probabilities": function(
                        "def probabilities():\n    return [0.5, 0.5]\n", []
                    ),
                    "outputs": {
                        "x": function(
                            "def draw(history, lottery):\n    return history\n",
                            ["history", "lottery"],
                        )
                    },
                },
            }
        }
    }
    fixture_map(fixture_map(_fields(original)["joint_transitions"])["original"])[
        "second_lottery"
    ] = {
        "kind": "dataclass",
        "type": "lcm.JointTransition",
        "fields": {
            "support_size": 2,
            "support": [0, 1],
            "probabilities": function(
                "def probabilities():\n    return [0.5, 0.5]\n", []
            ),
            "outputs": {
                "x": function(
                    "def draw_second(history, second_lottery):\n    return history\n",
                    ["history", "second_lottery"],
                )
            },
        },
    }
    with pytest.raises(ValueError, match="original joint producer output duplicated"):
        FibreSource(original)


def test_joint_producer_supersedes_ordinary_fallback() -> None:
    original = _original()
    _fields(original)["joint_transitions"] = {
        "original": {
            "lottery": {
                "kind": "dataclass",
                "type": "lcm.JointTransition",
                "fields": {
                    "support_size": 2,
                    "support": [0, 1],
                    "probabilities": function(
                        "def probabilities():\n    return [0.5, 0.5]\n", []
                    ),
                    "outputs": {
                        "x": function(
                            "def draw(history, lottery):\n    return history\n",
                            ["history", "lottery"],
                        )
                    },
                },
            }
        }
    }
    compare(original, {"original": ("x", "history")})


def test_pinned_cache_order_boundary_and_exact_streaming_fallback() -> None:
    from tests.regime_finder.implicit_cache_witness import (
        native_fixture,
        native_observations,
    )

    cold, warming, warm = native_observations()
    # These are the pinned OLD reference semantics, not the proposed optional
    # cache-height repair's semantics. The exact compiler must preserve them.
    assert cold == frozenset({"x", "history"})
    assert warming == warm == frozenset()
    compare(native_fixture(), {"original": ("history",)})
