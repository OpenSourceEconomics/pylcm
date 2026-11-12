"""Original graph fixtures, with no reduced graph/cut supplied to the adapter."""

from copy import deepcopy

from _lcm_regime_finder import find_regimes, verify_certificate
from _lcm_regime_finder.model_adapter import adapt_snapshot


def function(source, arguments, bindings=None):
    return {
        "kind": "callable",
        "source": source,
        "arguments": arguments,
        "bindings": bindings or {},
        "source_file": "original.py",
        "source_sha256": "original-fixture",
        "source_line": 1,
    }


def grid(values, codes=None):
    return {
        "kind": "object",
        "type": "lcm.DiscreteGrid",
        "fields": {
            "values": values,
            "codes": codes or list(range(len(values))),
            "categories": [str(x) for x in values],
        },
    }


def snapshot():
    source = function("def utility(x, history):\n    return x\n", ["x", "history"])
    constraint = function(
        "def allowed(choice, age):\n    return (choice == 0) | (age < CUTOFF)\n",
        ["choice", "age"],
        {"CUTOFF": 2},
    )
    regime = {
        "kind": "dataclass",
        "type": "lcm.Regime",
        "fields": {
            "states": {"x": grid([0, 1]), "history": grid([0, 1])},
            "actions": {"choice": grid([0, 1])},
            "functions": {"utility": source},
            "constraints": {"allowed": constraint},
            "state_transitions": {
                "x": function("def next_x(x):\n    return x\n", ["x"]),
                "history": function(
                    "def next_history(history):\n    return history\n", ["history"]
                ),
            },
        },
    }
    nodes = [[i, "original"] for i in range(6)]
    phase = {
        "kind": "dataclass",
        "fields": {"targets_by_period": [{"original": ["original"]} for _ in range(5)]},
    }
    reachability = {
        "kind": "dataclass",
        "fields": {"nodes": nodes, "solution": phase, "simulation": phase},
    }
    return {
        "schema": "constructed-pylcm-model-graph-1",
        "n_periods": 6,
        "ages": None,
        "user_regimes": {"original": regime},
        "reachability": reachability,
        "graph": {
            "kind": "dataclass",
            "fields": {
                "reachability": reachability,
                "initial_nodes": [[0, "original"]],
                "laws": {},
            },
        },
    }


def _check_analysis(source):
    problem = adapt_snapshot(source)
    result = find_regimes(problem)
    assert (result["status"]) == ("OPTIMAL"), result.get("error")
    checked = verify_certificate(problem, result)
    assert (checked["status"]) == ("PASS"), checked
    return problem, result


def test_original_constraint_discovers_cut_and_unused_state() -> None:
    problem, result = _check_analysis(snapshot())
    atoms = result["analysis"]["atoms"]
    assert all("history" not in atom["live_states"] for atom in atoms)
    assert ([a["age"] for a in atoms if a["forced_actions"]]) == ([2, 3, 4, 5])
    mutated = snapshot()
    mutated["user_regimes"]["original"]["fields"]["constraints"]["allowed"]["bindings"][
        "CUTOFF"
    ] = 3
    _, changed = _check_analysis(mutated)
    assert (
        [a["age"] for a in changed["analysis"]["atoms"] if a["forced_actions"]]
    ) == ([3, 4, 5])


def test_numeric_value_grid_code_is_not_its_value() -> None:
    source = snapshot()
    declaration = source["user_regimes"]["original"]["fields"]
    declaration["actions"]["choice"] = grid([10.0, 20.0])
    declaration["constraints"] = {
        "code": function(
            "def allowed(choice_code):\n    return choice_code == 0\n", ["choice_code"]
        )
    }
    _, result = _check_analysis(source)
    assert all(
        a["forced_actions"] == {"choice": 10.0} for a in result["analysis"]["atoms"]
    )


def test_period_is_index_and_age_is_physical_age() -> None:
    source = snapshot()
    source["ages"] = {
        "kind": "object",
        "fields": {"exact_values": [20, 21, 22, 23, 24, 25]},
    }
    nodes = [[20 + i, "original"] for i in range(6)]
    source["reachability"]["fields"]["nodes"] = nodes
    source["graph"]["fields"]["initial_nodes"] = [[20, "original"]]
    source["user_regimes"]["original"]["fields"]["constraints"] = {
        "index": function(
            "def allowed(choice, period):\n    return (choice == 0) | (period < 2)\n",
            ["choice", "period"],
        )
    }
    _, result = _check_analysis(source)
    assert ([a["age"] for a in result["analysis"]["atoms"] if a["forced_actions"]]) == (
        [22, 23, 24, 25]
    )


def test_new_original_consumer_prevents_state_deletion() -> None:
    source = snapshot()
    source["user_regimes"]["original"]["fields"]["functions"]["utility"] = function(
        "def utility(x, history):\n    return x + history\n", ["x", "history"]
    )
    _, result = _check_analysis(source)
    assert all("history" in a["live_states"] for a in result["analysis"]["atoms"])


def test_custom_aggregator_dependency_is_not_lost() -> None:
    source = snapshot()
    source["user_regimes"]["original"]["fields"]["koopmans_aggregator"] = function(
        "def aggregate(utility, CE, history):\n    return utility + CE + history\n",
        ["utility", "CE", "history"],
    )
    _, result = _check_analysis(source)
    assert all("history" in a["live_states"] for a in result["analysis"]["atoms"])


def test_unresolved_specialized_callback_retains_inputs() -> None:
    source = snapshot()
    source["user_regimes"]["original"]["fields"]["functions"]["utility"] = {
        "kind": "dataclass",
        "type": "lcm.AgeSpecializedFunction",
        "fields": {
            "build": function(
                "def build(age):\n    return make_function(age)\n", ["age"]
            )
        },
    }
    _, result = _check_analysis(source)
    assert all("history" in a["live_states"] for a in result["analysis"]["atoms"])


def test_dispatched_law_fingerprint_survives_fresh_adaptation() -> None:
    source = snapshot()
    source["graph"]["fields"]["laws"] = {
        "original": {
            "kind": "dataclass",
            "fields": {
                "transition": {
                    "kind": "dataclass",
                    "type": "engine._PeriodDispatch",
                    "fields": {
                        "case_by_period": [0, 0, 0, 0, 0],
                        "cases": [
                            {
                                "kind": "dataclass",
                                "type": "engine._RegimeCodeConstant",
                                "fields": {"value": 0},
                            }
                        ],
                    },
                }
            },
        }
    }
    assert (adapt_snapshot(source)) == (adapt_snapshot(deepcopy(source)))


def _create_alternating():
    source = snapshot()
    original = source["user_regimes"]["original"]
    original["fields"]["actions"] = {}
    original["fields"]["constraints"] = {}
    source["user_regimes"] = {"A": original, "B": deepcopy(original)}
    nodes = [[i, "A" if i % 2 == 0 else "B"] for i in range(6)]
    phase = {
        "kind": "dataclass",
        "fields": {
            "targets_by_period": [
                {"A" if i % 2 == 0 else "B": ["B" if i % 2 == 0 else "A"]}
                for i in range(5)
            ]
        },
    }
    reach = {
        "kind": "dataclass",
        "fields": {"nodes": nodes, "solution": phase, "simulation": phase},
    }
    source["reachability"] = reach
    source["graph"]["fields"].update(
        reachability=reach, initial_nodes=[[0, "A"]], coordinate_kind="period"
    )
    return source


def test_identical_repeated_stages_can_merge() -> None:
    problem, result = _check_analysis(_create_alternating())
    assert (problem["source_adapter"]["recurring_graph_cadence"]) == (2)
    assert (len(set(problem["source_adapter"]["stage_channels"].values()))) == (1)
    assert (result["recommended"]["cost"]["blocks"]) == (1)


def test_distinct_repeated_stages_keep_sparse_date_masks() -> None:
    source = _create_alternating()
    source["user_regimes"]["B"]["fields"]["functions"]["utility"] = function(
        "def utility(x, history):\n    return x + history\n", ["x", "history"]
    )
    problem, result = _check_analysis(source)
    assert (len(set(problem["source_adapter"]["stage_channels"].values()))) == (2)
    assert (result["recommended"]["cost"]["blocks"]) == (2)


def test_period_model_uses_its_economic_age_provider() -> None:
    source = snapshot()
    source["graph"]["fields"]["coordinate_kind"] = "period"
    declaration = source["user_regimes"]["original"]["fields"]
    declaration["functions"]["age"] = function(
        "def age(period):\n    return 20 + period\n", ["period"]
    )
    declaration["constraints"]["allowed"]["bindings"]["CUTOFF"] = 22
    _, result = _check_analysis(source)
    assert ([a["age"] for a in result["analysis"]["atoms"] if a["forced_actions"]]) == (
        [2, 3, 4, 5]
    )


def test_unbound_economic_age_cannot_be_replaced_with_period() -> None:
    source = snapshot()
    source["graph"]["fields"]["coordinate_kind"] = "period"
    _, result = _check_analysis(source)
    assert all(not a["forced_actions"] for a in result["analysis"]["atoms"])


def test_conditional_numerical_operand_drops_only_expired_history() -> None:
    source = snapshot()
    source["user_regimes"]["original"]["fields"]["functions"]["utility"] = function(
        "def utility(x, history, age):\n    return x + (history if age < 2 else 0)\n",
        ["x", "history", "age"],
    )
    _, result = _check_analysis(source)
    atoms = result["analysis"]["atoms"]
    assert ([a["age"] for a in atoms if "history" in a["live_states"]]) == ([0, 1])
    # Independent original numeric truth table, not the IR evaluator.
    for atom in atoms:
        if "history" not in atom["live_states"]:
            for x in (0, 1):
                outcomes = [
                    x + (history if atom["age"] < 2 else 0) for history in (0, 1)
                ]
                assert (outcomes[0]) == (outcomes[1])
    source["user_regimes"]["original"]["fields"]["functions"]["utility"]["source"] = (
        "def utility(x, history, age):\n    return x + (history if age < 2 else history)\n"
    )
    _, mutated = _check_analysis(source)
    assert all("history" in a["live_states"] for a in mutated["analysis"]["atoms"])


def test_future_payoff_dependency_crosses_original_routing_laws() -> None:
    source = snapshot()
    source["user_regimes"]["original"]["fields"]["functions"]["utility"] = function(
        "def utility(history, age):\n    return jnp.where(age == 5, history, 0)\n",
        ["history", "age"],
    )
    _, result = _check_analysis(source)
    assert all("history" in a["live_states"] for a in result["analysis"]["atoms"])


def test_default_aggregator_consumes_dag_discount_factor() -> None:
    source = snapshot()
    declaration = source["user_regimes"]["original"]["fields"]
    declaration["koopmans_aggregator"] = {
        "kind": "dataclass",
        "type": "lcm.LinearAggregator",
        "fields": {},
    }
    declaration["functions"]["discount_factor"] = function(
        "def discount_factor(history):\n    return 0.90 + 0.05 * history\n", ["history"]
    )
    _, result = _check_analysis(source)
    # Independent Bellman witness: even with U=x, history changes beta*V.
    assert ((0.90 + 0.05 * 0) * 10) != ((0.90 + 0.05 * 1) * 10)
    assert all("history" in a["live_states"] for a in result["analysis"]["atoms"])


def test_unavailable_zero_argument_constraint_remains_unknown() -> None:
    source = snapshot()
    source["user_regimes"]["original"]["fields"]["constraints"] = {
        "unknown": {"kind": "callable", "arguments": [], "source_unavailable": True}
    }
    _, result = _check_analysis(source)
    assert all(not a["forced_actions"] for a in result["analysis"]["atoms"])


def test_value_gate_targets_and_fallback_owners_are_preserved() -> None:
    source = _create_alternating()
    # A callback binding can legitimately be named `type`; it is not a
    # serialized object's type discriminator.
    source["user_regimes"]["A"]["fields"]["functions"]["utility"]["bindings"] = {
        "type": {"kind": "callable", "source_unavailable": True}
    }
    ref = lambda name: {
        "kind": "dataclass",
        "type": "lcm.ProjectedRegimeValue",
        "fields": {"regime": name},
    }
    source["graph"]["fields"]["laws"] = {
        "A": {
            "kind": "dataclass",
            "fields": {
                "gated_edges": {
                    "B": {
                        "kind": "dataclass",
                        "fields": {
                            "legs": {
                                "woman": {
                                    "kind": "dataclass",
                                    "fields": {"fallback": ref("A")},
                                },
                                "man": {
                                    "kind": "dataclass",
                                    "fields": {"fallback": ref("B")},
                                },
                            }
                        },
                    }
                }
            },
        }
    }
    problem, result = _check_analysis(source)
    assert (set(problem["source_adapter"]["value_route_protected_regimes"])) == (
        {"A", "B"}
    )
    assert (
        (problem["programs"][0]["routing_owner"])
        != (problem["programs"][1]["routing_owner"])
    )
    assert all("history" in atom["live_states"] for atom in result["analysis"]["atoms"])
