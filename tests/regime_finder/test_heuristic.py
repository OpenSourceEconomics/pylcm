"""Check heuristic layouts against small independent exhaustive oracles."""

import pytest
from copy import deepcopy
from fractions import Fraction
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence

from _lcm_regime_finder.frontier import Search, search_frontier
from _lcm_regime_finder.frontier_check import IndependentFamily
from _lcm_regime_finder.heuristic import search_heuristic
from _lcm_regime_finder.records import JSONValue
from tests.regime_finder.test_declaration_ingestion import (
    json_entries,
    json_list,
    json_member,
    json_object,
    json_rows,
)
from _lcm_regime_finder.heuristic_check import (
    check_heuristic_frontier,
    choose_candidate,
)
from .round6_frontier_oracle import block as literal_block, brute_family, canonical


METRICS = ("work", "storage", "clarity")


def fixture(
    n: int = 3,
    labels: Sequence[str] = ("a",),
    *,
    edges: Sequence[tuple[str, str, str, str, str]] = (),
    boundary_unit: int = 1,
) -> dict[str, JSONValue]:
    """Small exact cost input with physical ages deliberately unlike periods."""
    cells, interfaces = [], {}
    for label in labels:
        for period in range(n):
            cid = f"{label}:{period}"
            cell = {
                "id": cid,
                "regime": label,
                "period": period,
                "age": 21 + 3 * period,
                "role": {"actor": "household"},
                "ownership": "owner",
                "stage": "decision",
                "solver_identity": "same-solver",
                "states": {"x": "dx"},
                "actions": {"save": "da"},
            }
            cells.append(cell)
            interfaces[cid] = {
                "options": [
                    {
                        "id": "keep",
                        "states": dict(cell["states"]),
                        "actions": dict(cell["actions"]),
                        "state_projection": {"x": {"kind": "identity", "target": "x"}},
                        "action_reconstruction": {
                            "save": {"kind": "identity", "source": "save"}
                        },
                    }
                ]
            }
    return {
        "clarity_weights": {"block": 4, "characteristic": 2, "dispatch": 1},
        "declarations": {
            "cells": cells,
            "domains": {"dx": {"size": 3}, "da": {"size": 2}},
            "allocation": {
                "kind": "per_block",
                "status": "declared",
                "boundary_unit": boundary_unit,
            },
            "relations": [
                {
                    "source": source,
                    "targets": [target],
                    "support": support,
                    "phase": phase,
                    "kind": kind,
                }
                for source, target, support, phase, kind in edges
            ],
        },
        "interfaces": interfaces,
        "projection_policies": [
            {"id": "P0", "option_by_cell": {cell["id"]: "keep" for cell in cells}}
        ],
        "discovery_complete": True,
    }


def score(cost: JSONValue, objective: JSONValue) -> Fraction:
    """Weighted cost relative to the objective's reference scales."""
    assert isinstance(cost, Mapping)
    assert isinstance(objective, Mapping)
    weights, scales = objective["weights"], objective["reference_scales"]
    assert isinstance(weights, Mapping)
    assert isinstance(scales, Mapping)
    total = Fraction(0)
    for name in METRICS:
        value, scale = cost[name], scales[name]
        assert isinstance(value, int)
        assert isinstance(scale, int)
        total += Fraction(str(weights[name])) * Fraction(value, scale)
    return total


def _cells(analysis: dict[str, JSONValue]) -> list[dict[str, JSONValue]]:
    return json_rows(analysis, "declarations", "cells")


def _domains(analysis: dict[str, JSONValue]) -> dict[str, JSONValue]:
    return json_member(analysis, "declarations", "domains")


def _first_option(
    analysis: dict[str, JSONValue], cell_id: JSONValue
) -> dict[str, JSONValue]:
    assert isinstance(cell_id, str)
    return json_rows(analysis, "interfaces", cell_id, "options")[0]


def _first_policy(analysis: dict[str, JSONValue]) -> dict[str, JSONValue]:
    return json_member(json_rows(analysis, "projection_policies")[0], "option_by_cell")


def expected_cost(
    analysis: Mapping[str, JSONValue], point: dict[str, JSONValue]
) -> dict[str, int]:
    policies = analysis["projection_policies"]
    assert isinstance(policies, list)
    policy = next(
        row["option_by_cell"]
        for row in map(json_object, policies)
        if row["id"] == point["projection_policy"]
    )
    parts = [
        literal_block(analysis, part["cells"], policy)
        for part in json_rows(point, "blocks")
    ]
    if any(part is None for part in parts):
        raise AssertionError("literal oracle rejected heuristic plan")
    return {name: sum(part["cost"][name] for part in parts) for name in METRICS}


def pairwise_trap() -> dict[str, JSONValue]:
    """Every pair merge is worse; a three-way merge is worthwhile.

    Three singleton labels have distinct binary axes and all six directed
    same-period links.  The literal costs are original=(30,6,12), each pair
    merge=(34,10,9), all merged=(24,24,6).  At weights (1,.01,.01) and fixed
    unit scales, every pair merge is uphill but the three-way merge wins.
    """
    names = ("a", "b", "c")
    edges = [
        (f"{a}:0", f"{b}:0", "positive", "solve", "value_read")
        for a in names
        for b in names
        if a != b
    ]
    analysis = fixture(1, names, edges=edges, boundary_unit=2)
    json_member(analysis, "declarations")["domains"] = {
        name: {"size": 2} for name in names
    }
    for cell in _cells(analysis):
        name = cell["regime"]
        assert isinstance(name, str)
        cell["states"], cell["actions"] = {name: name}, {}
        option = _first_option(analysis, cell["id"])
        option.update(
            states={name: name},
            actions={},
            action_reconstruction={},
            state_projection={name: {"kind": "identity", "target": name}},
        )
    return analysis


def declarative_coarsening_fixture() -> dict[str, JSONValue]:
    """Full source declarations: utility uses floor(x/2), x persists exactly."""
    declarations = json_member(fixture(2), "declarations")
    declarations.update(
        schema="regime-declarations/v1",
        obligations=[],
        source_identity={"fixture": "coherent-categorical-identity-v1"},
        domains={
            "dx": {
                "kind": "categorical",
                "size": 3,
                "values": [0, 1, 2],
                "codes": [0, 1, 2],
            }
        },
        callbacks={
            "phi": {
                "kind": "callable",
                "arguments": ["x"],
                "reads": ["x"],
                "source": "def phi(x):\n    return x // 2\n",
                "expression": "x // 2",
                "status": "proved",
                "identity": "fixture-phi-source",
            },
            "utility": {
                "kind": "callable",
                "arguments": ["feature"],
                "reads": ["feature"],
                "source": "def utility(feature):\n    return feature\n",
                "expression": "feature",
                "status": "proved",
                "identity": "fixture-utility-source",
            },
        },
        characteristics={
            "feature": {
                "state": "x",
                "domain": "dx",
                "callback": "phi",
                "name": "feature",
            }
        },
        relations=[
            {
                "id": "carry",
                "source": "a:0",
                "targets": ["a:1"],
                "kind": "transition",
                "phase": "both",
                "support": "positive",
                "state_updates": {"x": {"kind": "identity", "source": "x"}},
            }
        ],
    )
    for cell in json_rows(declarations, "cells"):
        cell.update(
            actions={},
            functions={"feature": "phi", "utility": "utility"},
            consumers=[{"kind": "utility", "callback": "utility", "phase": "both"}],
        )
    return declarations


def _increment(record: dict[str, JSONValue], key: str) -> None:
    """Add one to the integer stored under `key`."""
    value = record[key]
    assert isinstance(value, int)
    record[key] = value + 1


def _increment_first(values: list[JSONValue]) -> None:
    """Add one to the leading integer of `values`."""
    value = values[0]
    assert isinstance(value, int)
    values[0] = value + 1


def _check_analysis(
    analysis: Mapping[str, JSONValue], **kwargs: Mapping[str, JSONValue]
) -> dict[str, JSONValue]:
    result = search_heuristic(analysis, **kwargs)
    receipt = check_heuristic_frontier(analysis, result)
    assert (receipt["status"]) == ("PASS"), receipt
    assert not (receipt["frontier_complete_verified"])
    assert (receipt["source_semantics"]) == ("NOT_CHECKED_HERE")
    assert (result["status"]) == ("HEURISTIC")
    assert (result["search"]["complete"]) is (False)
    assert (result["search"]["optimality_gap"]) == ("unknown")
    for point in result["candidates"]:
        assert (point["cost"]) == (expected_cost(analysis, point))
    assert isinstance(result, dict)
    return result


def test_fixed_group_scalar_interval_optimum_matches_tiny_exhaustive_oracle() -> None:
    for seed in range(8):
        rng = random.Random(seed)
        edges = [
            (
                f"a:{i}",
                f"a:{j}",
                rng.choice(("positive", "unknown", "zero")),
                "both",
                "transition",
            )
            for i in range(4)
            for j in range(4)
            if rng.random() < 0.25
        ]
        analysis = fixture(4, edges=edges)
        for period in range(4):
            if rng.random() < 0.5:
                option = _first_option(analysis, f"a:{period}")
                option["actions"] = {}
                option["action_reconstruction"] = {
                    "save": {"kind": "constant", "value": 0}
                }
        result = _check_analysis(
            analysis, recommendation={"allow_final_only_split": True}
        )
        exact = brute_family(analysis)
        assert (
            min(
                score(point["cost"], result["objective"])
                for point in json_rows(result, "candidates")
            )
        ) == (min(score(point["cost"], result["objective"]) for point in exact))


def test_tiny_multilabel_quality_reference_is_a_lower_bound_not_a_claim() -> None:
    for seed in range(6):
        rng = random.Random(seed)
        analysis = fixture(
            2,
            ("a", "b"),
            edges=[
                (f"{a}:0", f"{b}:{t}", "positive", "solve", "policy_read")
                for a in ("a", "b")
                for b in ("a", "b")
                for t in (0, 1)
                if rng.random() < 0.5
            ],
        )
        result = _check_analysis(analysis)
        optimum = min(
            score(point["cost"], result["objective"])
            for point in brute_family(analysis)
        )
        assert all(
            score(point["cost"], result["objective"]) >= optimum
            for point in json_rows(result, "candidates")
        )


def test_same_period_branching_and_duplicate_incidence_costs() -> None:
    edges = [
        ("a:0", "b:0", "positive", "solve", "value_read"),
        ("a:0", "b:0", "positive", "simulation", "policy_read"),
        ("a:0", "a:1", "unknown", "both", "transition"),
        ("a:0", "b:1", "positive", "both", "joint_transition"),
        ("a:0", "a:0", "positive", "solve", "self_read"),
        ("b:0", "a:1", "zero", "both", "transition"),
    ]
    analysis = fixture(2, ("a", "b"), edges=edges, boundary_unit=2)
    policy = _first_policy(analysis)
    assert (literal_block(analysis, ["a:0"], policy)["cost"]["work"]) == (42)
    before = canonical(analysis)
    _check_analysis(analysis)
    assert (canonical(analysis)) == (
        before
    )  # typed and phased metadata stays untouched


def test_per_block_storage_and_original_baseline() -> None:
    analysis = fixture(1, ("a", "b"))
    _domains(analysis)["dy"] = {"size": 5}
    cell = _cells(analysis)[1]
    cell["states"] = {"y": "dy"}
    option = _first_option(analysis, cell["id"])
    option["states"] = {"y": "dy"}
    option["state_projection"] = {"y": {"kind": "identity", "target": "y"}}
    result = _check_analysis(analysis)
    assert (json_member(result, "original", "cost")["storage"]) == (8)
    policy = _first_policy(analysis)
    assert (literal_block(analysis, ["a:0", "b:0"], policy)["cost"]["storage"]) == (30)
    assert (json_member(result, "objective", "reference_scales")["storage"]) == (8)


def test_original_baseline_is_not_the_only_offered_reduced_policy() -> None:
    analysis = fixture(2)
    _domains(analysis)["history"] = {"size": 100}
    for cell in _cells(analysis):
        json_member(cell, "states")["history"] = "history"
        option = _first_option(analysis, cell["id"])
        json_member(option, "state_projection")["history"] = {"kind": "drop"}
    result = _check_analysis(analysis)
    original_cost = json_member(result, "original")["cost"]
    assert (original_cost) == ({"work": 1200, "storage": 600, "clarity": 4})
    assert (json_member(result, "objective")["reference_scales"]) == (original_cost)
    works = [
        json_member(point, "cost")["work"] for point in json_rows(result, "candidates")
    ]
    assert all(isinstance(work, int) for work in works)
    assert (min(work for work in works if isinstance(work, int))) == (12)
    recommendation = json_member(result, "recommendation")
    assert (recommendation["graph_id"]) != ("retain-original")
    assert (recommendation["authority"]) == ("cost_only")


def test_incompatible_grids_and_common_execution_interfaces_cannot_merge() -> None:
    for kind in (
        "domain",
        "stage",
        "declaration_type",
        "structural_roles",
        "solver_identity",
    ):
        analysis = fixture(1, ("a", "b"))
        cell = _cells(analysis)[1]
        if kind == "domain":
            _domains(analysis)["other"] = {"size": 3}
            json_member(cell, "states")["x"] = "other"
            json_member(_first_option(analysis, "b:0"), "states")["x"] = "other"
        else:
            cell[kind] = (
                {"margin": "another"} if kind == "structural_roles" else "another"
            )
        result = _check_analysis(analysis)
        assert all(
            len(json_list(part, "cells")) == 1
            for point in json_rows(result, "candidates")
            for part in json_rows(point, "blocks")
        )


def test_coherent_policy_moves_do_not_mix_cell_options() -> None:
    analysis = fixture(2)
    _domains(analysis)["coarse"] = {"size": 2}
    for record in json_entries(analysis, "interfaces"):
        options = json_list(record, "options")
        reduced = deepcopy(json_object(options[0]))
        reduced.update(
            id="coarse",
            states={"class": "coarse"},
            state_projection={
                "x": {
                    "kind": "characteristic",
                    "callback": "known_map",
                    "target": "class",
                    "domain": "coarse",
                    "values": [0, 0, 1],
                }
            },
        )
        options.append(reduced)
    json_list(analysis, "projection_policies").append(
        {
            "id": "P1",
            "option_by_cell": {str(cell["id"]): "coarse" for cell in _cells(analysis)},
        }
    )
    result = _check_analysis(analysis)
    policies = {
        p["id"]: p["option_by_cell"] for p in json_rows(analysis, "projection_policies")
    }
    for point in json_rows(result, "candidates"):
        actual = {
            cid: option
            for part in json_rows(point, "blocks")
            for cid, option in json_member(part, "option_by_cell").items()
        }
        assert (actual) == (policies[point["projection_policy"]])
    bad = deepcopy(result)
    part = json_rows(json_rows(bad, "candidates")[0], "blocks")[0]
    cid = json_list(part, "cells")[0]
    assert isinstance(cid, str)
    option_by_cell = json_member(part, "option_by_cell")
    option_by_cell[cid] = "keep" if option_by_cell[cid] == "coarse" else "coarse"
    assert (check_heuristic_frontier(analysis, bad)["status"]) == ("FAIL")


def test_terminal_cosmetic_preference_has_explicit_override() -> None:
    analysis = fixture(3)
    _domains(analysis)["history"] = {"size": 20}
    for cell in _cells(analysis):
        json_member(cell, "states")["history"] = "history"
        option = _first_option(analysis, cell["id"])
        json_member(option, "state_projection")["history"] = {"kind": "drop"}
    last = _first_option(analysis, "a:2")
    last.update(
        actions={}, action_reconstruction={"save": {"kind": "constant", "value": 0}}
    )
    configuration = {"weights": {"clarity": "1/1000000"}}
    ordinary = _check_analysis(analysis, recommendation=configuration)
    selected = json_member(ordinary, "recommendation")["graph_id"]
    assert (selected) not in (
        json_list(
            ordinary, "recommendation", "endpoint_preference", "deferred_graph_ids"
        )
    )
    override = _check_analysis(
        analysis, recommendation=dict(configuration, allow_final_only_split=True)
    )
    assert (json_member(override, "recommendation")["graph_id"]) in (
        json_list(
            override, "recommendation", "endpoint_preference", "deferred_graph_ids"
        )
    )


def test_strict_improvement_and_source_eligible_filter_keep_original() -> None:
    analysis = fixture(2)
    result = _check_analysis(analysis)
    assert (json_member(result, "recommendation")["graph_id"]) == ("retain-original")
    assert (choose_candidate(analysis, result, allowed_ids=set())["operation"]) == (
        "retain_original"
    )
    assert (json_member(result, "objective")["reference_kind"]) == ("original")


def test_unpriceable_original_retains_identity_even_with_explicit_scales() -> None:
    analysis = fixture(2)
    # The original label crosses protected contexts.  Keeping the original
    # remains an identity action, but the block model cannot compare it as
    # one admissible regime.  No surrogate improvement is then asserted.
    _cells(analysis)[1]["stage"] = "second-decision-stage"
    result = _check_analysis(analysis)
    assert (json_member(result, "original")["cost"]) is None
    assert (json_member(result, "objective")["reference_kind"]) == ("unit_fallback")
    assert (json_member(result, "recommendation")["operation"]) == ("retain_original")
    explicit = _check_analysis(
        analysis, recommendation={"reference": dict.fromkeys(METRICS, 1)}
    )
    assert (json_member(explicit, "recommendation")["operation"]) == ("retain_original")


def test_per_policy_eligible_candidate_survives_a_cheaper_other_policy() -> None:
    analysis = fixture(2)
    _domains(analysis)["history"] = {"size": 100}
    for cell in _cells(analysis):
        json_member(cell, "states")["history"] = "history"
        option = _first_option(analysis, cell["id"])
        json_member(option, "state_projection")["history"] = {"kind": "drop"}
        alternative = deepcopy(option)
        alternative.update(
            id="drop-all",
            states={},
            state_projection={"x": {"kind": "drop"}, "history": {"kind": "drop"}},
        )
        json_list(analysis, "interfaces", str(cell["id"]), "options").append(
            alternative
        )
    json_list(analysis, "projection_policies").append(
        {
            "id": "P1",
            "option_by_cell": {
                str(cell["id"]): "drop-all" for cell in _cells(analysis)
            },
        }
    )
    result = _check_analysis(analysis)
    assert all(
        point["projection_policy"] == "P1" for point in json_rows(result, "frontier")
    )
    safe_policy_candidates = {
        point["id"]
        for point in json_rows(result, "candidates")
        if point["projection_policy"] == "P0"
    }
    assert safe_policy_candidates
    selected = choose_candidate(analysis, result, allowed_ids=safe_policy_candidates)
    assert (selected["graph_id"]) in (safe_policy_candidates)
    assert (selected["authority"]) == (
        "cost_only"
    )  # ranking never grants source authority


def test_pair_merge_local_minimum_is_not_global_optimality() -> None:
    analysis = pairwise_trap()
    policy = _first_policy(analysis)
    singleton = [
        literal_block(analysis, [f"{name}:0"], policy) for name in ("a", "b", "c")
    ]
    original = {name: sum(part["cost"][name] for part in singleton) for name in METRICS}
    objective: dict[str, JSONValue] = {
        "weights": {"work": "1", "storage": "1/100", "clarity": "1/100"},
        "reference_scales": dict.fromkeys(METRICS, 1),
    }
    assert (tuple(original[name] for name in METRICS)) == ((30, 6, 12))
    for a, b, other in (("a", "b", "c"), ("a", "c", "b"), ("b", "c", "a")):
        parts = [
            literal_block(analysis, [f"{a}:0", f"{b}:0"], policy),
            literal_block(analysis, [f"{other}:0"], policy),
        ]
        cost = {name: sum(part["cost"][name] for part in parts) for name in METRICS}
        assert (tuple(cost[name] for name in METRICS)) == ((34, 10, 9))
        assert (score(cost, objective)) > (score(original, objective))
    together = literal_block(analysis, ["a:0", "b:0", "c:0"], policy)["cost"]
    assert (tuple(together[name] for name in METRICS)) == ((24, 24, 6))
    assert (score(together, objective)) < (score(original, objective))
    result = _check_analysis(
        analysis,
        recommendation={
            "weights": objective["weights"],
            "reference": objective["reference_scales"],
        },
        effort={"rounds": 2},
    )
    assert (json_member(result, "search")["optimality_gap"]) == ("unknown")
    assert (json_member(result, "recommendation")["graph_id"]) == ("retain-original")
    best_examined = min(
        score(point["cost"], objective) for point in json_rows(result, "candidates")
    )
    assert (best_examined) > (
        min(score(point["cost"], objective) for point in brute_family(analysis))
    )


def test_width_one_chain_never_enters_exponential_assignment_or_checker_cover() -> None:
    # n-1 A signatures can each be assigned to A or A-union-B; the final
    # A-union-B provider is forced.  The native assignment count is 2**(n-1)
    # although every directed link goes to the immediately next period.
    n = 5
    labels = tuple(f"r{i}" for i in range(n))
    analysis = fixture(1, labels)
    declarations = json_member(analysis, "declarations")
    declarations["analysis_scope"] = "pylcm-structural-graph/v1"
    _domains(analysis)["dy"] = {"size": 2}
    for i, cell in enumerate(_cells(analysis)):
        cell["period"], cell["age"] = i, 20 + 2 * i
        cell["actions"] = {}
        option = _first_option(analysis, cell["id"])
        option.update(actions={}, action_reconstruction={})
        if i == n - 1:
            json_member(cell, "states")["y"] = "dy"
            json_member(option, "states")["y"] = "dy"
            json_member(option, "state_projection")["y"] = {
                "kind": "identity",
                "target": "y",
            }
    declarations["relations"] = [
        {"source": f"r{i}:0", "targets": [f"r{i + 1}:0"], "support": "positive"}
        for i in range(n - 1)
    ]
    # Instrumentation proves absence of expensive calls; it is not a
    # claimed defect under a monkeypatched scientific environment.
    with (
        patch.object(
            Search, "run", side_effect=AssertionError("exact producer invoked")
        ),
        patch.object(
            Search,
            "signature_frontier",
            side_effect=AssertionError("assignment cover invoked"),
        ),
        patch.object(
            IndependentFamily,
            "complete_frontier",
            side_effect=AssertionError("exact checker invoked"),
        ),
        patch.object(
            IndependentFamily,
            "signature_points",
            side_effect=AssertionError("checker assignment cover invoked"),
        ),
    ):
        _check_analysis(analysis, effort={"rounds": 1})


def test_input_permutations_keep_costs_and_selection() -> None:
    analysis = fixture(2, ("a", "b", "c"))
    before = _check_analysis(analysis)
    permuted = deepcopy(analysis)
    json_list(permuted, "declarations", "cells").reverse()
    json_list(permuted, "declarations", "relations").reverse()
    interfaces = json_member(permuted, "interfaces")
    permuted["interfaces"] = dict(reversed(list(interfaces.items())))
    after = _check_analysis(permuted)
    assert ([point["cost"] for point in json_rows(before, "frontier")]) == (
        [point["cost"] for point in json_rows(after, "frontier")]
    )
    assert (before["recommendation"]) == (after["recommendation"])


def test_candidate_and_source_binding_negative_controls(
    *, subtests: pytest.Subtests
) -> None:
    analysis = fixture(
        3, ("a", "b"), edges=[("a:0", "b:0", "positive", "solve", "value_read")]
    )
    result = _check_analysis(analysis)
    for mutation in (
        "cost",
        "date",
        "age",
        "coverage",
        "axes",
        "detail",
        "complete",
        "baseline",
        "reference",
    ):
        bad = deepcopy(result)
        first = json_rows(bad, "candidates")[0]
        part = json_rows(first, "blocks")[0]
        if mutation == "cost":
            _increment(json_member(first, "cost"), "work")
        elif mutation == "date":
            _increment_first(json_list(part, "periods"))
        elif mutation == "age":
            _increment_first(json_list(part, "ages"))
        elif mutation == "coverage":
            json_list(part, "cells").pop()
        elif mutation == "axes":
            part["state_axes"] = {}
        elif mutation == "detail":
            _increment(json_member(part, "cost_detail"), "boundary_incidences")
        elif mutation == "complete":
            json_member(bad, "search")["complete"] = True
        elif mutation == "baseline":
            _increment(json_member(bad, "original", "cost"), "storage")
        elif mutation == "reference":
            _increment(json_member(bad, "objective", "reference_scales"), "work")
        with subtests.test(mutation=mutation):
            assert (check_heuristic_frontier(analysis, bad)["status"]) == ("FAIL")
    for field, value in (
        ("period", 91),
        ("ownership", "new-owner"),
        ("solver_identity", "another-solver"),
    ):
        altered = deepcopy(analysis)
        _cells(altered)[0][field] = value
        assert (check_heuristic_frontier(altered, result)["status"]) == ("FAIL")
    changed = deepcopy(analysis)
    json_rows(changed, "declarations", "relations")[0]["phase"] = "simulation"
    assert (check_heuristic_frontier(changed, result)["status"]) == ("FAIL")


def test_recomputed_costs_do_not_authorize_noninterval_or_partial_date_blocks() -> None:
    for same_date in (False, True):
        analysis = fixture(3, ("a", "b") if same_date else ("a",))
        search = Search(analysis)
        policy = _first_policy(analysis)
        members = (
            [["a:0", "b:0", "a:1"], ["b:1", "a:2", "b:2"]]
            if same_date
            else [["a:0", "a:2"], ["a:1"]]
        )
        blocks = [search.block(ids, policy) for ids in members]
        bad_plan = {
            "projection_policy": "P0",
            "blocks": blocks,
            "cost": {
                name: sum(part["cost"][name] for part in blocks) for name in METRICS
            },
        }
        errors = IndependentFamily(analysis).admitted_plan(bad_plan)
        assert ("block is not an interval of complete exact date buckets") in (errors)


def test_cooperative_stop_keeps_literal_original_without_claiming_optimum() -> None:
    analysis = fixture(4, ("a", "b"))
    result = _check_analysis(analysis, limits={"seconds": 0})
    assert (result["candidates"]) == ([])
    assert (json_member(result, "recommendation")["operation"]) == ("retain_original")
    assert (json_member(result, "search")["evaluated_candidates"]) == (0)


def test_exact_api_remains_an_explicit_tiny_oracle() -> None:
    analysis = fixture(2, ("a", "b"))
    exact = search_frontier(analysis)
    assert (exact["status"]) == ("COMPLETE")
    assert exact["certificate"]["complete"]
    heuristic = _check_analysis(analysis, effort={"rounds": 0})
    assert (exact["schema"]) != (heuristic["schema"])


def test_disconnected_coarsening_regions_do_not_enumerate_the_powerset() -> None:
    from _lcm_regime_finder import analyze_production, verify_production_result
    from _lcm_regime_finder.production import MetadataAnalysis

    source = declarative_coarsening_fixture()
    template = json_rows(source, "cells")[0]
    source["cells"] = [
        dict(deepcopy(template), id=f"a:{i}", period=i, age=21 + i) for i in range(32)
    ]
    source["relations"] = []
    with patch.object(
        MetadataAnalysis,
        "closed_regions",
        side_effect=AssertionError("exact region powerset invoked"),
    ):
        result = analyze_production(
            source, facts_backend="metadata", search_mode="heuristic"
        )
        checked = verify_production_result(source, result)
    assert (checked["status"]) == ("PASS"), checked
    assert not (result["analysis"]["discovery_complete"])
    assert (len(result["analysis"]["projection_policies"])) == (2)
    assert any(
        row["authority"] == "source_checked" for row in result["recommendations"]
    )


def test_heuristic_label_region_is_closed_under_original_identity_links() -> None:
    from _lcm_regime_finder.production import MetadataAnalysis

    source = declarative_coarsening_fixture()
    json_rows(source, "cells")[1]["regime"] = "b"
    analysis = MetadataAnalysis(
        source, search_mode="heuristic", facts_backend="metadata"
    ).run()
    candidate = analysis["characteristic_candidates"][0]
    regions = MetadataAnalysis(
        source, search_mode="heuristic", facts_backend="metadata"
    ).heuristic_regions(candidate)[0]
    assert ({tuple(row["active_cells"]) for row in regions}) == (
        {("a:0", "a:1"), ("a:1",)}
    )
    from _lcm_regime_finder.semantic_check import check_semantics

    receipt = check_semantics(source, source, analysis, check_coverage=False)
    assert not (receipt["errors"]), receipt


def test_heuristic_coarsening_axes_do_not_form_a_cartesian_policy_family() -> None:
    from _lcm_regime_finder.production import MetadataAnalysis
    from _lcm_regime_finder.semantic_check import check_semantics

    source = declarative_coarsening_fixture()
    for axis in ("y", "z"):
        name = "feature_" + axis
        callbacks = json_member(source, "callbacks")
        callbacks[name] = dict(
            json_member(callbacks, "phi"),
            arguments=[axis],
            reads=[axis],
            source=f"def {name}({axis}):\n    return {axis} // 2\n",
            expression=f"{axis} // 2",
            identity=name,
        )
        json_member(source, "characteristics")[name] = {
            "state": axis,
            "domain": "dx",
            "callback": name,
            "name": name,
        }
        for cell in json_rows(source, "cells"):
            json_member(cell, "states")[axis] = "dx"
            json_member(cell, "functions")[name] = name
        json_member(json_rows(source, "relations")[0], "state_updates")[axis] = {
            "kind": "identity",
            "source": axis,
        }
    json_member(source, "callbacks", "utility").update(
        arguments=["feature", "feature_y", "feature_z"],
        reads=["feature", "feature_y", "feature_z"],
        expression="feature + feature_y + feature_z",
        source="def utility(feature, feature_y, feature_z):\n    return feature + feature_y + feature_z\n",
    )
    with patch(
        "_lcm_regime_finder.production.product",
        side_effect=AssertionError("Cartesian policies invoked"),
    ):
        analysis = MetadataAnalysis(
            source, search_mode="heuristic", facts_backend="metadata"
        ).run()
    assert (len(analysis["projection_policies"])) == (5)
    assert not (analysis["discovery_complete"])
    assert not (
        check_semantics(source, source, analysis, check_coverage=False)["errors"]
    )


def test_useful_change_is_source_checked_before_public_delivery() -> None:
    from _lcm_regime_finder import analyze_production, verify_production_result
    from _lcm_regime_finder.cli import candidate_boundary_maps

    path = (
        Path(__file__).resolve().parents[2]
        / "tests/regime_finder/examples/declarations/tiny-huge-grid.json"
    )
    source = json.loads(path.read_text())
    result = analyze_production(
        source,
        recommendation={"weights": {"work": 1, "storage": 1, "clarity": "1/10"}},
        facts_backend="metadata",
        search_mode="heuristic",
    )
    selected = result["recommendations"][0]
    assert (selected["authority"]) == ("source_checked")
    assert (selected["operation"]) == ("representation_specification")
    assert (verify_production_result(source, result)["status"]) == ("PASS")
    rows = candidate_boundary_maps(result)
    assert rows
    assert all(row["transformation_authorized"] for row in rows)
    changed = deepcopy(source)
    changed["cells"][0]["ownership"] = "different-owner"
    assert (verify_production_result(changed, result)["status"]) == ("FAIL")


def test_cost_admission_never_substitutes_for_global_policy_semantics() -> None:
    from _lcm_regime_finder.production import MetadataAnalysis
    from _lcm_regime_finder.proposal_decision import build_decisions
    from _lcm_regime_finder.semantic_check import check_semantics

    original = declarative_coarsening_fixture()
    analysis = MetadataAnalysis(
        original, facts_backend="metadata", search_mode="exact"
    ).run()
    source_receipt = check_semantics(original, original, analysis, check_coverage=False)
    assert not (source_receipt["errors"]), source_receipt
    assert all(item["scope"] == "family" for item in source_receipt["unresolved"])
    keep, coarse = {}, {}
    for cid, row in analysis["interfaces"].items():
        for option in row["options"]:
            target = (
                coarse
                if option["state_projection"]["x"]["kind"] == "characteristic"
                else keep
            )
            target[cid] = option["id"]
    changed = deepcopy(analysis)
    changed["projection_policies"] = [
        {
            "id": "invalid-offered-policy",
            "option_by_cell": {"a:0": coarse["a:0"], "a:1": keep["a:1"]},
        }
    ]
    scenario = search_heuristic(changed, effort={"rounds": 0})
    assert (check_heuristic_frontier(changed, scenario)["status"]) == ("PASS")
    semantic = check_semantics(original, original, changed, check_coverage=False)
    assert any(
        error["kind"] == "future projected identity law loses history"
        for error in semantic["errors"]
    ), semantic
    decisions = build_decisions(original, changed, [scenario])
    proposed = [
        row for row in decisions if row["operation"] == "representation_specification"
    ]
    assert proposed
    assert all(row["disposition"] == "reject" for row in proposed)


def test_actual_entry_points_keep_exact_optional_and_replay_source_guards() -> None:
    from _lcm_regime_finder import analyze_production, verify_production_result
    from .test_model_adapter import snapshot

    source = snapshot()
    result = analyze_production(
        source, facts_backend="metadata", search_mode="heuristic"
    )
    assert result["scenarios"]
    assert all(
        row["schema"] == "regime-heuristic-frontier/v1" for row in result["scenarios"]
    )
    assert (result["status"]) != ("COMPLETE")
    checked = verify_production_result(source, result)
    assert (checked["status"]) == ("PASS"), checked
    for row in result["recommendations"]:
        assert (row["authority"]) in (("source_checked", "retain_original"))
    exact = analyze_production(source, search_mode="exact", facts_backend="metadata")
    assert (exact["status"]) == ("COMPLETE")
    assert exact["scenarios"][0]["certificate"]["complete"]
    corrupted = deepcopy(result)
    callback = next(
        item
        for item in corrupted["analysis"]["declarations"]["callbacks"].values()
        if item.get("arguments") == ["x", "history"]
    )
    callback["arguments"] = []
    callback["reads"] = []
    assert (verify_production_result(source, corrupted)["status"]) == ("FAIL")
    # A producer flag cannot transform an unresolved/retained representation
    # into source authority; the independently replayed recommendation binds.
    corrupted = deepcopy(result)
    corrupted["recommendations"][0]["graph_id"] = "invented-graph"
    corrupted["recommendations"][0]["authority"] = "source_checked"
    assert (verify_production_result(source, corrupted)["status"]) == ("FAIL")
    corrupted = deepcopy(result)
    corrupted["status"] = "COMPLETE"
    assert (verify_production_result(source, corrupted)["status"]) == ("FAIL")
    expired = verify_production_result(source, result, limits={"seconds": 0})
    assert (expired["status"]) != ("PASS")
    assert all(
        row["operation"] == "retain_original"
        for row in expired.get("recommendations", [])
    )
