import pytest
from copy import deepcopy
from itertools import combinations, product
import hashlib
import json
from pathlib import Path
import tempfile

from _lcm_regime_finder import (
    find_cost_frontier,
    check_cost_frontier,
    query_cost_frontier,
    find_regimes,
    verify_certificate,
)
from _lcm_regime_finder.cost_family import encode_analysis, chain_cover
from tests.regime_finder.test_forest_frontier import example, literal
from tests.regime_finder.test_aca_separator_reference import fixture
from tests.regime_finder.test_borella_symbolic import toy
from tests.regime_finder.test_signature_integration import heterogeneous
from tests.regime_finder.round6_frontier_oracle import brute_family


def forest_analysis():
    original = example([{"x", "y"}, {"x", "z"}, {"y", "z"}])
    cells, interfaces, policy = [], {}, {}
    for c in original["cells"]:
        cell = c | {
            "regime": c["label"],
            "role": None,
            "ownership": None,
            "declaration_type": "native",
        }
        cells.append(cell)
        option = {
            "id": c["id"] + "/identity",
            "states": c["states"],
            "actions": c["actions"],
            "state_projection": {
                axis: {"kind": "identity", "target": axis} for axis in c["states"]
            },
            "action_reconstruction": {},
            "constants": {},
        }
        interfaces[c["id"]] = {"options": [option]}
        policy[c["id"]] = option["id"]
    return {
        "declarations": {
            "analysis_scope": "pylcm-structural-graph/v1",
            "allocation": {"kind": "per_block", "boundary_unit": 1},
            "cells": cells,
            "domains": original["domains"],
            "relations": [],
        },
        "interfaces": interfaces,
        "discovery_complete": True,
        "projection_policies": [{"id": "original", "option_by_cell": policy}],
    }, original


def test_all_directed_three_cell_graphs_preserve_the_literal_cost_family(
    *, subtests: pytest.Subtests
) -> None:
    from _lcm_regime_finder.cost_limits import DEFAULTS
    from _lcm_regime_finder.frontier import search_frontier
    from _lcm_regime_finder.frontier_check import check_frontier

    unlimited = dict.fromkeys(DEFAULTS)
    possible_edges = [(a + ":0", b + ":0") for a in "abc" for b in "abc" if a != b]
    for compatible, mask in product((False, True), range(64)):
        with subtests.test(compatible_domains=compatible, mask=mask):
            analysis = heterogeneous(
                edges=[
                    (*edge, "unknown")
                    for i, edge in enumerate(possible_edges)
                    if mask & (1 << i)
                ]
            )
            analysis["declarations"]["analysis_scope"] = "pylcm-structural-graph/v1"
            if compatible:
                cell = analysis["declarations"]["cells"][2]
                cell["states"]["other_health"] = cell["states"].pop("health")
                option = analysis["interfaces"][cell["id"]]["options"][0]
                option["states"]["other_health"] = option["states"].pop("health")
                option["state_projection"]["other_health"] = option[
                    "state_projection"
                ].pop("health")
            expected = sorted(
                tuple(point["cost"][key] for key in ("work", "storage", "clarity"))
                for point in brute_family(analysis)
            )
            result = find_cost_frontier(analysis, seconds=None, limits=unlimited)
            recurrence = search_frontier(analysis)
            assert (result["status"]) == ("COMPLETE_COST_FAMILY")
            assert (sorted(map(tuple, result["frontier"]))) == (expected)
            assert (
                [
                    tuple(point["cost"][key] for key in ("work", "storage", "clarity"))
                    for point in recurrence["frontier"]
                ]
            ) == (expected)
            assert check_cost_frontier(
                analysis, result, seconds=None, limits=unlimited
            )["cost_family_complete_verified"]
            assert check_frontier(analysis, recurrence)["frontier_complete_verified"]
            edges = {
                tuple(sorted((relation["source"], relation["targets"][0])))
                for relation in analysis["declarations"]["relations"]
            }
            backend = (
                "legacy-exact-component" if len(edges) == 3 else "singleton-forest"
            )
            assert ([component["backend"] for component in result["components"]]) == (
                [backend]
            )


def test_forest_and_cyclic_certificates_reject_source_frontier_and_witness_changes(
    *, subtests: pytest.Subtests
) -> None:
    from _lcm_regime_finder.cost_limits import DEFAULTS

    unlimited = dict.fromkeys(DEFAULTS)
    possible_edges = [(a + ":0", b + ":0") for a in "abc" for b in "abc" if a != b]
    for mask in (1, 11):
        analysis = heterogeneous(
            edges=[
                (*edge, "unknown")
                for i, edge in enumerate(possible_edges)
                if mask & (1 << i)
            ]
        )
        analysis["declarations"]["analysis_scope"] = "pylcm-structural-graph/v1"
        cell = analysis["declarations"]["cells"][2]
        cell["states"]["other_health"] = cell["states"].pop("health")
        option = analysis["interfaces"][cell["id"]]["options"][0]
        option["states"]["other_health"] = option["states"].pop("health")
        option["state_projection"]["other_health"] = option["state_projection"].pop(
            "health"
        )
        result = find_cost_frontier(analysis, seconds=None, limits=unlimited)
        for mutation in (
            "input_identity",
            "domain",
            "edge",
            "omitted_point",
            "component",
            "witness",
        ):
            with subtests.test(mask=mask, mutation=mutation):
                original, changed = deepcopy(analysis), deepcopy(result)
                if mutation == "input_identity":
                    changed["input_sha256"] = "wrong input"
                elif mutation == "domain":
                    original["declarations"]["domains"]["wealth"]["size"] += 1
                elif mutation == "edge":
                    original["declarations"]["relations"].append(
                        {"source": "c:0", "targets": ["outside"], "support": "positive"}
                    )
                elif mutation == "omitted_point":
                    changed["frontier"].pop()
                elif mutation == "component":
                    changed["components"] = []
                else:
                    component = changed["components"][0]
                    if component["backend"] == "singleton-forest":
                        component["proof"]["frontier"][0][
                            "normalized_witness_sha256"
                        ] = "forged witness"
                    else:
                        component["proof"]["frontier"][0]["blocks"][0]["cells"] = []
                assert (
                    check_cost_frontier(
                        original, changed, seconds=None, limits=unlimited
                    )["status"]
                ) == ("FAIL")


def test_no_time_or_count_limits_across_all_cost_backends(
    *, subtests: pytest.Subtests
) -> None:
    from _lcm_regime_finder.cost_limits import DEFAULTS

    unlimited = dict.fromkeys(DEFAULTS)
    legacy, _ = forest_analysis()
    for cell in legacy["declarations"]["cells"]:
        cell["regime"] = "one-dated-label"
    for analysis in (forest_analysis()[0], legacy, fixture(), toy()[0]):
        with subtests.test(symbolic=bool(analysis.get("characteristic_candidates"))):
            result = find_cost_frontier(analysis, seconds=None, limits=unlimited)
            assert (result["status"]) == ("COMPLETE_COST_FAMILY")
            assert (result["requested_bookkeeping_limits"]) == (unlimited)
            json.dumps(result, allow_nan=False)
            assert check_cost_frontier(
                analysis, result, seconds=None, limits=unlimited
            )["cost_family_complete_verified"]
            query = query_cost_frontier(
                analysis, result, seconds=None, limits=unlimited, weights=(1, 1, 1)
            )
            assert (query["status"]) == ("PASS")
            assert query["witness"]


def test_chain_coordinates_keep_layout_continuity_without_dropping_policies() -> None:
    candidate = {
        "cells": ["v0", "z0", "z1", "a0", "a1"],
        "requires": {
            "v0": ["a0", "z0"],
            "z0": ["a1", "z1"],
            "z1": [],
            "a0": ["a1"],
            "a1": [],
        },
    }
    cells = [
        {
            "id": node,
            "regime": "source"
            if node == "v0"
            else "left"
            if node.startswith("z")
            else "right",
            "states": {
                "claim": "claim",
                "x" if node in ("v0", "z0", "z1") else "y": "other",
            },
            "actions": {},
            "period": i,
        }
        for i, node in enumerate(candidate["cells"])
    ]
    cover = chain_cover(candidate, cells)
    assert ({tuple(c) for c in cover}) == ({("v0", "z0", "z1"), ("a0", "a1")})
    nodes = candidate["cells"]
    closed = lambda selected: all(
        target in selected
        for source in selected
        for target in candidate["requires"][source]
    )
    literal = {
        frozenset(s)
        for count in range(len(nodes) + 1)
        for s in combinations(nodes, count)
        if closed(s)
    }
    for chains in (cover, chain_cover(candidate)):
        represented = {
            frozenset(
                c
                for chain, threshold in zip(chains, thresholds)
                for c in chain[threshold:]
            )
            for thresholds in product(*(range(len(chain) + 1) for chain in chains))
        }
        assert ({s for s in represented if closed(s)}) == (literal)


def test_bookkeeping_caps_do_not_restrict_the_certified_family(
    *, subtests: pytest.Subtests
) -> None:
    analysis, _ = forest_analysis()
    partial = find_cost_frontier(analysis, limits={"forest_points": 1})
    assert (partial["status"]) == ("INCOMPLETE")
    assert check_cost_frontier(analysis, partial)["original_representation_verified"]
    result = find_cost_frontier(analysis, limits={"forest_points": 10000})
    assert (
        check_cost_frontier(analysis, result, limits={"frontier_points": 10000})[
            "status"
        ]
    ) == ("PASS")
    assert (result["analysis_sha256"]) == (partial["analysis_sha256"])
    for bad in (
        {"forest_points": -1},
        {"segments": True},
        {"unknown": 1},
        ["segments"],
    ):
        with subtests.test(bad=bad), pytest.raises(ValueError):
            find_cost_frontier(analysis, limits=bad)


def test_cost_parts_use_the_same_registered_reader_closure() -> None:
    from _lcm_regime_finder.cli import load_problem
    from _lcm_regime_finder.input_reads import discover_inputs
    from _lcm_regime_finder.frontier import digest

    analysis, _ = forest_analysis()
    packet = encode_analysis(analysis)
    records = []

    def intern(value):
        if not isinstance(value, (dict, list)):
            return value
        row = (
            {"mapping": [[k, intern(v)] for k, v in sorted(value.items())]}
            if isinstance(value, dict)
            else {"sequence": [intern(v) for v in value]}
        )
        records.append(row)
        return {"ref": len(records) - 1}

    root = intern(packet)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        part = directory / "part.json"
        part.write_text(json.dumps({"records": records}))
        source = directory / "input.json"
        source.write_text(
            json.dumps(
                {
                    "schema": "interned-json-cost-instance/v1",
                    "root": root,
                    "decoded_sha256": digest(packet),
                    "record_parts": [
                        {
                            "path": part.name,
                            "records": len(records),
                            "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
                        }
                    ],
                }
            )
        )
        reads = discover_inputs([source])
        assert (reads.bind(part)) == (part.resolve())
        assert (load_problem(source, reads=reads)) == (packet)
        part.write_text("{}")
        with pytest.raises(ValueError):
            load_problem(source, reads=reads)


def test_cli_cost_delivery_and_atomic_incomplete_checkpoints() -> None:
    from _lcm_regime_finder.cli import main

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        analysis, _ = forest_analysis()
        packet = encode_analysis(analysis)
        source = root / "source.json"
        source.write_text(json.dumps(packet))
        output = root / "result"
        code = main(
            [
                "find-regimes",
                str(source),
                "--output",
                str(output),
                "--worker",
                "--seconds",
                "20",
            ]
        )
        assert (code) == (0)
        report = json.loads((output / "report.json").read_text())
        assert report["independent_cost_check"]["cost_family_complete_verified"]
        checkpoints = list((output / "cost-checkpoints").rglob("ROOT.json"))
        assert checkpoints
        for path in checkpoints:
            record = json.loads(path.read_text())
            assert not (record["complete_cost_family"])
            assert (record["sha256"]) == (
                hashlib.sha256((path.parent / record["table"]).read_bytes()).hexdigest()
            )
        checked = root / "check.json"
        assert (
            main(
                [
                    "check-result",
                    str(source),
                    str(output / "report.json"),
                    "--output",
                    str(checked),
                    "--worker",
                ]
            )
        ) == (0)
        assert json.loads(checked.read_text())["cost_family_complete_verified"]


def test_partial_cost_incumbent_is_checked_and_cannot_be_forged() -> None:
    analysis, _ = forest_analysis()
    result = find_cost_frontier(analysis, seconds=0)
    assert (result["status"]) == ("INCOMPLETE")
    checked = check_cost_frontier(analysis, result)
    assert checked["original_representation_verified"]
    assert not (checked["cost_family_complete_verified"])
    result["original_representation"]["cost"]["work"] -= 1
    assert (check_cost_frontier(analysis, result)["status"]) == ("FAIL")


def test_forest_public_route_covers_unsupported_points() -> None:
    analysis, original = forest_analysis()
    result = find_cost_frontier(analysis, seconds=30)
    assert (result["status"]) == ("COMPLETE_COST_FAMILY")
    assert (result["frontier"]) == (sorted(literal(original)))
    assert (check_cost_frontier(analysis, result)["status"]) == ("PASS")
    assert ((20, 20, 9)) in (result["frontier"])
    omitted = deepcopy(result)
    omitted["frontier"] = [
        point for point in omitted["frontier"] if tuple(point) != (20, 20, 9)
    ]
    proof = omitted["components"][0]["proof"]
    proof["frontier"] = [
        point for point in proof["frontier"] if tuple(point["cost"]) != (20, 20, 9)
    ]
    assert (check_cost_frontier(analysis, omitted)["status"]) == ("FAIL")
    assert (result["legacy_endpoint_preference"]) == ("NOT_APPLIED")
    queried = query_cost_frontier(analysis, result, upper_bounds=(20, 20, 9))
    assert (queried["witness_cost"]) == ((20, 20, 9))
    assert queried["witness"]
    assert not (query_cost_frontier(analysis, result, upper_bounds=(0, 0, 0))["answer"])
    assert not (
        query_cost_frontier(analysis, result, dominated_vector=(20, 20, 9))["answer"]
    )
    assert query_cost_frontier(analysis, result, dominated_vector=(21, 21, 10))[
        "answer"
    ]


def test_exact_component_omission_and_source_promotion_fail() -> None:
    analysis, _ = forest_analysis()
    result = find_cost_frontier(analysis)
    for field, value in [
        ("components", []),
        ("frontier", []),
        ("source_semantics", "PASS"),
    ]:
        changed = deepcopy(result)
        changed[field] = value
        assert (check_cost_frontier(analysis, changed)["status"]) == ("FAIL")
    original_changed = deepcopy(analysis)
    original_changed["declarations"]["domains"]["x"]["size"] += 1
    assert (check_cost_frontier(original_changed, result)["status"]) == ("FAIL")


def test_separator_discovered_without_axis_or_label_names() -> None:
    analysis = fixture(2)
    for cell in analysis["declarations"]["cells"]:
        cell["regime"] = "renamed-" + cell["regime"]
    for record in analysis["interfaces"].values():
        for option in record["options"]:
            if "health" in option["states"]:
                option["states"]["different-axis"] = option["states"].pop("health")
                option["state_projection"]["different-axis"] = option[
                    "state_projection"
                ].pop("health")
    result = find_cost_frontier(analysis, seconds=30)
    assert (result["backend"]) == ("ordered-domain-separator")
    assert (check_cost_frontier(analysis, result)["status"]) == ("PASS")
    assert query_cost_frontier(analysis, result, weights=(1, 1, 1))["witness"]


def test_symbolic_complete_closure_and_source_boundary() -> None:
    packet, chains = toy()
    result = find_regimes(packet, budget={"seconds": 30})
    assert (result["backend"]) == ("symbolic-policy")
    assert (result["status"]) == ("COMPLETE_COST_FAMILY")
    assert (verify_certificate(packet, result)["status"]) == ("PASS")
    assert query_cost_frontier(packet, result, weights=(1, 1, 1))["witness"]
    cover = chain_cover(packet["characteristic_candidates"][0])
    assert (set(c for row in cover for c in row)) == (
        set(c for row in chains for c in row)
    )
    changed = deepcopy(result)
    changed["proof"]["frontier"] = []
    assert (check_cost_frontier(packet, changed)["status"]) == ("FAIL")


def test_cost_packet_roundtrip_retains_original_policy_choices() -> None:
    analysis, _ = forest_analysis()
    packet = encode_analysis(analysis)
    result = find_regimes(packet)
    assert (verify_certificate(packet, result)["status"]) == ("PASS")
    forged = deepcopy(packet)
    forged["policy_encoding"]["basis"][0] = -1
    with pytest.raises(ValueError):
        find_regimes(forged)
