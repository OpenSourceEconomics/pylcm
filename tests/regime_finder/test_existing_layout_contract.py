"""Existing-cell layout admission and raw regime-count objectives."""

from copy import deepcopy
from collections.abc import Callable, Sequence
from dataclasses import asdict
from importlib.resources import files
from itertools import product
import json
from pathlib import Path
import sys

import pytest
from lcm import Model

from _lcm_regime_finder.frontier import Search
from _lcm_regime_finder.production import analyze_production
from tests.regime_finder.test_round5_local_obligations import native_fixture
from tests.test_graphs import period_model
from tests.regime_finder.existing_family_oracle import (
    JSON,
    CellId,
    Label,
    Layout,
    at_most_curve,
    enumerate_layouts,
    pareto_layouts,
    _mapping,
    _rows,
    _name,
)


def _read_admission() -> dict[str, JSON]:
    value = json.loads(
        files("_lcm_regime_finder").joinpath("block_admission_v1.json").read_text()
    )
    assert isinstance(value, dict)
    return value


def _create_case(case: str) -> dict[str, JSON]:
    from tests.regime_finder.test_round6_frontier import fixture

    if case == "tradeoff":
        analysis = fixture(n=1)
        analysis["declarations"]["domains"].update(
            wealth={"size": 10}, choice={"size": 100}, coarse={"size": 2}
        )
        keep = analysis["interfaces"]["a:0"]["options"][0]
        coarse = deepcopy(keep)
        coarse.update(id="coarse", states={"wealth_class": "coarse"})
        coarse["state_projection"]["wealth"] = {
            "kind": "characteristic",
            "name": "wealth_class",
        }
        forced = deepcopy(keep)
        forced.update(id="forced", actions={})
        forced["action_reconstruction"]["choice"] = {"kind": "constant", "value": 0}
        analysis["interfaces"]["a:0"]["options"].extend((coarse, forced))
        analysis["projection_policies"].extend(
            {"id": option, "option_by_cell": {"a:0": option}}
            for option in ("coarse", "forced")
        )
    elif case in {"single_date", "equal_signatures"}:
        analysis = fixture(n=1, labels=("a", "b", "c"))
        analysis["declarations"]["domains"]["risk"] = {"size": 2}
        analysis["declarations"]["domains"]["wealth"]["size"] = 3
        analysis["interfaces"]["a:0"]["options"][0]["states"] = {}
        if case == "equal_signatures":
            analysis["interfaces"]["b:0"]["options"][0]["states"] = {}
        else:
            analysis["interfaces"]["c:0"]["options"][0]["states"] = {"risk": "risk"}
    else:
        analysis = fixture(n=3, labels=("a", "b") if case != "policies" else ("a",))
    analysis["clarity_weights"] = {"block": 1, "characteristic": 0, "dispatch": 0}
    analysis["declarations"]["analysis_scope"] = "pylcm-structural-graph/v1"
    if case == "bucket_union":
        analysis["declarations"]["domains"]["health"] = {"size": 2}
        for cell in analysis["declarations"]["cells"]:
            if cell["regime"] == "b":
                cell["states"]["health"] = "health"
                option = analysis["interfaces"][cell["id"]]["options"][0]
                option["states"]["health"] = "health"
                option["state_projection"]["health"] = {"kind": "identity"}
    if case in {"links", "axis_changes"}:
        analysis["declarations"]["relations"] = [
            {"source": "a:0", "targets": ["a:1"], "support": "positive"},
            {"source": "a:1", "targets": ["a:2"], "support": "unknown"},
            {"source": "a:1", "targets": ["a:2"], "support": "positive"},
            {"source": "b:0", "targets": ["b:2"], "support": "zero"},
            {"source": "a:2", "targets": ["external"], "support": "unknown"},
            {"source": "b:1", "targets": ["a:1"], "support": "positive"},
        ]
    if case == "axis_changes":
        for cid, row in analysis["interfaces"].items():
            option = row["options"][0]
            if cid.endswith(":2"):
                option["states"] = {}
                option["state_projection"]["wealth"] = {"kind": "drop", "value": 0}
                option["actions"] = {}
                option["action_reconstruction"]["choice"] = {
                    "kind": "constant",
                    "value": 0,
                }
    if case == "domains":
        analysis["declarations"]["domains"]["other"] = {"size": 13}
        for cid, row in analysis["interfaces"].items():
            if cid.startswith("b:"):
                row["options"][0]["states"]["wealth"] = "other"
    if case == "contexts":
        for cell in analysis["declarations"]["cells"]:
            cell["role"] = {"actor": cell["regime"]}
            cell["ownership"] = [cell["regime"]]
            cell["terminal"] = cell["period"] == 2
            if cell["id"] == "b:2":
                cell["stage"] = "another-stage"
    if case == "policies":
        analysis["declarations"]["domains"]["coarse"] = {"size": 2}
        analysis["declarations"]["domains"]["unused"] = {"size": 5}
        for cell in analysis["declarations"]["cells"]:
            cell["states"]["unused"] = "unused"
            keep = analysis["interfaces"][cell["id"]]["options"][0]
            keep["states"]["unused"] = "unused"
            keep["state_projection"]["unused"] = {"kind": "identity"}
            lean = deepcopy(keep)
            lean["id"] = "lean"
            lean["states"].pop("unused")
            lean["state_projection"]["unused"] = {"kind": "drop", "value": 0}
            analysis["interfaces"][cell["id"]]["options"].append(lean)
        for cid in ("a:1", "a:2"):
            keep = analysis["interfaces"][cid]["options"][1]
            coarse = deepcopy(keep)
            coarse.update(id="coarse", states={"wealth_class": "coarse"})
            coarse["state_projection"]["wealth"] = {
                "kind": "characteristic",
                "name": "wealth_class",
            }
            drop = deepcopy(keep)
            drop.update(id="drop", states={})
            drop["state_projection"]["wealth"] = {"kind": "drop", "value": 0}
            if cid == "a:2":
                for option in (coarse, drop):
                    option["actions"] = {}
                    option["action_reconstruction"]["choice"] = {
                        "kind": "constant",
                        "value": 0,
                    }
            analysis["interfaces"][cid]["options"].extend((coarse, drop))
        analysis["projection_policies"].append(
            {
                "id": "lean",
                "option_by_cell": {"a:0": "lean", "a:1": "lean", "a:2": "lean"},
            }
        )
        analysis["projection_policies"].extend(
            {
                "id": choice,
                "option_by_cell": {"a:0": "lean", "a:1": choice, "a:2": choice},
            }
            for choice in ("coarse", "drop")
        )
    return analysis


def _vector(point: dict[str, JSON]) -> tuple[int, int, int]:
    cost = point["cost"]
    assert isinstance(cost, dict)
    work, storage, regimes = (cost[key] for key in ("work", "storage", "clarity"))
    assert type(work) is int and type(storage) is int and type(regimes) is int
    return work, storage, regimes


def _create_curve_case(case: str) -> dict[str, JSON]:
    analysis = _create_case(case)
    allocation = _mapping(_mapping(analysis["declarations"])["allocation"])
    analysis["allocation_scenario"] = dict(allocation, boundary_unit=0)
    return analysis


class _LiteralMismatch(AssertionError):
    """The search lost or mispriced an attainable complete layout."""


def _assert_frontier_matches_literal(
    *, analysis: dict[str, JSON]
) -> tuple[Layout, ...]:
    from _lcm_regime_finder.frontier import search_frontier
    from _lcm_regime_finder.frontier_check import check_frontier

    layouts = enumerate_layouts(analysis=analysis, admission=_read_admission())
    expected = {layout.cost for layout in pareto_layouts(layouts)}
    result = search_frontier(analysis)
    assert result["status"] == "COMPLETE", result
    actual = {_vector(point) for point in result["frontier"]}
    if actual != expected:
        raise _LiteralMismatch({"expected": sorted(expected), "actual": sorted(actual)})
    for point in result["frontier"]:
        witness = sorted(
            (tuple(block["cells"]), tuple(sorted(block["option_by_cell"].items())))
            for block in point["blocks"]
        )
        assert any(
            layout.cost == _vector(point)
            and sorted((block.cells, block.selected_options) for block in layout.blocks)
            == witness
            for layout in layouts
        ), point
    checked = check_frontier(analysis, result)
    assert checked["status"] == "PASS", checked
    return layouts


def test_default_third_objective_counts_blocks_without_dispatch_surcharges() -> None:
    from tests.regime_finder.test_round6_frontier import fixture

    analysis = fixture(n=1)
    search = Search(analysis)
    policy = analysis["projection_policies"][0]["option_by_cell"]
    block = search.block(["a:0"], policy)
    assert block is not None
    assert block["cost"]["clarity"] == 1


def test_search_and_diagnostic_agree_for_missing_original_roles() -> None:
    original = native_fixture()
    for declaration in original["user_regimes"].values():
        declaration["fields"].pop("semantic_role")
    report = analyze_production(original, search_mode="exact")
    analysis = report["analysis"]
    search = Search(analysis)
    policy = analysis["projection_policies"][0]["option_by_cell"]
    ids = [cell["id"] for cell in analysis["declarations"]["cells"]]
    assert search.block(ids, policy) is not None
    pairs = analysis["family_boundary"]["conditional_merge_candidates"]
    assert not any(row["status"] == "CONDITIONAL_NOT_ADMITTED" for row in pairs)


def test_different_original_actors_can_share_allocation_with_source_contexts_intact() -> (
    None
):
    original = native_fixture()
    preserved = deepcopy(original)
    report = analyze_production(original, search_mode="exact")
    analysis = report["analysis"]
    cells = analysis["declarations"]["cells"]
    assert len({cell["role"]["actor"] for cell in cells}) == 2
    search = Search(analysis)
    policy = analysis["projection_policies"][0]["option_by_cell"]
    block = search.block([cell["id"] for cell in cells], policy)
    assert block is not None
    assert set(block["cells"]) == {cell["id"] for cell in cells}
    assert original == preserved


@pytest.mark.parametrize(
    "case",
    [
        "basic",
        "links",
        "axis_changes",
        "domains",
        "contexts",
        "policies",
        "single_date",
    ],
)
def test_complete_existing_family_matches_literal_enumeration(
    *,
    case: str,
    record_property: Callable[[str, str], None],
) -> None:
    layouts = _assert_frontier_matches_literal(analysis=_create_case(case))
    record_property(
        "literal_layout_witnesses", json.dumps([asdict(layout) for layout in layouts])
    )
    record_property(
        "literal_at_most_curve",
        json.dumps([asdict(layout) for layout in at_most_curve(layouts=layouts)]),
    )


def test_literal_policies_include_a_strict_dominator_of_the_unreduced_layout() -> None:
    analysis = _create_case("policies")
    layouts = enumerate_layouts(analysis=analysis, admission=_read_admission())
    unreduced = next(
        layout
        for layout in layouts
        if layout.policy == "P0" and len(layout.blocks) == 1
    )
    dominating = [
        layout
        for layout in layouts
        if layout.cost != unreduced.cost
        and all(
            left <= right
            for left, right in zip(layout.cost, unreduced.cost, strict=True)
        )
    ]
    assert dominating
    assert any(layout.policy == "drop" for layout in dominating)
    assert unreduced not in pareto_layouts(layouts)


def test_literal_family_covers_every_original_date_cut_and_empty_axis_product() -> None:
    analysis = _create_case("policies")
    layouts = enumerate_layouts(analysis=analysis, admission=_read_admission())
    assert {len(layout.blocks) for layout in layouts if layout.policy == "P0"} == {
        1,
        2,
        3,
    }
    empty = [
        block
        for layout in layouts
        for block in layout.blocks
        if block.cells == ("a:2",) and layout.policy == "drop"
    ]
    assert empty and all(
        block.state_rows == block.action_rows == block.evaluations == block.storage == 1
        for block in empty
    )


@pytest.mark.parametrize(
    "mutation",
    ["old_block_charge", "lose_cross_label_grouping", "ignore_reduction_policies"],
)
def test_literal_oracle_detects_semantic_search_mutations(
    *,
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = Search.run

    def altered_run(self: Search) -> dict[str, JSON]:
        if mutation == "old_block_charge":
            self.rules = {"block": 4, "characteristic": 2, "dispatch": 1}
        elif mutation == "ignore_reduction_policies":
            self.analysis = dict(
                self.analysis,
                projection_policies=self.analysis["projection_policies"][:1],
            )
        return original_run(self)

    if mutation == "lose_cross_label_grouping":

        def separate_labels(self: Search) -> list[dict[Label, list[CellId]]]:
            return [
                {
                    cell["regime"]: [
                        cid
                        for cid, other in self.cells.items()
                        if other["regime"] == cell["regime"]
                    ]
                }
                for cell in self.cells.values()
                if cell["period"] == 0
            ]

        monkeypatch.setattr(Search, "components", separate_labels)
    else:
        monkeypatch.setattr(Search, "run", altered_run)
    case = "policies" if mutation == "ignore_reduction_policies" else "basic"
    with pytest.raises(_LiteralMismatch):
        _assert_frontier_matches_literal(analysis=_create_case(case))


@pytest.mark.parametrize("field", ["role", "terminal"])
def test_independent_checker_rejects_a_forged_source_cell_context(
    field: str,
) -> None:
    from _lcm_regime_finder.frontier import search_frontier
    from _lcm_regime_finder.frontier_check import check_frontier

    analysis = _create_case("contexts")
    result = search_frontier(analysis)
    assert check_frontier(analysis, result)["status"] == "PASS"
    cells = {
        _name(cell["id"]): cell
        for cell in _rows(_mapping(analysis["declarations"])["cells"])
    }
    forged = deepcopy(result)
    changed = False
    for point in forged["frontier"]:
        for block in point["blocks"]:
            for left in block["cells"]:
                for right in block["cells"]:
                    if cells[left][field] != cells[right][field]:
                        bindings = block["admission"]["source_contexts"]
                        bindings[left] = bindings[right]
                        changed = True
    assert changed
    checked = check_frontier(analysis, forged)
    assert checked["status"] == "FAIL", checked
    assert any(
        error.endswith(": incorrect block admission") for error in checked["errors"]
    ), checked


def test_curve_reconstructs_an_offered_identity_layout_with_complete_membership() -> (
    None
):
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier

    analysis = _create_curve_case("basic")
    result = search_frontier(analysis, reporting_mode="curve")
    report = build_curve_report(analysis, result)
    assert report["status"] == "COMPLETE"
    own = report["our_layout"]
    assert own["status"] == "dominated"
    assert own["witness"]["projection_policy"] == "P0"
    assert sorted(
        cid for block in own["witness"]["blocks"] for cid in block["cells"]
    ) == sorted(_mapping(analysis["interfaces"]))


def test_raw_curve_search_does_not_call_the_scalar_recommendation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _lcm_regime_finder import frontier

    def forbidden_recommendation(*args: JSON, **kwargs: JSON) -> None:
        raise AssertionError("Raw evaluation curves cannot use normalized ranking")

    monkeypatch.setattr(frontier, "recommend", forbidden_recommendation)
    result = frontier.search_frontier(
        _create_curve_case("basic"), reporting_mode="curve"
    )
    assert result["status"] == "COMPLETE"
    assert "recommendation" not in result


@pytest.mark.parametrize(
    "case",
    [
        "basic",
        "links",
        "axis_changes",
        "domains",
        "contexts",
        "policies",
        "single_date",
        "tradeoff",
    ],
)
def test_raw_curve_matches_literal_at_most_regime_bounds(case: str) -> None:
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier

    analysis = _create_curve_case(case)
    layouts = enumerate_layouts(analysis=analysis, admission=_read_admission())
    result = search_frontier(analysis, reporting_mode="curve")
    report = build_curve_report(analysis, result)
    assert report["schema"] == "regime-evaluation-curve/v1"
    assert report["scope"] == "existing family, no dense start"
    assert report["status"] == "COMPLETE"
    assert not {"recommendation", "score", "threshold"}.intersection(report)
    assert {
        (point["evaluations"], point["storage"], point["regimes"])
        for point in report["frontier"]
    } == {layout.cost for layout in pareto_layouts(layouts)}
    for row in report["curve"]:
        eligible = [
            layout for layout in layouts if len(layout.blocks) <= row["at_most_regimes"]
        ]
        expected = min(
            (layout.cost[0], layout.cost[2], layout.cost[1]) for layout in eligible
        )
        assert (row["evaluations"], row["regimes"], row["storage"]) == expected
    cells = {
        _name(cell["id"]): cell
        for cell in _rows(_mapping(analysis["declarations"])["cells"])
    }
    for point in report["frontier"]:
        witness = sorted(
            (tuple(block["cells"]), tuple(sorted(block["option_by_cell"].items())))
            for block in point["witness"]["blocks"]
        )
        literal = next(
            layout
            for layout in layouts
            if sorted((block.cells, block.selected_options) for block in layout.blocks)
            == witness
        )
        assert point["grid_evaluations"] == literal.evaluations
        assert point["link_charge_evaluations"] == sum(
            block.link_charge for block in literal.blocks
        )
        assert point["directed_link_count"] == sum(
            block.directed_links for block in literal.blocks
        )
        assert point["unpriced_link_row_units"] == sum(
            block.unpriced_link_rows for block in literal.blocks
        )
        assert point["characteristic_maps"] == sum(
            block.characteristics for block in literal.blocks
        )
        assert point["dispatch_clauses"] == sum(
            block.dispatch for block in literal.blocks
        )
        largest = max(block.state_rows * block.action_rows for block in literal.blocks)
        largest_values = max(block.state_rows for block in literal.blocks)
        assert point["largest_single_array"] == largest
        assert point["largest_stored_value_array"] == largest_values
        at_largest = [
            cid
            for block in literal.blocks
            if block.state_rows * block.action_rows == largest
            for cid in block.cells
        ]
        assert point["periods_at_largest_array"] == len(
            {cells[cid]["period"] for cid in at_largest}
        )
        assert point["regime_period_arrays_at_largest_size"] == len(at_largest)


def test_our_layout_has_a_real_weak_dominance_witness() -> None:
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier

    analysis = _create_curve_case("policies")
    report = build_curve_report(
        analysis, search_frontier(analysis, reporting_mode="curve")
    )
    own = report["our_layout"]
    assert own["status"] == "dominated"
    rival = own["dominating_layout"]
    original_cost = tuple(own[key] for key in ("evaluations", "storage", "regimes"))
    rival_cost = tuple(rival[key] for key in ("evaluations", "storage", "regimes"))
    assert rival_cost != original_cost
    assert all(
        left <= right for left, right in zip(rival_cost, original_cost, strict=True)
    )
    assert own["curve_attained"] is False


def test_unoffered_original_identity_is_not_invented_as_a_family_member() -> None:
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier

    analysis = _create_curve_case("policies")
    policies = _rows(analysis["projection_policies"])
    analysis["projection_policies"] = [
        policy for policy in policies if policy["id"] != "P0"
    ]
    report = build_curve_report(
        analysis, search_frontier(analysis, reporting_mode="curve")
    )
    assert report["our_layout"]["status"] == "not_admitted"
    assert report["our_layout"]["curve_attained"] is None
    assert "dominating_layout" not in report["our_layout"]


@pytest.mark.parametrize("unfinished", ["search_budget", "discovery", "heuristic"])
def test_incomplete_curve_does_not_claim_complete_family_optimality(
    unfinished: str,
) -> None:
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier
    from _lcm_regime_finder.heuristic import search_heuristic

    analysis = _create_curve_case("axis_changes")
    if unfinished == "discovery":
        analysis["discovery_complete"] = False
    if unfinished == "heuristic":
        analysis = _create_curve_case("policies")
        result = search_heuristic(analysis)
    else:
        limits = {"segments": 0} if unfinished == "search_budget" else None
        result = search_frontier(analysis, limits=limits, reporting_mode="curve")
    report = build_curve_report(analysis, result)
    assert report["status"] == "PARTIAL"
    assert report["source_semantics"] == "NOT_ESTABLISHED_BY_COST_SEARCH"
    assert report["our_layout"]["status"] != "on_frontier"


def test_pareto_membership_does_not_imply_minimum_work_at_the_regime_bound() -> None:
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier

    analysis = _create_curve_case("tradeoff")
    result = search_frontier(analysis, reporting_mode="curve")
    own = max(result["frontier"], key=lambda point: point["cost"]["work"])
    report = build_curve_report(analysis, result, our_layout=own)
    assert report["our_layout"]["status"] == "on_frontier"
    assert report["our_layout"]["curve_attained"] is False
    assert report["our_layout"]["dominating_layout"] is None


@pytest.mark.parametrize("backend", ["recurrence", "symbolic", "separator"])
def test_independent_admission_replay_does_not_call_the_producer_predicate(
    *,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _lcm_regime_finder import block_admission
    from _lcm_regime_finder.frontier import search_frontier
    from _lcm_regime_finder.frontier_check import check_frontier
    from _lcm_regime_finder import separator_frontier, symbolic_frontier
    from tests.regime_finder.test_aca_separator_reference import (
        fixture as separator_fixture,
    )
    from tests.regime_finder.test_borella_symbolic import toy

    def forbidden_context(*args: JSON, **kwargs: JSON) -> str:
        raise AssertionError(
            "Independent admission must not call the producer predicate"
        )

    if backend == "recurrence":
        analysis = _create_case("contexts")
        result = search_frontier(analysis)
        monkeypatch.setattr(block_admission, "context_key", forbidden_context)
        checked = check_frontier(analysis, result)
    elif backend == "symbolic":
        packet, chains = toy()
        certificate = symbolic_frontier.solve(packet=packet, chains=chains)
        monkeypatch.setattr(block_admission, "context_key", forbidden_context)
        checked = symbolic_frontier.check(packet=packet, certificate=certificate)
    else:
        problem = separator_frontier.Problem(
            separator_fixture(left_count=1, right_count=1)
        )
        result = separator_frontier.produce(problem)
        monkeypatch.setattr(block_admission, "context_key", forbidden_context)
        checked = separator_frontier.check(problem, result)
    expected_status = "PASS_COST_FAMILY" if backend == "symbolic" else "PASS"
    assert checked["status"] == expected_status, checked


def test_heuristic_curve_does_not_present_dominated_policy_preferences_as_frontier() -> (
    None
):
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.heuristic import search_heuristic

    analysis = _create_curve_case("policies")
    result = search_heuristic(analysis)
    frontier_ids = {point["id"] for point in result["frontier"]}
    policy_winner_ids = {point["id"] for point in result["candidates"]}
    assert frontier_ids < policy_winner_ids
    report = build_curve_report(analysis, result)
    assert report["status"] == "PARTIAL"
    assert {point["graph_id"] for point in report["frontier"]} == frontier_ids


def test_curve_profile_rejects_posthoc_removal_of_priced_links() -> None:
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier

    priced = _create_case("links")
    with pytest.raises(ValueError, match="boundary|link"):
        search_frontier(priced, reporting_mode="curve")
    legacy = search_frontier(priced)
    with pytest.raises(ValueError, match="boundary|link"):
        build_curve_report(priced, legacy)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_zero_price_forest_certificates_match_the_same_literal_existing_family(
    *,
    version: str,
    record_property: Callable[[str, str], None],
) -> None:
    from _lcm_regime_finder.forest_frontier import optimize
    from _lcm_regime_finder.forest_check import verify
    from tests.regime_finder.test_cost_family import forest_analysis

    analysis, instance = forest_analysis()
    weights = {"block": 1, "characteristic": 0, "dispatch": 0}
    analysis["clarity_weights"] = dict(weights)
    instance["clarity_weights"] = dict(weights)
    if version == "v1":
        instance["cells"] = instance["cells"][:1]
        analysis["declarations"]["cells"] = analysis["declarations"]["cells"][:1]
        cid = instance["cells"][0]["id"]
        analysis["interfaces"] = {cid: analysis["interfaces"][cid]}
        policy = analysis["projection_policies"][0]["option_by_cell"]
        analysis["projection_policies"][0]["option_by_cell"] = {cid: policy[cid]}
    else:
        instance["pairs"] = [["0", "1"], ["1", "2"]]
        analysis["declarations"]["relations"] = [
            {"source": left, "targets": [right], "support": "unknown"}
            for left, right in instance["pairs"]
        ]
    layouts = enumerate_layouts(analysis=analysis, admission=_read_admission())
    expected = {layout.cost for layout in pareto_layouts(layouts)}
    certificate = optimize(instance, seconds=None)
    if version == "v1":
        assert certificate["resource"]["mask_dominated_points"] == 0
        certificate["schema"] = "regime-singleton-forest-mask-frontier/v1"
        certificate.pop("state_cover_rule")
    checked = verify(instance, certificate)
    assert checked["status"] == "PASS", checked
    assert set(map(tuple, checked["vectors"])) == expected
    record_property(
        "literal_layout_witnesses", json.dumps([asdict(layout) for layout in layouts])
    )


def test_zero_price_singleton_signature_matches_the_literal_existing_family() -> None:
    from _lcm_regime_finder.frontier import search_frontier

    analysis = _create_case("single_date")
    declaration = _mapping(analysis["declarations"])
    assert isinstance(declaration, dict)
    declaration["relations"] = [
        {"source": left, "targets": [right], "support": "unknown"}
        for left, right in (("a:0", "b:0"), ("b:0", "c:0"), ("c:0", "a:0"))
    ]
    _assert_frontier_matches_literal(analysis=analysis)
    result = search_frontier(analysis)
    assert result["certificate"]["signature_records"]


def test_zero_price_ordered_separator_matches_the_literal_existing_family(
    record_property: Callable[[str, str], None],
) -> None:
    from _lcm_regime_finder.separator_frontier import Problem, produce, check
    from tests.regime_finder.test_aca_separator_reference import fixture

    analysis = fixture(left_count=1, right_count=1)
    analysis["clarity_weights"] = {"block": 1, "characteristic": 0, "dispatch": 0}
    layouts = enumerate_layouts(analysis=analysis, admission=_read_admission())
    expected = {layout.cost for layout in pareto_layouts(layouts)}
    problem = Problem(analysis)
    result = produce(problem)
    assert result["status"] == "COMPLETE"
    assert check(problem, result)["status"] == "PASS"
    assert {_vector(point) for point in result["frontier"]} == expected
    record_property(
        "literal_layout_witnesses", json.dumps([asdict(layout) for layout in layouts])
    )


def test_zero_price_symbolic_policies_match_literal_closed_profile_enumeration(
    record_property: Callable[[str, str], None],
) -> None:
    from _lcm_regime_finder.symbolic_frontier import solve, check
    from tests.regime_finder.test_borella_symbolic import toy

    packet, chains = toy()
    packet["clarity_weights"] = {"block": 1, "characteristic": 0, "dispatch": 0}
    expanded = deepcopy(packet)
    ids = packet["policy_encoding"]["cells"]
    candidate = packet["characteristic_candidates"][0]
    requirements = candidate["requires"]
    for cid in ids:
        options = expanded["interfaces"][cid]["options"]
        if len(options) == 1:
            coarsened = deepcopy(options[0])
            coarsened["id"] = "literal-cost-only:" + cid
            coarsened["states"].pop(candidate["state"])
            coarsened["states"][candidate["target"]] = candidate["target_domain"]
            coarsened["state_projection"][candidate["state"]] = {
                "kind": "characteristic",
                "callback": candidate["callback"],
                "target": candidate["target"],
                "domain": candidate["target_domain"],
                "values": candidate["values"],
            }
            options.append(coarsened)
    policies = []
    for bits in product((False, True), repeat=len(ids)):
        selected = {cid for cid, bit in zip(ids, bits, strict=True) if bit}
        if any(
            cid in selected and target not in selected
            for cid, targets in requirements.items()
            for target in targets
        ):
            continue
        policies.append(
            {
                "id": "profile-" + "".join(str(int(bit)) for bit in bits),
                "option_by_cell": {
                    cid: expanded["interfaces"][cid]["options"][int(bit)]["id"]
                    for cid, bit in zip(ids, bits, strict=True)
                },
            }
        )
    expanded["projection_policies"] = policies
    layouts = enumerate_layouts(analysis=expanded, admission=_read_admission())
    expected = {layout.cost for layout in pareto_layouts(layouts)}
    certificate = solve(packet=packet, chains=chains)
    assert certificate["status"] == "COMPLETE_COST_FAMILY"
    assert check(packet=packet, certificate=certificate)["status"] == "PASS_COST_FAMILY"
    assert set(map(tuple, certificate["frontier"])) == expected
    record_property(
        "literal_layout_witnesses", json.dumps([asdict(layout) for layout in layouts])
    )


@pytest.mark.parametrize("ragged", [False, True])
def test_rectangle_comparison_does_not_reprice_source_cell_search(ragged: bool) -> None:
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier

    analysis = _create_curve_case("basic")
    declaration = _mapping(analysis["declarations"])
    assert isinstance(declaration, dict)
    if ragged:
        declaration["cells"] = [
            cell for cell in _rows(declaration["cells"]) if cell["id"] != "b:2"
        ]
        interfaces = _mapping(analysis["interfaces"])
        assert isinstance(interfaces, dict)
        interfaces.pop("b:2")
        policies = _rows(analysis["projection_policies"])
        policy = _mapping(policies[0]["option_by_cell"])
        assert isinstance(policy, dict)
        policy.pop("b:2")
    report = build_curve_report(
        analysis, search_frontier(analysis, reporting_mode="curve")
    )
    point = report["frontier"][0]
    assert point["regimes"] == 1
    blocks = point["block_reports"]
    assert len(blocks) == 1
    block = blocks[0]
    assert block["source_regime_count"] == 2
    assert block["period_count"] == 3
    assert block["source_cell_count"] == (5 if ragged else 6)
    rows = block["state_action_rows"]
    assert (
        point["source_cell_evaluations"]
        == point["grid_evaluations"]
        == (5 if ragged else 6) * rows
    )
    assert point["emitted_rectangle_evaluations"] == 2 * 3 * rows
    assert point["rectangle_extra_evaluations"] == (rows if ragged else 0)
    assert point["evaluations"] == point["source_cell_evaluations"]
    assert block["source_identity_dispatch_available"] is False
    assert block["emittable_as_one_regime"] is False
    assert block["reasons"]


def test_mixed_terminal_source_cells_report_unestablished_regime_emission() -> None:
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier

    analysis = _create_curve_case("basic")
    for cell in _rows(_mapping(analysis["declarations"])["cells"]):
        assert isinstance(cell, dict)
        cell["terminal"] = cell["regime"] == "b"
    result = search_frontier(analysis, reporting_mode="curve")
    report = build_curve_report(analysis, result)
    block = report["frontier"][0]["block_reports"][0]
    assert block["terminal_uniform_by_period"] is False
    assert block["emittable_as_one_regime"] is False
    assert block["reasons"]


@pytest.mark.parametrize(
    ("case", "rule"),
    [
        ("basic", "homogeneous-layout-dominance/v1"),
        ("bucket_union", "constant-date-bucket-union/v1"),
        ("axis_changes", "constant-prefix-single-cut/v1"),
        ("equal_signatures", "single-date-layout-contraction/v1"),
        ("domains", "incompatible-domain-components/v1"),
        ("bucket_union", "label-subset-recurrence/v1"),
    ],
)
def test_each_retained_shortcut_has_a_zero_surcharge_literal_family_control(
    *,
    case: str,
    rule: str,
) -> None:
    from _lcm_regime_finder.frontier import search_frontier

    analysis = _create_case(case)
    _assert_frontier_matches_literal(analysis=analysis)
    result = search_frontier(analysis)
    certificate = result["certificate"]
    observed = {record["rule"] for record in certificate["component_reductions"]}
    observed.update(record["rule"] for record in certificate["grouping_records"])
    observed.update(
        record["reduction"]["rule"]
        for record in certificate["interval_records"]
        if "reduction" in record
    )
    assert rule in observed


def test_synthetic_identity_packet_does_not_claim_a_constructed_regime_witness() -> (
    None
):
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier

    analysis = _create_curve_case("basic")
    declaration = _mapping(analysis["declarations"])
    assert isinstance(declaration, dict)
    declaration["production_metadata"] = {"native_model_constructed": False}
    for cell in _rows(declaration["cells"]):
        assert isinstance(cell, dict)
        cell["terminal"] = True
    report = build_curve_report(
        analysis, search_frontier(analysis, reporting_mode="curve")
    )
    own = report["our_layout"]["block_reports"]
    assert own and all(block["unchanged_original_interfaces"] for block in own)
    assert all(block["emittable_as_one_regime"] is False for block in own)


@pytest.mark.parametrize(
    "capture", ["valid", "missing_original", "mismatched_identity"]
)
def test_constructor_witness_requires_an_unchanged_bound_native_capture(
    *,
    capture: str,
    period_model: Model,
) -> None:
    from _lcm_regime_finder.curve import build_curve_report
    from _lcm_regime_finder.frontier import search_frontier

    production = analyze_production(period_model, search_mode="exact")
    analysis = production["analysis"]
    analysis["allocation_scenario"] = dict(
        analysis["declarations"]["allocation"], boundary_unit=0
    )
    if capture == "missing_original":
        analysis["declarations"].pop("original")
    elif capture == "mismatched_identity":
        analysis["declarations"]["source_identity"] = "not-the-captured-original"
    report = build_curve_report(
        analysis, search_frontier(analysis, reporting_mode="curve")
    )
    own = report["our_layout"]["block_reports"]
    assert own and all(
        block["emittable_as_one_regime"] is (capture == "valid") for block in own
    )


def _create_identity_package(tmp_path: Path) -> Path:
    package = tmp_path / "isolated-checker"
    package.mkdir()
    (package / "corpus.py").write_text("IMPLEMENTATION = 'unchanged'\n")
    (package / "pipeline.py").write_text("IMPLEMENTATION = 'unchanged'\n")
    (package / "block_admission_v1.json").write_text(json.dumps(_read_admission()))
    return package


def _change_only_admission_data(package: Path) -> None:
    path = package / "block_admission_v1.json"
    specification = json.loads(path.read_text())
    specification["block"]["missing_field_values"]["stage"] = "changed-default-stage"
    path.write_text(json.dumps(specification))


def test_corpus_implementation_identity_binds_the_normative_admission_data(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _lcm_regime_finder import corpus

    package = _create_identity_package(tmp_path)
    monkeypatch.setattr(corpus, "__file__", str(package / "corpus.py"))
    before = corpus._implementation_identity()
    _change_only_admission_data(package)
    assert corpus._implementation_identity() != before


def test_pipeline_source_map_binds_the_normative_admission_data(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _lcm_regime_finder import pipeline

    package = _create_identity_package(tmp_path)
    monkeypatch.setattr(pipeline, "__file__", str(package / "pipeline.py"))
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}")
    output = tmp_path / "result"
    output.mkdir()
    report = {"status": "INCOMPLETE"}
    before = pipeline.readable_summary(report, snapshot=snapshot, output=output)
    assert "block_admission_v1.json" in before["checker_module_sha256"]
    _change_only_admission_data(package)
    after = pipeline.readable_summary(report, snapshot=snapshot, output=output)
    assert after["checker_module_sha256"] != before["checker_module_sha256"]


@pytest.mark.parametrize("change_during_worker", [False, True])
def test_pipeline_worker_invalidates_a_changed_normative_admission_specification(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_during_worker: bool,
) -> None:
    from _lcm_regime_finder import cli, pipeline

    package = _create_identity_package(tmp_path)
    monkeypatch.setattr(pipeline, "__file__", str(package / "pipeline.py"))
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}")
    output = tmp_path / "result"
    output.mkdir()
    reached = []

    def emit_report(arguments: Sequence[str]) -> int:
        assert "--worker" in arguments
        reached.append(True)
        (output / "report.json").write_text(json.dumps({"status": "INCOMPLETE"}))
        if change_during_worker:
            _change_only_admission_data(package)
        return 3

    monkeypatch.setattr(cli, "main", emit_report)
    monkeypatch.setattr(
        sys, "argv", ["pipeline", str(snapshot), "--output", str(output), "--worker"]
    )
    exit_code = pipeline.main()
    assert reached == [True]
    assert exit_code == (1 if change_during_worker else 3)
