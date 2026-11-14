"""Public-API tests against independent finite and exhaustive partition oracles."""

import pytest
from copy import deepcopy
import json

from _lcm_regime_finder import find_regimes, verify_certificate
from .exhaustive_oracle import FiniteOracle
from . import fixtures


METRICS = {
    "original_problems_compared": 0,
    "cut_masks_enumerated": 0,
    "state_value_comparisons": 0,
    "feasible_q_comparisons": 0,
    "certificate_mutations_rejected": 0,
    "invalid_originals_rejected": 0,
}


maxDiff = 3000


def _check_analysis(problem, compare_optimum=True):
    oracle = FiniteOracle(problem)
    expected = oracle.exhaustive()
    report = find_regimes(problem)
    assert (report["status"]) == ("OPTIMAL"), report.get(
        "error", report.get("diagnostics")
    )
    verification = verify_certificate(problem, report)
    assert (verification["status"]) == ("PASS"), verification
    assert verification.get("optimum_verified"), verification
    assert report["search"]["complete"]
    interfaces = {}
    for atom in report["analysis"]["atoms"]:
        lane = tuple(atom["status"][name] for name in oracle.status_names)
        key = (atom["age"], lane)
        assert (set(atom["live_states"])) == (oracle.live[key]), (
            key,
            "dependency discrepancy",
        )
        assert (atom["forced_actions"]) == (oracle.atoms[key].forced), (
            key,
            "forcing discrepancy",
        )
        interfaces[key] = atom["live_states"]
    assert (set(interfaces)) == (set(oracle.atoms))
    economic = oracle.check_economic_pullback(interfaces)
    if compare_optimum:
        for key, value in expected["cost"].items():
            assert (report["recommended"]["cost"][key]) == (value), (
                problem["name"],
                key,
                report["recommended"]["cost"],
                expected["cost"],
            )
        observed = [
            (
                tuple(block["ages"]),
                tuple(block["status"][name] for name in oracle.status_names),
            )
            for block in report["recommended"]["regimes"]
        ]
        desired = [
            (tuple(block["ages"]), tuple(block["lane"]))
            for block in expected["segments"]
        ]
        assert (observed) == (desired), (
            problem["name"],
            "partition or deterministic tie discrepancy",
        )
    METRICS["original_problems_compared"] += 1
    METRICS["cut_masks_enumerated"] += expected["cut_masks_examined"]
    for key in ("state_value_comparisons", "feasible_q_comparisons"):
        METRICS[key] += economic[key]
    return report, expected, oracle


def test_required_named_mechanisms(*, subtests: pytest.Subtests) -> None:
    for name, problem in fixtures.named_cases().items():
        with subtests.test(case=name):
            # The original grouping is a separate coupled status alternative.
            # The independent cut-mask oracle is explicitly lane-only here.
            baseline_can_win = name in ("rare_incoming", "rare_simulation_incoming")
            _check_analysis(problem, compare_optimum=not baseline_can_win)


def test_generated_global_optima(*, subtests: pytest.Subtests) -> None:
    for seed in range(32):
        with subtests.test(seed=seed):
            _check_analysis(fixtures.generated(seed))


def test_renaming_does_not_change_partition() -> None:
    problem = fixtures.duplicate_coefficients()
    original, _, _ = _check_analysis(problem)
    renamed = deepcopy(problem)
    for index, program in enumerate(renamed["programs"]):
        program["id"] = f"different_label_{99 - index}"
    revised, _, _ = _check_analysis(renamed)
    assert (original["recommended"]["cost"]) == (revised["recommended"]["cost"])
    assert ([r["ages"] for r in original["recommended"]["regimes"]]) == (
        [r["ages"] for r in revised["recommended"]["regimes"]]
    )
    assert ("REDUNDANT_SPLIT") in ({d["kind"] for d in original["diagnostics"]})


def test_endpoint_exception_requires_real_role_boundary() -> None:
    report, expected, oracle = _check_analysis(fixtures.endpoint_with_redundant_label())
    assert (report["recommended"]["cost"]["work"]) == (13)
    assert (expected["cost"]["blocks"]) == (2)
    assert (oracle.exhaustive(clarity=False)["cost"]["work"]) == (12)


def test_nontransitive_greedy_counterexample() -> None:
    report, expected, oracle = _check_analysis(fixtures.nontransitive())
    assert (oracle.segment((), 0, 2)) is not None
    assert (oracle.segment((), 1, 3)) is not None
    assert (oracle.segment((), 0, 3)) is None
    left_greedy = [
        oracle.segment((), 0, 2),
        oracle.segment((), 2, 3),
        oracle.segment((), 3, 4),
    ]
    assert (sum(s["work"] for s in left_greedy)) == (14)
    assert (expected["cost"]["work"]) == (13)
    assert ([r["ages"] for r in report["recommended"]["regimes"]]) == (
        [[0], [1, 2], [3]]
    )


def test_positive_incoming_history_in_both_kernels(
    *, subtests: pytest.Subtests
) -> None:
    zero, _, _ = _check_analysis(fixtures.rare_incoming(False))
    for phase in ("both", "solve", "simulation"):
        with subtests.test(phase=phase):
            positive, _, _ = _check_analysis(
                fixtures.rare_incoming(True, phase), compare_optimum=False
            )
            active = [
                a
                for a in positive["analysis"]["atoms"]
                if a["status"]["status"] == "active"
            ]
            inactive = [
                a
                for a in zero["analysis"]["atoms"]
                if a["status"]["status"] == "active"
            ]
            assert ("history") in (active[0]["live_states"])
            assert ("history") not in (inactive[0]["live_states"])
            tiny = [
                e
                for e in positive["analysis"]["edges"]
                if e["probability"]
                == "1/100000000000000000000000000000000000000000000000000"
            ]
            assert tiny, "Every strictly positive edge must remain represented"


def test_role_order_and_routing_are_real_boundaries(
    *, subtests: pytest.Subtests
) -> None:
    changes = {
        "actor": "employer",
        "observations": ["signal_B", "signal_A"],
        "commitments": ["payment_pending"],
        "discount_boundary": "within_period",
    }
    for field, value in changes.items():
        with subtests.test(role_field=field):
            problem = fixtures.duplicate_coefficients()
            if field == "observations":
                problem["programs"][0]["role"][field] = ["signal_A", "signal_B"]
            problem["programs"][1]["role"][field] = value
            report, _, _ = _check_analysis(problem)
            assert ([r["ages"] for r in report["recommended"]["regimes"]]) == (
                [[0, 1], [2, 3], [4]]
            )


def test_full_allocations_and_required_outputs() -> None:
    allocated = fixtures.allocation()
    allocated["programs"][0]["allocation"] = "global_union"
    report, _, _ = _check_analysis(allocated)
    assert (report["recommended"]["cost"]["work"]) == (64)
    assert (report["recommended"]["regimes"][0]["allocated_state_axes"]) == (["x", "z"])
    outputs = fixtures.problem(
        "computed-output-is-a-consumer",
        4,
        states={"history": {"values": [0, 1]}},
        reward="0",
        updates={"history": "0"},
    )
    outputs["programs"][0]["outputs"] = {"required_history_report": "history"}
    report, _, _ = _check_analysis(outputs)
    assert ("history") in (report["analysis"]["atoms"][0]["live_states"])
    del outputs["programs"][0]["outputs"]
    outputs["external_report_tags"] = [
        {"subject": "person-7", "immutable_tag": "history-type"}
    ]
    report, _, _ = _check_analysis(outputs)
    assert ("history") not in (report["analysis"]["atoms"][0]["live_states"])
    assert (
        (report["recommended"]["external_report_tags"])
        == (outputs["external_report_tags"])
    )


def test_same_age_reads_and_dependency_cycle(*, subtests: pytest.Subtests) -> None:
    for weight, expected_live in (
        ("0", False),
        ("1/100000000000000000000000000000000000000000000000000", True),
    ):
        with subtests.test(weight=weight):
            problem = fixtures.external_read(weight)
            oracle = FiniteOracle(problem)
            report = find_regimes(problem)
            assert (report["status"]) == ("OPTIMAL"), report
            assert (verify_certificate(problem, report)["status"]) == ("PASS")
            for atom in report["analysis"]["atoms"]:
                lane = tuple(atom["status"][name] for name in oracle.status_names)
                assert (set(atom["live_states"])) == (oracle.live[(atom["age"], lane)])
                if atom["status"]["mode"] == "chooser":
                    assert ("history" in atom["live_states"]) == (expected_live)
    cyclic = find_regimes(fixtures.external_read(cyclic=True))
    assert (cyclic["status"]) == ("INVALID"), cyclic
    assert (cyclic["layers"]["admission"]["status"]) == ("FAIL")
    assert all(
        data["status"] == "NOT_RUN"
        for name, data in cyclic["layers"].items()
        if name != "admission"
    )


def test_node_admission_covers_every_claimed_status() -> None:
    report = find_regimes(fixtures.rare_incoming())
    assert (report["status"]) == ("OPTIMAL"), report
    atom_by_id = {a["id"]: a for a in report["analysis"]["atoms"]}
    for block in report["recommended"]["regimes"]:
        expected = [
            {"age": atom_by_id[key]["age"], "status": atom_by_id[key]["status"]}
            for key in block["atoms"]
        ]
        assert (block["admission"]) == (expected)
        assert ({json.dumps(item["status"], sort_keys=True) for item in expected}) == (
            {json.dumps(status, sort_keys=True) for status in block["status_admission"]}
        )


def test_all_unvisited_layers_are_not_run_after_admission_abort(
    *, subtests: pytest.Subtests
) -> None:
    for name, problem in fixtures.invalid_cases().items():
        with subtests.test(case=name):
            report = find_regimes(problem)
            assert (report["status"]) == ("INVALID"), report
            assert (report["layers"]["admission"]["status"]) == ("FAIL")
            for layer, data in report["layers"].items():
                if layer != "admission":
                    assert (data["status"]) == ("NOT_RUN"), (name, layer, data)
            METRICS["invalid_originals_rejected"] += 1


def test_budget_exhaustion_never_claims_optimum() -> None:
    problem = fixtures.fertility()
    report = find_regimes(problem, budget={"candidate_segments": 0})
    assert (report["status"]) == ("INCOMPLETE"), report
    assert not (report["search"]["complete"])
    assert (report["search"]["gap"]) is None
    assert not (report["certificate"]["complete"])
    verification = verify_certificate(problem, report)
    assert (verification["status"]) == ("INCOMPLETE"), verification
    assert not (verification["optimum_verified"])
    assert verification["incumbent_verified"]
    forged = deepcopy(report)
    forged["status"] = "OPTIMAL"
    assert (verify_certificate(problem, forged)["status"]) == ("FAIL")
    METRICS["certificate_mutations_rejected"] += 1


def test_certificate_rejects_independent_mutations(
    *, subtests: pytest.Subtests
) -> None:
    problem = fixtures.fertility()
    report, _, _ = _check_analysis(problem)

    def altered(mutator):
        result = deepcopy(report)
        mutator(result)
        return result

    first_lane = next(iter(report["certificate"]["potentials"]))
    first_atom = report["analysis"]["atoms"][0]["id"]
    forced_atom = next(
        a["id"] for a in report["analysis"]["atoms"] if a["forced_actions"]
    )
    forced_block = next(
        i
        for i, r in enumerate(report["recommended"]["regimes"])
        if forced_atom in r["atoms"]
    )
    mutations = {
        "false_work": lambda r: r["recommended"]["cost"].__setitem__("work", 0),
        "false_enumeration": lambda r: r["recommended"]["cost"].__setitem__(
            "enumeration", 0
        ),
        "false_original_cost": lambda r: r["original"]["cost"].__setitem__("work", 0),
        "missing_regime": lambda r: r["recommended"]["regimes"].pop(0),
        "lost_allocated_axis": lambda r: r["recommended"]["regimes"][0][
            "allocated_state_axes"
        ].clear(),
        "lost_semantic_history": lambda r: r["recommended"]["regimes"][0][
            "semantic_interfaces"
        ][first_atom]["states"].clear(),
        "wrong_forced_action": lambda r: r["recommended"]["regimes"][forced_block][
            "semantic_interfaces"
        ][forced_atom]["forced_actions"].__setitem__("birth", 1),
        "missing_projection": lambda r: r["recommended"]["transitions"][0][
            "projection"
        ].clear(),
        "missing_transition": lambda r: r["recommended"]["transitions"].pop(0),
        "altered_analysis": lambda r: r["analysis"]["atoms"][0]["live_states"].clear(),
        "wrong_problem_identity": lambda r: r["certificate"].__setitem__(
            "problem_sha256", "0" * 64
        ),
        "wrong_family_identity": lambda r: r["certificate"].__setitem__(
            "family_sha256", "0" * 64
        ),
        "unattained_lower_bound": lambda r: r["certificate"]["potentials"][first_lane][
            0
        ].__setitem__(0, 0),
        "invalid_lower_bound": lambda r: r["certificate"]["potentials"][first_lane][
            0
        ].__setitem__(0, 1000000),
        "false_path": lambda r: r["certificate"]["chosen_segments"][first_lane].clear(),
        "false_complete": lambda r: r["certificate"].__setitem__("complete", False),
        "inconsistent_search_complete": lambda r: r["search"].__setitem__(
            "complete", False
        ),
        "false_gap": lambda r: r["search"].__setitem__("gap", 1),
    }
    for name, mutation in mutations.items():
        with subtests.test(mutation=name):
            checked = verify_certificate(problem, altered(mutation))
            assert (checked["status"]) == ("FAIL"), checked
            assert checked["errors"], checked
            METRICS["certificate_mutations_rejected"] += 1
    tagged = deepcopy(problem)
    tagged["external_report_tags"] = [{"subject": "person-7", "tag": "pension-type-A"}]
    result = find_regimes(tagged)
    result["recommended"]["external_report_tags"] = []
    assert (verify_certificate(tagged, result)["status"]) == ("FAIL")
    METRICS["certificate_mutations_rejected"] += 1
