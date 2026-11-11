"""Actual inherited-checker regressions on automatically generated quotients."""

from copy import deepcopy
from fractions import Fraction
import json

from _lcm_regime_finder.core import find_regimes
from _lcm_regime_finder.transport_bridge import validate_generated_transport
from tests.regime_finder.fixtures import (
    external_read,
    fertility,
    problem,
    rare_incoming,
    retirement,
)


def interface_at(report, age, status=None):
    atoms = report["analysis"]["atoms"]
    atom = next(
        a
        for a in atoms
        if a["age"] == age and (status is None or a["status"] == status)
    )
    block = next(
        r for r in report["recommended"]["regimes"] if atom["id"] in r["atoms"]
    )
    return atom, block["semantic_interfaces"][atom["id"]]


def _run_problem(p):
    report = find_regimes(p)
    assert (report["status"]) in (("OPTIMAL", "INCOMPLETE")), report.get("error")
    return report, validate_generated_transport(p, report)


def _assert_real_pass(result):
    assert (result["status"]) == ("PASS"), result["diagnostics"]
    assert result["checker_executed"]
    assert (result["checker_result"]["status"]) == ("PASS")
    assert (result["checker_result"]["layers"]["finite_model"]["checks"]) > (0)
    assert (result["checker_result"]["cases"][0]["backward_evaluation"]["status"]) == (
        "passed"
    )
    json.dumps(result)  # The complete generated evidence is exportable JSON.


def test_fertility_actions_and_last_child_consumer() -> None:
    _, result = _run_problem(fertility())
    _assert_real_pass(result)
    assert (result["generated"]["original_cells"]) == (17)
    assert (result["generated"]["reduced_cells"]) == (14)
    assert (result["generated"]["original_feasible_action_rows"]) == (20)
    assert (result["generated"]["reduced_feasible_action_rows"]) == (17)
    data = result["checker_input"]["cases"][0]
    for old_id, cell in data["original"]["cells"].items():
        if 2 <= cell["stage"] < 8:
            assert (data["certificate"]["fixed_action_attributes"][old_id]) == (
                {"birth": 0}
            )
            reduced = data["reduced"]["cells"][data["certificate"]["state_map"][old_id]]
            assert all(
                "birth" not in action["decoded"]
                for action in reduced["actions"].values()
            )
    assert (len(data["starts"])) == (len(data["original"]["cells"]))


def test_retirement_keeps_lagged_bills_until_last_consumer() -> None:
    _, result = _run_problem(retirement())
    _assert_real_pass(result)
    assert (result["generated"]["original_cells"]) == (33)
    assert (result["generated"]["reduced_cells"]) == (18)
    assert (result["generated"]["original_feasible_action_rows"]) == (40)
    assert (result["generated"]["reduced_feasible_action_rows"]) == (25)
    data = result["checker_input"]["cases"][0]
    for cell in data["reduced"]["cells"].values():
        if cell["stage"] in (2, 3, 4):
            assert (set(cell["state"])) == ({"coverage_lag"})
        if cell["stage"] in (5, 6, 7):
            assert (cell["state"]) == ({})


def test_positive_rare_simulation_edges_are_not_tolerance_zeroed() -> None:
    _, result = _run_problem(rare_incoming(True, "simulation"))
    _assert_real_pass(result)
    data = result["checker_input"]["cases"][0]
    tiny = Fraction(1, 10**50)
    for side in ("original", "reduced"):
        actions = [
            a for c in data[side]["cells"].values() for a in c["actions"].values()
        ]
        simulation = [
            Fraction(w) for a in actions for w in a["kernels"]["simulation"].values()
        ]
        solve = [Fraction(w) for a in actions for w in a["kernels"]["solve"].values()]
        assert (tiny) in (simulation)
        assert (tiny) not in (solve)
        assert all(w > 0 for w in simulation + solve)


def test_corrupted_fertility_projection_fails_actual_reward_check() -> None:
    p = fertility()
    report = find_regimes(p)
    _, interface = interface_at(report, 0)
    interface["states"] = []
    result = validate_generated_transport(p, report)
    assert (result["status"]) == ("FAIL")
    assert result["checker_executed"]
    witnesses = result["checker_result"]["layers"]["finite_model"]["witnesses"]
    assert any(w["condition"] == "reward_transport" for w in witnesses), witnesses


def test_corrupted_rare_history_projection_fails_simulation_pushforward() -> None:
    p = rare_incoming(True, "simulation")
    report = find_regimes(p)
    _, interface = interface_at(report, 2, {"status": "active"})
    assert ("history") in (interface["states"])
    interface["states"] = []
    result = validate_generated_transport(p, report)
    assert (result["status"]) == ("FAIL")
    assert result["checker_executed"]
    witnesses = result["checker_result"]["layers"]["finite_model"]["witnesses"]
    assert any(
        w["condition"] == "kernel_pushforward" and w.get("phase") == "simulation"
        for w in witnesses
    ), witnesses


def test_wrong_fixed_action_is_not_repaired_by_rederiving_interface() -> None:
    p = fertility()
    report = find_regimes(p)
    _, interface = interface_at(report, 2)
    interface["forced_actions"]["birth"] = 1
    result = validate_generated_transport(p, report)
    assert (result["status"]) == ("FAIL")
    assert result["checker_executed"]
    witnesses = result["checker_result"]["layers"]["finite_model"]["witnesses"]
    assert any(w["condition"] == "feasible_action_surjection" for w in witnesses), (
        witnesses
    )


def test_changed_actor_role_is_rejected() -> None:
    p = retirement()
    report = find_regimes(p)
    report["recommended"]["regimes"][0]["actor_information_role"] = {
        "actor": "different"
    }
    result = validate_generated_transport(p, report)
    assert (result["status"]) == ("FAIL")
    assert not (result["checker_executed"])
    assert (result["diagnostics"][0]["code"]) == ("economic_role_changed")


def test_budget_does_not_substitute_reachability_or_sampling() -> None:
    p = fertility()
    result = validate_generated_transport(p, find_regimes(p), max_cells=1)
    assert (result["status"]) == ("NOT_RUN")
    assert not (result["checker_executed"])
    assert (result["checker_input"]) is None
    assert (result["generated"]["original_cells"]) == (0)
    assert (result["diagnostics"][0]["code"]) == ("finite_generation_budget")


def test_symbolic_size_grid_is_not_executed() -> None:
    p = fertility()
    p["states"]["child"] = {"size": 2, "grid_id": "child-grid-without-values"}
    _, result = _run_problem(p)
    assert (result["status"]) == ("NOT_RUN")
    assert not (result["checker_executed"])
    assert (result["diagnostics"][0]["code"]) == ("nonenumerable_grid")


def test_opaque_primitive_is_not_replaced_by_a_stub() -> None:
    p = fertility()
    p["programs"][0]["reward"] = {"opaque": "actual_native_payoff", "reads": ["child"]}
    _, result = _run_problem(p)
    assert (result["status"]) == ("NOT_RUN")
    assert not (result["checker_executed"])
    assert (result["diagnostics"][0]["code"]) == ("opaque_primitive")


def test_ranked_same_period_extension_is_explicitly_unresolved() -> None:
    _, result = _run_problem(external_read())
    assert (result["status"]) == ("UNRESOLVED")
    assert not (result["checker_executed"])
    assert (result["layers"]["finite_transport"]["status"]) == ("NOT_RUN")
    assert (result["diagnostics"][0]["code"]) == ("same_period_or_external_read")


def test_nonunit_rows_are_preserved_and_not_called_exact_markov() -> None:
    p = problem("rounded-row", 1)
    p["programs"][0]["transitions"][0]["probability"] = "0.9999999999999"
    _, result = _run_problem(p)
    _assert_real_pass(result)
    assert not (result["exact_markov_normalization"])
    assert (result["guarantee"]) == ("exact_raw_operator_transport_with_mass_error")
    assert not (
        result["checker_result"]["probability_normalization"]["weights_changed"]
    )
    assert (
        result["checker_result"]["probability_normalization"]["max_absolute_mass_error"]
    ) == ("1/10000000000000")
    data = result["checker_input"]["cases"][0]
    for side in ("original", "reduced"):
        weights = [
            Fraction(w)
            for c in data[side]["cells"].values()
            for a in c["actions"].values()
            for w in a["kernels"]["solve"].values()
        ]
        assert (weights) == ([Fraction(9999999999999, 10000000000000)])


def test_declared_action_dependent_outputs_use_full_decoded_actions() -> None:
    p = fertility()
    p["programs"][0]["outputs"] = {
        "chosen_birth": "birth",
        "child_plus_birth": "child + birth",
    }
    _, result = _run_problem(p)
    _assert_real_pass(result)
    data = result["checker_input"]["cases"][0]
    assert all(
        "outputs_by_decoded_action" in c["observables"]
        for c in data["original"]["cells"].values()
        if c["terminal"] is None
    )


def test_state_and_action_grid_decoding_is_not_integer_index_substitution() -> None:
    p = problem(
        "decoded-grids",
        2,
        states={"wealth": {"values": [10, 20]}},
        actions={"consume": {"values": [5, 7]}},
        reward="wealth + consume",
        updates={"wealth": "wealth"},
    )
    _, result = _run_problem(p)
    _assert_real_pass(result)
    data = result["checker_input"]["cases"][0]
    rewards = {
        Fraction(a["reward"]["value"])
        for c in data["original"]["cells"].values()
        for a in c["actions"].values()
    }
    assert (rewards) == ({Fraction(15), Fraction(17), Fraction(25), Fraction(27)})


def test_inputs_and_report_are_immutable() -> None:
    p = retirement()
    report = find_regimes(p)
    old_problem, old_report = deepcopy(p), deepcopy(report)
    _assert_real_pass(validate_generated_transport(p, report))
    assert (p) == (old_problem)
    assert (report) == (old_report)
