"""Independent numerical admission and error-bound regressions (stdlib only)."""

from copy import deepcopy
from fractions import Fraction as F

from _lcm_regime_finder.numerics import (
    compare_fibers,
    finite_horizon_error_bounds,
    one_step_error_bound,
)


def terminal(value):
    return {"reward": 0, "discount": 0, "kernel": {}, "terminal": value}


def problem(reward=0, kernel=None):
    rows = {
        "a": {
            "reward": 0,
            "discount": "19/20",
            "kernel": {"low": "2/5", "high": "3/5"},
        },
        "b": {
            "reward": reward,
            "discount": "19/20",
            "kernel": kernel or {"low": "2/5", "high": "3/5"},
        },
        "low": terminal(-2),
        "high": terminal(3),
    }
    return rows, {"a": "a", "b": "a", "low": "low", "high": "high"}


def conditions(report):
    return {w["condition"] for w in report["witnesses"]}


def test_exact_projection_aggregates_fibers_without_losing_support() -> None:
    rows, groups = problem()
    rows["low_copy"] = terminal(-2)
    groups["low_copy"] = "low"
    rows["b"]["kernel"] = {"low": "1/5", "low_copy": "1/5", "high": "3/5"}
    report = compare_fibers(rows, groups, strict=True)
    assert (report["status"]) == ("PASS"), report
    assert (report["guarantee"]) == ("exact_represented_rows")
    assert (len(report["positive_edges"])) == (5)


def test_observed_float_rounding_admitted_without_renormalizing() -> None:
    rows, groups = problem()
    rows["a"]["reward"] = 0.3
    rows["b"]["reward"] = 0.1 + 0.2
    rows["a"]["kernel"] = {"low": 0.4, "high": 0.6}
    rows["b"]["kernel"] = {"low": 0.4 + 1e-16, "high": 0.6}
    original = deepcopy(rows)
    practical = compare_fibers(rows, groups)
    assert (practical["status"]) == ("PASS"), practical
    assert (practical["guarantee"]) == ("bounded_represented_rows")
    assert (F(practical["mass_errors"]["b"]["absolute_error"])) > (0)
    assert (F(practical["max_errors"]["reward"])) > (0)
    assert not (practical["contract"]["renormalized"])
    assert (rows) == (original)
    assert (compare_fibers(rows, groups, strict=True)["status"]) == ("FAIL")


def test_material_reward_and_probability_perturbations_fail() -> None:
    rows, groups = problem(reward="1/100000")
    assert ("reward_transport") in (conditions(compare_fibers(rows, groups)))
    rows, groups = problem(kernel={"low": "41/100", "high": "59/100"})
    assert ("projected_kernel_l1") in (conditions(compare_fibers(rows, groups)))


def test_admission_abort_leaves_unvisited_layers_not_run() -> None:
    rows, groups = problem(kernel={"low": "2/5", "high": "600000001/1000000000"})
    report = compare_fibers(rows, groups)
    assert (report["layers"]) == (
        {
            "admission": "FAIL",
            "projected_support": "NOT_RUN",
            "numerical_transport": "NOT_RUN",
        }
    )
    assert (report["comparisons"]) == ([])


def test_negative_and_zero_rows_cannot_be_admitted_by_large_tolerance() -> None:
    for kernel in ({"low": 0, "high": 0}, {"low": -1, "high": 2}):
        rows, groups = problem(kernel=kernel)
        assert (
            compare_fibers(rows, groups, probability_mass_absolute=10)["status"]
        ) == ("FAIL")


def test_rare_positive_edge_is_structural_even_with_loose_budgets() -> None:
    tiny = F(1, 10**50)
    rows, groups = problem(kernel={"low": 1 - tiny, "high": tiny})
    rows["a"]["kernel"] = {"low": 1, "high": 0}
    report = compare_fibers(rows, groups, absolute=1, relative=1)
    assert (report["status"]) == ("FAIL")
    assert ("positive_support_mismatch") in (conditions(report))
    assert any(
        e["source"] == "b" and e["destination"] == "high" and F(e["weight"]) == tiny
        for e in report["positive_edges"]
    )


def test_nontransitive_chain_is_not_a_fiber_certificate() -> None:
    tol = F(1, 100)
    rows, groups = problem(reward=3 * tol / 4)
    rows["c"] = deepcopy(rows["b"])
    rows["c"]["reward"] = 3 * tol / 2
    groups["c"] = "a"
    report = compare_fibers(rows, groups, absolute=tol, relative=0)
    assert (report["status"]) == ("FAIL")
    assert any(
        w.get("row") == "c" and w["condition"] == "reward_transport"
        for w in report["witnesses"]
    )
    assert not (any(w.get("row") == "b" for w in report["witnesses"]))
    groups["c"] = "b"
    chain = compare_fibers(rows, groups, absolute=tol, relative=0)
    assert (chain["layers"]["admission"]) == ("FAIL")


def test_terminal_rounding_is_bounded_not_exact() -> None:
    rows = {"a": terminal(0.3), "b": terminal(0.1 + 0.2)}
    groups = {"a": "a", "b": "a"}
    report = compare_fibers(rows, groups)
    assert (report["status"]) == ("PASS")
    assert (report["guarantee"]) == ("bounded_represented_rows")
    assert (compare_fibers(rows, groups, strict=True)["status"]) == ("FAIL")


def test_nonunit_identity_is_never_labeled_exact_markov_transport() -> None:
    rows, groups = problem()
    rows["a"]["kernel"]["low"] = F(2, 5) + F(1, 10**14)
    rows["b"] = deepcopy(rows["a"])
    report = compare_fibers(rows, groups)
    assert (report["status"]) == ("PASS")
    assert (report["max_errors"]["kernel_l1"]) == ("0")
    assert (report["guarantee"]) == ("bounded_represented_rows")


def test_bound_dominates_an_independently_evaluated_q_difference() -> None:
    p = [F(2, 5), F(3, 5)]
    q = [F(2, 5) + F(1, 10**14), F(3, 5) - F(1, 10**14)]
    vbar = [F(100), F(-50)]
    v = [vbar[0] + F(1, 10**8), vbar[1] - F(1, 10**8)]
    beta, betabar = F(19, 20), F(19, 20) + F(1, 10**14)
    reward, rbar = F(2), F(2) + F(1, 10**13)
    direct = abs(
        reward
        + beta * sum(x * y for x, y in zip(p, v))
        - rbar
        - betabar * sum(x * y for x, y in zip(q, vbar))
    )
    bound = one_step_error_bound(
        reward_error=abs(reward - rbar),
        discount_error=abs(beta - betabar),
        kernel_l1_error=sum(abs(x - y) for x, y in zip(p, q)),
        discount_max=beta,
        row_mass_max=1,
        reduced_row_mass_max=1,
        next_value_abs_bound=100,
        next_error=F(1, 10**8),
    )
    assert (direct) <= (bound)


def test_finite_horizon_bound_against_direct_recursion() -> None:
    common = dict(
        discount_error=0,
        kernel_l1_error=0,
        row_mass_max=1,
        reduced_row_mass_max=1,
        next_value_abs_bound=0,
    )
    stages = [
        {**common, "reward_error": F(1, 5), "discount_max": F(1, 2)},
        {**common, "reward_error": F(1, 10), "discount_max": F(3, 4)},
    ]
    bound = finite_horizon_error_bounds(stages, terminal_error=F(1, 5))
    original_terminal = F(1, 5)
    original_later = F(1, 10) + F(3, 4) * original_terminal
    original_first = F(1, 5) + F(1, 2) * original_later
    assert (bound) == ([original_first, original_later, original_terminal])
    assert (bound[0]) == (F(13, 40))


def test_nonfinite_scalar_rejected_and_not_hidden_as_unknown() -> None:
    rows, groups = problem(reward=float("nan"))
    assert (compare_fibers(rows, groups)["layers"]["admission"]) == ("FAIL")
