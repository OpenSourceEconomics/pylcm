"""Independent numerical admission and error-bound regressions (stdlib only)."""
from copy import deepcopy
from fractions import Fraction as F
import unittest

from _lcm_regime_finder.numerics import compare_fibers, finite_horizon_error_bounds, one_step_error_bound


def terminal(value):
    return {"reward": 0, "discount": 0, "kernel": {}, "terminal": value}


def problem(reward=0, kernel=None):
    rows = {
        "a": {"reward": 0, "discount": "19/20", "kernel": {"low": "2/5", "high": "3/5"}},
        "b": {"reward": reward, "discount": "19/20", "kernel": kernel or {"low": "2/5", "high": "3/5"}},
        "low": terminal(-2), "high": terminal(3),
    }
    return rows, {"a": "a", "b": "a", "low": "low", "high": "high"}


def conditions(report):
    return {w["condition"] for w in report["witnesses"]}


class NumericalFiberTests(unittest.TestCase):
    def test_exact_projection_aggregates_fibers_without_losing_support(self):
        rows, groups = problem()
        rows["low_copy"] = terminal(-2)
        groups["low_copy"] = "low"
        rows["b"]["kernel"] = {"low": "1/5", "low_copy": "1/5", "high": "3/5"}
        report = compare_fibers(rows, groups, strict=True)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["guarantee"], "exact_represented_rows")
        self.assertEqual(len(report["positive_edges"]), 5)

    def test_observed_float_rounding_admitted_without_renormalizing(self):
        rows, groups = problem()
        rows["a"]["reward"] = 0.3
        rows["b"]["reward"] = 0.1 + 0.2
        rows["a"]["kernel"] = {"low": 0.4, "high": 0.6}
        rows["b"]["kernel"] = {"low": 0.4 + 1e-16, "high": 0.6}
        original = deepcopy(rows)
        practical = compare_fibers(rows, groups)
        self.assertEqual(practical["status"], "PASS", practical)
        self.assertEqual(practical["guarantee"], "bounded_represented_rows")
        self.assertGreater(F(practical["mass_errors"]["b"]["absolute_error"]), 0)
        self.assertGreater(F(practical["max_errors"]["reward"]), 0)
        self.assertFalse(practical["contract"]["renormalized"])
        self.assertEqual(rows, original)
        self.assertEqual(compare_fibers(rows, groups, strict=True)["status"], "FAIL")

    def test_material_reward_and_probability_perturbations_fail(self):
        rows, groups = problem(reward="1/100000")
        self.assertIn("reward_transport", conditions(compare_fibers(rows, groups)))
        rows, groups = problem(kernel={"low": "41/100", "high": "59/100"})
        self.assertIn("projected_kernel_l1", conditions(compare_fibers(rows, groups)))

    def test_admission_abort_leaves_unvisited_layers_not_run(self):
        rows, groups = problem(kernel={"low": "2/5", "high": "600000001/1000000000"})
        report = compare_fibers(rows, groups)
        self.assertEqual(report["layers"], {"admission": "FAIL", "projected_support": "NOT_RUN", "numerical_transport": "NOT_RUN"})
        self.assertEqual(report["comparisons"], [])

    def test_negative_and_zero_rows_cannot_be_admitted_by_large_tolerance(self):
        for kernel in ({"low": 0, "high": 0}, {"low": -1, "high": 2}):
            rows, groups = problem(kernel=kernel)
            self.assertEqual(compare_fibers(rows, groups, probability_mass_absolute=10)["status"], "FAIL")

    def test_rare_positive_edge_is_structural_even_with_loose_budgets(self):
        tiny = F(1, 10**50)
        rows, groups = problem(kernel={"low": 1 - tiny, "high": tiny})
        rows["a"]["kernel"] = {"low": 1, "high": 0}
        report = compare_fibers(rows, groups, absolute=1, relative=1)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("positive_support_mismatch", conditions(report))
        self.assertTrue(any(e["source"] == "b" and e["destination"] == "high" and F(e["weight"]) == tiny
                            for e in report["positive_edges"]))

    def test_nontransitive_chain_is_not_a_fiber_certificate(self):
        tol = F(1, 100)
        rows, groups = problem(reward=3 * tol / 4)
        rows["c"] = deepcopy(rows["b"])
        rows["c"]["reward"] = 3 * tol / 2
        groups["c"] = "a"
        report = compare_fibers(rows, groups, absolute=tol, relative=0)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(w.get("row") == "c" and w["condition"] == "reward_transport" for w in report["witnesses"]))
        self.assertFalse(any(w.get("row") == "b" for w in report["witnesses"]))
        groups["c"] = "b"
        chain = compare_fibers(rows, groups, absolute=tol, relative=0)
        self.assertEqual(chain["layers"]["admission"], "FAIL")

    def test_terminal_rounding_is_bounded_not_exact(self):
        rows = {"a": terminal(0.3), "b": terminal(0.1 + 0.2)}
        groups = {"a": "a", "b": "a"}
        report = compare_fibers(rows, groups)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["guarantee"], "bounded_represented_rows")
        self.assertEqual(compare_fibers(rows, groups, strict=True)["status"], "FAIL")

    def test_nonunit_identity_is_never_labeled_exact_markov_transport(self):
        rows, groups = problem()
        rows["a"]["kernel"]["low"] = F(2, 5) + F(1, 10**14)
        rows["b"] = deepcopy(rows["a"])
        report = compare_fibers(rows, groups)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["max_errors"]["kernel_l1"], "0")
        self.assertEqual(report["guarantee"], "bounded_represented_rows")

    def test_bound_dominates_an_independently_evaluated_q_difference(self):
        p = [F(2, 5), F(3, 5)]
        q = [F(2, 5) + F(1, 10**14), F(3, 5) - F(1, 10**14)]
        vbar = [F(100), F(-50)]
        v = [vbar[0] + F(1, 10**8), vbar[1] - F(1, 10**8)]
        beta, betabar = F(19, 20), F(19, 20) + F(1, 10**14)
        reward, rbar = F(2), F(2) + F(1, 10**13)
        direct = abs(reward + beta * sum(x * y for x, y in zip(p, v))
                     - rbar - betabar * sum(x * y for x, y in zip(q, vbar)))
        bound = one_step_error_bound(reward_error=abs(reward - rbar), discount_error=abs(beta - betabar),
            kernel_l1_error=sum(abs(x - y) for x, y in zip(p, q)), discount_max=beta,
            row_mass_max=1, reduced_row_mass_max=1, next_value_abs_bound=100, next_error=F(1, 10**8))
        self.assertLessEqual(direct, bound)

    def test_finite_horizon_bound_against_direct_recursion(self):
        common = dict(discount_error=0, kernel_l1_error=0, row_mass_max=1,
                      reduced_row_mass_max=1, next_value_abs_bound=0)
        stages = [{**common, "reward_error": F(1, 5), "discount_max": F(1, 2)},
                  {**common, "reward_error": F(1, 10), "discount_max": F(3, 4)}]
        bound = finite_horizon_error_bounds(stages, terminal_error=F(1, 5))
        original_terminal = F(1, 5)
        original_later = F(1, 10) + F(3, 4) * original_terminal
        original_first = F(1, 5) + F(1, 2) * original_later
        self.assertEqual(bound, [original_first, original_later, original_terminal])
        self.assertEqual(bound[0], F(13, 40))

    def test_nonfinite_scalar_rejected_and_not_hidden_as_unknown(self):
        rows, groups = problem(reward=float("nan"))
        self.assertEqual(compare_fibers(rows, groups)["layers"]["admission"], "FAIL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
