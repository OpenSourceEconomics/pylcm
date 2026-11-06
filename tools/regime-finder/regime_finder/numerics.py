"""Bounded numerical checks on an explicit, already chosen finite projection.

This module never discovers approximate equivalence classes. Every row is
compared directly with its fixed representative. Kernel support uses strict
positivity; numerical tolerances never delete an edge or renormalize a row.
Finite floats denote their exact observed binary values, not unknown source
probabilities. The caller separately establishes action, timing, and source
correspondence. All returned errors/budgets are exact rational strings.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Mapping


def scalar(value) -> Fraction:
    """Interpret finite observed scalars without decimal rounding of floats."""
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numerical scalar")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite float")
        return Fraction.from_float(value)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid exact scalar {value!r}") from exc
    raise ValueError(f"Unsupported numerical scalar {value!r}")


def _nonnegative(value, name):
    result = scalar(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _budget(a, b, absolute, relative):
    return absolute + relative * max(abs(a), abs(b))


def _project(kernel, groups):
    result = {}
    for destination, weight in kernel.items():
        # Admission has rejected negative weights. Zero is exact, never a cutoff.
        if weight > 0:
            projected = groups[destination]
            result[projected] = result.get(projected, Fraction()) + weight
    return result


def compare_fibers(rows, groups, *, probability_mass_absolute="1e-12",
                   absolute="1e-12", relative="1e-10", strict=False):
    """Check numerical transport to fixed representatives of aligned rows.

    ``rows`` maps string ids to ``{reward, discount, kernel, terminal?}``.
    A non-None terminal is a scalar terminal payoff and requires an empty kernel.
    Every other row is a decision/action row with a nonzero probability mass.
    ``groups`` covers exactly the row ids and maps to existing row ids, with
    ``groups[representative] == representative``. Kernel destinations must also
    be rows. All feasible action pairings must have been established by the
    caller; a comparison of scalar rows cannot establish that correspondence.

    Strict mode sets every budget to zero. Practical mode separately checks
    row-total error and aggregate projected-law l1 error. The latter uses
    ``absolute + relative * max(original_mass, representative_mass)``.
    No argument is mutated. This is finite-data validation, not a source proof.
    """
    report = {
        "format_version": "1.0",
        "status": "FAIL",
        "guarantee": "none",
        "layers": {name: "NOT_RUN" for name in ("admission", "projected_support", "numerical_transport")},
        "comparisons": [], "mass_errors": {}, "witnesses": [],
        "positive_edges": [],
        "trust_boundary": "Aligned finite rows only; no action-bijection, source, continuous, policy-tie or runtime inference.",
    }

    def fail(layer, condition, **witness):
        report["layers"][layer] = "FAIL"
        report["witnesses"].append({"condition": condition, **witness})

    try:
        if not isinstance(strict, bool):
            raise ValueError("strict must be Boolean")
        mass_tol = _nonnegative(probability_mass_absolute, "probability_mass_absolute")
        abs_tol = _nonnegative(absolute, "absolute")
        rel_tol = _nonnegative(relative, "relative")
        if strict:
            mass_tol = abs_tol = rel_tol = Fraction()
        report["contract"] = {
            "mode": "strict" if strict else "practical",
            "probability_mass_absolute": str(mass_tol),
            "absolute": str(abs_tol), "relative": str(rel_tol),
            "float_interpretation": "exact observed binary ratio",
            "structural_support": "strictly positive projected mass; no tolerance zeroing",
            "renormalized": False,
            "fiber_rule": "every member directly against one fixed representative",
        }
        if not isinstance(rows, Mapping) or not rows:
            raise ValueError("rows must be a nonempty mapping")
        if not isinstance(groups, Mapping) or set(groups) != set(rows):
            raise ValueError("groups must cover exactly the row ids")
        if any(not isinstance(name, str) or not name for name in rows):
            raise ValueError("row ids must be nonempty strings")
        for name, representative in groups.items():
            if representative not in rows or groups[representative] != representative:
                raise ValueError(f"{name}: representative must be an existing fixed point")
        parsed = {}
        for name, row in rows.items():
            if not isinstance(row, Mapping) or not {"reward", "discount", "kernel"} <= set(row):
                raise ValueError(f"{name}: reward, discount and kernel are required")
            if set(row) - {"reward", "discount", "kernel", "terminal"}:
                raise ValueError(f"{name}: unsupported fields")
            if not isinstance(row["kernel"], Mapping) or not set(row["kernel"]) <= set(rows):
                raise ValueError(f"{name}: kernel destinations must be declared rows")
            kernel = {key: scalar(value) for key, value in row["kernel"].items()}
            terminal = None if row.get("terminal") is None else scalar(row["terminal"])
            parsed[name] = {"reward": scalar(row["reward"]),
                            "discount": _nonnegative(row["discount"], "discount"),
                            "kernel": kernel, "terminal": terminal}
            if any(p < 0 for p in kernel.values()):
                fail("admission", "negative_probability", row=name)
            if terminal is not None:
                if kernel:
                    fail("admission", "terminal_has_kernel", row=name)
                continue
            mass = sum(kernel.values(), Fraction())
            error = abs(mass - 1)
            report["mass_errors"][name] = {"mass": str(mass), "absolute_error": str(error),
                                          "budget": str(mass_tol)}
            if mass <= 0 or error > mass_tol:
                fail("admission", "probability_mass", row=name, mass=str(mass),
                     absolute_error=str(error), budget=str(mass_tol))
        if report["layers"]["admission"] == "FAIL":
            return report
        report["layers"]["admission"] = "PASS"
    except (ValueError, TypeError, KeyError) as exc:
        fail("admission", "schema_or_admission", error=str(exc))
        return report

    report["layers"]["projected_support"] = "PASS"
    report["layers"]["numerical_transport"] = "PASS"
    maxima = {key: Fraction() for key in ("reward", "discount", "kernel_l1", "terminal", "row_mass")}
    for row in report["mass_errors"].values():
        maxima["row_mass"] = max(maxima["row_mass"], Fraction(row["absolute_error"]))
    for name in sorted(parsed):
        original = parsed[name]
        representative_id = groups[name]
        representative = parsed[representative_id]
        entry = {"row": name, "representative": representative_id}
        for destination, weight in original["kernel"].items():
            if weight > 0:
                report["positive_edges"].append({"source": name, "destination": destination,
                    "projected_destination": groups[destination], "weight": str(weight)})
        if (original["terminal"] is None) != (representative["terminal"] is None):
            fail("numerical_transport", "terminal_kind", row=name, representative=representative_id)
            entry["status"] = "FAIL"
            report["comparisons"].append(entry)
            continue
        fields = ("reward", "discount") if original["terminal"] is None else ("terminal",)
        failed = False
        for field in fields:
            a, b = original[field], representative[field]
            error, budget = abs(a - b), _budget(a, b, abs_tol, rel_tol)
            maxima[field] = max(maxima[field], error)
            entry[field] = {"original": str(a), "representative": str(b),
                            "absolute_error": str(error), "budget": str(budget)}
            if error > budget:
                failed = True
                fail("numerical_transport", field + "_transport", row=name,
                     representative=representative_id, absolute_error=str(error), budget=str(budget))
        if original["terminal"] is None:
            p, q = _project(original["kernel"], groups), _project(representative["kernel"], groups)
            p_support, q_support = set(p), set(q)
            entry["projected_support"] = {"original": sorted(p_support), "representative": sorted(q_support)}
            if p_support != q_support:
                failed = True
                fail("projected_support", "positive_support_mismatch", row=name,
                     representative=representative_id, original=sorted(p_support), reduced=sorted(q_support))
            error = sum((abs(p.get(k, 0) - q.get(k, 0)) for k in p_support | q_support), Fraction())
            mass, reduced_mass = sum(p.values(), Fraction()), sum(q.values(), Fraction())
            budget = abs_tol + rel_tol * max(mass, reduced_mass)
            maxima["kernel_l1"] = max(maxima["kernel_l1"], error)
            entry["kernel_l1"] = {"absolute_error": str(error), "budget": str(budget),
                                  "original_mass": str(mass), "representative_mass": str(reduced_mass)}
            if error > budget:
                failed = True
                fail("numerical_transport", "projected_kernel_l1", row=name,
                     representative=representative_id, absolute_error=str(error), budget=str(budget))
        entry["status"] = "FAIL" if failed else "PASS"
        report["comparisons"].append(entry)
    report["max_errors"] = {name: str(value) for name, value in maxima.items()}
    if not report["witnesses"]:
        report["status"] = "PASS"
        report["guarantee"] = "exact_represented_rows" if not any(maxima.values()) else "bounded_represented_rows"
    return report


def one_step_error_bound(*, reward_error, discount_error, kernel_l1_error,
                         discount_max, row_mass_max, reduced_row_mass_max,
                         next_value_abs_bound, next_error) -> Fraction:
    """Return a rigorous Q/value bound conditional on the supplied uniform bounds.

    Every argument is a nonnegative upper bound over the admitted fiber/action
    class. The function does not infer those bounds or a continuation norm from
    a scalar probe. Max over aligned finite actions is 1-Lipschitz, so the same
    expression bounds values. Exact argmax/tie correspondence is not inferred.
    """
    values = {name: _nonnegative(value, name) for name, value in {
        "reward_error": reward_error, "discount_error": discount_error,
        "kernel_l1_error": kernel_l1_error, "discount_max": discount_max,
        "row_mass_max": row_mass_max, "reduced_row_mass_max": reduced_row_mass_max,
        "next_value_abs_bound": next_value_abs_bound, "next_error": next_error,
    }.items()}
    return (values["reward_error"] + values["discount_max"] * values["row_mass_max"] * values["next_error"]
            + (values["discount_max"] * values["kernel_l1_error"]
               + values["discount_error"] * values["reduced_row_mass_max"]) * values["next_value_abs_bound"])


def finite_horizon_error_bounds(stages, *, terminal_error=0):
    """Return error bounds for stages 0..T, including the terminal bound.

    ``stages`` is chronological and each element supplies the keyword upper
    bounds of ``one_step_error_bound`` except ``next_error``. This calculation
    is conditional on those uniform bounds and does not certify their source.
    """
    errors = [_nonnegative(terminal_error, "terminal_error")]
    for stage in reversed(stages):
        if not isinstance(stage, Mapping) or "next_error" in stage:
            raise ValueError("stage must supply bound keywords without next_error")
        errors.append(one_step_error_bound(**stage, next_error=errors[-1]))
    return list(reversed(errors))
