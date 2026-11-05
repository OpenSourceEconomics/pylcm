"""Hand-derived contracts for integration, coupling, signed CEs and EGM replay.

All primitives here are literal fixture declarations. The references import no
production lowerer, transition, mask, reducer, solver or interpolation routine.
"""

import ast
import math
from fractions import Fraction as Frac
from pathlib import Path

import pytest

from tests.solution import _iid_shock_oracles as oracle


def test_finite_hard_backward_induction() -> None:
    """Terminal E max is wealth+2; earlier continuation adds 3/2 then 9/4."""
    values, _ = oracle.finite_backward(mode="hard")
    assert values == {
        (2, 0): (Frac(2),),
        (2, 1): (Frac(3),),
        (1, 0): (Frac(7, 2),),
        (1, 1): (Frac(9, 2),),
        (0, 0): (Frac(17, 4),),
        (0, 1): (Frac(21, 4),),
    }


def test_finite_hard_tie_uses_lowest_action() -> None:
    _, policies = oracle.finite_backward(mode="hard", periods=1)
    assert tuple(
        policies[0, wealth, node] for wealth in (0, 1) for node in range(3)
    ) == (0, 1, 1, 0, 1, 1)


def test_node_decision_precedes_integration() -> None:
    """Nodewise action values (1,1), (0,2), (-1,3) tie only at low z.

    A crossing fixture separately makes E max strictly exceed max E.
    """
    values = ((Frac(3), Frac(0)), (Frac(1), Frac(1)), (Frac(0), Frac(3)))
    weights = (Frac(1, 4), Frac(1, 2), Frac(1, 4))
    integrated_max = sum(
        weight * max(node) for weight, node in zip(weights, values, strict=True)
    )
    max_integrated = max(
        sum(weight * node[action] for weight, node in zip(weights, values, strict=True))
        for action in (0, 1)
    )
    assert (integrated_max, max_integrated) == (Frac(2), Frac(5, 4))


def test_collective_channels_share_one_selected_action() -> None:
    """Terminal Pareto sums tie; earlier periods prefer wealth'=1."""
    values, policies = oracle.finite_backward(mode="collective")
    assert (values, tuple(policies[period, 0, 0] for period in range(3))) == (
        {
            (2, 0): (Frac(0), Frac(2)),
            (2, 1): (Frac(1), Frac(3)),
            (1, 0): (Frac(5, 2), Frac(3, 2)),
            (1, 1): (Frac(7, 2), Frac(5, 2)),
            (0, 0): (Frac(15, 4), Frac(5, 4)),
            (0, 1): (Frac(19, 4), Frac(9, 4)),
        },
        (1, 1, 0),
    )


def test_ev1_integrates_logsum_after_decision() -> None:
    """For terminal z=-1,0,1 the logsum is log(2)+1, log(1+e²), log(e^-1+e³)."""
    values, _ = oracle.finite_backward(mode="ev1", periods=1)
    expected = (
        (math.log(2) + 1) / 4
        + math.log1p(math.exp(2)) / 2
        + math.log(math.exp(-1) + math.exp(3)) / 4
    )
    assert values[0, 0][0] == pytest.approx(expected, rel=1e-14)


def test_ev1_nonlinear_integration_does_not_commute() -> None:
    values, _ = oracle.finite_backward(mode="ev1", periods=1)
    assert values[0, 0][0] > math.log1p(math.exp(2))


def test_cdf_joint_cells_are_exact() -> None:
    assert oracle.categorical_cells(
        ((Frac(4, 5), Frac(1, 5)), (Frac(1, 5), Frac(4, 5)))
    ) == (
        (Frac(1, 5), (0, 0)),
        (Frac(3, 5), (0, 1)),
        (Frac(1, 5), (1, 1)),
    )


def test_cdf_refines_multiple_targets_and_recovers_marginals() -> None:
    probabilities = (
        (Frac(0), Frac(1, 3), Frac(2, 3)),
        (Frac(1, 4), Frac(3, 4)),
        (Frac(1, 2), Frac(0), Frac(1, 2)),
    )
    cells = oracle.categorical_cells(probabilities)
    recovered = tuple(
        tuple(
            sum(mass for mass, categories in cells if categories[law] == category)
            for category in range(len(probs))
        )
        for law, probs in enumerate(probabilities)
    )
    assert recovered == probabilities


@pytest.mark.parametrize(
    ("uniform", "expected"), [(Frac(0), 1), (Frac(1, 3), 3), (Frac(2, 3), 3)]
)
def test_cdf_cutpoints_skip_zero_mass(*, uniform: Frac, expected: int) -> None:
    assert (
        oracle.categorical_draw(
            probabilities=(Frac(0), Frac(1, 3), Frac(0), Frac(2, 3)), uniform=uniform
        )
        == expected
    )


@pytest.mark.parametrize("probabilities", [(Frac(-1), Frac(2)), (Frac(1, 2),), ()])
def test_cdf_refuses_invalid_probabilities(probabilities: tuple[Frac, ...]) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        oracle.categorical_cells((probabilities,))


@pytest.mark.parametrize("uniform", [Frac(-1, 10), Frac(1)])
def test_cdf_refuses_uniform_outside_half_open_unit_interval(uniform: Frac) -> None:
    with pytest.raises(ValueError, match="uniform"):
        oracle.categorical_draw(probabilities=(Frac(1),), uniform=uniform)


@pytest.mark.parametrize(
    ("q", "expected"),
    [(Frac(1, 4), Frac(1, 16)), (Frac(1, 3), Frac(0)), (Frac(1), Frac(-1, 2))],
)
def test_signed_power_domain_boundary(*, q: Frac, expected: Frac) -> None:
    """For V=(1,2), sum((-q,1+q)*V^-2)=(1-3q)/4."""
    assert oracle.signed_power_statistic(q=q, high=Frac(2)) == expected


def test_signed_linear_sum_can_leave_value_range() -> None:
    assert -Frac(1) * Frac(1) + Frac(2) * Frac(2) > Frac(2)


@pytest.mark.parametrize("wealth", [Frac(1, 2), Frac(1), Frac(3, 2)])
def test_resources_only_integral_and_economic_marginal(wealth: Frac) -> None:
    """V(m)=m², m=b+z, z∈{0,2}, weights (1/4,3/4): W=b²+3b+3."""
    assert oracle.resources_integral(wealth=wealth) == (
        wealth**2 + 3 * wealth + 3,
        2 * wealth + 3,
    )


def test_resources_projection_error_is_separate_from_economic_marginal() -> None:
    left, right = (oracle.resources_integral(wealth=Frac(k))[0] for k in (0, 2))
    value, marginal = oracle.resources_integral(wealth=Frac(1, 2))
    projected = left + (right - left) * Frac(1, 4)
    assert (projected - value, (right - left) / 2, marginal) == (
        Frac(3, 4),
        Frac(5),
        Frac(4),
    )


def test_case_u_reconstruction_keeps_the_original_candidate() -> None:
    candidate = oracle.log_savings_candidate(resources=Frac(1))
    grid = (Frac(1, 4), Frac(3, 4))
    selected = max(
        grid, key=lambda c: oracle.log_savings_value(resources=Frac(1), consumption=c)
    )
    assert (candidate, selected, (candidate - selected) / candidate) == (
        Frac(1, 2),
        Frac(1, 4),
        Frac(1, 2),
    )


def test_case_u_grid_omission_changes_discrete_branch() -> None:
    """A at m=1 beats B at m=1/2 plus half-log(7/2); a sparse grid reverses it."""
    bonus = math.log(7 / 2) / 2
    exact_a = oracle.log_savings_value(resources=Frac(1), consumption=Frac(1, 2))
    exact_b = (
        oracle.log_savings_value(resources=Frac(1, 2), consumption=Frac(1, 4)) + bonus
    )
    grid_a = oracle.log_savings_value(resources=Frac(1), consumption=Frac(1, 4))
    assert exact_a > exact_b > grid_a


@pytest.mark.parametrize("shock", [-2.0, 0.0, 3.0])
def test_case_u_replay_at_original_shock_nodes(shock: float) -> None:
    candidate = oracle.log_savings_candidate(resources=Frac(1))
    value = oracle.log_savings_value(
        resources=Frac(1), consumption=candidate, shock=shock
    )
    assert value == pytest.approx(-math.log(2) + shock, rel=1e-14, abs=1e-14)


def test_reference_modules_import_only_independent_dependencies() -> None:
    modules = (
        Path(oracle.__file__),
        Path(__file__).with_name("test_exit_reference_household.py"),
    )
    imports = {
        node.module.split(".")[0]
        for module in modules
        for node in ast.walk(ast.parse(module.read_text()))
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for module in modules
        for node in ast.walk(ast.parse(module.read_text()))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports <= {
        "collections",
        "dataclasses",
        "fractions",
        "typing",
        "math",
        "itertools",
        "numpy",
        "pytest",
    }


def test_state_dependent_weight_marginal_includes_probability_derivative() -> None:
    """w=(1-b,b), V=(b,b²): W=b-b²+b³; W'=1-2b+3b².

    At b=1/4 the frozen-weight marginal is 7/8, while W'=11/16.
    """
    wealth = Frac(1, 4)
    value, derivative = oracle.weighted_value_and_derivative(
        values=(wealth, wealth**2),
        marginals=(Frac(1), 2 * wealth),
        weights=(1 - wealth, wealth),
        weight_derivatives=(Frac(-1), Frac(1)),
    )
    frozen = (1 - wealth) + wealth * (2 * wealth)
    assert (value, derivative, frozen) == (Frac(13, 64), Frac(11, 16), Frac(7, 8))


def test_zero_primal_mass_with_nonzero_derivative_keeps_probability_term() -> None:
    """w=(1-b,b), V=(1,3): at b=0, W=1 while W'=2."""
    assert oracle.weighted_value_and_derivative(
        values=(Frac(1), Frac(3)),
        marginals=(Frac(0), Frac(0)),
        weights=(Frac(1), Frac(0)),
        weight_derivatives=(Frac(-1), Frac(1)),
    ) == (Frac(1), Frac(2))


def test_zero_mass_and_zero_weight_derivative_ignore_poisoned_node() -> None:
    assert oracle.weighted_value_and_derivative(
        values=(Frac(3), float("-inf")),
        marginals=(Frac(2), float("nan")),
        weights=(Frac(1), Frac(0)),
        weight_derivatives=(Frac(0), Frac(0)),
    ) == (Frac(3), Frac(2))


@pytest.mark.parametrize("degree", range(7))
def test_standard_normal_raw_moments(degree: int) -> None:
    expected = (Frac(1), Frac(0), Frac(1), Frac(0), Frac(3), Frac(0), Frac(15))
    assert oracle.base_moment(measure="normal", degree=degree) == expected[degree]


@pytest.mark.parametrize("degree", range(7))
def test_unit_uniform_raw_moments(degree: int) -> None:
    assert oracle.base_moment(measure="uniform", degree=degree) == Frac(1, degree + 1)


def test_resources_integral_transforms_kinks_and_owns_jumps_on_the_right() -> None:
    """V(m)=max(m-1,0)+2*1[m>=2]; z=(0,2), w=(1/4,3/4).

    Transforming the row creates kinks at b=-1 and b=1 and jumps at b=0 and b=2.
    Values at the jumps include their right-hand term before integration.
    """

    def row(resources: Frac) -> Frac:
        return max(resources - 1, Frac(0)) + 2 * int(resources >= 2)

    wealth_nodes = (Frac(-1, 2), Frac(0), Frac(1, 2), Frac(1), Frac(3, 2), Frac(2))
    values = tuple(row(wealth) / 4 + 3 * row(wealth + 2) / 4 for wealth in wealth_nodes)
    assert values == (
        Frac(3, 8),
        Frac(9, 4),
        Frac(21, 8),
        Frac(3),
        Frac(7, 2),
        Frac(9, 2),
    )
