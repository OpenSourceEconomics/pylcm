"""Independent uncertainty, timing and KKT controls for log-utility accuracy."""

from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from tests.solution import _nb_scientific_reference as reference


def _point(value: str) -> reference.Interval:
    return reference.Interval(lower=Decimal(value), upper=Decimal(value))


def _log_two_bounds() -> tuple[Fraction, Fraction]:
    """Bound ln2 independently by the positive atanh series and geometric tail."""
    count = 100
    z = Fraction(1, 3)
    lower = 2 * sum(
        (z ** (2 * index + 1) / (2 * index + 1) for index in range(count)),
        start=Fraction(0),
    )
    tail = 2 * z ** (2 * count + 1) / ((2 * count + 1) * (1 - z * z))
    return lower, lower + tail


@pytest.mark.parametrize("sign", [1, -1])
def test_log_enclosure_contains_an_independent_rational_series(sign: int) -> None:
    """Correct log bounds contain a much tighter independent atanh enclosure."""
    lower, upper = _log_two_bounds()
    if sign < 0:
        lower, upper = -upper, -lower
    got = reference.log_fraction(value=Fraction(2) ** sign)
    assert Fraction(got.lower) <= lower <= upper <= Fraction(got.upper)
    assert Fraction(got.upper) - Fraction(got.lower) < Fraction(1, 10**75)


@pytest.mark.parametrize("coefficient", [Fraction(7, 13), Fraction(-7, 13)])
def test_signed_scaling_encloses_negative_log_intervals(coefficient: Fraction) -> None:
    """Exact rational scaling preserves outwardness even for negative logarithms."""
    interval = reference.Interval(
        lower=Decimal("-0.69314718055994530941723212145817656807550013436026"),
        upper=Decimal("-0.69314718055994530941723212145817656807550013436025"),
    )
    exact = sorted(
        (Fraction(interval.lower) * coefficient, Fraction(interval.upper) * coefficient)
    )
    with localcontext() as hostile:
        hostile.prec = 6
        got = reference.scale(interval=interval, coefficient=coefficient)
    assert Fraction(got.lower) <= exact[0] <= exact[1] <= Fraction(got.upper)
    assert Fraction(got.upper) - Fraction(got.lower) < Fraction(1, 10**45)


def test_sign_reversal_is_exact_under_a_hostile_ambient_context() -> None:
    """Negation neither rounds long endpoints nor preserves their old order."""
    interval = reference.Interval(
        lower=Decimal("-1.23456789012345678901234567890123456789"),
        upper=Decimal("-0.12345678901234567890123456789012345678"),
    )
    with localcontext() as hostile:
        hostile.prec = 6
        got = reference.negate(interval)
    assert got.lower == interval.upper.copy_negate()
    assert got.upper == interval.lower.copy_negate()


def test_directed_cancellation_uses_private_precision() -> None:
    """Adding and subtracting long endpoints encloses their exact rational result."""
    left = _point("123456789012345678901234567890.123456789")
    right = _point("-123456789012345678901234567890.123456788")
    with localcontext() as hostile:
        hostile.prec = 6
        total = reference.add(left=left, right=right)
        difference = reference.subtract(left=left, right=reference.negate(right))
    exact = Fraction(1, 10**9)
    for got in (total, difference):
        assert Fraction(got.lower) <= exact <= Fraction(got.upper)
        assert Fraction(got.upper) - Fraction(got.lower) < Fraction(1, 10**60)


def test_empty_nonnegative_regret_intersection_is_refused() -> None:
    """An action interval strictly above the alleged optimum is inconsistent."""
    with pytest.raises(reference.OracleInconsistentError):
        reference.nonnegative_regret(optimum=_point("0"), action=_point("1"))


def test_oracle_uncertainty_cannot_pass_an_unresolved_budget() -> None:
    """An interval crossing the ceiling remains unresolved."""
    uncertain = reference.Interval(lower=Decimal("0.9999"), upper=Decimal("1.0001"))
    with pytest.raises(reference.OracleUnresolvedError):
        reference.require_loss_budget(uncertain)
    reference.require_loss_budget(_point("1"))
    with pytest.raises(reference.OracleBudgetExceededError):
        reference.require_loss_budget(_point("1.0001"))


def test_zero_regret_has_zero_consumption_equivalent_loss() -> None:
    """The utility-to-consumption conversion encloses exact zero at zero regret."""
    got = reference.loss_basis_points(regret=_point("0"), beta=Fraction(19, 20))
    assert got.lower <= 0 <= got.upper
    assert Fraction(got.upper) - Fraction(got.lower) < Fraction(1, 10**70)


@pytest.mark.parametrize(
    ("wealth", "consumption", "savings"),
    [
        (Fraction(1, 100), Fraction(51, 100), Fraction(-1, 2)),
        (Fraction(1, 38), Fraction(10, 19), Fraction(-1, 2)),
        (Fraction(1, 2), Fraction(10, 13), Fraction(-7, 26)),
    ],
)
def test_last_period_budget_and_one_sided_kkt(
    *, wealth: Fraction, consumption: Fraction, savings: Fraction
) -> None:
    """Binding, switch and slack tuples satisfy the original saving budget."""
    economics = reference.Economics()
    got = reference.last_period(economics=economics, wealth=wealth, income=Fraction(1))
    assert got.consumption == consumption
    assert got.savings == savings
    assert got.terminal_wealth == savings + 1
    assert got.savings == wealth - got.consumption
    derivative = 1 / got.consumption - economics.beta / got.terminal_wealth
    assert derivative >= 0 if savings == Fraction(-1, 2) else derivative == 0


def test_tax_kink_has_genuine_distinct_wealth_slopes() -> None:
    """Current tax equality keeps the left unit slope and the right net-tax slope."""
    got = reference.last_period(
        economics=reference.Economics(), wealth=Fraction(10), income=Fraction(1)
    )
    assert got.consumption == Fraction(220, 39)
    assert got.slope_left == Fraction(39, 220)
    assert got.slope_right == Fraction(39, 275)


def test_child_tax_kink_reverses_consumption_derivative_sides() -> None:
    """Increasing consumption approaches the child's wealth kink from the right."""
    left, right = reference.objective0_slopes(
        economics=reference.Economics(),
        wealth=Fraction(15),
        income=Fraction(1),
        consumption=Fraction(5),
    )
    expected_child = Fraction(4, 5) * Fraction(39, 220) + Fraction(1, 5) * Fraction(
        13, 80
    )
    assert left == Fraction(1, 5) - Fraction(19, 20) * Fraction(4, 5) * expected_child
    assert right == Fraction(1, 5) - Fraction(19, 20) * expected_child
    assert left > right


def test_zero_discount_optimum_is_the_actual_borrowing_endpoint() -> None:
    """Without continuation utility the positive feasible upper consumption wins."""
    got = reference.optimum0(
        economics=reference.Economics(beta=Fraction(0)),
        wealth=Fraction(1, 2),
        income=Fraction(1),
    )
    assert got.consumption_lower == got.consumption_upper == 1
    assert got.value.lower <= 0 <= got.value.upper


def test_reference_calibration_and_label_permutation_preserve_economics() -> None:
    """Permuting income labels, amounts and probabilities preserves value."""
    economics = reference.Economics(
        beta=Fraction(9, 10),
        tax_rate=Fraction(3, 10),
        share=Fraction(2, 5),
        incomes=(Fraction(5, 4), Fraction(5, 2)),
        probability_rows=(
            (Fraction(7, 10), Fraction(3, 10)),
            (Fraction(3, 10), Fraction(7, 10)),
        ),
    )
    permuted = replace(
        economics,
        incomes=economics.incomes[::-1],
        probability_rows=tuple(
            tuple(row[::-1]) for row in economics.probability_rows[::-1]
        ),
    )
    left = reference.objective0(
        economics=economics,
        wealth=Fraction(3),
        income=Fraction(5, 4),
        consumption=Fraction(1),
    )
    right = reference.objective0(
        economics=permuted,
        wealth=Fraction(3),
        income=Fraction(5, 4),
        consumption=Fraction(1),
    )
    assert max(left.lower, right.lower) <= min(left.upper, right.upper)


@pytest.mark.parametrize("beta", [Fraction(-1), Fraction(1)])
def test_reference_parameter_validation_distinguishes_valid_endpoints(
    beta: Fraction,
) -> None:
    """Nonnegative beta is supported, while negative discounting is refused."""
    if beta < 0:
        with pytest.raises(ValueError, match="beta"):
            reference.Economics(beta=beta)
    else:
        assert reference.Economics(beta=beta).beta == beta


def _positive_log_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    """Bound a positive rational log by range reduction and an atanh tail."""
    power = 0
    while value >= 2:
        value /= 2
        power += 1
    while value < 1:
        value *= 2
        power -= 1
    count = 120
    z = (value - 1) / (value + 1)
    lower = 2 * sum(
        (z ** (2 * index + 1) / (2 * index + 1) for index in range(count)),
        start=Fraction(0),
    )
    tail = 2 * z ** (2 * count + 1) / ((2 * count + 1) * (1 - z * z))
    log_two_lower, log_two_upper = _log_two_bounds()
    scaled = sorted((power * log_two_lower, power * log_two_upper))
    return lower + scaled[0], lower + tail + scaled[1]


def test_current_income_transition_has_an_independent_analytic_objective() -> None:
    """Current income sets next wealth before future income changes continuation."""
    got = reference.objective0(
        economics=reference.Economics(beta=Fraction(1)),
        wealth=Fraction(1),
        income=Fraction(1),
        consumption=Fraction(1),
    )
    lower, upper = _positive_log_bounds(Fraction(3, 2))
    assert Fraction(got.lower) <= Fraction(2, 5) * lower
    assert Fraction(2, 5) * upper <= Fraction(got.upper)
    assert Fraction(got.upper) - Fraction(got.lower) < Fraction(1, 10**70)


def test_interior_optimum_encloses_independent_consumption_and_value() -> None:
    """The continuous concave optimum encloses c=4/3 and value=3 ln(4/3)."""
    economics = reference.Economics(
        beta=Fraction(1),
        probability_rows=((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
    )
    got = reference.optimum0(
        economics=economics,
        wealth=Fraction(2),
        income=Fraction(1),
    )
    assert got.consumption_lower <= Fraction(4, 3) <= got.consumption_upper
    assert got.consumption_upper - got.consumption_lower < Fraction(1, 10**18)
    lower, upper = _positive_log_bounds(Fraction(4, 3))
    assert Fraction(got.value.lower) <= 3 * lower
    assert 3 * upper <= Fraction(got.value.upper)
    assert Fraction(got.value.upper) - Fraction(got.value.lower) < Fraction(1, 10**35)
    with pytest.raises(reference.OracleUnresolvedError, match="bisection"):
        reference.optimum0(
            economics=economics,
            wealth=Fraction(2),
            income=Fraction(1),
            max_bisections=0,
        )
    with pytest.raises(reference.OracleUnresolvedError, match="halving"):
        reference.optimum0(
            economics=economics,
            wealth=Fraction(2),
            income=Fraction(1),
            max_halvings=0,
        )


def test_high_wealth_terminal_budget_is_not_taxed_again() -> None:
    """Current resources are taxed once and terminal utility uses actual next wealth."""
    got = reference.last_period(
        economics=reference.Economics(),
        wealth=Fraction(40),
        income=Fraction(1),
    )
    assert got.consumption == Fraction(700, 39)
    assert got.savings == Fraction(34) - got.consumption == Fraction(626, 39)
    assert got.terminal_wealth == got.savings + 1 == Fraction(665, 39)
    c_lower, c_upper = _positive_log_bounds(Fraction(700, 39))
    q_lower, q_upper = _positive_log_bounds(Fraction(665, 39))
    assert Fraction(got.value.lower) <= c_lower + Fraction(19, 20) * q_lower
    assert c_upper + Fraction(19, 20) * q_upper <= Fraction(got.value.upper)


@pytest.mark.parametrize("precision", [80, 160])
def test_exact_transcendental_identities_and_value_budget(precision: int) -> None:
    """Exact zero identities and conservative structural budgets are preserved."""
    assert reference.log_fraction(value=Fraction(1), precision=precision) == _point("0")
    exponential = reference.exp_interval(interval=_point("0"), precision=precision)
    assert exponential == _point("1")
    assert reference.value_budget(
        observed=Fraction(1),
        truth=_point("1"),
        precision=precision,
    )
    assert not reference.value_budget(
        observed=Fraction(2),
        truth=_point("1"),
        precision=precision,
    )
    with pytest.raises(reference.OracleUnresolvedError, match="value"):
        reference.value_budget(
            observed=Fraction(0),
            truth=reference.Interval(lower=Decimal("0.004"), upper=Decimal("0.006")),
            precision=precision,
        )


@pytest.mark.parametrize(
    "make_economics",
    [
        lambda: reference.Economics(tax_rate=Fraction(1)),
        lambda: reference.Economics(share=Fraction(1)),
        lambda: reference.Economics(tax_kink=Fraction(-1)),
        lambda: reference.Economics(incomes=(Fraction(0), Fraction(2))),
        lambda: reference.Economics(
            probability_rows=((Fraction(-1), Fraction(2)), (Fraction(0), Fraction(1))),
        ),
        lambda: reference.Economics(
            probability_rows=(
                (Fraction(4, 5), Fraction(1, 4)),
                (Fraction(1, 5), Fraction(4, 5)),
            ),
        ),
    ],
)
def test_invalid_economics_are_refused(
    make_economics: Callable[[], reference.Economics],
) -> None:
    """Concavity and nominal-probability assumptions are explicit input obligations."""
    with pytest.raises(ValueError, match="economics"):
        make_economics()


def test_shadow_retains_actual_nonunit_weight_mass() -> None:
    """Raw shadow coefficients scale continuation without probability normalization."""
    nominal = reference.Economics()
    shadow = replace(
        nominal,
        shadow=True,
        probability_rows=tuple(
            tuple(2 * weight for weight in row) for row in nominal.probability_rows
        ),
    )
    original = reference.objective0(
        economics=nominal,
        wealth=Fraction(1),
        income=Fraction(1),
        consumption=Fraction(1),
    )
    doubled = reference.objective0(
        economics=shadow,
        wealth=Fraction(1),
        income=Fraction(1),
        consumption=Fraction(1),
    )
    assert Fraction(doubled.lower) <= 2 * Fraction(original.upper)
    assert 2 * Fraction(original.lower) <= Fraction(doubled.upper)
    assert doubled.lower > original.upper > 0


@pytest.mark.parametrize("consumption", [Fraction(0), Fraction(2)])
def test_infeasible_actions_are_refused_without_clipping(consumption: Fraction) -> None:
    """The true objective refuses actions outside the original borrowing budget."""
    with pytest.raises(ValueError, match="consumption"):
        reference.objective0(
            economics=reference.Economics(),
            wealth=Fraction(1, 2),
            income=Fraction(1),
            consumption=consumption,
        )


@pytest.mark.parametrize("value", [Fraction(0), Fraction(-1)])
def test_nonpositive_log_arguments_are_refused(value: Fraction) -> None:
    """Log-domain errors cannot become finite scientific scores."""
    with pytest.raises(ValueError, match="positive"):
        reference.log_fraction(value=value)


def test_nonfinite_or_reversed_intervals_are_refused() -> None:
    """Only finite ordered uncertainty bounds enter scientific classifications."""
    with pytest.raises(ValueError, match="finite"):
        reference.Interval(lower=Decimal("NaN"), upper=Decimal(1))
    with pytest.raises(reference.OracleInconsistentError, match="reversed"):
        reference.Interval(lower=Decimal(2), upper=Decimal(1))


def test_exact_value_boundary_is_accepted_and_uncertainty_is_not_hidden() -> None:
    """The unchanged absolute-relative value budget is applied conservatively."""
    assert reference.value_budget(observed=Fraction(503, 500), truth=_point("1"))
    assert not reference.value_budget(
        observed=Fraction(503, 500) + Fraction(1, 10**20),
        truth=_point("1"),
    )


def _small_exponential_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    """Enclose exp at a small rational argument by exact Taylor remainder bounds."""
    assert abs(value) < 1
    count = 60
    term = Fraction(1)
    total = term
    for index in range(1, count + 1):
        term *= value / index
        total += term
    next_term = term * value / (count + 1)
    if value < 0:
        return total + next_term, total
    tail_bound = next_term / (1 - value / (count + 2))
    return total, total + tail_bound


@pytest.mark.parametrize("precision", [80, 160])
@pytest.mark.parametrize("exponent_text", ["0.00005", "0.0002", "-0.00005", "-0.0002"])
def test_nonzero_exponential_contains_independent_taylor_bounds(
    *,
    precision: int,
    exponent_text: str,
) -> None:
    """Positive and negative nonzero exponentials enclose exact Taylor bounds."""
    lower, upper = _small_exponential_bounds(Fraction(Decimal(exponent_text)))
    got = reference.exp_interval(interval=_point(exponent_text), precision=precision)
    assert Fraction(got.lower) <= lower <= upper <= Fraction(got.upper)
    assert Fraction(got.upper) - Fraction(got.lower) < Fraction(
        1, 10 ** (precision - 5)
    )
    if Fraction(Decimal(exponent_text)) > 0:
        assert got.lower > 1
    else:
        assert got.upper < 1


@pytest.mark.parametrize("precision", [80, 160])
@pytest.mark.parametrize(
    ("argument", "regret", "passes"),
    [
        (Fraction(1, 20000), "0.000142625", True),
        (Fraction(1, 5000), "0.0005705", False),
    ],
)
def test_nonzero_loss_preserves_discount_scale_and_one_bp_classification(
    *,
    precision: int,
    argument: Fraction,
    regret: str,
    passes: bool,
) -> None:
    """Discounted-log losses enclose nonzero independent consumption losses."""
    beta = Fraction(19, 20)
    assert Fraction(Decimal(regret)) == (1 + beta + beta * beta) * argument
    exponential_lower, exponential_upper = _small_exponential_bounds(-argument)
    expected_lower = 10000 * (1 - exponential_upper)
    expected_upper = 10000 * (1 - exponential_lower)
    got = reference.loss_basis_points(
        regret=_point(regret),
        beta=beta,
        precision=precision,
    )
    assert (
        Fraction(got.lower) <= expected_lower <= expected_upper <= Fraction(got.upper)
    )
    assert Fraction(got.upper) - Fraction(got.lower) < Fraction(
        1, 10 ** (precision - 10)
    )
    assert got.lower > 0
    if passes:
        assert expected_upper < 1
        reference.require_loss_budget(got)
    else:
        assert expected_lower > 1
        with pytest.raises(reference.OracleBudgetExceededError, match="one bp"):
            reference.require_loss_budget(got)


def test_positive_discount_optimum_is_the_actual_borrowing_endpoint() -> None:
    """Positive continuation utility can leave the original borrowing bound binding."""
    wealth, income = Fraction(1, 100), Fraction(1)
    economics = reference.Economics()
    consumption = Fraction(51, 100)
    assert wealth - consumption == -economics.share * income == Fraction(-1, 2)
    assert wealth - consumption + income == Fraction(1, 2)
    left, right = reference.objective0_slopes(
        economics=economics,
        wealth=wealth,
        income=income,
        consumption=consumption,
    )
    expected = Fraction(100, 51) - Fraction(5681, 5000)
    assert left == right == expected > 0
    got = reference.optimum0(economics=economics, wealth=wealth, income=income)
    assert got.consumption_lower == got.consumption_upper == consumption
    assert got.iterations == 0


def test_child_tax_kink_is_an_exact_straddling_optimum() -> None:
    """The continuous optimizer retains a kink whose one-sided slopes straddle zero."""
    economics = reference.Economics(beta=Fraction(1, 2))
    wealth, income = Fraction(685, 24), Fraction(1)
    consumption = Fraction(95, 6)
    resources = wealth - economics.tax_rate * (wealth - economics.tax_kink)
    assert resources == Fraction(149, 6)
    savings = resources - consumption
    assert savings == 9 > -economics.share * income
    assert savings + income == economics.tax_kink == 10
    left, right = reference.objective0_slopes(
        economics=economics,
        wealth=wealth,
        income=income,
        consumption=consumption,
    )
    assert left == Fraction(199, 20900) > 0
    assert right == Fraction(-13, 3344) < 0
    got = reference.optimum0(economics=economics, wealth=wealth, income=income)
    assert got.consumption_lower == got.consumption_upper == consumption
    assert got.iterations == 3
