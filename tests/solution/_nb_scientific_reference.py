"""Independent exact-budget and outward-rounded reference for the log economy."""

from dataclasses import dataclass
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    Clamped,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    Subnormal,
    Underflow,
)
from fractions import Fraction


class OracleInconsistentError(ValueError):
    """The purported mathematical enclosures have an empty feasible intersection."""


class OracleUnresolvedError(ValueError):
    """The available uncertainty crosses the declared scientific budget."""


class OracleBudgetExceededError(ValueError):
    """The complete uncertainty interval exceeds the scientific budget."""


@dataclass(frozen=True, kw_only=True)
class Interval:
    """Finite lower and upper Decimal endpoints, ordered independently of context."""

    lower: Decimal
    """An outward lower bound on the exact real quantity."""

    upper: Decimal
    """An outward upper bound on the exact real quantity."""

    def __post_init__(self) -> None:
        if not self.lower.is_finite() or not self.upper.is_finite():
            raise ValueError("Interval endpoints must be finite.")
        if self.lower > self.upper:
            raise OracleInconsistentError("Interval endpoints are reversed.")


@dataclass(frozen=True, kw_only=True)
class Economics:
    """Exact intended primitives; shadow weights retain their separate meaning."""

    beta: Fraction = Fraction(19, 20)
    """Nonnegative per-period utility discount factor."""

    tax_rate: Fraction = Fraction(1, 5)
    """Marginal resource tax above the wealth kink."""

    tax_kink: Fraction = Fraction(10)
    """Nonnegative wealth at which the marginal resource tax begins."""

    share: Fraction = Fraction(1, 2)
    """Fraction of current income that can finance borrowing."""

    incomes: tuple[Fraction, Fraction] = (Fraction(1), Fraction(2))
    """Distinct positive income amounts in transition-row order."""

    probability_rows: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] = (
        (Fraction(4, 5), Fraction(1, 5)),
        (Fraction(1, 5), Fraction(4, 5)),
    )
    """Nonnegative exact transition coefficients, retaining raw shadow mass."""

    shadow: bool = False
    """Whether raw nonunit transition masses are admitted for sensitivity scoring."""

    def __post_init__(self) -> None:
        if not isinstance(self.beta, Fraction) or self.beta < 0:
            raise ValueError("economics beta must be an exact nonnegative Fraction.")
        _validate_economics(self)


@dataclass(frozen=True, kw_only=True)
class LastPeriod:
    """An independently optimized last-period tuple and its wealth-side slopes."""

    consumption: Fraction
    """Exact optimal positive consumption under the original saving constraint."""

    savings: Fraction
    """Current net resources minus consumption, before income arrives."""

    terminal_wealth: Fraction
    """Saving plus current income, entering terminal log utility without tax."""

    value: Interval
    """An enclosure of log consumption plus discounted terminal log wealth."""

    slope_left: Fraction
    """The left derivative with respect to current physical wealth."""

    slope_right: Fraction
    """The right derivative with respect to current physical wealth."""


@dataclass(frozen=True, kw_only=True)
class Optimum:
    """A continuous optimum bracket and global objective enclosure."""

    consumption_lower: Fraction
    """A positive exact lower bracket endpoint containing the maximizer."""

    consumption_upper: Fraction
    """An exact upper bracket endpoint containing the maximizer."""

    value: Interval
    """A feasible objective lower bound and a global concavity upper bound."""

    iterations: int
    """The number of halving and bisection iterations used for certification."""


def add(*, left: Interval, right: Interval, precision: int = 80) -> Interval:
    """Enclose addition with directed private-context rounding."""
    return Interval(
        lower=_context(precision=precision).add(left.lower, right.lower),
        upper=_context(precision=precision, upward=True).add(left.upper, right.upper),
    )


def subtract(*, left: Interval, right: Interval, precision: int = 80) -> Interval:
    """Enclose subtraction with reversed right endpoints."""
    return add(left=left, right=negate(right), precision=precision)


def negate(interval: Interval) -> Interval:
    """Reverse endpoints and their signs without ambient-context rounding."""
    return Interval(
        lower=interval.upper.copy_negate(), upper=interval.lower.copy_negate()
    )


def scale(
    *, interval: Interval, coefficient: Fraction, precision: int = 80
) -> Interval:
    """Enclose signed exact-rational multiplication without rounding its coefficient."""
    if not isinstance(coefficient, Fraction):
        raise TypeError("Scaling requires an exact Fraction coefficient.")
    if coefficient < 0:
        interval = negate(interval)
        coefficient = -coefficient
    lower_context, upper_context = (
        _context(precision=precision),
        _context(precision=precision, upward=True),
    )
    numerator = Decimal(coefficient.numerator)
    denominator = Decimal(coefficient.denominator)
    return Interval(
        lower=lower_context.divide(
            lower_context.multiply(interval.lower, numerator), denominator
        ),
        upper=upper_context.divide(
            upper_context.multiply(interval.upper, numerator), denominator
        ),
    )


def log_fraction(*, value: Fraction, precision: int = 80) -> Interval:
    """Enclose ln(p)-ln(q) using exact integers and adjacent rounded results."""
    if not isinstance(value, Fraction) or value <= 0:
        raise ValueError("Logarithm requires a positive rational argument.")
    _context(precision=precision)
    return subtract(
        left=_log_integer(value=value.numerator, precision=precision),
        right=_log_integer(value=value.denominator, precision=precision),
        precision=precision,
    )


def exp_interval(*, interval: Interval, precision: int = 80) -> Interval:
    """Enclose exp by monotone endpoint evaluation in an explicit private context."""
    _context(precision=precision)
    return Interval(
        lower=_exp_point(value=interval.lower, precision=precision).lower,
        upper=_exp_point(value=interval.upper, precision=precision).upper,
    )


def nonnegative_regret(
    *, optimum: Interval, action: Interval, precision: int = 80
) -> Interval:
    """Intersect regret bounds with nonnegative values, refusing emptiness."""
    difference = subtract(left=optimum, right=action, precision=precision)
    if difference.upper < 0:
        raise OracleInconsistentError("Nonnegative regret intersection is empty.")
    return Interval(lower=max(Decimal(0), difference.lower), upper=difference.upper)


def loss_basis_points(
    *, regret: Interval, beta: Fraction, precision: int = 80
) -> Interval:
    """Enclose the three-period proportional discounted-log consumption loss."""
    if not isinstance(beta, Fraction) or beta < 0 or regret.lower < 0:
        raise ValueError("Loss requires nonnegative beta and regret.")
    argument = scale(
        interval=regret, coefficient=-1 / (1 + beta + beta * beta), precision=precision
    )
    loss = nonnegative_regret(
        optimum=_point(1),
        action=exp_interval(interval=argument, precision=precision),
        precision=precision,
    )
    return scale(interval=loss, coefficient=Fraction(10000), precision=precision)


def require_loss_budget(interval: Interval) -> None:
    """Require the conservative upper endpoint to meet the exact one-bp ceiling."""
    if interval.lower < 0:
        raise ValueError("Consumption loss must be nonnegative.")
    if interval.upper <= 1:
        return
    if interval.lower > 1:
        raise OracleBudgetExceededError("The complete loss interval exceeds one bp.")
    raise OracleUnresolvedError("Oracle uncertainty crosses the one-bp loss budget.")


def value_budget(*, observed: Fraction, truth: Interval, precision: int = 80) -> bool:
    """Apply the .005 absolute and .001 relative structural value budgets."""
    _context(precision=precision)
    if not isinstance(observed, Fraction):
        raise TypeError("Observed value must be an exact Fraction.")
    lower, upper = Fraction(truth.lower), Fraction(truth.upper)
    max_error = max(abs(observed - lower), abs(observed - upper))
    min_error = max(lower - observed, observed - upper, Fraction(0))
    min_abs = max(lower, -upper, Fraction(0))
    max_abs = max(abs(lower), abs(upper))
    if max_error <= Fraction(1, 200) + Fraction(1, 1000) * min_abs:
        return True
    if min_error > Fraction(1, 200) + Fraction(1, 1000) * max_abs:
        return False
    raise OracleUnresolvedError(
        "Oracle uncertainty crosses the structural value budget."
    )


def last_period(
    *, economics: Economics, wealth: Fraction, income: Fraction, precision: int = 80
) -> LastPeriod:
    """Optimize the original budget and retain exact one-sided envelope derivatives."""
    consumption, savings, terminal = _last_quantities(
        economics=economics, wealth=wealth, income=income
    )
    left, right = _wealth_slopes(
        economics=economics, wealth=wealth, consumption=consumption
    )
    value = add(
        left=log_fraction(value=consumption, precision=precision),
        right=scale(
            interval=log_fraction(value=terminal, precision=precision),
            coefficient=economics.beta,
            precision=precision,
        ),
        precision=precision,
    )
    return LastPeriod(
        consumption=consumption,
        savings=savings,
        terminal_wealth=terminal,
        value=value,
        slope_left=left,
        slope_right=right,
    )


def objective0(
    *,
    economics: Economics,
    wealth: Fraction,
    income: Fraction,
    consumption: Fraction,
    precision: int = 80,
) -> Interval:
    """Enclose Q0 using current income in the physical next-wealth law."""
    child_wealth, weights = _action_domain(
        economics=economics, wealth=wealth, income=income, consumption=consumption
    )
    continuation = _point(0)
    for future_income, weight in zip(economics.incomes, weights, strict=True):
        if weight:
            child = last_period(
                economics=economics,
                wealth=child_wealth,
                income=future_income,
                precision=precision,
            )
            continuation = add(
                left=continuation,
                right=scale(
                    interval=child.value, coefficient=weight, precision=precision
                ),
                precision=precision,
            )
    return add(
        left=log_fraction(value=consumption, precision=precision),
        right=scale(
            interval=continuation, coefficient=economics.beta, precision=precision
        ),
        precision=precision,
    )


def objective0_slopes(
    *, economics: Economics, wealth: Fraction, income: Fraction, consumption: Fraction
) -> tuple[Fraction, Fraction]:
    """Return exact left/right consumption slopes, reversing child-wealth sides."""
    child_wealth, weights = _action_domain(
        economics=economics, wealth=wealth, income=income, consumption=consumption
    )
    left, right = Fraction(1) / consumption, Fraction(1) / consumption
    for future_income, weight in zip(economics.incomes, weights, strict=True):
        child_consumption, _, _ = _last_quantities(
            economics=economics, wealth=child_wealth, income=future_income
        )
        child_left, child_right = _wealth_slopes(
            economics=economics, wealth=child_wealth, consumption=child_consumption
        )
        left -= economics.beta * weight * child_right
        right -= economics.beta * weight * child_left
    return left, right


def optimum0(
    *,
    economics: Economics,
    wealth: Fraction,
    income: Fraction,
    precision: int = 80,
    max_halvings: int = 128,
    max_bisections: int = 128,
) -> Optimum:
    """Enclose the continuous maximum with exact KKT and concavity tangent bounds."""
    _context(precision=precision)
    if max_halvings < 0 or max_bisections < 0:
        raise ValueError("Iteration budgets must be nonnegative.")
    upper = _resources(economics=economics, wealth=wealth) + economics.share * income
    left, _ = objective0_slopes(
        economics=economics, wealth=wealth, income=income, consumption=upper
    )
    if left >= 0:
        return _optimum_at(
            economics=economics,
            wealth=wealth,
            income=income,
            consumption=upper,
            precision=precision,
            iterations=0,
        )
    lower, halvings = _bracket_lower(
        economics=economics,
        wealth=wealth,
        income=income,
        upper=upper,
        max_halvings=max_halvings,
    )
    for iteration in range(1, max_bisections + 1):
        middle = (lower + upper) / 2
        left, right = objective0_slopes(
            economics=economics, wealth=wealth, income=income, consumption=middle
        )
        if left >= 0 >= right:
            return _optimum_at(
                economics=economics,
                wealth=wealth,
                income=income,
                consumption=middle,
                precision=precision,
                iterations=halvings + iteration,
            )
        slope = right if right > 0 else left
        tangent_gap = max(slope * (lower - middle), slope * (upper - middle))
        if tangent_gap <= Fraction(1, 10 ** (precision // 2)):
            value = objective0(
                economics=economics,
                wealth=wealth,
                income=income,
                consumption=middle,
                precision=precision,
            )
            ceiling = add(
                left=value,
                right=_rational_interval(value=tangent_gap, precision=precision),
                precision=precision,
            )
            return Optimum(
                consumption_lower=lower,
                consumption_upper=upper,
                value=Interval(lower=value.lower, upper=ceiling.upper),
                iterations=halvings + iteration,
            )
        if right > 0:
            lower = middle
        else:
            upper = middle
    raise OracleUnresolvedError("The optimum bisection budget is exhausted.")


def _validate_economics(economics: Economics) -> None:
    """Validate the increasing-concave resource and nonnegative-weight domain."""
    scalars = (economics.tax_rate, economics.tax_kink, economics.share)
    if any(not isinstance(value, Fraction) for value in scalars):
        raise ValueError("economics primitives must be exact Fractions.")
    if not 0 <= economics.tax_rate < 1 or not 0 <= economics.share < 1:
        raise ValueError("economics tax_rate and share must lie in [0,1).")
    if economics.tax_kink < 0:
        raise ValueError("economics tax_kink must be nonnegative.")
    if len(economics.incomes) != 2 or len(set(economics.incomes)) != 2:
        raise ValueError("economics requires two distinct incomes.")
    if any(
        not isinstance(value, Fraction) or value <= 0 for value in economics.incomes
    ):
        raise ValueError("economics incomes must be exact positive Fractions.")
    if len(economics.probability_rows) != 2:
        raise ValueError("economics requires two transition rows.")
    for row in economics.probability_rows:
        if len(row) != 2 or any(
            not isinstance(weight, Fraction) or weight < 0 for weight in row
        ):
            raise ValueError(
                "economics transition weights must be nonnegative Fractions."
            )
        if not economics.shadow and sum(row, start=Fraction(0)) != 1:
            raise ValueError(
                "economics nominal transition rows must sum exactly to one."
            )


def _context(*, precision: int, upward: bool = False) -> Context:
    """Create a bounded private context independent of caller rounding and traps."""
    if not 80 <= precision <= 160:
        raise ValueError("Oracle precision must be between 80 and 160 digits.")
    return Context(
        prec=precision,
        rounding=ROUND_CEILING if upward else ROUND_FLOOR,
        Emin=-999999,
        Emax=999999,
        traps=[
            InvalidOperation,
            DivisionByZero,
            Overflow,
            Underflow,
            Subnormal,
            Clamped,
        ],
        flags=[],
    )


def _rational_interval(*, value: Fraction, precision: int) -> Interval:
    """Convert an exact rational outward without a rounded coefficient."""
    numerator, denominator = Decimal(value.numerator), Decimal(value.denominator)
    return Interval(
        lower=_context(precision=precision).divide(numerator, denominator),
        upper=_context(precision=precision, upward=True).divide(numerator, denominator),
    )


def _point(value: int) -> Interval:
    """Return an exact integer singleton interval."""
    return Interval(lower=Decimal(value), upper=Decimal(value))


def _log_integer(*, value: int, precision: int) -> Interval:
    """Enclose a correctly rounded integer logarithm by adjacent context values."""
    if value == 1:
        return _point(0)
    context = _context(precision=precision)
    rounded = context.ln(Decimal(value))
    return Interval(lower=context.next_minus(rounded), upper=context.next_plus(rounded))


def _exp_point(*, value: Decimal, precision: int) -> Interval:
    """Enclose a correctly rounded exponential by adjacent context values."""
    if value.is_zero():
        return _point(1)
    context = _context(precision=precision)
    rounded = context.exp(value)
    return Interval(lower=context.next_minus(rounded), upper=context.next_plus(rounded))


def _resources(*, economics: Economics, wealth: Fraction) -> Fraction:
    """Apply current saving-regime tax once to positive physical wealth."""
    if not isinstance(wealth, Fraction) or wealth <= 0:
        raise ValueError("Physical wealth must be positive.")
    return wealth - economics.tax_rate * max(wealth - economics.tax_kink, Fraction(0))


def _last_quantities(
    *, economics: Economics, wealth: Fraction, income: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    """Solve exact last-period budgets before evaluating any logarithms."""
    if not isinstance(income, Fraction) or income not in economics.incomes:
        raise ValueError("Current income must identify a declared transition row.")
    resources = _resources(economics=economics, wealth=wealth)
    consumption = min(
        (resources + income) / (1 + economics.beta),
        resources + economics.share * income,
    )
    savings = resources - consumption
    terminal = savings + income
    if consumption <= 0 or terminal <= 0 or savings < -economics.share * income:
        raise ValueError(
            "Last-period budgets require positive log arguments and legal saving."
        )
    return consumption, savings, terminal


def _wealth_slopes(
    *, economics: Economics, wealth: Fraction, consumption: Fraction
) -> tuple[Fraction, Fraction]:
    """Return exact wealth slopes on both sides of the resource tax kink."""
    left = 1 - economics.tax_rate if wealth > economics.tax_kink else Fraction(1)
    right = 1 - economics.tax_rate if wealth >= economics.tax_kink else Fraction(1)
    return left / consumption, right / consumption


def _action_domain(
    *, economics: Economics, wealth: Fraction, income: Fraction, consumption: Fraction
) -> tuple[Fraction, tuple[Fraction, Fraction]]:
    """Check the actual current budget and bind its future-income weight row."""
    if not isinstance(income, Fraction) or income not in economics.incomes:
        raise ValueError("Current income must identify a declared transition row.")
    resources = _resources(economics=economics, wealth=wealth)
    if not isinstance(consumption, Fraction) or not (
        0 < consumption <= resources + economics.share * income
    ):
        raise ValueError("consumption violates the original positive borrowing budget.")
    return resources - consumption + income, economics.probability_rows[
        economics.incomes.index(income)
    ]


def _optimum_at(
    *,
    economics: Economics,
    wealth: Fraction,
    income: Fraction,
    consumption: Fraction,
    precision: int,
    iterations: int,
) -> Optimum:
    """Enclose the value at an exactly certified endpoint or kink optimum."""
    return Optimum(
        consumption_lower=consumption,
        consumption_upper=consumption,
        value=objective0(
            economics=economics,
            wealth=wealth,
            income=income,
            consumption=consumption,
            precision=precision,
        ),
        iterations=iterations,
    )


def _bracket_lower(
    *,
    economics: Economics,
    wealth: Fraction,
    income: Fraction,
    upper: Fraction,
    max_halvings: int,
) -> tuple[Fraction, int]:
    """Find a positive increasing-side endpoint within a finite halving budget."""
    lower = upper
    for iteration in range(1, max_halvings + 1):
        lower /= 2
        _, right = objective0_slopes(
            economics=economics, wealth=wealth, income=income, consumption=lower
        )
        if right > 0:
            return lower, iteration
    raise OracleUnresolvedError("The optimum halving budget is exhausted.")
