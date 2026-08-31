"""A lottery that pays one amount everywhere has that amount as its mean.

The weighted power mean is a certainty equivalent: what a household would accept
in place of the lottery. When every node pays the same `c` there is no risk to
price, so the answer is `c` at every risk aversion and under every set of
weights — and it is `c` *exactly*, because nothing in the problem has to be
rounded away. There is no trade-off between nodes to resolve, so an answer one
unit in the last place below `c` is not an approximation of the right answer,
it is the wrong one.

The invariant has to hold on a large common level too. Adding the same amount to
every node's payoff reprices no risk and reverses no ordering, so it carries no
economic information — but it consumes the working format's significance, which
is where a mean reconstructed from logarithms loses the value it was handed.
"""

import jax.numpy as jnp
import pytest

from _lcm.power_mean import weighted_power_mean

# Risk aversions spanning both reductions the mean dispatches on. `1.0` is the
# logarithmic household, whose exponent is exactly zero and whose mean is taken
# through the geometric branch; the others go through the power branch.
_RISK_AVERSIONS = [0.0, 0.5, 1.0, 2.0, 5.0]

# Payoff levels, in units of the constant every node pays. `1.0` is the level at
# which a logarithm costs nothing, so it is the control: a mean that fails there
# is failing for some other reason than the one under test.
_LEVELS = [1.0, 7.5, 100.0, 1e12, 1e30]


def _constant_lottery_mean(*, payoff: float, risk_aversion: float) -> float:
    """Return the certainty equivalent of a lottery paying `payoff` everywhere.

    The weights are deliberately unequal and do not sum to one, so the mean has
    to normalize by the total mass rather than inheriting it.
    """
    dtype = jnp.zeros(()).dtype
    return float(
        weighted_power_mean(
            values=jnp.full((3,), payoff, dtype=dtype),
            weights=jnp.asarray([2.0, 1.0, 0.5], dtype=dtype),
            exponent=jnp.asarray(1.0 - risk_aversion, dtype=dtype),
            shifts=jnp.zeros((), dtype=jnp.int32),
        )
    )


@pytest.mark.parametrize("risk_aversion", _RISK_AVERSIONS)
@pytest.mark.parametrize("payoff", _LEVELS)
def test_a_lottery_paying_one_amount_returns_that_amount_exactly(
    *, payoff: float, risk_aversion: float
) -> None:
    """The certainty equivalent of a riskless lottery is its payoff, bit for bit."""
    expected = float(jnp.asarray(payoff, dtype=jnp.zeros(()).dtype))
    assert (
        _constant_lottery_mean(payoff=payoff, risk_aversion=risk_aversion) == expected
    )


@pytest.mark.parametrize("risk_aversion", _RISK_AVERSIONS)
def test_the_largest_representable_payoff_does_not_become_infinite(
    risk_aversion: float,
) -> None:
    """A riskless lottery at the top of the range stays at the top of the range.

    The largest representable payoff is an ordinary finite answer — the mean has
    to return it rather than overflow, since a certainty equivalent can never
    exceed the largest payoff on offer.
    """
    largest = float(jnp.finfo(jnp.zeros(()).dtype).max)
    assert (
        _constant_lottery_mean(payoff=largest, risk_aversion=risk_aversion) == largest
    )
