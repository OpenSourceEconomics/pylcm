"""One Euler contract covers both built-in temporal aggregators.

An EGM step chooses an action `c` at a post-decision node `s`, and its value is
`V(c) = H(q(c), nu(s))` with `s = resources - c`. The interior first-order
condition is therefore

```{math}
\\frac{\\partial H}{\\partial U}\\, q_c(c)
  = \\frac{\\partial H}{\\partial CE}\\, \\frac{d\\nu}{ds},
```

whatever `H` is. What decides whether the endogenous grid method *applies* is
whether that condition separates: the action must be recoverable from a single
scalar carried back from the continuation.

Writing the condition as `MRS(q(c), nu) q_c(c) = dnu/ds`, with `MRS` the
marginal rate of substitution `(dH/dU) / (dH/dCE)`, it separates exactly when
that ratio factors multiplicatively, `MRS(U, CE) = A(U) B(CE)`. The condition is
then `A(q(c)) q_c(c) = (dnu/ds) / B(nu)` — an action-only left side and a
node-only right side. An additively separable `H(U, CE) = f(a(U) + b(CE))` is a
sufficient special case, where `f'` cancels and `MRS = a'(U) / b'(CE)`; it is not
the boundary of the admissible class, and an aggregator that couples the two
arguments can still qualify.

That criterion is executable: a product factors if and only if its logarithm is
a sum, so `H` admits an endogenous-grid step if and only if

```{math}
\\frac{\\partial^2}{\\partial U \\, \\partial CE}
  \\log\\frac{\\partial H / \\partial U}{\\partial H / \\partial CE} = 0 .
```

Both built-in aggregators satisfy it, and their two closed-form inversions —
the time-separable Euler equation and the Epstein-Zin one — are recovered from
the same automatic-differentiation residual with no aggregator-specific algebra.
An aggregator that fails the criterion is one whose optimum an EGM step cannot
recover from a scalar, and it is rejected rather than silently mis-solved.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.crra import crra_utility
from _lcm.egm.ez_kernel import ez_consumption_from_euler
from lcm.koopmans_aggregation import W_epstein_zin, W_linear

# Exact algebra: closed-form identities and a cross-partial that must vanish, so
# this module runs with x64 regardless of the suite's `--precision` flag.
pytestmark = pytest.mark.usefixtures("x64_enabled")

_DISCOUNT_FACTOR = 0.94
_EIS = 2.0
_INVERSE_EIS = 1.0 / _EIS
_CRRA = 2.0

# The criterion is a second derivative evaluated by automatic differentiation,
# so it is zero to a few rounding errors rather than bit-zero.
_SEPARABLE_ATOL = 1e-10


def _linear(utility, continuation):
    return W_linear(
        utility=jnp.asarray(utility),
        CE=jnp.asarray(continuation),
        discount_factor=jnp.asarray(_DISCOUNT_FACTOR),
    )


def _epstein_zin(utility, continuation):
    return W_epstein_zin(
        utility=jnp.asarray(utility),
        CE=jnp.asarray(continuation),
        discount_factor=jnp.asarray(_DISCOUNT_FACTOR),
        intertemporal_elasticity_of_substitution=jnp.asarray(_EIS),
    )


def _factoring_interaction(utility, continuation):
    """A coupled aggregator that nonetheless admits an endogenous-grid step.

    Its marginal rate of substitution is `(1 + 0.1 CE) / (beta + 0.1 U)`, which
    factors, so the continuation still enters the action's first-order condition
    through one scalar. Interaction between the two arguments does not by itself
    put an aggregator outside the contract.
    """
    return utility + _DISCOUNT_FACTOR * continuation + 0.1 * utility * continuation


def _non_factoring(utility, continuation):
    """An aggregator whose marginal rate of substitution does not factor.

    Its MRS is `(1 + 0.2 U CE) / (beta + 0.1 U^2)`, whose numerator mixes the
    two arguments in a way no product of a flow term and a continuation term
    reproduces. The action then cannot be recovered from any single scalar.
    """
    return utility + _DISCOUNT_FACTOR * continuation + 0.1 * utility**2 * continuation


def _identity_flow(consumption):
    """The basic single-good period flow `q(c) = c`."""
    return consumption


def _log_mrs(aggregator, utility, continuation):
    """Log marginal rate of substitution between flow and continuation."""
    d_utility = jax.grad(aggregator, argnums=0)(utility, continuation)
    d_continuation = jax.grad(aggregator, argnums=1)(utility, continuation)
    return jnp.log(d_utility) - jnp.log(d_continuation)


def _flow_curvature_of_mrs(aggregator, utility, continuation):
    """`d/dU log MRS`, which is `A'(U) / A(U)` once the MRS factors.

    Where factorization holds this is independent of the continuation, so it is
    a property of the aggregator's flow side alone.
    """
    return float(
        jax.grad(_log_mrs, argnums=1)(
            aggregator, jnp.asarray(utility), jnp.asarray(continuation)
        )
    )


def _separability_defect(aggregator, utility, continuation):
    """The cross partial that vanishes exactly when the FOC separates."""
    cross = jax.grad(jax.grad(_log_mrs, argnums=1), argnums=2)
    return float(cross(aggregator, jnp.asarray(utility), jnp.asarray(continuation)))


def _euler_residual(aggregator, *, consumption, flow, flow_marginal, nu, dnu_ds):
    """Log-form interior FOC residual, decreasing in the action.

    Taking logs cancels the monotone outer transform, so the residual is the
    same object for every separable aggregator and its root is the optimum.
    """
    return (
        _log_mrs(aggregator, flow(consumption), nu)
        + jnp.log(flow_marginal(consumption))
        - jnp.log(dnu_ds)
    )


def _bisect_euler(aggregator, *, flow, flow_marginal, nu, dnu_ds):
    """Solve the AD-derived FOC for the action, with no closed form assumed."""
    lower, upper = 1e-8, 1e4
    for _ in range(200):
        mid = 0.5 * (lower + upper)
        residual = _euler_residual(
            aggregator,
            consumption=jnp.asarray(mid),
            flow=flow,
            flow_marginal=flow_marginal,
            nu=jnp.asarray(nu),
            dnu_ds=jnp.asarray(dnu_ds),
        )
        # Decreasing in the action: a positive residual means the action is
        # still too small.
        lower, upper = (mid, upper) if residual > 0 else (lower, mid)
    return 0.5 * (lower + upper)


@pytest.mark.parametrize(
    "aggregator",
    [
        pytest.param(_linear, id="W_linear"),
        pytest.param(_epstein_zin, id="W_epstein_zin"),
    ],
)
@pytest.mark.parametrize(("utility", "continuation"), [(0.7, 1.3), (2.5, 0.4)])
def test_built_in_aggregators_admit_an_endogenous_grid_step(
    aggregator, utility, continuation
):
    """Both shipped aggregators separate, so a scalar determines the action."""
    assert abs(_separability_defect(aggregator, utility, continuation)) < (
        _SEPARABLE_ATOL
    )


@pytest.mark.parametrize(("utility", "continuation"), [(0.7, 1.3), (2.5, 0.4)])
def test_a_non_factoring_aggregator_is_detected_as_outside_the_contract(
    utility, continuation
):
    """An aggregator whose MRS does not factor is rejected.

    The defect is a finite quantity, not a rounding-scale one, so the criterion
    separates the two cases by orders of magnitude rather than by a threshold
    chosen to make it do so.
    """
    defect = abs(_separability_defect(_non_factoring, utility, continuation))
    assert defect > 1e-3


@pytest.mark.parametrize(("utility", "continuation"), [(0.7, 1.3), (2.5, 0.4)])
def test_coupling_the_two_arguments_does_not_by_itself_disqualify(
    utility, continuation
):
    """An aggregator may couple flow and continuation and still separate.

    This pins the contract's boundary where it actually is. Reading the
    criterion as "additively separable" would reject this aggregator, whose
    optimum an endogenous-grid step recovers perfectly well.
    """
    defect = abs(_separability_defect(_factoring_interaction, utility, continuation))
    assert defect < _SEPARABLE_ATOL


@pytest.mark.parametrize("dnu_ds", [0.35, 1.2])
def test_the_derived_condition_reproduces_the_time_separable_euler_equation(dnu_ds):
    """With `W_linear` the contract yields `c = (beta * dnu/ds)^(-1/crra)`."""
    got = _bisect_euler(
        _linear,
        flow=lambda c: crra_utility(c, _CRRA),
        flow_marginal=lambda c: c ** (-_CRRA),
        nu=1.0,
        dnu_ds=dnu_ds,
    )
    expected = (_DISCOUNT_FACTOR * dnu_ds) ** (-1.0 / _CRRA)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


@pytest.mark.parametrize(("nu", "dnu_ds"), [(1.4, 0.6), (0.8, 1.9)])
def test_the_derived_condition_reproduces_the_epstein_zin_closed_form(nu, dnu_ds):
    """With `W_epstein_zin` the contract yields `ez_consumption_from_euler`.

    The kernel's closed form covers a period flow whose risk-adjusted marginal
    `q^(-rho) q_c` is a single power of the action; the basic single-good flow
    `q = c` is that case with a unit coefficient. The contract reaches the same
    action without knowing the flow is a power at all.
    """
    got = _bisect_euler(
        _epstein_zin,
        flow=_identity_flow,
        flow_marginal=jnp.ones_like,
        nu=nu,
        dnu_ds=dnu_ds,
    )
    expected = ez_consumption_from_euler(
        nu=jnp.asarray(nu),
        dnu_ds=jnp.asarray(dnu_ds),
        discount_factor=_DISCOUNT_FACTOR,
        inverse_eis=_INVERSE_EIS,
        log_flow_coefficient=0.0,
        flow_exponent=-_INVERSE_EIS,
    )
    np.testing.assert_allclose(got, float(expected), rtol=1e-6)


def _convex_flow_side(utility, continuation):
    """A factoring aggregator whose flow-side factor `A` increases.

    Additively separable, so it factors and admits an endogenous-grid step. But
    its `A(U) = 2U / beta` grows with the flow, and `g = A(q(c)) q_c(c)` then
    increases in the action for an ordinary concave flow — the opposite of what
    a bracketed root find assumes.
    """
    return utility**2 + _DISCOUNT_FACTOR * continuation


@pytest.mark.parametrize(
    "aggregator",
    [pytest.param(_linear, id="W_linear"), pytest.param(_epstein_zin, id="H_ez")],
)
@pytest.mark.parametrize(("utility", "continuation"), [(0.7, 1.3), (2.5, 0.4)])
def test_shipped_aggregators_also_admit_numeric_inversion(
    aggregator, utility, continuation
):
    """Both shipped aggregators have a non-increasing flow-side factor.

    With `g = A(q(c)) q_c(c)`, the derivative is
    `g' = A'(q) q_c^2 + A(q) q_cc`. A strictly increasing, strictly concave flow
    makes the second term negative, so `A' <= 0` is what makes `g` strictly
    decreasing and its root unique on a bracket. `W_linear` has a constant `A`;
    Epstein-Zin has `A(U) = (1-beta) U^(-rho)` with `rho > 0`.
    """
    assert _flow_curvature_of_mrs(aggregator, utility, continuation) <= 0.0


@pytest.mark.parametrize(("utility", "continuation"), [(0.7, 1.3), (2.5, 0.4)])
def test_factorization_alone_does_not_license_numeric_inversion(utility, continuation):
    """An aggregator can factor and still forbid a bracketed root find.

    This is why admissibility and invertibility are two gates rather than one:
    the first decides whether an endogenous-grid step exists at all, the second
    whether its action may be recovered numerically instead of from a declared
    closed form.
    """
    assert abs(_separability_defect(_convex_flow_side, utility, continuation)) < (
        _SEPARABLE_ATOL
    )
    assert _flow_curvature_of_mrs(_convex_flow_side, utility, continuation) > 0.0


def test_the_flow_side_curvature_predicts_the_direction_of_g():
    """`d/dU log MRS` has the sign of `g`'s slope for a concave flow.

    Checked directly rather than inferred: `g` is sampled across a bracket for
    an aggregator on each side of the criterion, with the same concave flow.
    The flow is `c**0.75` -- strictly increasing, strictly concave, and strictly
    positive, which Epstein-Zin requires of both its arguments. The exponent is
    deliberately above one half: at exactly one half, `q q_c` is constant, so
    the convex-flow-side aggregator's `g` is flat rather than increasing and the
    case would be a knife edge rather than a discriminator.
    """
    actions = jnp.linspace(0.2, 4.0, 40)

    def g_values(aggregator):
        return jnp.asarray(
            [
                jnp.exp(_log_mrs(aggregator, c**0.75, jnp.asarray(1.0)))
                * 0.75
                * c ** (-0.25)
                for c in actions
            ]
        )

    decreasing = np.diff(np.asarray(g_values(_epstein_zin)))
    increasing = np.diff(np.asarray(g_values(_convex_flow_side)))
    assert (decreasing < 0).all()
    assert (increasing > 0).all()
