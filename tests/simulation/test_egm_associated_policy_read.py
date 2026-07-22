"""The off-grid branch action is read from the conditional value, not a separate row.

A branch is ranked by its cubic-Hermite conditional value; the action returned for the
winning branch must *attain* that value. Reading the action from a separately-linear
policy row can return an action worth less than the ranked value — so a value-ranked
branch's returned pair may be dominated by a competitor. The associated read instead
takes the value read's own resource-derivative `V'(R_q)` — the marginal value of
resources, which by the envelope theorem equals `u'(c*)` — and inverts it with the
regime's `inverse_marginal_utility` to recover the branch optimum `c*` that attains the
ranked value.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.simulation.simulate import _hermite_value_derivative_rows


def _branch_a_rows() -> EGMSimPolicy:
    """A branch whose optimal policy is nonlinear in resources.

    Value nodes `(2, 0)`, `(3, 0.9)` with marginal-value slopes `(1, 0.8)`; the
    linear policy row `(1, 1.25)` does not attain the Hermite value between nodes.
    """
    return EGMSimPolicy(
        endog_grid=jnp.array([2.0, 3.0]),
        policy=jnp.array([1.0, 1.25]),
        value=jnp.array([0.0, 0.9]),
        marginal_utility=jnp.array([1.0, 0.8]),
    )


def test_hermite_value_derivative_equals_the_value_reads_own_slope():
    """`V'(R_q)` from the associated-read derivative is the Hermite read's slope.

    At `R = 2.5` the value read's resource-derivative is `0.9`, matching the
    custom-JVP query tangent (`jax.grad` of the value read).
    """
    sim_policy = _branch_a_rows()
    deriv = _hermite_value_derivative_rows(
        sim_policy=sim_policy, index=(), resources=jnp.array([2.5]), n_subjects=1
    )
    expected = jax.grad(
        lambda x: interp_on_padded_grid(
            x_query=x,
            xp=sim_policy.endog_grid,
            fp=sim_policy.value,
            fp_slopes=sim_policy.marginal_utility,
        )
    )(jnp.asarray(2.5))
    np.testing.assert_allclose(float(deriv[0]), float(expected), atol=1e-6)
    np.testing.assert_allclose(float(deriv[0]), 0.9, atol=1e-6)


def test_inverting_the_value_derivative_recovers_the_attaining_action():
    """`c = (u')^{-1}(V'(R_q))` attains the ranked value; the linear read does not.

    With log utility `(u')^{-1}(m) = 1/m`, inverting the value derivative at
    `R = 2.5` gives `c = 1/0.9 ≈ 1.111`, whose branch objective exceeds the
    objective of the separately-linear policy read `1.125`.
    """
    sim_policy = _branch_a_rows()
    deriv = _hermite_value_derivative_rows(
        sim_policy=sim_policy, index=(), resources=jnp.array([2.5]), n_subjects=1
    )
    c_associated = 1.0 / float(deriv[0])

    linear_policy = float(
        interp_on_padded_grid(
            x_query=jnp.asarray(2.5),
            xp=sim_policy.endog_grid,
            fp=sim_policy.policy,
        )
    )

    # The branch objective u(c) + W_A(R - c) that generated the published rows.
    def w_a(s: float) -> float:
        y = ((7.0 - s) + np.sqrt((7.0 - s) ** 2 - 20.0)) / 10.0
        t = 5.0 * (1.0 - y)
        return t - 0.1 * t**2 + float(np.log(y))

    def objective(c: float) -> float:
        return float(np.log(c)) + w_a(2.5 - c)

    assert objective(c_associated) > objective(linear_policy)
    np.testing.assert_allclose(c_associated, 1.0 / 0.9, atol=1e-5)


def test_unsupported_all_nan_row_yields_a_nan_derivative():
    """An all-NaN (fail-closed) row's value-derivative is NaN, never a finite slope.

    A row published all-NaN is rejected off-grid so the read falls back to the
    constraint-masked grid pair. The associated read maps the derivative through
    `inverse_marginal_utility`, so the derivative of an unsupported row must stay
    NaN — otherwise a finite slope would invert to a spurious finite action that
    passes the acceptance checks and admits a fabricated read.
    """
    nan_row = EGMSimPolicy(
        endog_grid=jnp.array([jnp.nan, jnp.nan]),
        policy=jnp.array([jnp.nan, jnp.nan]),
        value=jnp.array([jnp.nan, jnp.nan]),
        marginal_utility=jnp.array([jnp.nan, jnp.nan]),
    )
    deriv = _hermite_value_derivative_rows(
        sim_policy=nan_row, index=(), resources=jnp.array([2.5]), n_subjects=1
    )
    assert bool(jnp.isnan(deriv[0]))
