"""The continuation operator generalizes the linear expectation in `Q_and_F`.

A regime may supply a `value_transform` / `inverse_value_transform` pair to
replace the linear continuation `E[V']` with the certainty equivalent
`g_inv(E[g(V')])` (the Epstein-Zin / power-mean and entropic-risk forms). The
default — no transform — is the identity, so the expectation is the plain
weighted average and the solved value function is byte-identical. A genuine
transform changes the value function, and a degenerate (no-risk) continuation
collapses back to the linear expectation regardless of the transform.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.Q_and_F import (
    _apply_continuation_operator,
    _invert_joint,
    _transform_and_average,
)
from lcm import AgeGrid, Model
from lcm.exceptions import ModelInitializationError
from tests.test_models import stochastic


def _identity(continuation_value):
    return continuation_value


def _shift_inverse(continuation_value, continuation_bonus):
    return continuation_value + continuation_bonus


def _entropic_transform(continuation_value, risk_sensitivity):
    return jnp.exp(risk_sensitivity * continuation_value)


def _entropic_inverse(continuation_value, risk_sensitivity):
    return jnp.log(continuation_value) / risk_sensitivity


def _model_with_operator(n_periods, value_transform, inverse_value_transform):
    """The stochastic model with a continuation operator on its non-terminal regimes."""
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    operator = {
        "value_transform": value_transform,
        "inverse_value_transform": inverse_value_transform,
    }
    working = stochastic.working_life.replace(
        active=lambda age, la=last_age: age < la,
        functions={**dict(stochastic.working_life.functions), **operator},
    )
    retirement = stochastic.retirement.replace(
        active=lambda age, la=last_age: age < la,
        functions={**dict(stochastic.retirement.functions), **operator},
    )
    return Model(
        regimes={
            "working_life": working,
            "retirement": retirement,
            "dead": stochastic.dead,
        },
        ages=ages,
        regime_id_class=stochastic.RegimeId,
    )


def test_identity_continuation_operator_is_byte_identical():
    """An identity value-transform pair reproduces the linear-expectation solution."""
    n_periods = 3
    params = stochastic.get_params(n_periods)
    base = stochastic.get_model(n_periods).solve(params=params, log_level="off")

    # The identity transforms carry no params, so the operator model's template
    # equals the base model's and the same params dict solves both.
    operator_model = _model_with_operator(n_periods, _identity, _identity)
    operator_solution = operator_model.solve(params=params, log_level="off")

    assert base.keys() == operator_solution.keys()
    for period in base:
        for regime in base[period]:
            np.testing.assert_array_equal(
                np.asarray(operator_solution[period][regime]),
                np.asarray(base[period][regime]),
            )


def test_entropic_certainty_equivalent_penalizes_a_risky_lottery():
    """`g_inv(E[g(V')])` is the entropic CE: below the mean on a non-degenerate lottery.

    For outcomes `[1, 3]` with equal weight and the entropic pair
    `g(v) = exp(θ v)`, `g_inv(x) = log(x)/θ` at `θ = -1`, the operator returns
    `-log(½(e^{-1} + e^{-3}))`, which is strictly below the linear mean `2.0` — the
    risk penalty. The default (no transform) returns the plain weighted mean.
    """
    values = jnp.array([1.0, 3.0])
    weights = jnp.array([0.5, 0.5])

    linear = _apply_continuation_operator(
        values=values,
        weights=weights,
        value_transform=None,
        inverse_value_transform=None,
        params={},
    )
    np.testing.assert_allclose(float(linear), 2.0)

    entropic = _apply_continuation_operator(
        values=values,
        weights=weights,
        value_transform=_entropic_transform,
        inverse_value_transform=_entropic_inverse,
        params={"risk_sensitivity": jnp.asarray(-1.0)},
    )
    expected = np.log(0.5 * (np.exp(-1.0) + np.exp(-3.0))) / (-1.0)
    np.testing.assert_allclose(float(entropic), expected, rtol=1e-6)
    assert float(entropic) < 2.0


def test_joint_certainty_equivalent_mixes_regimes_inside_the_transform():
    """Over multiple target regimes the operator is the joint CE, not per-target.

    The seam accumulates `sum_r p_r E_z[g(V_rz)]` across targets and applies `g_inv`
    once, giving the joint certainty equivalent `g_inv(sum_rz p_r p_z g(V_rz))` — the
    sound form for a non-linear operator over both regime and shock uncertainty.
    With the entropic pair `g(v)=exp(theta v)`, `g_inv(x)=log(x)/theta`, this equals
    the hand-computed joint value and is strictly below the (unsound) per-target form
    `sum_r p_r g_inv(E_z[g(V_rz)])` when the two regimes' continuations differ.
    """
    theta = -1.0
    p_regime = (0.6, 0.4)  # regime mixing weights
    # Two targets, each a two-outcome shock lottery (equal shock weights).
    target_values = (jnp.array([0.0, 2.0]), jnp.array([3.0, 5.0]))
    shock_weights = jnp.array([0.5, 0.5])
    params = {"risk_sensitivity": jnp.asarray(theta)}

    transformed_mix = sum(
        p
        * _transform_and_average(
            values=vals,
            weights=shock_weights,
            value_transform=_entropic_transform,
            params=params,
        )
        for p, vals in zip(p_regime, target_values, strict=True)
    )
    joint = float(
        _invert_joint(
            transformed_mix,
            inverse_value_transform=_entropic_inverse,
            params=params,
        )
    )

    # Hand value: g_inv( sum_rz p_r p_z exp(theta V_rz) ).
    g_sum = sum(
        p * 0.5 * (np.exp(theta * float(vals[0])) + np.exp(theta * float(vals[1])))
        for p, vals in zip(p_regime, target_values, strict=True)
    )
    expected = np.log(g_sum) / theta
    np.testing.assert_allclose(joint, expected, rtol=1e-6)

    # The unsound per-target-then-mix form differs (g_inv inside the regime mix).
    per_target = sum(
        p
        * float(
            _invert_joint(
                _transform_and_average(
                    values=vals,
                    weights=shock_weights,
                    value_transform=_entropic_transform,
                    params=params,
                ),
                inverse_value_transform=_entropic_inverse,
                params=params,
            )
        )
        for p, vals in zip(p_regime, target_values, strict=True)
    )
    assert abs(joint - per_target) > 1e-3


def test_non_identity_operator_changes_the_solved_value_function():
    """A non-identity continuation operator is applied through a full solve.

    A level-shift operator (`g = id`, `g_inv(x) = x + bonus`) adds `bonus` to every
    continuation `E[V']`, so the solved value function rises relative to the linear
    solution everywhere a continuation enters — proving the operator reaches the
    backward induction and reads its param, regardless of the lottery's spread.
    """
    n_periods = 3
    base_params = stochastic.get_params(n_periods)
    base = stochastic.get_model(n_periods).solve(params=base_params, log_level="off")

    operator_model = _model_with_operator(n_periods, _identity, _shift_inverse)
    params = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in base_params.items()
    }
    for regime in ("working_life", "retirement"):
        params[regime]["inverse_value_transform"] = {"continuation_bonus": 1.0}
    operator_solution = operator_model.solve(params=params, log_level="off")

    raised_somewhere = False
    for period in base:
        for regime in base[period]:
            base_v = np.asarray(base[period][regime])
            op_v = np.asarray(operator_solution[period][regime])
            finite = np.isfinite(base_v) & np.isfinite(op_v)
            if np.any(op_v[finite] > base_v[finite] + 1e-6):
                raised_somewhere = True
    assert raised_somewhere


def test_supplying_only_one_transform_is_rejected():
    """A `value_transform` without its inverse (or vice versa) is a build error."""
    n_periods = 2
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    working = stochastic.working_life.replace(
        active=lambda age, la=last_age: age < la,
        functions={
            **dict(stochastic.working_life.functions),
            "value_transform": _identity,
        },
    )
    with pytest.raises(ModelInitializationError, match="inverse_value_transform"):
        Model(
            regimes={
                "working_life": working,
                "retirement": stochastic.retirement.replace(
                    active=lambda age, la=last_age: age < la
                ),
                "dead": stochastic.dead,
            },
            ages=ages,
            regime_id_class=stochastic.RegimeId,
        ).solve(params=stochastic.get_params(n_periods), log_level="off")
