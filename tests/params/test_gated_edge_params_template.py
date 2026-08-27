"""A gated edge's own functions contribute parameters like any other function.

A gated edge is declared with three kinds of user callable: the gate predicate,
one projection per gate reference, and one projection per leg fallback. Each is
an ordinary DAG function, so every scalar it reads beyond the values and states
the engine wires in is a model parameter: it is listed by
`get_params_template()` and the value the user supplies for it is what the
solved model uses.

The topology is a mutual-consent edge. A single woman reaches a married couple
only if both partners prefer marriage to their own single life (the gate), each
partner's single life is valued by a gate reference read at a projected wage,
and a woman whose proposal is refused falls back to her own single value, also
read at a projected wage. Wages live on the two-point grid $\\{1, 2\\}$ and every
value below is exact on it.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    DiscreteGrid,
    GatedEdge,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserFunction,
)
from tests.collective_fixtures import DISCOUNT_FACTOR, Work
from tests.conftest import DECIMAL_PRECISION


def test_gate_scalar_parameter_is_a_model_parameter():
    """A scalar the gate predicate reads is listed and used at the supplied value.

    A marriage premium of 1.5 lifts both partners' married values above their
    single ones at every wage, so the gate is open throughout and the wife's
    continuation is her married value everywhere:
    `V = wage + 0.95 * V_married_f = [1 + 0.95*2, 2 + 0.95*4] = [2.9, 5.8]`.
    Without the premium the husband refuses at the high wage and the value there
    is 4.85 instead.
    """
    model = _build_model(
        gate=_consent_gate_with_premium,
        husband_reference_projection=_wage_itself,
        wife_fallback_projection=_wage_itself,
    )
    assert "marriage_premium" in _parameter_names(
        model.get_params_template()["single_f"]
    )
    solution = model.solve(
        params={"discount_factor": DISCOUNT_FACTOR, "marriage_premium": 1.5},
        log_level="debug",
    )
    aaae(
        np.asarray(solution[0]["single_f"]),
        np.array([2.9, 5.8]),
        decimal=DECIMAL_PRECISION,
    )


def test_gate_ref_projection_scalar_parameter_is_a_model_parameter():
    """A scalar a gate reference's projection reads is listed and used.

    A weight of 0.5 has the husband value his single life halfway between the
    couple's own wage and the top of the wage grid, which raises his outside
    option enough to refuse at both wages. The wife then takes her own fallback
    everywhere: `V = wage + 0.95 * 1.5 * wage = [2.425, 4.85]`. At a weight of
    zero he would accept at the low wage and the value there would be 2.9.
    """
    model = _build_model(
        gate=_consent_gate,
        husband_reference_projection=_husband_reference_wage,
        wife_fallback_projection=_wage_itself,
    )
    assert "husband_reference_weight" in _parameter_names(
        model.get_params_template()["single_f"]
    )
    solution = model.solve(
        params={
            "discount_factor": DISCOUNT_FACTOR,
            "husband_reference_weight": 0.5,
        },
        log_level="debug",
    )
    aaae(
        np.asarray(solution[0]["single_f"]),
        np.array([2.425, 4.85]),
        decimal=DECIMAL_PRECISION,
    )


def test_leg_fallback_projection_scalar_parameter_is_a_model_parameter():
    """A scalar a leg fallback's projection reads is listed and used.

    The husband refuses at the high wage, so the wife falls back on her single
    value there. A weight of 0.4 has her read it at
    `2 - 0.4 * (2 - 1) = 1.6`, worth `1.5 * 1.6 = 2.4`, giving
    `V = [2.9, 2 + 0.95 * 2.4] = [2.9, 4.28]`. At a weight of zero she would
    read it at her own wage, worth 3.0, and the value there would be 4.85.
    """
    model = _build_model(
        gate=_consent_gate,
        husband_reference_projection=_wage_itself,
        wife_fallback_projection=_wife_fallback_wage,
    )
    assert "wife_fallback_weight" in _parameter_names(
        model.get_params_template()["single_f"]
    )
    solution = model.solve(
        params={"discount_factor": DISCOUNT_FACTOR, "wife_fallback_weight": 0.4},
        log_level="debug",
    )
    aaae(
        np.asarray(solution[0]["single_f"]),
        np.array([2.9, 4.28]),
        decimal=DECIMAL_PRECISION,
    )


@categorical(ordered=False)
class _RegimeId:
    """Regime ids of the mutual-consent model."""

    single_f: ScalarInt
    married_terminal: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt


# The one continuous state every regime carries.
_WAGE = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)

# `_WAGE`'s two nodes, so a projection can name the grid's ends directly.
_WAGE_LOW = 1.0
_WAGE_HIGH = 2.0

# Three ages: the single woman decides at age 0, everyone else pays out from age 1.
_AGES = AgeGrid(start=0, stop=2, step="Y")


def _parameter_names(branch: Mapping[str, object]) -> set[str]:
    """Return every parameter name a regime's params-template branch asks for.

    The branch nests parameters under function keys, and per-target transition
    cells nest one level deeper still, so the leaves are collected at any depth:
    what a caller supplies is a parameter name, not a path.

    Args:
        branch: One regime's entry of `Model.get_params_template()`.

    Returns:
        Set of the parameter names the branch's leaves carry.

    """
    names: set[str] = set()
    for name, value in branch.items():
        if isinstance(value, Mapping):
            names |= _parameter_names(value)
        else:
            names.add(name)
    return names


def _build_model(
    *,
    gate: UserFunction,
    husband_reference_projection: UserFunction,
    wife_fallback_projection: UserFunction,
) -> Model:
    """Build the mutual-consent model around the three edge callables given.

    Args:
        gate: The edge's boolean consent predicate.
        husband_reference_projection: Wage at which the husband's single value is
            read for the gate.
        wife_fallback_projection: Wage at which the wife's single value is read
            when consent fails.

    Returns:
        The model, ready to solve.

    """
    single_f = Regime(
        transition={"married_terminal": MarkovTransition(_marry_for_sure)},
        active=lambda age: age < 1,
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_single_f},
        gated_edges={
            "married_terminal": GatedEdge(
                gate=gate,
                legs={
                    "f": StakeholderRoute(
                        target_stakeholder="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f_terminal",
                            projection={"wage": wife_fallback_projection},
                        ),
                    )
                },
                gate_refs={
                    "V_single_f_ref": ProjectedRegimeValue(
                        regime="single_f_terminal",
                        projection={"wage": _wage_itself},
                    ),
                    "V_single_m_ref": ProjectedRegimeValue(
                        regime="single_m_terminal",
                        projection={"wage": husband_reference_projection},
                    ),
                },
            )
        },
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_married_f, "utility_m": _utility_married_m},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _utility_single_f_terminal},
    )
    single_m_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _utility_single_m_terminal},
    )
    return Model(
        regimes={
            "single_f": single_f,
            "married_terminal": married_terminal,
            "single_f_terminal": single_f_terminal,
            "single_m_terminal": single_m_terminal,
        },
        ages=_AGES,
        regime_id_class=_RegimeId,
    )


def _consent_gate(
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
) -> BoolND:
    """Marriage happens only if both partners strictly prefer it."""
    return (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)


def _consent_gate_with_premium(
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
    marriage_premium: float,
) -> BoolND:
    """Both partners value marriage at its value plus a common premium."""
    return ((V_target_f + marriage_premium) > V_single_f_ref) & (
        (V_target_m + marriage_premium) > V_single_m_ref
    )


def _wage_itself(wage: ContinuousState) -> ContinuousState:
    """Read the referenced regime at the couple's own wage."""
    return wage


def _husband_reference_wage(
    wage: ContinuousState, husband_reference_weight: float
) -> ContinuousState:
    """Wage at which the husband values single life.

    A weight of zero is the couple's own wage; a weight of one is the top of the
    wage grid, where single life is worth most to him.
    """
    return wage + husband_reference_weight * (_WAGE_HIGH - wage)


def _wife_fallback_wage(
    wage: ContinuousState, wife_fallback_weight: float
) -> ContinuousState:
    """Wage at which a refused woman values single life.

    A weight of zero is the couple's own wage; a weight of one is the bottom of
    the wage grid, so a larger weight is a costlier refusal.
    """
    return wage + wife_fallback_weight * (_WAGE_LOW - wage)


def _utility_single_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """A single woman earns her wage when she works and nothing otherwise."""
    return wage * work


def _utility_single_f_terminal(wage: ContinuousState) -> FloatND:
    """Terminal single-life payoff of the woman: 1.5 per unit of wage."""
    return 1.5 * wage


def _utility_single_m_terminal(wage: ContinuousState) -> FloatND:
    """Terminal single-life payoff of the man: 0.5 at the low wage, 3.0 at the high."""
    return jnp.where(wage < 1.5, 0.5, 3.0)


def _utility_married_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """The wife's married payoff: twice the wage, whatever the couple does."""
    return 2.0 * wage + 0.0 * work


def _utility_married_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """The husband's married payoff: the wage itself, whatever the couple does."""
    return wage + 0.0 * work


def _marry_for_sure(age: FloatND) -> FloatND:
    """The single woman faces the married couple with probability one."""
    return jnp.ones_like(age, dtype=float)
