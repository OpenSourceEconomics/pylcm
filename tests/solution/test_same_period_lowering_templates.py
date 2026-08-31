"""Same-period reference values are lowered against the reference regime's own V.

A collective regime may read another regime's same-period value function (through
`same_period_refs`, feeding its `value_constraints`) and, in the same declaration,
route its continuation into that very regime through a `GatedEdge`. The two reads
are different objects: the same-period slot takes the reference regime's value
array of THIS period, while the continuation slot takes the gated object ``Wbar``
folded on the target's grid. For a collective source into a singleton target the
two even differ in rank — ``Wbar`` carries the source's trailing stakeholder axis
— so lowering the same-period slot against ``Wbar`` compiles a program the solve
loop can never call.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.engine import StateActionSpace
from _lcm.solution.grid_search import _GridSearchPeriodKernel
from _lcm.typing import RegimeName
from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentConstraint,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_DISCOUNT_FACTOR = 0.5

# Wage nodes `{1.0, 2.0}`; the gate opens above the midpoint, so it is closed at
# the low node and open at the high one.
_WAGE = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
_WAGE_FLOOR = 1.5

# `couple`'s period-0 value per (wage node, stakeholder). See `_build_model`.
_EXPECTED_COUPLE_V = np.array([[5.5, 0.75], [6.0, -4.0]])


@categorical(ordered=True)
class _Work:
    """The couple's single discrete action."""

    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class _RegimeId:
    """Regime ids of the couple / single / outside-option model."""

    couple: ScalarInt
    single: ScalarInt
    outside_f: ScalarInt
    outside_m: ScalarInt


def test_solve_reads_a_reference_regime_that_is_also_a_gated_edge_target() -> None:
    """`couple` prices its participation constraint on `single`'s own value.

    At the high wage node the household would rather take leisure, but the wife's
    action value there falls short of her single value, so the constraint removes
    leisure and the pair works. The constraint is evaluated against `single`'s
    period-0 value function — not against the gated continuation folded on
    `single`'s grid, which the couple's continuation reads instead.
    """
    model = _build_model()
    solution = model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR}, log_level="debug"
    )
    aaae(
        np.asarray(solution[0]["couple"]),
        _EXPECTED_COUPLE_V,
        decimal=DECIMAL_PRECISION,
    )


def test_build_lower_args_templates_the_same_period_slot_with_the_reference_V() -> None:
    """The same-period lowering template is the reference regime's V template.

    A kernel whose regime both references and gates into the same regime lowers
    its continuation slot against that regime's ``Wbar`` and its same-period slot
    against that regime's own V — two arrays of different rank whenever the source
    is collective.
    """
    kernel = _GridSearchPeriodKernel(
        core=lambda **kwargs: kwargs,
        regime_name="couple",
        collective=True,
        same_period_ref_regimes=("single",),
        edge_target_regimes=("single",),
    )
    reference_V = jnp.zeros(3)
    lower_args = kernel.build_lower_args(
        state_action_space=StateActionSpace(
            states=MappingProxyType({"wage": jnp.zeros(3)}),
            discrete_actions=MappingProxyType({}),
            continuous_actions=MappingProxyType({}),
            state_and_discrete_action_names=("wage",),
        ),
        next_regime_to_V_arr=MappingProxyType({"single": reference_V}),
        next_regime_to_continuation=MappingProxyType({}),
        flat_params=MappingProxyType(
            {"couple": MappingProxyType({}), "single": MappingProxyType({})}
        ),
        period=0,
        ages=AgeGrid(start=0, stop=2, step="Y"),
        edge_regime_to_V_arr=MappingProxyType({"single": jnp.zeros((3, 2))}),
    )
    same_period = cast(
        "Mapping[RegimeName, FloatND]", lower_args["same_period_regime_to_V_arr"]
    )
    assert same_period["single"] is reference_V


def _build_model() -> Model:
    r"""Build a couple that both references and gates into the same single regime.

    Topology over ages 0-2: `couple` is active at age 0 and moves into `single`
    with probability one. `single`, `outside_f` and `outside_m` are terminal and
    active at every age. The couple declares

    - `same_period_refs` on `single`, feeding the wife's participation
      constraint with `single`'s value at the couple's OWN period, and
    - a `gated_edges` entry on `single`, folding the couple's continuation

    ```{math}
    \bar{W}^s(w) = \mathbb{1}[w > 1.5] \, V_{\text{single}}(w)
                   + \mathbb{1}[w \le 1.5] \, V_{\text{outside}_s}(w).
    ```

    Hand computation, $\beta = 0.5$, equal Pareto weights:

    - Terminal values on the wage grid `[1, 2]`: `single = [2, 4]`,
      `outside_f = [11, 12]`, `outside_m = [1.5, 2.5]`.
    - The gate is closed at wage `1` and open at wage `2`, so each fold takes the
      outside option at the low node and `single` at the high one:
      $\bar{W}^f = [11, 4]$ and $\bar{W}^m = [1.5, 4]$.
    - Action values $Q^s = u_s + 0.5 \bar{W}^s$ over (wage node, leisure/work):
      the wife has `[[5.5, 9.5], [2.0, 6.0]]` and the husband
      `[[0.75, -5.25], [2.0, -4.0]]`.
    - The household objective $0.5 Q^f + 0.5 Q^m$ prefers leisure at both wage
      nodes (`3.125` over `2.125`, and `2.0` over `1.0`), but at wage `2` leisure
      gives the wife `2.0` against a single value of `4`, so her participation
      constraint removes it and the pair works.
    - Hence `V = [[5.5, 0.75], [6.0, -4.0]]`.

    Returns:
        The model, which `{"discount_factor": 0.5}` solves.

    """
    couple = Regime(
        transition={
            "single": ValueDependentTransition(
                probability=MarkovTransition(_probability_of_separating),
                gate=_wage_clears_the_floor,
                routes={
                    "f": StakeholderRoute(
                        fallback=ProjectedRegimeValue(
                            regime="outside_f", projection={"wage": _identity_wage}
                        )
                    ),
                    "m": StakeholderRoute(
                        fallback=ProjectedRegimeValue(
                            regime="outside_m", projection={"wage": _identity_wage}
                        )
                    ),
                },
            )
        },
        active=lambda age: age < 1,
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(_Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _couple_utility_f, "m": _couple_utility_m}
            )
        },
        constraints={
            "participation_f": ValueDependentConstraint(
                predicate=_wife_participates,
                references={
                    "V_single_ref": ProjectedRegimeValue(
                        regime="single", projection={"wage": _identity_wage}
                    )
                },
            )
        },
    )
    single = Regime(
        transition=None,
        states={"wage": _WAGE},
        functions={"utility": _single_utility},
    )
    outside_f = Regime(
        transition=None,
        states={"wage": _WAGE},
        functions={"utility": _outside_f_utility},
    )
    outside_m = Regime(
        transition=None,
        states={"wage": _WAGE},
        functions={"utility": _outside_m_utility},
    )
    return Model(
        regimes={
            "couple": couple,
            "single": single,
            "outside_f": outside_f,
            "outside_m": outside_m,
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
    )


def _probability_of_separating(age: FloatND) -> FloatND:
    """The couple moves into the single regime with probability one."""
    return jnp.ones_like(age, dtype=float)


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    """Every projection carries the wage node across unchanged."""
    return wage


def _couple_utility_f(work: DiscreteAction) -> FloatND:
    """The wife gains from working."""
    return 4.0 * work


def _couple_utility_m(work: DiscreteAction) -> FloatND:
    """The husband loses from working, and by more than the wife gains."""
    return -6.0 * work


def _single_utility(wage: ContinuousState) -> FloatND:
    """The single regime pays twice the wage."""
    return 2.0 * wage


def _outside_f_utility(wage: ContinuousState) -> FloatND:
    """The wife's outside option is far above her single value."""
    return 10.0 + wage


def _outside_m_utility(wage: ContinuousState) -> FloatND:
    """The husband's outside option is below his single value."""
    return 0.5 + wage


def _wage_clears_the_floor(wage: ContinuousState) -> BoolND:
    """The couple stays together only at wage nodes above the floor."""
    return wage > _WAGE_FLOOR


def _wife_participates(Q_f: FloatND, V_single_ref: FloatND) -> BoolND:
    """The wife accepts an action only if it beats her same-period single value."""
    return Q_f >= V_single_ref
