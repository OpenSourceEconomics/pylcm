"""A gated edge's parameters are discovered against its own target regime.

An edge's gate and projections run on the target regime's grid, so the target's
state names are bound by the engine rather than supplied by the user. Every
other regime's state names are not: a gate argument that happens to share a name
with some unrelated regime's state is an ordinary parameter of this edge, and
resolving it against a model-wide union of state names removes it from the
template — leaving a model whose gate reads a value nothing can supply.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt

AGES = AgeGrid(start=40, stop=50, step="5Y")
X = LinSpacedGrid(start=0.0, stop=2.0, n_points=2)
BONUS_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


@categorical(ordered=False)
class RegimeIdWithBystander:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt
    bystander: ScalarInt


def _certain_target(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _utility_source(*, x: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.zeros_like(x) * work


def _utility_target(x: ContinuousState) -> FloatND:
    return 10.0 + jnp.zeros_like(x)


def _utility_fallback(x: ContinuousState) -> FloatND:
    return jnp.zeros_like(x)


def _utility_bystander(marriage_bonus: ContinuousState) -> FloatND:
    """The bystander regime's own payoff, so its state is genuinely used."""
    return marriage_bonus


def _identity_x(x: ContinuousState) -> ContinuousState:
    return x


def _gate(*, V_target: FloatND, marriage_bonus: float) -> BoolND:
    """Opens where the target beats a bonus the SOURCE supplies as a parameter."""
    return V_target > marriage_bonus


def _build_model(*, with_bystander: bool) -> Model:
    regimes = {
        "source": Regime(
            transition={
                "target": ValueDependentTransition(
                    probability=MarkovTransition(_certain_target),
                    gate=_gate,
                    routes={
                        "only": StakeholderRoute(
                            fallback=ProjectedRegimeValue(
                                regime="fallback", projection={"x": _identity_x}
                            )
                        )
                    },
                )
            },
            active=lambda age: age < 45,
            states={"x": X},
            state_transitions={"x": fixed_transition("x")},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _utility_source},
        ),
        "target": Regime(
            transition=None,
            active=lambda age: age >= 45,
            states={"x": X},
            functions={"utility": _utility_target},
        ),
        "fallback": Regime(
            transition=None,
            active=lambda age: age >= 45,
            states={"x": X},
            functions={"utility": _utility_fallback},
        ),
    }
    if with_bystander:
        regimes["bystander"] = Regime(
            transition=None,
            active=lambda age: age >= 45,
            states={"marriage_bonus": BONUS_GRID},
            functions={"utility": _utility_bystander},
        )
    return Model(
        regimes=regimes,
        ages=AGES,
        regime_id_class=RegimeIdWithBystander if with_bystander else RegimeId,
    )


@pytest.mark.parametrize("with_bystander", [False, True])
def test_a_gate_parameter_survives_an_unrelated_regimes_state_of_the_same_name(
    *,
    with_bystander: bool,
) -> None:
    """`marriage_bonus` is a parameter of this edge either way.

    The `False` case is the control: it fixes what the template holds when no
    other regime uses the name at all, so the `True` case is a comparison rather
    than a bare assertion about one model.
    """
    template = _build_model(with_bystander=with_bystander).get_params_template()

    assert template["source"]["target"]["gate"] == {"marriage_bonus": "float"}


def test_the_bystanders_state_is_not_a_parameter_of_its_own_regime() -> None:
    """The bystander still owns `marriage_bonus` as a state, not a parameter.

    Without this, the test above would also pass for a change that simply
    stopped treating declared states as engine-bound anywhere.
    """
    template = _build_model(with_bystander=True).get_params_template()

    assert template["bystander"]["utility"] == {}


def test_the_edge_gate_parameter_is_solvable() -> None:
    """A model with the bystander still solves once the gate param is supplied.

    A template entry nothing consumes would satisfy the assertions above while
    leaving the gate reading an unbound name, so the parameter is exercised.
    """
    model = _build_model(with_bystander=True)
    params = {
        "source": {
            "koopmans_aggregator": {"discount_factor": 0.5},
            "target": {"gate": {"marriage_bonus": 1.0}},
        },
        "target": {},
        "fallback": {},
        "bystander": {},
    }

    solution = model.solve(params=params, log_level="debug").values

    assert set(solution[1]) == {"target", "fallback", "bystander"}
