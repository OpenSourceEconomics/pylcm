"""A gated edge's projected operands are read where the source lands.

A gate reference and a leg fallback both name ANOTHER regime's value at
coordinates a projection produces. Tabulating that composition on the target's
grid and interpolating the surface afterwards computes
`interpolate(V_ref o projection)`, while the branch pays
`V_ref(projection(landing))`. The two agree only where the projection is
affine, so a curved projection would have solve price a branch at a number no
branch pays — and forward simulation, which evaluates the projection at the
realized point, would route on the other number.

The target's own value and its dissolution flag are genuine arrays on the
target's grid and are interpolated there, but they have to answer together: the
value is `-inf` exactly where the flag is set, so a landing carrying any weight
on a dissolved node is worth `-inf` however small that weight is.

Every number below is an exact small rational, so the assertions are on the
value itself rather than on a tolerance around a fitted one.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentConstraint,
    ValueDependentTransition,
    categorical,
)
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION

# Nodes 0, 1, 2 on every regime's grid.
_X = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)

# The one action lands at x = 0.5, strictly inside the cell [0, 1].
_SAVING = IrregSpacedGrid(points=(0.5, 2.0))


@categorical(ordered=False)
class ProjectionRegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _certain_target(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _utility_source(x: ContinuousState, saving: ContinuousAction) -> FloatND:
    return -2.0 * saving + 0.0 * x


def _next_x(saving: ContinuousAction) -> ContinuousState:
    return saving


def _zero_utility(x: ContinuousState) -> FloatND:
    return 0.0 * x


def _fallback_value(x: ContinuousState) -> FloatND:
    return x


def _square_x(x: ContinuousState) -> ContinuousState:
    return x**2


def _half_x(x: ContinuousState) -> ContinuousState:
    return 0.5 * x


def _closed_above_one(x: ContinuousState) -> BoolND:
    return x > 1.0


def _projection_model(projection) -> Model:
    """A source whose gate is closed at its landing, so the fallback pays.

    The fallback regime is worth `V(x) = x`, so with `projection = x**2` the
    pullback surface over the nodes `{0, 1, 2}` is `[0, 1, 4]` and interpolating
    it at `0.5` gives `0.5`, while the branch pays `V(0.5**2) = 0.25`.
    """
    return Model(
        regimes={
            "source": Regime(
                transition={
                    "target": ValueDependentTransition(
                        probability=MarkovTransition(_certain_target),
                        gate=_closed_above_one,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={"x": projection}
                                )
                            )
                        },
                        off_grid="pointwise",
                    )
                },
                active=lambda age: age < 1,
                states={"x": _X},
                state_transitions={"x": _next_x},
                actions={"saving": _SAVING},
                functions={"utility": _utility_source},
            ),
            "target": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"x": _X},
                functions={"utility": _zero_utility},
            ),
            "fallback": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"x": _X},
                functions={"utility": _fallback_value},
            ),
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=ProjectionRegimeId,
    )


_PROJECTION_PARAMS = {
    "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "target": {},
    "fallback": {},
}


def test_a_curved_fallback_projection_is_priced_where_the_source_lands():
    """The closed branch is worth `V(pi(landing))`, not the pullback's blend.

    Source utility is `-2 * saving` and the action lands at `0.5`, so the
    branch pays `-1 + 0.25`. Reading a surface tabulated over the target's
    nodes would pay `-1 + 0.5` instead.
    """
    model = _projection_model(_square_x)

    solution = model.solve(params=_PROJECTION_PARAMS, log_level="off")

    aaae(np.asarray(solution[0]["source"]), [-0.75] * 3, decimal=DECIMAL_PRECISION)


def test_the_simulated_row_lands_where_the_solved_value_priced_it():
    """Simulation realizes `pi(0.5) = 0.25`, the coordinate solve was priced at.

    Solve and simulation evaluate one operator, so the state the row arrives in
    is the state whose value the source maximized against.
    """
    model = _projection_model(_square_x)
    solution = model.solve(params=_PROJECTION_PARAMS, log_level="off")

    result = model.simulate(
        params=_PROJECTION_PARAMS,
        initial_conditions={
            "x": jnp.zeros(1),
            "age": jnp.zeros(1),
            "regime_id": jnp.full(
                1, model.regime_names_to_ids["source"], dtype=jnp.int32
            ),
        },
        period_to_regime_to_V_arr=solution,
        log_level="off",
        seed=0,
    )
    landed = result.to_dataframe().query("period == 1")

    aaae(landed["x"].to_numpy(), [0.25], decimal=DECIMAL_PRECISION)


def test_an_affine_projection_is_left_exactly_where_it_was():
    """A straight projection commutes with interpolation, so nothing moves.

    With `pi(x) = x / 2` the pullback surface `[0, 0.5, 1]` interpolates at
    `0.5` to `0.25`, which is also `V(pi(0.5))`. The source is worth
    `-1 + 0.25`, and reading the operand at the landing has to reproduce
    exactly the number the tabulated route already gave.
    """
    model = _projection_model(_half_x)

    solution = model.solve(params=_PROJECTION_PARAMS, log_level="off")

    aaae(np.asarray(solution[0]["source"]), [-0.75] * 3, decimal=DECIMAL_PRECISION)


@categorical(ordered=False)
class Work:
    off: ScalarInt
    on: ScalarInt


@categorical(ordered=False)
class CoupledRegimeId:
    source: ScalarInt
    pair: ScalarInt
    alone_f: ScalarInt
    alone_m: ScalarInt


_WAGE = IrregSpacedGrid(points=(1.0, 2.0, 3.0))

# The fallback pays this everywhere, so any finite source value identifies the
# closed branch and any `-inf` identifies the open one.
_FALLBACK_PAYOFF = -100.0


def _to_pair(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _u_source(wage: ContinuousState, saving: ContinuousAction) -> FloatND:
    return 0.0 * wage + 0.0 * saving


def _next_wage(saving: ContinuousAction) -> ContinuousState:
    return saving


def _pair_payoff(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage + 0.0 * work


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _outside_option_f(wage: ContinuousState) -> FloatND:
    """Beats anything the pair can pay at wage 2 alone, emptying that cell."""
    return jnp.where(jnp.isclose(wage, 2.0), 100.0, _FALLBACK_PAYOFF) + 0.0 * wage


def _outside_option_m(wage: ContinuousState) -> FloatND:
    return _FALLBACK_PAYOFF + 0.0 * wage


def _participation_f(Q_f: FloatND, V_alone_f: FloatND) -> BoolND:
    return Q_f >= V_alone_f


def _no_dissolution(D_target: BoolND) -> BoolND:
    return ~D_target


def _coupled_model(saving_points) -> Model:
    """A collective target whose feasible set is empty at the middle node."""
    return Model(
        regimes={
            "source": Regime(
                transition={
                    "pair": ValueDependentTransition(
                        probability=MarkovTransition(_to_pair),
                        gate=_no_dissolution,
                        routes={
                            "only": StakeholderRoute(
                                target_stakeholder="f",
                                fallback=ProjectedRegimeValue(
                                    regime="alone_m",
                                    projection={"wage": _identity_wage},
                                ),
                            )
                        },
                        off_grid="pointwise",
                    )
                },
                active=lambda age: age < 1,
                states={"wage": _WAGE},
                state_transitions={"wage": _next_wage},
                actions={"saving": IrregSpacedGrid(points=saving_points)},
                functions={"utility": _u_source},
            ),
            "pair": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"wage": _WAGE},
                actions={"work": DiscreteGrid(Work)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _pair_payoff, "m": _pair_payoff}
                    )
                },
                constraints={
                    "participation_f": ValueDependentConstraint(
                        predicate=_participation_f,
                        references={
                            "V_alone_f": ProjectedRegimeValue(
                                regime="alone_f", projection={"wage": _identity_wage}
                            )
                        },
                    )
                },
            ),
            "alone_f": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"wage": _WAGE},
                functions={"utility": _outside_option_f},
            ),
            "alone_m": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"wage": _WAGE},
                functions={"utility": _outside_option_m},
            ),
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=CoupledRegimeId,
    )


_COUPLED_PARAMS = {
    "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "pair": {},
    "alone_f": {},
    "alone_m": {},
}


@pytest.mark.parametrize(
    ("saving_points", "expected"),
    [
        # Landings just above the feasible node 1 keep positive weight on the
        # empty node 2, so the edge is closed and the fallback pays.
        ((1.1, 1.25), _FALLBACK_PAYOFF),
        # Node 3 carries no weight on the empty node, so the edge opens there
        # and the pair pays 3 — beating the 2.5 landing beside it, which
        # straddles the empty node and so falls back to -100.
        ((2.5, 3.0), 3.0),
    ],
)
def test_a_landing_touching_an_empty_cell_takes_the_branch_that_pays(
    saving_points, expected
):
    """A branch is opened only where the value behind it is one a branch pays.

    The target's feasible set is empty at wage 2, where its value is `-inf` and
    its dissolution flag is set. A landing that carries any weight on that node
    is worth `-inf`, so opening the edge there would price a source action at a
    value no branch delivers — while the fallback beside it is finite.
    """
    model = _coupled_model(saving_points)

    solution, flags = model.solve(
        params=_COUPLED_PARAMS, log_level="off", return_dissolution_flags=True
    )

    np.testing.assert_array_equal(np.asarray(flags[1]["pair"]), [False, True, False])
    aaae(np.asarray(solution[0]["source"]), [expected] * 3, decimal=DECIMAL_PRECISION)
