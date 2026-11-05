"""A gated edge's projected operands are read where the source lands.

A gate reference and a leg fallback both name ANOTHER regime's value at
coordinates a projection produces. Tabulating that composition on the target's
grid and interpolating the surface afterwards computes
`interpolate(V_ref o projection)`, while the branch pays
`V_ref(projection(landing))`. The two agree only where the projection is
affine, so a curved projection would have solve price a branch at a number no
branch pays — and forward simulation, which evaluates the projection at the
realized point, would route on the other number.

Every number below is an exact small rational, so the assertions are on the
value itself rather than on a tolerance around a fitted one.
"""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    ByAge,
    Gate,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION

# Nodes 0, 1, 2 on every regime's grid.
_X = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)

# The one action lands at x = 0.5, strictly inside the cell [0, 1].
_SAVING = IrregSpacedGrid(points=np.array([0.5, 2.0]))


@categorical(ordered=False)
class ProjectionRegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _certain_target(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _utility_source(*, x: ContinuousState, saving: ContinuousAction) -> FloatND:
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
        edges={
            "source": Transition(
                targets={"target": 0, "fallback": 0},
                law=ByAge(
                    cases={0: {"target": StochasticTransition(func=_certain_target)}}
                ),
                gates={
                    "target": Gate(
                        predicate=_closed_above_one,
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
            )
        },
        regimes={
            "source": Regime(
                states={"x": _X},
                state_transitions={"x": _next_x},
                actions={"saving": _SAVING},
                functions={"utility": _utility_source},
            ),
            "target": Regime(
                states={"x": _X},
                functions={"utility": _zero_utility},
            ),
            "fallback": Regime(
                states={"x": _X},
                functions={"utility": _fallback_value},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=ProjectionRegimeId,
        initial_nodes={0: "source"},
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

    aaae(
        np.asarray(solution.values[0]["source"]), [-0.75] * 3, decimal=DECIMAL_PRECISION
    )


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
        solution=solution,
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

    aaae(
        np.asarray(solution.values[0]["source"]), [-0.75] * 3, decimal=DECIMAL_PRECISION
    )
