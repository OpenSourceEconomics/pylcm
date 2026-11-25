"""Two legs of one gate may share a fallback regime when they write the same row.

Simulation writes every leg's projected fallback state into its fallback regime's
per-subject slot. Two legs naming one regime write the same value only when their
fallbacks are the same declaration, function for function, and read nothing that is
the leg's own, i.e. no parameter, which lives under the leg's route. Then sharing is
allowed; otherwise the gate is refused.

The shape is widowhood as a closed gate: whichever spouse a row followed, it
continues in the survivor's household. The husband's leg prices the household's
end at another regime in the solve and lands in the survivor's in simulation.
"""

from collections.abc import Mapping
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    LinSpacedGrid,
    MemberRoute,
    Model,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION
from tests.regime_building.test_gated_edges_collective_solve import (
    EKLRegimeId,
    _make_full_topology_regimes,
    _with_full_topology_laws,
)

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")

type RouteForm = Literal[
    "survivor", "both_phases", "free_parameter", "different_functions"
]


@categorical(ordered=True)
class Work:
    rest: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class RegimeId:
    couple: ScalarInt
    single_f_ret: ScalarInt
    dead_single: ScalarInt


WEALTH = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)
DISCOUNT_FACTOR = 0.9


def utility_f(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return wealth + 0.1 * work


def utility_m(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.5 * wealth - 0.05 * work


def utility_survivor(wealth: ContinuousState) -> FloatND:
    return 2.0 * wealth


def utility_dead(wealth: ContinuousState) -> FloatND:
    return 0.0 * wealth


def certain(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def never_open(wealth: ContinuousState) -> BoolND:
    return jnp.zeros_like(wealth, dtype=bool)


def keep_wealth(wealth: ContinuousState) -> ContinuousState:
    return wealth


def keep_wealth_too(wealth: ContinuousState) -> ContinuousState:
    return wealth


def shifted_wealth(*, wealth: ContinuousState, shift: FloatND) -> ContinuousState:
    return wealth + shift


SURVIVOR_PROJECTION = {"wealth": keep_wealth}


def _survivor(projection: Mapping[str, UserFunction]) -> ProjectedRegimeValue:
    return ProjectedRegimeValue(regime="single_f_ret", projection=projection)


def _routes(route_form: RouteForm) -> dict[str, MemberRoute]:
    if route_form == "survivor":
        return {
            "f": MemberRoute(fallback=_survivor(SURVIVOR_PROJECTION)),
            "m": MemberRoute(
                fallback=Phased(
                    solve=ProjectedRegimeValue(
                        regime="dead_single", projection={"wealth": keep_wealth}
                    ),
                    simulate=_survivor(SURVIVOR_PROJECTION),
                )
            ),
        }
    if route_form == "both_phases":
        return {
            "f": MemberRoute(fallback=_survivor(SURVIVOR_PROJECTION)),
            "m": MemberRoute(fallback=_survivor(SURVIVOR_PROJECTION)),
        }
    if route_form == "free_parameter":
        shifted = {"wealth": shifted_wealth}
        return {
            "f": MemberRoute(fallback=_survivor(shifted)),
            "m": MemberRoute(fallback=_survivor(shifted)),
        }
    return {
        "f": MemberRoute(fallback=_survivor(SURVIVOR_PROJECTION)),
        "m": MemberRoute(fallback=_survivor({"wealth": keep_wealth_too})),
    }


def build_model(*, route_form: RouteForm) -> Model:
    """A couple at age 0 whose household ends at age 1 through a closed gate."""
    return Model(
        regimes={
            "couple": Regime(
                states={"wealth": WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                actions={"work": DiscreteGrid(Work)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": utility_f, "m": utility_m}
                    )
                },
            ),
            "single_f_ret": Regime(
                states={"wealth": WEALTH}, functions={"utility": utility_survivor}
            ),
            "dead_single": Regime(
                states={"wealth": WEALTH}, functions={"utility": utility_dead}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "couple"},
        edges={
            "couple": Transition(
                law={"single_f_ret": StochasticTransition(func=certain)},
                gates={
                    "single_f_ret": Gate(
                        predicate=never_open, routes=_routes(route_form)
                    )
                },
            )
        },
    )


PARAMS = {
    "couple": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
    "single_f_ret": {},
    "dead_single": {},
}


def test_survivor_legs_price_each_spouse_at_their_own_solve_fallback() -> None:
    """Both work: her value continues as the survivor, his ends at zero.

    V_f = wealth + 0.1 + 0.9 * 2 * wealth and V_m = 0.5 * wealth - 0.05.
    """
    solution = build_model(route_form="survivor").solve(params=PARAMS, log_level="off")
    aaae(
        solution.values[0]["couple"],
        np.array([[2.9, 0.45], [5.7, 0.95], [8.5, 1.45]]),
        decimal=DECIMAL_PRECISION,
    )


def test_rows_following_either_spouse_land_in_the_shared_regime() -> None:
    """At period 1 every row is in the survivor's regime at its own wealth."""
    model = build_model(route_form="survivor")
    roles = model.member_names_to_ids
    result = model.simulate(
        params=PARAMS,
        initial_conditions={
            "regime_id": jnp.zeros(4, dtype=jnp.int32),
            "age": jnp.zeros(4),
            "wealth": jnp.asarray([1.0, 2.0, 3.0, 2.0]),
            "own_member": jnp.asarray(
                [roles["f"], roles["f"], roles["m"], roles["m"]], dtype=jnp.int32
            ),
        },
        log_level="off",
        seed=0,
    )
    panel = result.to_dataframe()
    landed = panel.loc[panel["period"] == 1, ["regime_name", "wealth"]]
    assert [
        (str(name), float(wealth)) for name, wealth in landed.to_numpy().tolist()
    ] == [
        ("single_f_ret", 1.0),
        ("single_f_ret", 2.0),
        ("single_f_ret", 3.0),
        ("single_f_ret", 2.0),
    ]


def test_legs_sharing_one_fallback_in_both_phases_price_both_at_it() -> None:
    """Both spouses continue as the survivor: V_m = 0.5 wealth - 0.05 + 1.8 wealth."""
    solution = build_model(route_form="both_phases").solve(
        params=PARAMS, log_level="off"
    )
    aaae(
        solution.values[0]["couple"],
        np.array([[2.9, 2.25], [5.7, 4.55], [8.5, 6.85]]),
        decimal=DECIMAL_PRECISION,
    )


def test_legs_sharing_a_regime_with_different_projections_are_refused() -> None:
    """Two functions with the same body are two declarations; either may write last."""
    with pytest.raises(
        ModelInitializationError,
        match=r"two or more legs share the same fallback regime \['single_f_ret'\]"
        r".*Legs may share one only where",
    ):
        build_model(route_form="different_functions")


def test_legs_sharing_a_projection_that_reads_a_parameter_are_refused() -> None:
    """A parameter lives under each leg's own route, so the legs may write apart."""
    with pytest.raises(
        ModelInitializationError,
        match=r"share the same fallback regime \['single_f_ret'\].*'wealth'.*"
        r"reads the parameter 'shift'",
    ):
        build_model(route_form="free_parameter")


def test_an_accepted_gate_model_keeps_its_values() -> None:
    """The gated consent topology still prices single_f at its consent continuation.

    Each single declares one leg, so the relaxed rule has nothing to admit there:
    V_single_f(0) = wage + 0.95 * Wbar_f with Wbar_f = [1.5, 5.5, 6], the values
    `test_full_ekl_topology_via_public_model_api` derives for the same regimes.
    """
    model = Model(
        regimes=_make_full_topology_regimes(),
        edges=_with_full_topology_laws(
            {
                "single_f": {"married": 0, "single_f_p1": 0},
                "single_m": {"married": 0, "single_m_p1": 0},
                "single_f_p1": {"single_f_terminal": 1},
                "single_m_p1": {"single_m_terminal": (0, 1, 2)},
                "married": {"married_terminal": 1},
            }
        ),
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=EKLRegimeId,
        initial_nodes={0: ("single_f", "single_m")},
    )
    solution = model.solve(
        params={"discount_factor": 0.95},
        log_level="off",
    )
    aaae(
        solution.values[0]["single_f"][..., 0],
        np.array([2.425, 7.225, 8.7]),
        decimal=DECIMAL_PRECISION,
    )
