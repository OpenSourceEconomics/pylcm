"""F3: a closed gate whose fallback lands back in its own target regime.

`simulation_gate_route_delta` infers `closed` from `routed_ids != target_id`
instead of carrying the actual per-row closed-gate mask the routing pass
already computed. When a leg's `realized_fallback.regime` equals the
transition's own target regime, a closed row keeps the same regime ID on
both branches, so the ID-inequality proxy reads it as open and the delta
never publishes the fallback's projected state — the row keeps its
unprojected candidate coordinate instead of the projection it is priced at.
"""

import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
    categorical,
)
from lcm.transition import MarkovTransition
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_X = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)
# Utility strictly prefers 1.0, so every solved row lands at the same
# candidate x=1 before any gate/fallback logic runs; 2.0 is dominated.
_SAVING = IrregSpacedGrid(points=(1.0, 2.0))


@categorical(ordered=False)
class SameTargetFallbackRegimeId:
    source: ScalarInt
    target: ScalarInt


def _always_true(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _always_closed(x: ContinuousState) -> FloatND:
    return jnp.zeros_like(x, dtype=bool)


def _costly_saving(*, x: ContinuousState, saving: ContinuousAction) -> FloatND:
    """Strictly prefers the smaller `saving`, independent of any continuation
    value — so the optimal action (and hence the candidate landing) is fixed
    at `saving=1.0` regardless of what the target-side value function is,
    keeping this test isolated from the (unrelated, solve-side) value path.
    """
    return -saving + 0.0 * x


def _next_x(saving: ContinuousAction) -> ContinuousState:
    return saving


def _constant_utility(x: ContinuousState) -> FloatND:
    return 0.0 * x


def _half_x(x: ContinuousState) -> ContinuousState:
    return 0.5 * x


def _same_target_fallback_model() -> Model:
    """A gate that is always closed, whose one leg falls back into its own
    target regime with a nonidentity (halving) projection.

    The candidate landing (via the ordinary transition) is x=1; a closed gate
    must publish the projected x=0.5 instead.
    """
    return Model(
        regimes={
            "source": Regime(
                transition={
                    "target": ValueDependentTransition(
                        probability=MarkovTransition(_always_true),
                        gate=_always_closed,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="target", projection={"x": _half_x}
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
                functions={"utility": _costly_saving},
            ),
            "target": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"x": _X},
                functions={"utility": _constant_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=SameTargetFallbackRegimeId,
    )


_PARAMS = {
    "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "target": {},
}


def test_a_closed_gate_publishes_the_projected_state_into_its_own_target_regime():
    """A closed gate whose fallback IS its own target must still project.

    Candidate x is 1 (the ordinary transition's landing); the closed gate's
    one leg falls back into "target" with projection x -> x/2, so the row
    must arrive at x=0.5, not the unprojected candidate x=1.
    """
    model = _same_target_fallback_model()
    solution = model.solve(params=_PARAMS, log_level="off")

    result = model.simulate(
        params=_PARAMS,
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

    aaae(landed["x"].to_numpy(), [0.5], decimal=DECIMAL_PRECISION)
