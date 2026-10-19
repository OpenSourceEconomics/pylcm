"""Boundary augmentation retains the saving domain and represented candidate tuples."""

import logging
import math
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

import lcm
from _lcm.egm.carry import EGMCarry
from _lcm.solution.egm import _EGMPeriodKernel
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    ExecutionConfig,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
    post_decision_lower_bound,
)
from lcm.regime import ConstraintEntry
from lcm.solver_api import EGM_CONTINUATION
from lcm.solvers import EGM, NBEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    FunctionName,
    ScalarInt,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION
from tests.solution._nbegm_direct_oracle import ride_along_kernel

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@categorical(ordered=False)
class RegimeId:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def resources(liquid: ContinuousState) -> FloatND:
    return liquid


def savings(*, liquid: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return liquid - consumption


def parent_law(savings: FloatND) -> ContinuousState:
    return savings + 4.0


def child_law(savings: FloatND) -> ContinuousState:
    return 3.0 * savings + 4.0


def terminal(liquid: ContinuousState) -> FloatND:
    return jnp.log(liquid)


def build_model(
    *,
    floor: float = 0.0,
    callable_floor: bool = True,
    boundary_upper: float | None = 10.0,
    interior: bool = True,
    template_origin: float = 0.0,
    enable_jit: bool = True,
) -> Model:
    """Represent two corner or interior parent queries without changing child
    economics.
    """
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )

    def lower(period: ScalarInt) -> FloatND:
        return jnp.asarray(floor) + 0.0 * period

    parent_nodes = (
        (template_origin, template_origin + 0.5, template_origin + 1.0)
        if callable_floor
        else (floor, floor + 0.5, floor + 1.0)
    )
    child_constraints: dict[FunctionName, ConstraintEntry] = {
        "borrowing": post_decision_lower_bound(margin=margin, lower=0.0),
    }
    if boundary_upper is not None:
        child_constraints["liquid_ceiling"] = cast(
            "UserFunction", lcm.ref("liquid") <= boundary_upper
        )
    return Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={
                    "liquid": IrregSpacedGrid(
                        points=(425.0, 425.25)
                        if interior
                        else (floor + 0.25, floor + 0.5)
                    )
                },
                actions={
                    "consumption": IrregSpacedGrid(
                        points=(424.25, 424.5, 424.75, 425.0)
                        if interior
                        else (0.125, 0.25, 0.5, 1.0)
                    )
                },
                functions={
                    "utility": utility,
                    "resources": resources,
                    "savings": savings,
                },
                state_transitions={"liquid": parent_law},
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=margin, lower=lower if callable_floor else floor
                    )
                },
                solver=EGM(savings_grid=IrregSpacedGrid(points=parent_nodes)),
            ),
            "child": ConsumptionSavingsRegime(
                liquid=margin,
                states={
                    "liquid": IrregSpacedGrid(
                        points=(3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
                    )
                },
                actions={"consumption": IrregSpacedGrid(points=(0.25, 1.0, 3.0, 6.0))},
                functions={
                    "utility": utility,
                    "resources": resources,
                    "savings": savings,
                },
                state_transitions={"liquid": child_law},
                constraints=child_constraints,
                solver=NBEGM(savings_grid=IrregSpacedGrid(points=(0.0, 0.5, 1.0))),
            ),
            "done": Regime(
                states={
                    "liquid": IrregSpacedGrid(points=(4.0, 4.75, 5.5, 6.25, 7.0, 21.25))
                },
                functions={"utility": terminal},
            ),
        },
        ages=AgeGrid(exact_values=(10, 20, 30)),
        edges={"parent": {"child": (10,)}, "child": {"done": (20,)}},
        initial_nodes={10: "parent"},
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def expected_interior_savings(
    *, active_boundary: bool, precision_epsilon: float
) -> NDArray[np.float64]:
    """Interpolate the two Euler endpoints under the child's declared linear
    marginal.
    """
    cash = np.asarray((425.0, 425.25))
    if not active_boundary:
        return (cash - 400.0) / 101.0
    # At the represented left neighbor of the root, linear marginal interpolation
    # differs from evaluating the analytical marginal at the physical landing.
    boundary_left = 0.5 - 18.0 * precision_epsilon
    assert 0.0 < boundary_left < 0.5
    child_marginal = 0.25 - boundary_left / 18.0
    boundary_consumption = 100.0 / child_marginal
    boundary_cash = boundary_left + boundary_consumption
    assert np.all((cash > 400.0) & (cash < boundary_cash))
    return boundary_left * (cash - 400.0) / (boundary_cash - 400.0)


def expected_interior_values(
    *, active_boundary: bool, precision_epsilon: float
) -> NDArray[np.float64]:
    """Value the represented policy through the declared linear child value read."""
    cash = np.asarray((425.0, 425.25))
    saving = expected_interior_savings(
        active_boundary=active_boundary, precision_epsilon=precision_epsilon
    )
    consumption = cash - saving
    # The child consumes all. Its declared linear value read between 4 and 4.5
    # is distinct from evaluating the exact log value at the reconstructed saving.
    child_value = (
        (1.0 - 2.0 * saving) * math.log(4.0)
        + 2.0 * saving * math.log(4.5)
        + 0.01 * math.log(4.0)
    )
    return np.log(consumption) + 0.01 * child_value


@pytest.mark.parametrize("callable_floor", [False, True])
@pytest.mark.parametrize(
    ("boundary_upper", "active_boundary"), [(10.0, False), (4.5, True)]
)
def test_boundary_augmented_interior_uses_actual_law_value(
    *, callable_floor: bool, boundary_upper: float, active_boundary: bool
) -> None:
    """A represented interior values its interpolated saving at the actual child
    landing.
    """
    model = build_model(callable_floor=callable_floor, boundary_upper=boundary_upper)
    solution = model.solve(params={"discount_factor": 0.01}, log_level="off")
    actual = np.asarray(solution.values[0]["parent"])
    np.testing.assert_array_almost_equal(
        actual,
        expected_interior_values(
            active_boundary=active_boundary,
            precision_epsilon=float(np.finfo(actual.dtype).eps),
        ),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize(
    ("boundary_upper", "active_boundary"), [(10.0, False), (4.5, True)]
)
def test_boundary_augmented_carry_keeps_original_saving_association(
    *, boundary_upper: float, active_boundary: bool
) -> None:
    """The actual producer keeps each saving selected from its represented Euler row."""
    model = build_model(boundary_upper=boundary_upper, enable_jit=False)
    kernel, context = ride_along_kernel(
        model=model,
        params={"discount_factor": 0.01},
        regime_name="parent",
        period=0,
    )
    assert isinstance(kernel, _EGMPeriodKernel)
    output = kernel(
        compiled_cores={"main": kernel.core_programs()["main"].function},
        **context,
        logger=logging.getLogger(__name__),
    )
    carry = output.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    actual = np.asarray(carry.leaves()[("query_data", "savings")])
    np.testing.assert_array_almost_equal(
        actual,
        expected_interior_savings(
            active_boundary=active_boundary,
            precision_epsilon=float(np.finfo(actual.dtype).eps),
        ),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("floor", [-0.5, 0.5])
@pytest.mark.parametrize("template_origin", [0.0, 8.0])
def test_inactive_boundary_keeps_translated_domain_corner(
    *, floor: float, template_origin: float
) -> None:
    """The evaluated floor owns the corner independently of the template origin."""
    model = build_model(floor=floor, template_origin=template_origin, interior=False)
    solution = model.solve(params={"discount_factor": 0.01}, log_level="off")
    continuation = math.log(floor + 4.0) + 0.01 * math.log(4.0)
    np.testing.assert_array_almost_equal(
        solution.values[0]["parent"],
        np.log(np.asarray((0.25, 0.5))) + 0.01 * continuation,
        decimal=DECIMAL_PRECISION,
    )
