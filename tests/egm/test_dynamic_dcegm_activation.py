"""Public DC floors and actual EGM-family child reads share canonical economics."""

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
    post_decision_lower_bound,
)
from lcm.solver_api import EGM_CONTINUATION, SIMULATION_POLICY
from lcm.solvers import DCEGM, EGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    RegimeName,
    ScalarInt,
    UserFunction,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution._nbegm_direct_oracle import interpolate_row, ride_along_kernel

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class RegimeId:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class ChildRegimeId:
    child: ScalarInt
    done: ScalarInt


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _terminal(liquid: ContinuousState) -> FloatND:
    return jnp.log(liquid)


def _resources(liquid: ContinuousState) -> FloatND:
    return 2.0 * liquid + 1.0


def _parent_resources(liquid: ContinuousState) -> FloatND:
    return liquid


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _inverse(marginal_continuation: FloatND) -> ContinuousAction:
    return 1.0 / marginal_continuation


def _to_child(savings: FloatND) -> ContinuousState:
    return 5.0 * savings - 1.0


def _ordinary_law(savings: FloatND) -> ContinuousState:
    return 3.0 * savings + 4.0


def _asset_law(*, savings: FloatND, liquid: ContinuousState) -> ContinuousState:
    return 3.0 * savings + 4.0 + liquid


def _lower(floor: float) -> FloatND:
    return jnp.asarray(floor)


def _period_lower(*, floor: float, period: ScalarInt) -> FloatND:
    return floor + 0.5 * period


def build_activation_model(
    *,
    floor: float,
    literal: bool,
    asset: bool,
    parent: bool = False,
    plain_parent: bool = False,
    lower_function: UserFunction = _lower,
) -> Model:
    """Use exact terminal abscissae and a binding current lower corner."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    child_liquid_nodes = (0.0, 0.125, 0.25, 0.5)
    source_nodes = (0.0, 0.5, 1.0)
    saving_nodes = tuple(floor + offset for offset in source_nodes)
    terminal_nodes = tuple(
        sorted(
            {
                3.0 * saving + 4.0 + (liquid if asset else 0.0)
                for saving in saving_nodes
                for liquid in child_liquid_nodes
            }
        )
    )
    regimes = {
        "child": ConsumptionSavingsRegime(
            liquid=margin,
            states={"liquid": IrregSpacedGrid(points=child_liquid_nodes)},
            actions={
                "consumption": IrregSpacedGrid(points=(0.025, 0.05, 0.25, 0.5, 1.0))
            },
            functions={
                "utility": _utility,
                "resources": _resources,
                "savings": _savings,
                "inverse_marginal_utility": _inverse,
            },
            constraints={
                "borrowing": post_decision_lower_bound(
                    margin=margin, lower=floor if literal else lower_function
                )
            },
            state_transitions={"liquid": _asset_law if asset else _ordinary_law},
            solver=DCEGM(
                savings_grid=IrregSpacedGrid(
                    points=saving_nodes if literal else source_nodes
                ),
                n_constrained_points=4,
            ),
        ),
        "done": Regime(
            states={"liquid": IrregSpacedGrid(points=terminal_nodes)},
            functions={"utility": _terminal},
        ),
    }
    if parent:
        parent_solver = EGM if plain_parent else DCEGM
        regimes["parent"] = ConsumptionSavingsRegime(
            liquid=margin,
            states={"liquid": IrregSpacedGrid(points=(0.225, 0.25))},
            actions={"consumption": IrregSpacedGrid(points=(0.025, 0.05, 0.1))},
            functions={
                "utility": _utility,
                "resources": _parent_resources,
                "savings": _savings,
                "inverse_marginal_utility": _inverse,
            },
            constraints={
                "parent_borrowing": post_decision_lower_bound(margin=margin, lower=0.2)
            },
            state_transitions={"liquid": _to_child},
            solver=parent_solver(savings_grid=IrregSpacedGrid(points=(0.2, 0.3, 0.4))),
        )
    return Model(
        regimes=regimes,
        ages=AgeGrid(start=0, inclusive_stop=2 if parent else 1, step="Y"),
        edges={"parent": {"child": 0}, "child": {"done": 1}}
        if parent
        else {"child": {"done": 0}},
        initial_nodes={0: "parent" if parent else "child"},
        regime_id_class=RegimeId if parent else ChildRegimeId,
    )


def activation_params(
    *, floor: float, literal: bool, parent: bool = False
) -> UserParams:
    return {
        "child": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            **({} if literal else {"borrowing_lower": {"floor": floor}}),
        },
        "done": {},
        **(
            {"parent": {"koopmans_aggregator": {"discount_factor": 1.0}}}
            if parent
            else {}
        ),
    }


@pytest.mark.parametrize("literal", [True, False])
@pytest.mark.parametrize("asset", [False, True])
@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_public_dc_corner_has_nonidentity_analytic_value(
    *, literal: bool, asset: bool, floor: float
) -> None:
    """Keep exact corners and the declared coarse represented publication distinct."""
    model = build_activation_model(floor=floor, literal=literal, asset=asset)
    solution = model.solve(
        params=activation_params(floor=floor, literal=literal), log_level="off"
    )
    expected = np.log((1.0 - floor) * (3.0 * floor + 4.0))
    if not asset and floor == 0.0:
        expected = _represented_dc_value(
            model=model,
            params=activation_params(floor=floor, literal=literal),
            regime_name="child",
            period=0,
            borrowing_limit=floor,
            floor_continuation=np.log(3.0 * floor + 4.0),
            resources=1.0,
        )
    np.testing.assert_array_almost_equal(
        solution.values[0]["child"][0], expected, decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("literal", [True, False])
@pytest.mark.parametrize("plain_parent", [False, True])
@pytest.mark.parametrize("asset", [False, True])
def test_egm_family_parent_consumes_dc_child_corner(
    *, literal: bool, plain_parent: bool, asset: bool
) -> None:
    """Consume the actual child row in physical resource and marginal units."""
    model = build_activation_model(
        floor=0.5,
        literal=literal,
        asset=asset,
        parent=True,
        plain_parent=plain_parent,
    )
    solution = model.solve(
        params=activation_params(floor=0.5, literal=literal, parent=True),
        log_level="off",
    )
    expected = _parent_represented_value(
        model=model,
        params=activation_params(floor=0.5, literal=literal, parent=True),
        literal=literal,
        asset=asset,
        plain_parent=plain_parent,
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["parent"][1], expected, decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("asset", [False, True])
def test_parent_uses_child_period_floor_and_qualified_parameter(*, asset: bool) -> None:
    """The target at period one uses floor one half from its own namespace."""
    model = build_activation_model(
        floor=0.5,
        literal=False,
        asset=asset,
        parent=True,
        lower_function=_period_lower,
    )
    solution = model.solve(
        params=activation_params(floor=0.0, literal=False, parent=True),
        log_level="off",
    )
    np.testing.assert_array_almost_equal(
        solution.values[1]["child"][0], np.log(2.75), decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["parent"][1],
        _parent_represented_value(
            model=model,
            params=activation_params(floor=0.0, literal=False, parent=True),
            literal=False,
            asset=asset,
            plain_parent=False,
        ),
        decimal=DECIMAL_PRECISION,
    )


def _row_read(*, carry: EGMCarry, resources: float, hermite: bool) -> float:
    """Read the actual produced row with the independent scalar carry oracle."""
    leaves = carry.leaves()
    return interpolate_row(
        query=resources,
        xp=np.asarray(leaves[("endog_grid",)]),
        fp=np.asarray(leaves[("value",)]),
        slopes=np.asarray(leaves[("marginal_utility",)]) if hermite else None,
    )


def _represented_dc_value(
    *,
    model: Model,
    params: UserParams,
    regime_name: RegimeName,
    period: int,
    borrowing_limit: float,
    floor_continuation: float,
    resources: float,
) -> float:
    """Check a real floor candidate, then apply the declared publication operator."""
    kernel, context = ride_along_kernel(
        model=model, params=params, regime_name=regime_name, period=period
    )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    program = kernel.core_programs()["replay"]
    output = kernel(
        compiled_cores={"replay": program.function},
        **context,
        logger=logging.getLogger(__name__),
    )
    carry = output.continuations[EGM_CONTINUATION]
    policy = output.replay[SIMULATION_POLICY]
    assert isinstance(carry, EGMCarry)
    assert isinstance(policy, EGMSimPolicy)
    leaves = carry.leaves()
    grid = np.asarray(leaves[("endog_grid",)])
    value = np.asarray(leaves[("value",)])
    marginal = np.asarray(leaves[("marginal_utility",)])
    consumption = np.asarray(policy.policy)
    data = carry.query_data_for_transport()
    if data is not None:
        recorded_savings = np.asarray(data.savings)
        np.testing.assert_array_equal(
            recorded_savings[0],
            np.asarray(borrowing_limit, dtype=recorded_savings.dtype),
        )
        assert data.corner_continuation is not None
        np.testing.assert_array_almost_equal(
            np.asarray(data.corner_continuation)[0],
            floor_continuation,
            decimal=DECIMAL_PRECISION,
        )
    assert grid.ndim == value.ndim == marginal.ndim == consumption.ndim == 1
    assert grid[0] > borrowing_limit
    np.testing.assert_array_almost_equal(
        grid[0] - consumption[0], borrowing_limit, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        value[0],
        np.log(consumption[0]) + floor_continuation,
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        marginal[0], 1.0 / consumption[0], decimal=DECIMAL_PRECISION
    )
    corner = np.log(resources - borrowing_limit) + floor_continuation
    represented = _row_read(carry=carry, resources=resources, hermite=True)
    return corner if resources <= grid[0] else max(corner, represented)


def _parent_represented_value(
    *,
    model: Model,
    params: UserParams,
    literal: bool,
    asset: bool,
    plain_parent: bool,
) -> float:
    """Keep child representation, actual floor candidates and parent optima separate."""
    _, context = ride_along_kernel(
        model=model, params=params, regime_name="parent", period=0
    )
    child = context["next_regime_to_continuation"]["child"]
    assert isinstance(child, EGMCarry)
    if plain_parent:
        return _plain_parent_value(carry=child, literal=literal, asset=asset)
    floor_continuation = _row_read(carry=child, resources=1.0, hermite=True)
    if not literal:
        floor_continuation = max(floor_continuation, np.log(2.75))
    return _represented_dc_value(
        model=model,
        params=params,
        regime_name="parent",
        period=0,
        borrowing_limit=0.2,
        floor_continuation=floor_continuation,
        resources=0.25,
    )


def _plain_parent_value(*, carry: EGMCarry, literal: bool, asset: bool) -> float:
    """Invert the first two Euler nodes using physical x and dR/dA exactly once."""
    leaves = carry.leaves()
    grid = np.asarray(leaves[("endog_grid",)])
    marginal = np.asarray(leaves[("marginal_utility",)])
    savings = np.asarray([0.2, 0.3], dtype=grid.dtype)
    child_liquid = 5.0 * savings - 1.0
    child_resources = 2.0 * child_liquid + 1.0
    if literal:
        continuation = np.asarray(
            [
                _row_read(carry=carry, resources=float(point), hermite=False)
                for point in child_resources
            ]
        )
        resource_marginal = np.asarray(
            [
                interpolate_row(query=float(point), xp=grid, fp=marginal, slopes=None)
                for point in child_resources
            ]
        )
        saving_marginal = 10.0 * resource_marginal
    else:
        pairs = [
            _represented_plain_child_pair(
                carry=carry,
                resources=float(resources),
                liquid=float(liquid),
                asset=asset,
            )
            for resources, liquid in zip(child_resources, child_liquid, strict=True)
        ]
        continuation = np.asarray([value for value, _ in pairs])
        saving_marginal = 5.0 * np.asarray([marginal for _, marginal in pairs])
    parent_resources = 0.25
    consumption = 1.0 / saving_marginal
    endogenous = savings + consumption
    candidate_value = np.log(consumption) + continuation
    assert endogenous[0] < endogenous[1]
    assert endogenous[1] > parent_resources
    if endogenous[0] >= parent_resources:
        return float(np.log(parent_resources - savings[0]) + continuation[0])
    fraction = (parent_resources - endogenous[0]) / (endogenous[1] - endogenous[0])
    return float((1.0 - fraction) * candidate_value[0] + fraction * candidate_value[1])


def _represented_plain_child_pair(
    *, carry: EGMCarry, resources: float, liquid: float, asset: bool
) -> tuple[float, float]:
    """Rank the associated represented tuple against the exact finite corner."""
    leaves = carry.leaves()
    grid = np.asarray(leaves[("endog_grid",)])
    marginal = np.asarray(leaves[("marginal_utility",)])
    data = carry.query_data_for_transport()
    assert data is not None
    count = int(np.count_nonzero(np.isfinite(grid)))
    assert count > 1
    assert grid[0] <= resources <= grid[count - 1]
    insertion = int(np.searchsorted(grid[:count], resources, side="right"))
    exact = min(max(insertion - 1, 0), count - 1)
    endpoint = grid[exact] == resources
    upper = min(max(insertion, 1), count - 1)
    lower = upper - 1
    selected = [exact] if endpoint else [lower, upper]
    kinds = np.asarray(data.node_kind)[selected]
    assert np.isin(kinds, [0, 1]).all()
    supported = bool(np.all(kinds == 1))
    if supported and not endpoint:
        owners = np.asarray(data.owner_id)[selected]
        supported = bool(np.asarray(data.link_valid)[lower] and owners[0] == owners[1])
    consumption = resources - 0.5
    terminal = 5.5 + (liquid if asset else 0.0)
    corner_value = float(np.log(consumption) + np.log(terminal))
    corner_native_marginal = 1.0 / consumption + (
        1.0 / (2.0 * terminal) if asset else 0.0
    )
    if supported:
        row_value = _row_read(carry=carry, resources=resources, hermite=True)
        row_marginal = interpolate_row(
            query=resources, xp=grid, fp=marginal, slopes=None
        )
        row_savings = interpolate_row(
            query=resources, xp=grid, fp=np.asarray(data.savings), slopes=None
        )
        assert np.isfinite([row_value, row_marginal, row_savings]).all()
        legal = 0.5 <= row_savings <= 1.5 and resources > row_savings
        if legal and row_value >= corner_value:
            return row_value, 2.0 * row_marginal
    return corner_value, 2.0 * corner_native_marginal
