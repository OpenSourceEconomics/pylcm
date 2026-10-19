"""Keep feasible parent candidates when earlier child landings are impossible."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.one_asset_egm_step import RetiredEGMResult, egm_one_asset_step
from _lcm.egm.preferences import Preferences
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
from lcm.exceptions import InvalidValueFunctionError
from lcm.solvers import EGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@dataclass(frozen=True, kw_only=True)
class Allocation:
    """Original three-period log allocation with a binding child saving floor."""

    cash: float = 4.0
    """Parent cash before consumption and saving."""
    child_floor: float = 2.0
    """Minimum saving chosen by the child."""
    parent_income: float = 1.0
    """Income added to parent saving to obtain child cash."""
    child_income: float = 1.0
    """Income added to child saving to obtain terminal wealth."""

    @property
    def consumption(self) -> float:
        """Return equal nonterminal consumption from the allocation conditions."""
        return (self.cash - self.child_floor + self.parent_income) / 2.0

    @property
    def parent_saving(self) -> float:
        """Return the saving that divides available consumption equally."""
        return self.cash - self.consumption

    @property
    def value(self) -> float:
        """Evaluate original utility at the strictly concave boundary optimum."""
        assert self.consumption > 0.0
        assert self.parent_saving >= 0.0
        assert self.consumption <= self.child_floor + self.child_income
        return 2.0 * math.log(self.consumption) + math.log(
            self.child_floor + self.child_income
        )


@categorical(ordered=False)
class RegimeId:
    parent: ScalarInt
    saving: ScalarInt
    done: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def inverse_marginal_utility(marginal_continuation: FloatND) -> ContinuousAction:
    return 1.0 / marginal_continuation


def marginal_utility(consumption: FloatND) -> FloatND:
    return 1.0 / consumption


def terminal_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def savings(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def floor(floor_value: float) -> FloatND:
    return jnp.asarray(floor_value)


def canonical_nodes(values: Iterable[float]) -> tuple[float, ...]:
    """Represent grid nodes in the solve precision before declaring their order."""
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    return tuple(
        float(value) for value in np.unique(np.asarray(tuple(values), dtype=dtype))
    )


def build_allocation_model(
    *,
    allocation: Allocation | None = None,
    enable_jit: bool = True,
    literal_child: bool = False,
    subdivisions: int = 8,
    parent_nodes: tuple[float, ...] | None = None,
) -> tuple[Model, UserParams]:
    """Represent the analytical optimum and every full-grid child landing exactly."""
    allocation = Allocation() if allocation is None else allocation
    full_nodes = canonical_nodes(
        (
            0.0,
            allocation.parent_saving,
            *(
                allocation.cash * index / subdivisions
                for index in range(subdivisions + 1)
            ),
        )
    )
    child_nodes = canonical_nodes(
        node + allocation.parent_income for node in full_nodes
    )
    terminal_nodes = canonical_nodes(
        allocation.child_floor + node + allocation.child_income for node in full_nodes
    )
    actions = canonical_nodes(
        (allocation.cash / 16.0, allocation.consumption, allocation.cash)
    )
    margin = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="wealth",
        post_decision_state="savings",
    )

    def parent_law(savings: FloatND) -> ContinuousState:
        return savings + allocation.parent_income

    def child_law(savings: FloatND) -> ContinuousState:
        return savings + allocation.child_income

    functions = {
        "utility": utility,
        "inverse_marginal_utility": inverse_marginal_utility,
        "savings": savings,
    }
    model = Model(
        regimes={
            "parent": ConsumptionSavingsRegime(
                liquid=margin,
                states={
                    "wealth": IrregSpacedGrid(
                        points=(0.75 * allocation.cash, allocation.cash)
                    )
                },
                actions={"consumption": IrregSpacedGrid(points=actions)},
                functions=functions,
                state_transitions={"wealth": parent_law},
                constraints={},
                solver=EGM(
                    savings_grid=IrregSpacedGrid(
                        points=full_nodes if parent_nodes is None else parent_nodes
                    )
                ),
            ),
            "saving": ConsumptionSavingsRegime(
                liquid=margin,
                states={"wealth": IrregSpacedGrid(points=child_nodes)},
                actions={"consumption": IrregSpacedGrid(points=actions)},
                functions=functions,
                state_transitions={"wealth": child_law},
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=margin,
                        lower=allocation.child_floor if literal_child else floor,
                    )
                },
                solver=EGM(
                    savings_grid=IrregSpacedGrid(
                        points=tuple(
                            allocation.child_floor + node for node in full_nodes
                        )
                        if literal_child
                        else full_nodes
                    )
                ),
            ),
            "done": Regime(
                states={"wealth": IrregSpacedGrid(points=terminal_nodes)},
                functions={"utility": terminal_utility},
            ),
        },
        ages=AgeGrid(exact_values=(10, 20, 30)),
        edges={"parent": {"saving": 10}, "saving": {"done": 20}},
        initial_nodes={10: "parent"},
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )
    params: UserParams = {
        "parent": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "saving": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            **(
                {}
                if literal_child
                else {
                    "borrowing_lower": {
                        "floor_value": jnp.asarray(allocation.child_floor)
                    }
                }
            ),
        },
        "done": {},
    }
    return model, params


def assert_allocation_value(
    *,
    allocation: Allocation,
    enable_jit: bool = True,
    subdivisions: int = 8,
    literal_child: bool = False,
) -> None:
    """Compare the published parent value with the original constrained allocation."""
    model, params = build_allocation_model(
        allocation=allocation,
        enable_jit=enable_jit,
        subdivisions=subdivisions,
        literal_child=literal_child,
    )
    result = model.solve(params=params, runtime_checks=False, log_level="off")
    np.testing.assert_array_almost_equal(
        result.values[0]["parent"][1], allocation.value, decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("literal_child", [False, True])
def test_parent_retains_represented_finite_optimum(
    *, enable_jit: bool, literal_child: bool
) -> None:
    """Keep utility log(6.75) at saving 2.5 despite an impossible first landing."""
    assert_allocation_value(
        allocation=Allocation(), enable_jit=enable_jit, literal_child=literal_child
    )


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("translation", [0.0, 1.0])
@pytest.mark.parametrize("subdivisions", [4, 8, 16])
def test_represented_optimum_survives_scale_translation_and_impossible_prefix(
    *, scale: float, translation: float, subdivisions: int
) -> None:
    """Preserve the optimum under matched floor/law translations and grid prefixes."""
    assert_allocation_value(
        allocation=Allocation(
            cash=4.0 * scale,
            child_floor=(2.0 + translation) * scale,
            parent_income=(1.0 + translation) * scale,
            child_income=scale,
        ),
        subdivisions=subdivisions,
    )


@pytest.mark.parametrize("side", [-1, 0, 1])
def test_first_child_landing_respects_adjacent_representable_floor(
    *, side: int
) -> None:
    """Keep the parent optimum across exact child affordability boundaries."""
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    bound = dtype(2.0)
    landing = (
        bound
        if side == 0
        else np.nextafter(bound, dtype(-np.inf if side < 0 else np.inf))
    )
    if side:
        assert bool(landing < bound) is (side < 0)
    assert_allocation_value(
        allocation=Allocation(parent_income=float(landing)), subdivisions=7
    )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_short_parent_support_is_mandatory_coverage_failure(
    *, enable_jit: bool
) -> None:
    """Reject all-impossible represented support at legal parent cash states."""
    model, params = build_allocation_model(
        enable_jit=enable_jit, parent_nodes=(0.0, 0.5, 1.0)
    )
    with pytest.raises(InvalidValueFunctionError, match="query_coverage_missing"):
        model.solve(params=params, runtime_checks=False, log_level="off")


def run_step(
    *,
    values: tuple[float, ...],
    marginals: tuple[float, ...],
    saving_nodes: tuple[float, ...],
    cash_nodes: tuple[float, ...],
    feature_floor: bool = True,
    discount_factor: float = 1.0,
    compiled: bool = False,
) -> RetiredEGMResult:
    """Read a supplied continuation row through the production Euler consumer."""

    def solve() -> RetiredEGMResult:
        return egm_one_asset_step(
            next_value=jnp.asarray(values),
            next_marginal=jnp.asarray(marginals),
            liquid_grid=jnp.asarray(cash_nodes),
            next_liquid_grid=jnp.asarray(saving_nodes),
            savings_grid=jnp.asarray(saving_nodes),
            discount_factor=discount_factor,
            preferences=Preferences(
                utility=utility,
                marginal_utility=marginal_utility,
                inverse_marginal_utility=inverse_marginal_utility,
            ),
            next_liquid=jnp.asarray(saving_nodes),
            marginal_return=jnp.ones(len(saving_nodes)),
            continuation_value=jnp.asarray(values),
            continuation_marginal=jnp.asarray(marginals),
            feature_floor=feature_floor,
        )

    return jax.jit(solve)() if compiled else solve()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("channel", ["value", "consumption", "savings", "marginal"])
def test_impossible_prefix_keeps_original_candidate_tuple(
    *, compiled: bool, channel: str
) -> None:
    """Retain each viable candidate's value, consumption and original saving."""
    result = run_step(
        values=(-np.inf, -np.inf, math.log(3.0), math.log(4.0)),
        marginals=(0.0, 0.0, 1.0, 0.5),
        saving_nodes=(0.0, 1.0, 2.0, 3.0),
        cash_nodes=(3.0, 4.0, 5.0),
        compiled=compiled,
    )
    expected = {
        "value": (math.log(3.0), 0.5 * (math.log(3.0) + math.log(8.0)), math.log(8.0)),
        "consumption": (1.0, 1.5, 2.0),
        "savings": (2.0, 2.5, 3.0),
        "marginal": (1.0, 2.0 / 3.0, 0.5),
    }
    np.testing.assert_array_almost_equal(
        getattr(result, channel), expected[channel], decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("channel", ["value", "consumption", "savings", "marginal"])
def test_finite_support_upper_clamp_keeps_the_last_candidate_tuple(
    *, channel: str
) -> None:
    """Clamp every associated field above support to the last finite endpoint."""
    result = run_step(
        values=(0.0, math.log(2.0), math.log(3.0)),
        marginals=(1.0, 0.5, 1.0 / 3.0),
        saving_nodes=(0.0, 1.0, 2.0),
        cash_nodes=(7.0,),
    )
    expected = {
        "value": (math.log(9.0),),
        "consumption": (3.0,),
        "savings": (2.0,),
        "marginal": (1.0 / 3.0,),
    }
    np.testing.assert_array_almost_equal(
        getattr(result, channel), expected[channel], decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("channel", ["value", "consumption", "marginal"])
def test_all_finite_feature_support_preserves_optional_none_results(
    *, channel: str
) -> None:
    """Feature activation preserves the all-finite one-asset interpolation result."""
    arguments = {
        "values": (0.0, math.log(2.0), math.log(3.0)),
        "marginals": (1.0, 0.5, 1.0 / 3.0),
        "saving_nodes": (0.0, 1.0, 2.0),
        "cash_nodes": (0.5, 1.0, 2.0, 3.0, 5.0, 7.0),
    }
    feature = run_step(
        values=arguments["values"],
        marginals=arguments["marginals"],
        saving_nodes=arguments["saving_nodes"],
        cash_nodes=arguments["cash_nodes"],
    )
    legacy = run_step(
        values=arguments["values"],
        marginals=arguments["marginals"],
        saving_nodes=arguments["saving_nodes"],
        cash_nodes=arguments["cash_nodes"],
        feature_floor=False,
    )
    np.testing.assert_array_equal(getattr(feature, channel), getattr(legacy, channel))


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
@pytest.mark.parametrize("invalid_channel", ["value", "marginal"])
def test_invalid_active_continuation_survives_finite_corner(
    *, invalid: float, invalid_channel: str
) -> None:
    """Preserve invalid active arithmetic even when the floor corner is finite."""
    result = run_step(
        values=(0.0, invalid if invalid_channel == "value" else 1.0, 2.0),
        marginals=(1.0, invalid if invalid_channel == "marginal" else 0.5, 0.25),
        saving_nodes=(0.0, 1.0, 2.0),
        cash_nodes=(0.5,),
    )
    assert bool(jnp.any(jnp.isnan(result.value)))


def test_original_saving_survives_cash_consumption_cancellation() -> None:
    """Keep original saving when cash minus consumption loses the saving node."""
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    exponent = np.finfo(dtype).nmant
    consumption = tuple(float(2.0 ** (exponent - 2 + index)) for index in range(3))
    saving_nodes = (0.03125, 0.0625, 0.125)
    cash = tuple(
        float(dtype(action + saving))
        for action, saving in zip(consumption, saving_nodes, strict=True)
    )
    assert dtype(cash[0] - consumption[0]) != dtype(saving_nodes[0])
    result = run_step(
        values=(0.0, 1.0, 2.0),
        marginals=tuple(1.0 / action for action in consumption),
        saving_nodes=saving_nodes,
        cash_nodes=cash,
    )
    np.testing.assert_array_equal(result.savings, saving_nodes)


@pytest.mark.parametrize(
    ("values", "marginals", "cash"),
    [
        ((-np.inf, -np.inf, -np.inf, -np.inf), (0.0, 0.0, 0.0, 0.0), 4.0),
        ((-np.inf, -np.inf, math.log(3.0), math.log(4.0)), (0.0, 0.0, 1.0, 0.5), 2.5),
        ((0.0, -np.inf, math.log(3.0), math.log(4.0)), (1.0, 0.0, 1.0, 0.5), 2.0),
    ],
    ids=["no-viable-node", "below-first-viable", "hole-between-original-indices"],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_unrepresented_support_is_coverage_missing(
    *,
    values: tuple[float, ...],
    marginals: tuple[float, ...],
    cash: float,
    compiled: bool,
) -> None:
    """Missing viable brackets never certify economic impossibility or bridge a gap."""
    result = run_step(
        values=values,
        marginals=marginals,
        saving_nodes=(0.0, 1.0, 2.0, 3.0),
        cash_nodes=(cash,),
        compiled=compiled,
    )
    assert result.coverage_missing is not None
    assert bool(jnp.any(result.coverage_missing))


@pytest.mark.parametrize("channel", ["value", "consumption", "savings", "marginal"])
def test_impossible_suffix_preserves_upper_endpoint_clamp(*, channel: str) -> None:
    """An impossible suffix cannot replace the last finite represented endpoint."""
    result = run_step(
        values=(0.0, math.log(2.0), math.log(3.0), -np.inf),
        marginals=(1.0, 0.5, 1.0 / 3.0, 0.0),
        saving_nodes=(0.0, 1.0, 2.0, 3.0),
        cash_nodes=(7.0,),
    )
    expected = {
        "value": (math.log(9.0),),
        "consumption": (3.0,),
        "savings": (2.0,),
        "marginal": (1.0 / 3.0,),
    }
    np.testing.assert_array_almost_equal(
        getattr(result, channel), expected[channel], decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("cash", [1.5, 2.0])
def test_known_current_budget_emptiness_needs_no_numerical_bracket(
    *, cash: float
) -> None:
    """Current cash at or below its declared floor remains an economic empty state."""
    result = run_step(
        values=(-np.inf, -np.inf),
        marginals=(0.0, 0.0),
        saving_nodes=(2.0, 3.0),
        cash_nodes=(cash,),
    )
    np.testing.assert_array_equal(result.value, [-np.inf])
    np.testing.assert_array_equal(result.marginal, [0.0])
    assert result.coverage_missing is not None
    assert not bool(jnp.any(result.coverage_missing))
