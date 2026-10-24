"""Finite outer candidates share the current stock's borrowing floor."""

import logging
import math
from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import Literal, TypedDict, Unpack, cast

import jax.numpy as jnp
import numpy as np
import pytest
from dags import rename_arguments

from _lcm.egm.carry import EGMCarry
from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.nested_floor import PreparedNestedFloor
from _lcm.egm.preferences import get_discount_factor_reader
from _lcm.egm.published_policy import EGMSimPolicy, NBEGMGridPolicy
from _lcm.execution.core_program import core_program_graph, select_programs
from _lcm.solution import backward_induction
from _lcm.solution.contract import PeriodKernel
from _lcm.solution.feasibility import BoundedResult
from _lcm.solution.kernel_output import ConsumedKernelOutput
from _lcm.solution.nbegm import _NBEGMNestedKernelOutput, _RideAlongNBEGMPeriodKernel
from _lcm.solution.negm import _NEGMPeriodKernel, _with_outer_post_decision
from _lcm.solution.nnbegm import _NNBEGMPeriodKernel
from _lcm.typing import EconFunction
from lcm import (
    AgeGrid,
    CESAggregator,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    PowerMean,
    Regime,
    categorical,
    outer_unchanged,
    post_decision_lower_bound,
)
from lcm.exceptions import ModelInitializationError
from lcm.solver_api import (
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    ArtifactAuthority,
    ArtifactKey,
    KernelOutput,
)
from lcm.solvers import DCEGM, NBEGM, NEGM, NNBEGM, FiniteOuterGrid
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    RegimeName,
    ScalarInt,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution._nbegm_direct_oracle import OracleContext, ride_along_kernel

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

type Family = Literal["negm", "nnbegm"]

_LIQUID = IrregSpacedGrid(points=np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0]))
_STOCK = IrregSpacedGrid(points=np.array([0.0, 1.0, 2.0]))
_OUTER = IrregSpacedGrid(points=np.array([0.0, 1.0, 2.0]))
_SAVINGS = IrregSpacedGrid(points=np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0]))
_CONSUMPTION = IrregSpacedGrid(
    points=np.array([0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0])
)
_INVESTMENT = IrregSpacedGrid(points=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
_PARAMS = {"discount_factor": 0.0}
_LOGGER = logging.getLogger(__name__)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    done: ScalarInt


def new_illiquid(
    *, illiquid: ContinuousState, investment: ContinuousAction
) -> ContinuousState:
    return illiquid + investment


def credited(*, illiquid: ContinuousState, new_illiquid: ContinuousState) -> FloatND:
    return (new_illiquid - illiquid) ** 2 / 2.0


def resources_before_outer_cost(*, liquid: ContinuousState) -> FloatND:
    return liquid + 1.25


def resources(
    *,
    liquid: ContinuousState,
    illiquid: ContinuousState,
    new_illiquid: ContinuousState,
) -> FloatND:
    return liquid + 1.25 - (new_illiquid - illiquid) ** 2 / 2.0


def savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def service(*, new_illiquid: ContinuousState) -> FloatND:
    return new_illiquid * jnp.log(2.0)


def utility(*, consumption: ContinuousAction, service: FloatND) -> FloatND:
    return jnp.log(consumption) + service


def inverse_marginal_utility(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_liquid(*, savings: FloatND) -> ContinuousState:
    return savings


def next_illiquid(*, new_illiquid: ContinuousState) -> ContinuousState:
    return new_illiquid


def terminal(*, liquid: ContinuousState, illiquid: ContinuousState) -> FloatND:
    return liquid + illiquid


def current_floor(*, illiquid: ContinuousState) -> FloatND:
    return illiquid / 4.0


def chosen_floor_helper(*, new_illiquid: ContinuousState) -> FloatND:
    return new_illiquid / 4.0


def chosen_floor(*, chosen_floor_helper: FloatND) -> FloatND:
    return chosen_floor_helper


def _model(*, family: Family, lower: float | UserFunction) -> Model:
    """Use the finite nested toy's public margin and resource declarations."""
    resource = (
        NetOfAdjustmentCost(
            output="resources",
            before_cost="resources_before_outer_cost",
            cost="credited",
        )
        if family == "negm"
        else "resources"
    )
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources=resource,
        post_decision_state="savings",
    )
    functions: dict[str, UserFunction] = {
        "utility": utility,
        "service": service,
        "savings": savings,
        "new_illiquid": new_illiquid,
        "inverse_marginal_utility": inverse_marginal_utility,
        "chosen_floor_helper": chosen_floor_helper,
    }
    solver: NEGM | NNBEGM
    if family == "negm":
        functions |= {
            "resources_before_outer_cost": resources_before_outer_cost,
            "credited": credited,
        }
        solver = NEGM(inner=DCEGM(savings_grid=_SAVINGS), outer_grid=_OUTER)
    else:
        functions["resources"] = resources
        solver = NNBEGM(
            inner=NBEGM(savings_grid=_SAVINGS),
            outer_search=FiniteOuterGrid(grid=_OUTER),
        )
    alive = NestedConsumptionSavingsRegime(
        states={"liquid": _LIQUID, "illiquid": _STOCK},
        actions={"consumption": _CONSUMPTION, "investment": _INVESTMENT},
        state_transitions={"liquid": next_liquid, "illiquid": next_illiquid},
        functions=functions,
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=lower)
        },
        solver=solver,
        liquid=margin,
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="investment",
            post_decision_state="new_illiquid",
            no_adjustment=outer_unchanged,
        ),
    )
    return Model(
        regimes={
            "alive": alive,
            "done": Regime(
                states={"liquid": _LIQUID, "illiquid": _STOCK},
                functions={"utility": terminal},
            ),
        },
        ages=AgeGrid(exact_values=(10, 20)),
        edges={"alive": {"done": (10,)}},
        initial_nodes={10: "alive"},
        regime_id_class=RegimeId,
    )


def _inner(
    *,
    kernel: PeriodKernel,
    context: OracleContext,
    prepared_source_floor: PreparedNestedFloor | None = None,
) -> KernelOutput:
    """Run an installed inner kernel on the same actual operands as its producer."""
    programs = select_programs(
        graph=core_program_graph(kernel=kernel), retain_replay=True
    )
    compiled_cores = {
        name: partial(
            program.function,
            **{axis.width_keyword: axis.extent for axis in program.requirements.axes},
        )
        for name, program in programs.items()
    }
    if prepared_source_floor is not None:
        assert type(prepared_source_floor) is BoundedResult
        assert len(prepared_source_floor.bound_status) == 1
        assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
        assert kernel.retain_nested_feasibility is True
        return kernel(
            compiled_cores=compiled_cores,
            **context,
            logger=_LOGGER,
            prepared_source_floor=prepared_source_floor,
        )
    return kernel(
        compiled_cores=compiled_cores,
        **context,
        logger=_LOGGER,
    )


class _ConsumeArgs(TypedDict):
    continuation_key: ArtifactKey | None
    regime_name: RegimeName
    period: int
    artifact_authorities: Mapping[ArtifactKey, ArtifactAuthority]


def _branches(
    *,
    model: Model,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[KernelOutput, tuple[KernelOutput, ...]]:
    """Capture a public solve, then run its own keeper and three node kernels."""
    published: list[KernelOutput] = []
    consume = backward_induction.consume_kernel_output

    def observe(
        *,
        output: KernelOutput,
        **kwargs: Unpack[_ConsumeArgs],
    ) -> ConsumedKernelOutput:
        if kwargs["regime_name"] == "alive":
            published.append(output)
        return consume(output=output, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(backward_induction, "consume_kernel_output", observe)
        kernel, context = ride_along_kernel(model=model, params=_PARAMS, period=0)
    assert len(published) == 1
    assert isinstance(kernel, _NEGMPeriodKernel | _NNBEGMPeriodKernel)
    keeper = _inner(kernel=kernel.keeper_kernel, context=context)
    prepared_source_floor = None
    if isinstance(kernel, _NNBEGMPeriodKernel) and kernel.query_model is not None:
        assert isinstance(keeper, _NBEGMNestedKernelOutput)
        prepared_source_floor = keeper.prepared_source_floor
        assert type(prepared_source_floor) is BoundedResult
        assert len(prepared_source_floor.bound_status) == 1
    adjusters = tuple(
        _inner(
            kernel=kernel.adjuster_kernel,
            context={
                **context,
                "flat_params": _with_outer_post_decision(
                    flat_params=context["flat_params"],
                    regime_name="alive",
                    outer_post_decision="new_illiquid",
                    value=node,
                ),
            },
            prepared_source_floor=prepared_source_floor,
        )
        for node in _OUTER.to_jax()
    )
    return published[0], (keeper, *adjusters)


def _assert_published_bank(
    *,
    published: KernelOutput,
    outputs: tuple[KernelOutput, ...],
    family: Family,
) -> None:
    """The actual outer producer retains each candidate's complete row identity."""
    carry = published.continuations[EGM_CONTINUATION]
    assert isinstance(carry, EGMCarry)
    leaves = carry.leaves()
    assert leaves[("value",)].shape[-2] == len(outputs)
    for index, output in enumerate(outputs):
        source = output.continuations[EGM_CONTINUATION]
        assert isinstance(source, EGMCarry)
        native = source.leaves()
        value, marginal = native[("value",)], native[("marginal_utility",)]
        if family == "nnbegm":
            marginal = jnp.where(jnp.isnan(value), 0.0, marginal)
            value = jnp.where(jnp.isnan(value), -jnp.inf, value)
        np.testing.assert_array_equal(leaves[("value",)][..., index, :], value)
        np.testing.assert_array_equal(
            leaves[("marginal_utility",)][..., index, :],
            marginal,
        )
        grid = native[("endog_grid",)]
        if family == "negm" and index:
            shift = (_OUTER.to_jax()[index - 1] - _STOCK.to_jax()) ** 2 / 2.0
            grid = grid + shift[:, None]
        np.testing.assert_array_equal(leaves[("endog_grid",)][..., index, :], grid)
    np.testing.assert_array_equal(
        published.value,
        jnp.max(jnp.stack([output.value for output in outputs]), axis=0),
    )


def _consumption_at_zero(
    *,
    output: KernelOutput,
    stock_index: int,
    resource: float,
) -> float:
    """Read a real inner policy in its declared native coordinate."""
    policy = output.replay[SIMULATION_POLICY]
    if isinstance(policy, NBEGMGridPolicy):
        state_index = tuple(
            stock_index if name == "illiquid" else 0 for name in policy.state_names
        )
        return float(policy.action[state_index])
    assert isinstance(policy, EGMSimPolicy)
    assert policy.row_passive_state_names == ("illiquid",)
    return float(
        interp_on_padded_grid(
            x_query=jnp.asarray(resource),
            xp=policy.endog_grid[stock_index],
            fp=policy.policy[stock_index],
        )
    )


def _assert_candidate_economics(
    *,
    outputs: tuple[KernelOutput, ...],
    dynamic: bool,
) -> None:
    """Each actual producer keeps its own target, budget, consumption and floor."""
    for stock_index, stock in enumerate((0.0, 1.0)):
        bound = stock / 4.0 if dynamic else 0.0
        targets = (stock, 0.0, 1.0, 2.0)
        for output, target in zip(outputs, targets, strict=True):
            resource = 1.25 - (target - stock) ** 2 / 2.0
            consumption = resource - bound
            expected = (
                math.log(consumption) + target * math.log(2.0)
                if consumption > 0.0
                else -math.inf
            )
            actual = np.asarray(output.value)[0, stock_index]
            np.testing.assert_array_almost_equal(
                actual, expected, decimal=DECIMAL_PRECISION
            )
            carry = output.continuations[EGM_CONTINUATION]
            assert isinstance(carry, EGMCarry)
            if consumption > 0.0:
                chosen_c = _consumption_at_zero(
                    output=output,
                    stock_index=stock_index,
                    resource=resource,
                )
                np.testing.assert_array_almost_equal(
                    [chosen_c, resource - chosen_c],
                    [consumption, bound],
                    decimal=DECIMAL_PRECISION,
                )
            if dynamic:
                data = carry.query_data_for_transport()
                assert data is not None
                np.testing.assert_array_equal(
                    np.asarray(data.floor_used)[stock_index], bound
                )
    # At liquid=.75, stock=0 and target=2 the candidate has R=b=0.
    # Zero consumption is outside the log domain even though savings equals b.
    np.testing.assert_array_equal(np.asarray(outputs[3].value)[3, 0], -np.inf)


@pytest.mark.parametrize("family", ["negm", "nnbegm"])
def test_finite_nested_literal_candidates_have_their_own_costs(
    *,
    family: Family,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admitted literal model ranks cost-adjusted inner candidates."""
    model = _model(family=family, lower=0.0)
    published, outputs = _branches(model=model, monkeypatch=monkeypatch)
    _assert_published_bank(published=published, outputs=outputs, family=family)
    _assert_candidate_economics(outputs=outputs, dynamic=False)
    at_zero = np.stack([np.asarray(output.value)[0, :2] for output in outputs])
    np.testing.assert_array_equal(np.argmax(at_zero, axis=0), [2, 3])
    np.testing.assert_array_almost_equal(
        np.max(at_zero, axis=0),
        np.log([1.5, 3.0]),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("family", ["negm", "nnbegm"])
def test_finite_nested_current_stock_floor_preserves_candidate_identity(
    *,
    family: Family,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed current stock sets every candidate floor, including a keeper tie."""
    literal, control = _branches(
        model=_model(family=family, lower=0.0),
        monkeypatch=monkeypatch,
    )
    _assert_published_bank(published=literal, outputs=control, family=family)
    _assert_candidate_economics(outputs=control, dynamic=False)
    published, outputs = _branches(
        model=_model(family=family, lower=current_floor),
        monkeypatch=monkeypatch,
    )
    _assert_published_bank(published=published, outputs=outputs, family=family)
    _assert_candidate_economics(outputs=outputs, dynamic=True)
    at_zero = np.stack([np.asarray(output.value)[0, :2] for output in outputs])
    # The dtype's actual arithmetic must preserve this exact tie; no tolerance
    # chooses its owner or converts distinct values into a tie.
    np.testing.assert_array_equal(at_zero[0, 1], at_zero[3, 1])
    np.testing.assert_array_equal(np.argmax(at_zero, axis=0), [2, 0])
    np.testing.assert_array_almost_equal(
        np.max(at_zero, axis=0),
        np.log([1.5, 2.0]),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        [
            _consumption_at_zero(output=outputs[0], stock_index=1, resource=1.25),
            _consumption_at_zero(output=outputs[3], stock_index=1, resource=0.75),
        ],
        [1.0, 0.5],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("family", ["negm", "nnbegm"])
def test_nested_floor_rejects_chosen_outer_ancestry(*, family: Family) -> None:
    """A floor cannot reach a chosen outer target through a helper."""
    _model(family=family, lower=0.0)
    with pytest.raises(
        ModelInitializationError,
        match=(
            r"Borrowing bound 'borrowing'.*forbidden dependency 'new_illiquid'"
            r" through borrowing_lower -> chosen_floor_helper -> new_illiquid\."
        ),
    ):
        _model(family=family, lower=chosen_floor)


def _ces_floor(*, illiquid: ContinuousState) -> FloatND:
    return 0.0 * illiquid


def _ces_utility(*, consumption: ContinuousAction) -> FloatND:
    return consumption


def _ces_terminal(*, liquid: ContinuousState, illiquid: ContinuousState) -> FloatND:
    return liquid + illiquid + 1.0


def _ces_model(*, lower: float | UserFunction) -> Model:
    """Two alive periods so the finite nested corner reads a CES continuation."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    stock = IrregSpacedGrid(points=np.array([0.0, 1.0, 2.0]))
    liquid = IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0]))
    alive = NestedConsumptionSavingsRegime(
        states={"liquid": liquid, "illiquid": stock},
        actions={
            "consumption": IrregSpacedGrid(
                points=np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
            ),
            "investment": _INVESTMENT,
        },
        state_transitions={"liquid": next_liquid, "illiquid": next_illiquid},
        functions={
            "utility": _ces_utility,
            "savings": savings,
            "new_illiquid": new_illiquid,
            "resources": resources,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=lower)
        },
        solver=NNBEGM(
            inner=NBEGM(savings_grid=_SAVINGS), outer_search=FiniteOuterGrid(grid=stock)
        ),
        liquid=margin,
        koopmans_aggregator=CESAggregator(),
        certainty_equivalent=PowerMean(),
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="investment",
            post_decision_state="new_illiquid",
            no_adjustment=outer_unchanged,
        ),
    )
    return Model(
        regimes={
            "alive": alive,
            "done": Regime(
                states={"liquid": liquid, "illiquid": stock},
                functions={"utility": _ces_terminal},
            ),
        },
        ages=AgeGrid(exact_values=(10, 20, 30)),
        edges={"alive": {"alive": (10,), "done": (20,)}},
        initial_nodes={10: "alive"},
        regime_id_class=RegimeId,
    )


_CES_PARAMS = {
    "alive": {
        "koopmans_aggregator": {
            "discount_factor": 0.9,
            "intertemporal_elasticity_of_substitution": 0.5,
        },
        "certainty_equivalent": {"risk_aversion": 2.0},
    }
}


def test_finite_nested_ces_callable_floor_matches_its_literal_value() -> None:
    """A callable floor equal to a literal one solves the same CES model."""
    literal = _ces_model(lower=0.0).solve(params=_CES_PARAMS, log_level="off")
    callable_floor = _ces_model(lower=_ces_floor).solve(
        params=_CES_PARAMS, log_level="off"
    )
    for period in (0, 1):
        np.testing.assert_array_almost_equal(
            callable_floor.values[period]["alive"],
            literal.values[period]["alive"],
            decimal=DECIMAL_PRECISION,
        )


def test_discount_factor_reader_selects_beta_among_ces_parameters() -> None:
    """The CES aggregator's elasticity is not mistaken for its discount factor."""
    aggregator = rename_arguments(
        func=CESAggregator(),
        mapper={
            name: f"koopmans_aggregator__{name}"
            for name in ("discount_factor", "intertemporal_elasticity_of_substitution")
        },
    )
    read = get_discount_factor_reader(
        functions=MappingProxyType({}),
        koopmans_aggregator=cast("EconFunction", aggregator),
    )
    beta = read(
        {
            "koopmans_aggregator__discount_factor": jnp.asarray(0.9),
            "koopmans_aggregator__intertemporal_elasticity_of_substitution": (
                jnp.asarray(0.5)
            ),
        }
    )
    np.testing.assert_array_equal(beta, jnp.asarray(0.9))
