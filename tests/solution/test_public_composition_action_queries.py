"""Composition actions compare the mapped common policies, with independent anchors."""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import lcm
from _lcm import composition_queries, policy_queries
from _lcm.composition_queries import (
    _compose,
    _QueryGraphBuilder,
    bind_composition_query_graph,
)
from _lcm.current_iid import CurrentIIDDrawPlan
from _lcm.execution.core_program import CoreBuildContext
from _lcm.policy_queries import (
    PolicyAnswer,
    QueryKind,
    ScalarPolicyQueries,
    _ChunkArguments,
    _own_raw_evaluation_sources,
    _PolicyQueryKernel,
    _QueryChunks,
    _QueryCore,
    build_policy_query_kernel,
)
from _lcm.regime_building.Q_and_F import CanonicalActionKernel, EvaluatedQAndF
from _lcm.solution.fingerprint import _semantic_fingerprint
from _lcm.typing import FlatParams, PytreeValue
from lcm import (
    AgeGrid,
    CollectiveUtility,
    CompositionAction,
    Consent,
    Counterparty,
    Destination,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    ParetoObjective,
    Regime,
    categorical,
)
from lcm.composition_actions import CompositionRule, PCOnActions
from lcm.exceptions import (
    InvalidParamsError,
    OffGridStateError,
    RegimeInitializationError,
    ShockDeclarationError,
)
from lcm.shocks import (
    Discrete,
    GaussHermite,
    Normal,
    PhysicalValues,
    Product,
    ShockGroup,
    UserNodes,
)
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.transition import fixed_transition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    IntND,
    OffGrid,
    Period,
    ScalarInt,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class _Household:
    single_f: ScalarInt
    single_m: ScalarInt
    couple: ScalarInt


@categorical(ordered=False)
class _Offer:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class _Singles:
    single_f: ScalarInt
    single_m: ScalarInt


def _wealth(wealth: FloatND) -> FloatND:
    return wealth


def _newlywed() -> int:
    return 0


def _incumbent() -> int:
    return 1


def _single(wealth: FloatND) -> FloatND:
    return 5.0 + 0 * wealth


def _female(*, action: IntND, wealth: FloatND, married_last_period: IntND) -> FloatND:
    return 20.0 - 14.0 * action + 0 * wealth + 0 * married_last_period


def _male(*, action: IntND, wealth: FloatND, married_last_period: IntND) -> FloatND:
    return 4.0 + 2.0 * action + 0 * wealth + 0 * married_last_period


def _cost(*, married_last_period: IntND, cost: FloatND) -> FloatND:
    return -cost * married_last_period


def _school_single(wealth: FloatND) -> FloatND:
    return 5.0 + 2 * wealth


def _school_projection(married_last_period: IntND) -> FloatND:
    return 1.0 - married_last_period


def _household(
    *,
    newlywed: bool,
    enable_jit: bool,
    source_only: bool = False,
    outside_lag: bool = False,
) -> Model:
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    return Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                functions={"utility": _school_single if outside_lag else _single},
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="couple",
                            role="f",
                            projection={
                                "wealth": _wealth,
                                "married_last_period": _newlywed
                                if newlywed
                                else _incumbent,
                            },
                        )
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="single_m",
                            role="m",
                            projection={"wealth": _wealth},
                        )
                    ),
                ),
            ),
            "single_m": Regime(
                states={"wealth": wealth},
                functions={"utility": _school_single if outside_lag else _single},
            ),
            "couple": Regime(
                states={
                    "wealth": wealth,
                    "married_last_period": DiscreteGrid(values=jnp.asarray([0, 1])),
                },
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={
                    "utility": CollectiveUtility(utilities={"f": _female, "m": _male})
                },
                composition_action=CompositionAction(
                    destinations={
                        "f": Destination(
                            regime="single_f",
                            projection={
                                "wealth": _school_projection if outside_lag else _wealth
                            },
                            utility_adjustment=_cost,
                        ),
                        "m": Destination(
                            regime="single_m",
                            projection={
                                "wealth": _school_projection if outside_lag else _wealth
                            },
                            utility_adjustment=_cost,
                        ),
                    },
                    rule=PCOnActions(),
                ),
            ),
        },
        regime_id_class=_Household,
        n_periods=1,
        initial_nodes=InitialNodes(
            by_period={
                0: ("single_f",) if source_only else ("single_f", "single_m", "couple")
            }
        ),
        edges={},
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        fixed_params={
            "couple": {
                "composition_action": {
                    "destinations": {
                        "f": {"utility_adjustment": {"cost": 0.0}},
                        "m": {"utility_adjustment": {"cost": 2.0}},
                    }
                }
            }
        },
    )


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("newlywed", [False, True])
def test_public_formation_uses_mapped_lagged_state_and_selected_policy(
    *,
    newlywed: bool,
    enable_jit: bool,
) -> None:
    """Lag zero selects (6,6); positive-cost (20,4) cannot pass fresh consent."""
    assert (
        Path(lcm.__file__).resolve().parents[2] == Path(__file__).resolve().parents[2]
    )
    model = _household(newlywed=newlywed, enable_jit=enable_jit)
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    # Literal independent S0 discriminator: (20,4) beats (6,6) after the male
    # threshold falls to 3, but male4 does not strictly beat his Raw stay5.
    np.testing.assert_allclose(result.values[0]["single_f"], 6.0 if newlywed else 5.0)
    couple = model._regimes["couple"]
    lag_axis = couple.solution.state_names.index("married_last_period")
    by_lag = np.moveaxis(np.asarray(result.values[0]["couple"]), lag_axis, 0)
    np.testing.assert_allclose(by_lag[0], np.broadcast_to([6.0, 6.0], by_lag[0].shape))
    np.testing.assert_allclose(by_lag[1], np.broadcast_to([20.0, 4.0], by_lag[1].shape))
    assert jax.default_backend() in {"cpu", "gpu"}


def _dated_female(
    *,
    action: IntND,
    wealth: FloatND,
    married_last_period: IntND,
    period: Period,
    solve_bonus: FloatND,
) -> FloatND:
    return (
        _female(action=action, wealth=wealth, married_last_period=married_last_period)
        + solve_bonus[period]
    )


@lcm.time_varying_params("replay_bonus")
def _dated_replay_female(
    *,
    action: IntND,
    wealth: FloatND,
    married_last_period: IntND,
    replay_bonus: FloatND,
) -> FloatND:
    return (
        _female(action=action, wealth=wealth, married_last_period=married_last_period)
        + replay_bonus
    )


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
@pytest.mark.parametrize("fixed", [False])
@pytest.mark.parametrize("missing", [None, "solve_bonus", "replay_bonus"])
def test_query_only_temporal_params_cover_the_consuming_phase_date(
    *, fixed: bool, missing: str | None
) -> None:
    """A mapped household reads dated payoffs without requiring unread dates."""
    base = _household(newlywed=True, enable_jit=True, source_only=True)
    params = {
        "solve_bonus": pd.Series(
            [100.0] if missing == "solve_bonus" else [10.0],
            index=pd.Index([0] if missing == "solve_bonus" else [1], name="period"),
        ),
        "replay_bonus": lcm.TimeVarying(
            values=jnp.asarray([100.0] if missing == "replay_bonus" else [20.0]),
            periods=(0,) if missing == "replay_bonus" else (1,),
        ),
    }

    def evaluate() -> tuple[FloatND, pd.DataFrame]:
        model = Model(
            regimes={
                **base.user_regimes,
                "couple": replace(
                    base.user_regimes["couple"],
                    functions={
                        "utility": CollectiveUtility(
                            utilities={
                                "f": lcm.Phased(
                                    solve=_dated_female, simulate=_dated_replay_female
                                ),
                                "m": _male,
                            }
                        )
                    },
                ),
            },
            regime_id_class=_Household,
            n_periods=2,
            initial_nodes=InitialNodes(by_period={1: "single_f"}),
            edges={},
            fixed_params={**base.fixed_params, **(params if fixed else {})},
            execution_config=ExecutionConfig(device_memory_bytes=None),
        )
        runtime = {} if fixed else params
        solution = model.solve(params=runtime, log_level="off", runtime_checks=False)
        assert set(solution.values[1]) == {"single_f"}
        replay = model.simulate(
            params=runtime,
            solution=solution,
            initial_conditions={
                "period": jnp.asarray([1], dtype=jnp.int32),
                "regime_id": jnp.asarray([_Household.single_f], dtype=jnp.int32),
                "wealth": jnp.asarray([0.0]),
            },
            seed=3,
            log_level="off",
            runtime_checks=False,
        ).to_dataframe(use_categorical_codes=True)
        return solution.values[1]["single_f"], replay

    if missing is None:
        value, replay = evaluate()
        # Male6 beats Raw5, hence common action1 yields female6 plus her bonus.
        np.testing.assert_allclose(value, 16.0)
        np.testing.assert_array_equal(replay["regime_name"], "couple")
        np.testing.assert_allclose(replay["value_f"], 26.0)
        np.testing.assert_allclose(replay["value_m"], 6.0)
    else:
        with pytest.raises(InvalidParamsError, match=r"missing required period.*1"):
            evaluate()


def _dated_raw_spouse(
    *, wealth: FloatND, period: Period, solve_outside: FloatND
) -> FloatND:
    return solve_outside[period] + 0 * wealth


@lcm.time_varying_params("replay_outside")
def _dated_replay_raw_spouse(*, wealth: FloatND, replay_outside: FloatND) -> FloatND:
    return replay_outside + 0 * wealth


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
@pytest.mark.parametrize("fixed", [False])
@pytest.mark.parametrize("missing", [None, "solve_outside", "replay_outside"])
def test_raw_counterparty_temporal_params_require_the_queried_date(
    *, fixed: bool, missing: str | None
) -> None:
    """An unoccupied spouse's dated outside option governs current consent."""
    base = _household(newlywed=True, enable_jit=True, source_only=True)
    params = {
        "solve_outside": pd.Series(
            [100.0] if missing == "solve_outside" else [6.5],
            index=pd.Index([0] if missing == "solve_outside" else [1], name="period"),
        ),
        "replay_outside": lcm.TimeVarying(
            values=jnp.asarray([100.0] if missing == "replay_outside" else [5.0]),
            periods=(0,) if missing == "replay_outside" else (1,),
        ),
    }

    def evaluate() -> tuple[FloatND, pd.DataFrame]:
        model = Model(
            regimes={
                **base.user_regimes,
                "single_m": replace(
                    base.user_regimes["single_m"],
                    functions={
                        "utility": lcm.Phased(
                            solve=_dated_raw_spouse,
                            simulate=_dated_replay_raw_spouse,
                        )
                    },
                ),
            },
            regime_id_class=_Household,
            n_periods=2,
            initial_nodes=InitialNodes(by_period={1: "single_f"}),
            edges={},
            fixed_params={**base.fixed_params, **(params if fixed else {})},
            execution_config=ExecutionConfig(device_memory_bytes=None),
        )
        assert (1, "single_m") not in model.graph.nodes
        assert (
            "single_m" not in model.reachability.simulation.active_regimes_by_period[1]
        )
        runtime = {} if fixed else params
        solution = model.solve(params=runtime, log_level="off", runtime_checks=False)
        assert set(solution.values[1]) == {"single_f"}
        replay = model.simulate(
            params=runtime,
            solution=solution,
            initial_conditions={
                "period": jnp.asarray([1], dtype=jnp.int32),
                "regime_id": jnp.asarray([_Household.single_f], dtype=jnp.int32),
                "wealth": jnp.asarray([0.0]),
            },
            seed=3,
            log_level="off",
            runtime_checks=False,
        ).to_dataframe(use_categorical_codes=True)
        return solution.values[1]["single_f"], replay

    if missing is None:
        value, replay = evaluate()
        # Male6 cannot beat6.5 in solve; in replay male6 beats his dated Raw5.
        np.testing.assert_allclose(value, 5.0)
        np.testing.assert_array_equal(replay["regime_name"], "couple")
        np.testing.assert_allclose(replay[["value_f", "value_m"]], 6.0)
    else:
        with pytest.raises(InvalidParamsError, match=r"missing required period.*1"):
            evaluate()


def _hidden_offer(*, offer_code: IntND, action: IntND, wealth: FloatND) -> FloatND:
    return offer_code + action * (2 * offer_code - 1) + 0 * wealth


def _stay(wealth: FloatND) -> FloatND:
    return 0.5 + 0 * wealth


def _stay_with_offer(*, wealth: FloatND, offer_code: IntND) -> FloatND:
    return 0.5 + 0 * wealth + 0 * offer_code


def _opposite(wealth: FloatND) -> FloatND:
    return 1 - wealth


def _offer_probs(wealth: FloatND) -> FloatND:
    return jnp.stack([0.8 - 0.6 * wealth, 0.2 + 0.6 * wealth])


def _fee() -> float:
    return -0.25


def _offer_model(
    *,
    observed: bool,
    shared: bool = False,
    flipped: bool = False,
) -> Model:
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    return Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                functions={"utility": _stay_with_offer if shared else _stay},
                shocks=("offers",) if shared else (),
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="single_m",
                            projection={"wealth": _opposite if flipped else _wealth},
                            utility_adjustment=_fee,
                        )
                    },
                    rule=PCOnActions(),
                    observes=("offer",) if observed else (),
                ),
            ),
            "single_m": Regime(
                states={"wealth": wealth},
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _hidden_offer},
                shocks=("offers",),
            ),
        },
        regime_id_class=_Singles,
        shocks={
            "offers": ShockGroup(
                shocks={
                    "offer": Discrete(
                        categories=_Offer,
                        probs=_offer_probs if shared else np.asarray([0.5, 0.5]),
                    )
                },
            )
        },
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
@pytest.mark.parametrize("observed", [False, True])
def test_hidden_query_prices_selected_policy_before_source_decision(
    *, observed: bool
) -> None:
    """E[max(0,-1),max(1,2)]-.25=.75; a sampled offer cannot choose for source."""
    result = _offer_model(observed=observed).solve(
        params={}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(
        result.values[0]["single_f"], 1.125 if observed else 0.75
    )


def test_shared_offer_uses_the_same_category_at_identical_law_inputs() -> None:
    result = _offer_model(observed=False, shared=True).solve(
        params={}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(result.values[0]["single_f"][..., 0], [0.75, 1.5])


def test_shared_offer_refuses_a_projection_that_changes_its_cdf() -> None:
    with pytest.raises(ShockDeclarationError, match="reads state 'wealth'"):
        _offer_model(observed=False, shared=True, flipped=True)


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
@pytest.mark.parametrize("observed", [False, True])
def test_scalar_query_prices_hidden_frame_without_hindsight(*, observed: bool) -> None:
    model = _offer_model(observed=observed)
    kernel = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    scalar = ScalarPolicyQueries(
        definitions=kernel.definitions, root=kernel.root, period=0
    )
    target = next(
        node for node in kernel.definitions.values() if node.read.target == "single_m"
    )
    plan = target.context.current_iid_plans[0]
    assert plan is not None
    draw = CurrentIIDDrawPlan(
        bindings={name: binding for name, binding, _ in plan.bindings}
    )
    arguments = scalar.arguments(
        context=CoreBuildContext(
            state_action_space=kernel.root_context.state_action_space,
            next_regime_to_V_arr={},
            next_regime_to_continuation={},
            flat_params={},
            period=0,
            ages=kernel.root_context.ages,
        ),
        fixed_flat_params=kernel.fixed_flat_params,
    )
    for code in (0, 1):
        frame = draw.realise(
            keys={"offer": jax.random.key(123)},
            group_params={"offers": {}},
            arguments={
                "wealth": jnp.asarray(0.0),
                "period": jnp.int32(0),
                "age": jnp.int32(0),
            },
            overrides={"offer": PhysicalValues(values=jnp.asarray(code, jnp.int32))},
        )
        value, action, support, valid, landings = jax.jit(scalar)(
            states={"wealth": jnp.asarray(0.0)},
            arguments=arguments,
            frame=frame,
        )
        assert landings.dtype == jnp.int32
        np.testing.assert_array_equal(landings, np.zeros(landings.shape, np.int32))
        np.testing.assert_allclose(
            value, (0.5 if code == 0 else 1.75) if observed else 0.75
        )
        assert int(action) == (code if observed else 1)
        assert bool(support)
        assert bool(valid)


def _positive(wealth: FloatND) -> BoolND:
    return wealth > 0


def _transfer(wealth: FloatND) -> FloatND:
    return 2 * wealth - 1


def _taste_target(*, action: IntND, wealth: FloatND) -> FloatND:
    return jnp.log(3.0) * action + 0 * wealth


def _taste_stay(wealth: FloatND) -> FloatND:
    return 1.2 + 0 * wealth


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
def test_hidden_target_taste_is_priced_at_logsum_then_drawn_when_acting() -> None:
    """log(4)>1.2>log(3): a hard target max would make the wrong decision to leave."""
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    model = Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                functions={"utility": _taste_stay},
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="single_m", projection={"wealth": _wealth}
                        )
                    },
                    rule=PCOnActions(),
                ),
            ),
            "single_m": Regime(
                states={"wealth": wealth},
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _taste_target},
                taste_shocks=ExtremeValueTasteShocks(),
            ),
        },
        regime_id_class=_Singles,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        fixed_params={"single_m": {"taste_shocks": {"scale": 1.0}}},
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(result.values[0]["single_f"], np.log(4.0))
    kernel = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    definitions, root = bind_composition_query_graph(
        contexts={
            node.read.target: node.context for node in kernel.definitions.values()
        },
        canonical_regimes=model._regimes,
        root="single_m",
        period=0,
        kind=QueryKind.PARTICIPATING,
    )
    scalar = ScalarPolicyQueries(definitions=definitions, root=root, period=0)
    arguments = dict(
        scalar.arguments(
            context=CoreBuildContext(
                state_action_space=definitions[root].context.state_action_space,
                next_regime_to_V_arr={},
                next_regime_to_continuation={},
                flat_params={},
                period=0,
                ages=kernel.root_context.ages,
            ),
            fixed_flat_params=kernel.fixed_flat_params,
        )
    )
    key = jax.random.key(3)
    arguments[root] = {**arguments[root], "taste_shock_key": key}
    value, action, support, valid, landings = jax.jit(scalar)(
        states={"wealth": jnp.asarray(0.0)}, arguments=arguments
    )
    assert landings.dtype == jnp.int32
    np.testing.assert_array_equal(landings, np.zeros(landings.shape, np.int32))
    expected = jnp.argmax(
        jnp.asarray([0.0, jnp.log(3.0)]) + jax.random.gumbel(key, (2,))
    )
    assert int(action) == int(expected)
    np.testing.assert_allclose(value, np.log(3.0) * int(expected))
    assert bool(support)
    assert bool(valid)


def _joint_value(*, action: IntND, x: FloatND, y: FloatND, wealth: FloatND) -> FloatND:
    return action * (x + y) + 0 * wealth


def _quarter(wealth: FloatND) -> FloatND:
    return 0.25 + 0 * wealth


def _joint_information_model(*, observes: tuple[str, ...], correlated: bool) -> Model:
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    return Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                functions={"utility": _quarter},
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="single_m", projection={"wealth": _wealth}
                        )
                    },
                    rule=PCOnActions(),
                    observes=observes,
                ),
            ),
            "single_m": Regime(
                states={"wealth": wealth},
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _joint_value},
                shocks=("joint",),
            ),
        },
        shocks={
            "joint": ShockGroup(
                shocks={"x": Normal(mu=0, sigma=1), "y": Normal(mu=0, sigma=1)},
                **({"correlation": ((1.0, 0.5), (0.5, 1.0))} if correlated else {}),
                nodes=UserNodes(
                    nodes=np.asarray([[-1.0, -1.0], [1.0, 1.0]]),
                    weights=np.asarray([0.5, 0.5]),
                ),
            )
        },
        regime_id_class=_Singles,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


@pytest.mark.parametrize("correlated", [False, True])
def test_wholly_hidden_joint_group_keeps_its_original_rows(*, correlated: bool) -> None:
    """The joint numerical rule is kept intact when the source sees neither draw."""
    result = _joint_information_model(observes=(), correlated=correlated).solve(
        params={}, log_level="off", runtime_checks=False
    )
    # Rows are (-1,-1),(1,1). Correlation transforms y=.5*x+sqrt(.75)*base_y.
    expected = 0.5 * (1.5 + np.sqrt(0.75)) if correlated else 1.0
    np.testing.assert_allclose(result.values[0]["single_f"], expected)


def test_a_composition_action_observing_part_of_a_shock_group_is_refused() -> None:
    """A group is observed whole or not at all."""
    with pytest.raises(
        ShockDeclarationError, match=r"\['x'\] of shock group 'joint' but not \['y'\]"
    ):
        _joint_information_model(observes=("x",), correlated=False)


def test_complete_correlated_observation_reuses_its_actual_joint_realization() -> None:
    """The source sees both coordinates, so no conditional Gaussian rule is needed."""
    result = _joint_information_model(observes=("x", "y"), correlated=True).solve(
        params={}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(
        result.values[0]["single_f"], 0.125 + 0.5 * (1.5 + np.sqrt(0.75))
    )


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
def test_nested_outside_projection_reads_the_already_mapped_marital_history() -> None:
    """Newlywed lag0 exposes schooling7; incumbent lag1 outside remains5."""
    model = _household(newlywed=True, enable_jit=True, outside_lag=True)
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    # At lag0 neither couple action gives both members their schooling outside7.
    # A root source must therefore stay, rather than price the lag1 policy.
    np.testing.assert_allclose(result.values[0]["single_f"][..., 0], [5.0, 7.0])
    couple = model._regimes["couple"]
    by_lag = np.moveaxis(
        np.asarray(result.values[0]["couple"]),
        couple.solution.state_names.index("married_last_period"),
        0,
    )
    np.testing.assert_allclose(by_lag[0], np.broadcast_to([7.0, 7.0], by_lag[0].shape))
    np.testing.assert_allclose(by_lag[1], np.broadcast_to([20.0, 4.0], by_lag[1].shape))


def _one() -> float:
    return 1.0


def _unused_helper(toll: FloatND) -> FloatND:
    return toll


def _ignored_formation_available(*, unused_helper: FloatND, fee: FloatND) -> BoolND:
    return unused_helper + fee > 0


def _reporter(report_toll: FloatND) -> FloatND:
    return report_toll


@pytest.mark.parametrize("preserve", [False, True])
def test_raw_counterparty_omits_unread_formation_params(*, preserve: bool) -> None:
    """A synthetic Raw/Part spouse keeps declarations, but never calls Consent."""
    base = _household(newlywed=True, enable_jit=True, source_only=True)
    spouse = base.user_regimes["single_m"]
    ignored = CompositionAction(
        destinations={
            "self": Destination(
                regime="couple",
                role="m",
                projection={"wealth": _wealth, "married_last_period": _newlywed},
            )
        },
        rule=Consent(
            counterparty=Counterparty(
                regime="single_f", role="f", projection={"wealth": _wealth}
            )
        ),
        available=_ignored_formation_available,
    )
    model = Model(
        regimes={
            **base.user_regimes,
            "single_m": replace(
                spouse,
                functions={
                    **spouse.functions,
                    "unused_helper": _unused_helper,
                    **({"reporter": _reporter} if preserve else {}),
                },
                output_only_functions=("reporter",) if preserve else (),
                composition_action=ignored,
            ),
        },
        regime_id_class=_Household,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        fixed_params=base.fixed_params,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    template = model.get_params_template()["single_m"]
    assert "composition_action" not in template
    assert "unused_helper" not in template
    if preserve:
        assert template["reporter"] == {"report_toll": "FloatND"}
    else:
        assert "reporter" not in template
    assert model.user_regimes["single_m"].composition_action is not None
    assert model._model_structure_fingerprint != base._model_structure_fingerprint
    result = model.solve(
        params={"single_m": {"reporter": {"report_toll": 0.0}}} if preserve else {},
        log_level="off",
        runtime_checks=False,
    )
    np.testing.assert_allclose(result.values[0]["single_f"], 6.0)


def _strict_stay(*, action: IntND, wealth: FloatND) -> FloatND:
    return action + 0 * wealth


def _rule_model(
    *,
    rule: CompositionRule,
    target_taste: bool = False,
    available: UserFunction | None = None,
) -> Model:
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    target = Destination(regime="single_m", projection={"wealth": _wealth})
    return Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _strict_stay},
                composition_action=CompositionAction(
                    destinations={"self": target}, rule=rule, available=available
                ),
            ),
            "single_m": Regime(
                states={"wealth": wealth},
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))}
                if target_taste
                else {},
                functions={"utility": _taste_target if target_taste else _single_tie},
                taste_shocks=ExtremeValueTasteShocks() if target_taste else None,
            ),
        },
        regime_id_class=_Singles,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        fixed_params=(
            {"single_m": {"taste_shocks": {"scale": 1.0}}} if target_taste else {}
        ),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _single_tie(wealth: FloatND) -> FloatND:
    return 1.0 + 0 * wealth


def test_strict_tie_has_empty_participating_policy_and_full_composition_action() -> (
    None
):
    """Strict source choice rejects the tied stay; Raw still chooses stayID1."""
    model = _rule_model(rule=PCOnActions(tie="leave"))
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(result.values[0]["single_f"], 1.0)
    kernel = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    contexts = {node.read.target: node.context for node in kernel.definitions.values()}
    for kind, expected_id, expected_support in [
        (QueryKind.RAW, 1, True),
        (QueryKind.PARTICIPATING, -1, False),
        (QueryKind.FULL, 2, True),
    ]:
        definitions, root = bind_composition_query_graph(
            contexts=contexts,
            canonical_regimes=model._regimes,
            root="single_f",
            period=0,
            kind=kind,
        )
        scalar = ScalarPolicyQueries(definitions=definitions, root=root, period=0)
        arguments = scalar.arguments(
            context=CoreBuildContext(
                state_action_space=kernel.root_context.state_action_space,
                next_regime_to_V_arr={},
                next_regime_to_continuation={},
                flat_params={},
                period=0,
                ages=kernel.root_context.ages,
            ),
            fixed_flat_params=kernel.fixed_flat_params,
        )
        value, action, support, valid, landings = jax.jit(scalar)(
            states={"wealth": jnp.asarray(0.0)},
            arguments=arguments,
        )
        assert landings.dtype == jnp.int32
        np.testing.assert_array_equal(landings, np.zeros(landings.shape, np.int32))
        assert int(action) == expected_id
        assert bool(support) == expected_support
        assert bool(valid)
        np.testing.assert_allclose(value, 1.0 if expected_support else -jnp.inf)


def test_strict_composition_action_prices_a_hidden_target_taste_at_its_logsum() -> None:
    """Staying is compared with the target's expected maximum, log 4."""
    result = _rule_model(rule=PCOnActions(tie="leave"), target_taste=True).solve(
        params={}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(result.values[0]["single_f"], np.log(4.0))


def _never_available() -> bool:
    return False


def _vector_available() -> BoolND:
    return jnp.asarray([False, True])


def test_public_availability_accepts_python_scalar_but_refuses_vector_reduction() -> (
    None
):
    """False is a scalar guard; a vector cannot silently become all(vector)."""
    result = _rule_model(rule=PCOnActions(), available=_never_available).solve(
        params={}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(result.values[0]["single_f"], 1.0)
    with pytest.raises(ValueError, match="must be a scalar Boolean"):
        _rule_model(rule=PCOnActions(), available=_vector_available).solve(
            params={}, log_level="off", runtime_checks=False
        )


def _female_offer(
    *, action: IntND, wealth: FloatND, married_last_period: IntND, offer_code: IntND
) -> FloatND:
    return (
        _female(action=action, wealth=wealth, married_last_period=married_last_period)
        + offer_code
    )


def _male_offer(
    *, action: IntND, wealth: FloatND, married_last_period: IntND, offer_code: IntND
) -> FloatND:
    return (
        _male(action=action, wealth=wealth, married_last_period=married_last_period)
        + offer_code
    )


@pytest.mark.parametrize("observed", [False, True])
def test_formation_requires_target_policy_information_at_the_actual_choice(
    *, observed: bool
) -> None:
    """Hidden dissolution pricing does not authorize an uninformed formation policy."""
    base = _household(newlywed=True, enable_jit=True, source_only=True)
    source = base.user_regimes["single_f"]
    formation = replace(
        cast("CompositionAction", source.composition_action),
        observes=("offer",) if observed else (),
    )
    couple = base.user_regimes["couple"]
    regimes = {
        **base.user_regimes,
        "single_f": replace(source, composition_action=formation),
        "couple": replace(
            couple,
            shocks=("offers",),
            functions={
                "utility": CollectiveUtility(
                    utilities={"f": _female_offer, "m": _male_offer}
                )
            },
        ),
    }

    def build() -> Model:
        return Model(
            regimes=regimes,
            shocks={
                "offers": ShockGroup(
                    shocks={
                        "offer": Discrete(
                            categories=_Offer, probs=np.asarray([0.5, 0.5])
                        )
                    }
                )
            },
            regime_id_class=_Household,
            n_periods=1,
            edges={},
            initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
            fixed_params=base.fixed_params,
            execution_config=ExecutionConfig(device_memory_bytes=None),
        )

    if not observed:
        with pytest.raises(
            RegimeInitializationError, match="unobserved decision information"
        ):
            build()
    else:
        result = build().solve(params={}, log_level="off", runtime_checks=False)
        # Offer0 forms at (6,6). Offer1 selects (21,5), failing male strict Raw5.
        np.testing.assert_allclose(result.values[0]["single_f"], 5.5)


def test_statically_false_composition_needs_no_inactive_destination_context() -> None:
    """A proof of false guards the same missing context admitted by graph validation."""
    model = Model(
        regimes={
            "single_f": Regime(
                functions={"utility": _one},
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(regime="single_m", projection={})
                    },
                    rule=PCOnActions(),
                    available=_never_available,
                ),
            ),
            "single_m": Regime(functions={"utility": _one}),
        },
        regime_id_class=_Singles,
        n_periods=3,
        edges={"single_m": {"single_f": lcm.Periods(values=(1,))}},
        initial_nodes=InitialNodes(by_period={0: ("single_f",), 1: ("single_m",)}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    result = model.solve(
        params={"discount_factor": 0.9}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(result.values[0]["single_f"], 1.0)
    np.testing.assert_allclose(result.values[1]["single_m"], 1.9)


def _unsafe_available(wealth: FloatND) -> BoolND:
    return jnp.sqrt(wealth) > 0


def test_false_ancestor_protects_a_poisoned_mapped_child_availability() -> None:
    """The child predicate cannot inspect sqrt(-1) after its ancestor refuses."""
    base = _household(newlywed=True, enable_jit=True, source_only=True)
    source = base.user_regimes["single_f"]
    declaration = cast("CompositionAction", source.composition_action)
    formation = replace(
        declaration,
        available=_positive,
        destinations={
            "self": replace(
                declaration.destinations["self"],
                projection={"wealth": _transfer, "married_last_period": _newlywed},
            )
        },
    )
    couple = base.user_regimes["couple"]
    model = Model(
        regimes={
            **base.user_regimes,
            "single_f": replace(source, composition_action=formation),
            "couple": replace(
                couple,
                composition_action=replace(
                    cast("CompositionAction", couple.composition_action),
                    available=_unsafe_available,
                ),
            ),
        },
        regime_id_class=_Household,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        fixed_params=base.fixed_params,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(result.values[0]["single_f"][..., 0], [5.0, 6.0])


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
@pytest.mark.parametrize("n_points", [64, 65])
def test_outside_choice_ignores_padded_query_lanes(*, n_points: int) -> None:
    """A composition action keeps every live source value on a partial chunk."""
    wealth = LinSpacedGrid(start=0.0, stop=1.0, n_points=n_points)
    model = Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                actions={
                    "action": DiscreteGrid(values=np.asarray([0], dtype=np.int32))
                },
                functions={"utility": _strict_stay},
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="single_m", projection={"wealth": _wealth}
                        )
                    },
                    rule=PCOnActions(),
                ),
            ),
            "single_m": Regime(
                states={"wealth": wealth}, functions={"utility": _single}
            ),
        },
        regime_id_class=_Singles,
        n_periods=1,
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        edges={},
        enable_jit=True,
        execution_config=ExecutionConfig(
            device_memory_bytes=None, axis_widths={"query_cell": 64}
        ),
    )
    kernel = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    assert kernel.chunk_size == 64
    assert n_points % kernel.chunk_size == (0 if n_points == 64 else 1)
    assert any(
        node.decision is not None and node.available is None
        for node in kernel.definitions.values()
    )
    result = model.solve(params={}, log_level="off", runtime_checks=True)
    np.testing.assert_array_equal(
        result.values[0]["single_f"], np.full((n_points, 1), 5.0)
    )


def _ancestor_transfer(*, wealth: FloatND, ancestor_shift: FloatND) -> FloatND:
    return wealth + ancestor_shift


def _descendant_transfer(descendant_shift: FloatND) -> FloatND:
    return descendant_shift


def _ancestor_available(*, wealth: FloatND, ancestor_threshold: FloatND) -> BoolND:
    return wealth >= ancestor_threshold


def _information_wealth(*, wealth: FloatND, middle_draw: FloatND) -> FloatND:
    return 10.0 + wealth + middle_draw


def _information_benchmark(*, wealth: FloatND, leaf_draw: FloatND) -> FloatND:
    return wealth + leaf_draw


def _information_stay(root_draw: FloatND) -> FloatND:
    return -100.0 + root_draw


@categorical(ordered=False)
class _NestedHousehold:
    single_f: ScalarInt
    single_m: ScalarInt
    couple: ScalarInt
    alone_f: ScalarInt


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
@pytest.mark.parametrize("fixed", [False, True])
def test_nested_information_uses_each_ancestors_transfer_operands(
    *, fixed: bool
) -> None:
    """A constant descendant map cannot supply its ancestor's shift or guard.

    The single forms a couple by consent at wealth `w + ancestor_shift`, where
    `w >= ancestor_threshold`, and observes the couple's middle draw `m = -1, 1`.
    Each spouse of the couple is worth `10 + w + ancestor_shift + m` and leaves for
    a single's problem at wealth `descendant_shift`, worth `descendant_shift` once
    its hidden mean-zero leaf draw is priced. The couple therefore forms exactly
    where the guard holds and `10 + w + ancestor_shift + m >= descendant_shift`:
    both spouses then strictly beat their stay values, at most 1 for the husband
    and `-100 + root_draw` for the single. Elsewhere the single stays at
    `-100 + root_draw`, whose mean-zero draw integrates to -100.
    """
    wealth = LinSpacedGrid(start=0, stop=12, n_points=13)
    regimes = {
        "single_f": Regime(
            states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
            functions={"utility": _information_stay},
            shocks=("root_noise",),
            composition_action=CompositionAction(
                destinations={
                    "self": Destination(
                        regime="couple",
                        role="f",
                        projection={"wealth": _ancestor_transfer},
                    )
                },
                available=_ancestor_available,
                observes=("middle_draw",),
                rule=Consent(
                    counterparty=Counterparty(
                        regime="single_m", role="m", projection={"wealth": _wealth}
                    )
                ),
            ),
        ),
        "single_m": Regime(
            states={"wealth": wealth},
            functions={"utility": _information_benchmark},
            shocks=("leaf_noise",),
        ),
        "couple": Regime(
            states={"wealth": wealth},
            functions={
                "utility": CollectiveUtility(
                    utilities={"f": _information_wealth, "m": _information_wealth}
                )
            },
            shocks=("middle_noise",),
            composition_action=CompositionAction(
                destinations={
                    "f": Destination(
                        regime="alone_f",
                        projection={"wealth": _descendant_transfer},
                    ),
                    "m": Destination(
                        regime="single_m",
                        projection={"wealth": _descendant_transfer},
                    ),
                },
                rule=PCOnActions(),
            ),
        ),
        "alone_f": Regime(
            states={"wealth": wealth},
            functions={"utility": _information_benchmark},
            shocks=("leaf_noise",),
        ),
    }
    runtime_model = None
    for shift, descendant, threshold in (
        (0.25, 1.5, 0.5),
        (0.75, 10.0, -0.5),
    ):
        params = {
            "ancestor_shift": shift,
            "descendant_shift": descendant,
            "ancestor_threshold": threshold,
        }
        if runtime_model is None or fixed:
            model = Model(
                regimes=regimes,
                regime_id_class=_NestedHousehold,
                n_periods=1,
                initial_nodes=InitialNodes(by_period={0: "single_f"}),
                edges={},
                shocks={
                    group: ShockGroup(
                        shocks={name: Normal(mu=0.0, sigma=1.0)},
                        nodes=UserNodes(
                            nodes=np.asarray([[-1.0], [1.0]]),
                            weights=np.asarray([0.5, 0.5]),
                        ),
                    )
                    for group, name in (
                        ("root_noise", "root_draw"),
                        ("middle_noise", "middle_draw"),
                        ("leaf_noise", "leaf_draw"),
                    )
                },
                fixed_params=params if fixed else {},
                enable_jit=True,
                execution_config=ExecutionConfig(device_memory_bytes=None),
            )
            runtime_model = model
        else:
            model = runtime_model
        kernel = model._regimes["single_f"].solution.period_kernels[0]
        assert isinstance(kernel, _PolicyQueryKernel)
        ancestor_key, ancestor_node = next(
            (key, node)
            for key, node in kernel.definitions.items()
            if node.read.target == "couple"
        )
        descendant_node = next(
            node
            for node in kernel.definitions.values()
            if node.read.target == "alone_f"
        )
        assert ancestor_key in descendant_node.ancestors
        assert ancestor_node.available is not None
        assert ancestor_node.read.operand_paths.keys() - (
            descendant_node.read.operand_paths.keys()
        )
        solution = model.solve(
            params={} if fixed else params, log_level="off", runtime_checks=True
        )
        source_wealth = np.asarray([0.0, 1.0])
        middle = 10.0 + source_wealth[:, None] + shift + np.asarray([-1.0, 1.0])
        forms = (source_wealth[:, None] >= threshold) & (middle >= descendant)
        expected = np.where(forms, middle, -100.0).mean(axis=1)
        np.testing.assert_allclose(solution.values[0]["single_f"][..., 0], expected)
        assert np.all(np.isfinite(solution.values[0]["single_f"]))


def test_leaf_composition_action_projection_does_not_capture_a_dag() -> None:
    """A projection with no computed inputs retains only its authored callback."""

    builder = cast(
        "_QueryGraphBuilder",
        SimpleNamespace(
            contexts={"owner": SimpleNamespace(functions={"unrelated": _wealth})},
            canonical_regimes={
                "owner": SimpleNamespace(
                    regime_params_template=MappingProxyType(
                        {"projection": MappingProxyType({})}
                    )
                )
            },
            aliases={},
        ),
    )
    bound = _QueryGraphBuilder.bound(
        builder, owner="owner", path=("projection",), function=_wealth
    )
    assert bound is _wealth
    np.testing.assert_array_equal(bound(wealth=jnp.asarray(3.0)), 3.0)


def test_destination_projection_keeps_computed_inputs_and_parameter_aliases() -> None:
    """Computed dependencies and their caller-owned aliases retain their values."""

    def resources(wealth: FloatND) -> FloatND:
        return 2 * wealth

    def projection(resources: FloatND) -> FloatND:
        return resources + 1

    builder = cast(
        "_QueryGraphBuilder",
        SimpleNamespace(
            contexts={"owner": SimpleNamespace(functions={"resources": resources})},
            canonical_regimes={
                "owner": SimpleNamespace(
                    regime_params_template=MappingProxyType(
                        {"projection": MappingProxyType({})}
                    )
                )
            },
            aliases={("owner", "wealth"): "cash"},
        ),
    )
    bound = _QueryGraphBuilder.bound(
        builder, owner="owner", path=("projection",), function=projection
    )
    np.testing.assert_array_equal(bound(cash=jnp.asarray(3.0)), 7.0)


def test_callable_composition_action_projection_keeps_fingerprint_admission() -> None:
    """Callable identity declarations keep the existing composed-hash support."""

    builder = cast(
        "_QueryGraphBuilder",
        SimpleNamespace(
            contexts={"owner": SimpleNamespace(functions={})},
            canonical_regimes={
                "owner": SimpleNamespace(
                    regime_params_template=MappingProxyType(
                        {"projection": MappingProxyType({})}
                    )
                )
            },
            aliases={},
        ),
    )
    bound = _QueryGraphBuilder.bound(
        builder,
        owner="owner",
        path=("projection",),
        function=fixed_transition("wealth"),
    )
    assert inspect.isfunction(bound)
    composed = _compose(projection={}, function=bound, source="owner")
    np.testing.assert_array_equal(composed(wealth=jnp.asarray(3.0)), 3.0)
    assert _semantic_fingerprint(composed)


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
def test_own_raw_and_full_evaluate_canonical_problem_once(  # noqa: PLR0915 - keep one graph and runtime population proof together
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical conditional single work is shared while both callers still decide.

    The couple reads the single's outside value at its point.
    """
    monkeypatch.setattr(composition_queries, "_READ_STORED_OUTSIDE_VALUES", False)
    model = _household(
        newlywed=True, enable_jit=True, source_only=True, outside_lag=True
    )
    kernel = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    root = kernel.definitions[kernel.root]
    own_key = root.inputs["own_raw"]
    own = kernel.definitions[own_key]
    couple = kernel.definitions[root.inputs["composition_action_self"]]
    outside_key = couple.inputs["composition_action_f"]
    outside = kernel.definitions[outside_key]
    assert outside.read.kind is QueryKind.RAW
    assert outside.read.target == "single_f"
    assert not outside.inputs
    assert outside_key != own_key
    canonical = root.context.Q_and_F_functions[0]
    assert root.own_raw == own_key
    assert kernel.chunk_size == 2
    assert own.context is root.context
    assert own.context.Q_and_F_functions[0] is canonical
    assert own.read.kind is QueryKind.RAW
    assert own.available is root.available is None
    assert own.read.operand_paths == root.read.operand_paths
    assert own.read.phase == root.read.phase == "solve"
    assert own.read.period == root.read.period == 0
    assert own.read.role == root.read.role == ""
    assert all(
        own.read.projection[name] is helper
        for name, helper in root.read.projection.items()
    )
    numerical_calls: list[float] = []
    caller_answers: dict[str, tuple[np.ndarray, ...]] = {}
    consent_calls: list[bool] = []
    evaluate = policy_queries._shared_evaluation
    core = _QueryCore.__call__
    eligible = composition_queries._CompositionReducer._eligible

    def record_numerical(value: np.ndarray) -> None:
        numerical_calls.append(float(value[0]))

    def observed_evaluation(
        *, function: CanonicalActionKernel, arguments: Mapping[str, PytreeValue]
    ) -> EvaluatedQAndF:
        evaluated = evaluate(function=function, arguments=arguments)
        if function is canonical:
            jax.debug.callback(record_numerical, evaluated.parts.Q)
        return evaluated

    def observed_core(self: _QueryCore, **arguments: PytreeValue) -> PolicyAnswer:
        answer = core(self, **arguments)
        if self.name in {kernel.root, own_key, outside_key}:

            def record_caller(*leaves: np.ndarray) -> None:
                caller_answers[self.name] = leaves

            jax.debug.callback(record_caller, *answer)
        return answer

    # keyword-only-exempt: library-callback=Python.descriptor_binding
    def observed_eligible(
        self: composition_queries._CompositionReducer,
        *,
        operands: Mapping[str, PytreeValue],
        payoffs: Mapping[str, FloatND],
        active: BoolND,
    ) -> tuple[BoolND, BoolND]:
        result = eligible(self, operands=operands, payoffs=payoffs, active=active)
        if isinstance(cast("CompositionAction", self.declaration).rule, Consent):
            jax.debug.callback(
                lambda value: consent_calls.append(bool(value)), result[0]
            )
        return result

    monkeypatch.setattr(policy_queries, "_shared_evaluation", observed_evaluation)
    monkeypatch.setattr(_QueryCore, "__call__", observed_core)
    monkeypatch.setattr(
        composition_queries._CompositionReducer, "_eligible", observed_eligible
    )
    program = kernel.programs["_chunks"]
    builder = cast("_ChunkArguments", program.argument_builder)
    operands = builder.operands(
        CoreBuildContext(
            state_action_space=kernel.root_context.state_action_space,
            next_regime_to_V_arr={},
            next_regime_to_continuation={},
            flat_params={},
            period=0,
            ages=kernel.root_context.ages,
        )
    )
    bank, statuses, counts = jax.jit(program.function)(
        bank=jnp.zeros((2, 1)),
        **operands,
    )
    jax.block_until_ready((bank, statuses, counts))
    jax.effects_barrier()
    expected = np.asarray([[5.0], [7.0]], dtype=bank.dtype)
    assert np.asarray(bank).tobytes() == expected.tobytes()
    assert set(caller_answers) == {kernel.root, own_key, outside_key}
    for name, answer in caller_answers.items():
        caller_expected = (
            np.asarray([[7.0], [7.0]], dtype=expected.dtype)
            if name == outside_key
            else expected
        )
        assert answer[0].dtype == expected.dtype
        assert answer[0].shape == (2, 1)
        assert answer[0].tobytes() == caller_expected.tobytes()
        np.testing.assert_array_equal(answer[1], np.asarray([0, 0], np.int32))
        np.testing.assert_array_equal(answer[2], [True, True])
        assert bool(answer[3])
        assert answer[4].shape == (0,)
        assert answer[4].dtype == np.int32
    assert consent_calls == [False, False]
    np.testing.assert_array_equal(statuses, np.ones(len(kernel.definitions), bool))
    assert len(counts) == len(kernel.definitions)
    assert all(count.shape == (0,) and count.dtype == jnp.int32 for count in counts)
    # Two own-state lanes and two distinct lagged-history outside-option lanes.
    assert len(numerical_calls) == 4, numerical_calls
    np.testing.assert_array_equal(sorted(numerical_calls), [5.0, 7.0, 7.0, 7.0])


def _shifted_single(*, wealth: FloatND, level: FloatND) -> FloatND:
    return level + 2 * wealth


def _positive_wealth(wealth: FloatND) -> BoolND:
    return wealth > 0


def _reversed_wealth(wealth: FloatND) -> FloatND:
    return 1.0 - wealth


def _sharing_control_model(
    *,
    state_count: int = 2,
    guarded: bool = False,
    parameterized: bool = False,
    constrained: bool = False,
) -> Model:
    base = _household(
        newlywed=True, enable_jit=True, source_only=True, outside_lag=True
    )
    source = base.user_regimes["single_f"]
    return Model(
        regimes={
            **base.user_regimes,
            "single_f": replace(
                source,
                constraints={"positive_wealth": _positive_wealth}
                if constrained
                else source.constraints,
                states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=state_count)},
                functions={
                    "utility": _shifted_single if parameterized else _school_single
                },
                composition_action=replace(
                    cast("CompositionAction", source.composition_action),
                    available=_positive_wealth if guarded else None,
                ),
            ),
        },
        regime_id_class=_Household,
        n_periods=1,
        initial_nodes=InitialNodes(by_period={0: "single_f"}),
        edges={},
        fixed_params=base.fixed_params,
        enable_jit=True,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _sharing_chunk_arguments(
    *, kernel: _PolicyQueryKernel, level: FloatND | None = None
) -> Mapping[str, PytreeValue]:

    flat_params: FlatParams = (
        MappingProxyType({})
        if level is None
        else MappingProxyType({"single_f": MappingProxyType({"utility__level": level})})
    )
    return cast(
        "Mapping[str, PytreeValue]",
        cast("_ChunkArguments", kernel.programs["_chunks"].argument_builder).operands(
            CoreBuildContext(
                state_action_space=kernel.root_context.state_action_space,
                next_regime_to_V_arr={},
                next_regime_to_continuation={},
                flat_params=flat_params,
                period=0,
                ages=kernel.root_context.ages,
            )
        ),
    )


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
@pytest.mark.parametrize("state_count", [3])
@pytest.mark.parametrize("chunk_size", [2])
@pytest.mark.parametrize("compiled", [True])
@pytest.mark.parametrize("constrained", [True])
def test_static_single_sharing_preserves_uninstrumented_chunks(
    *,
    state_count: int,
    chunk_size: int,
    compiled: bool,
    constrained: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw sharing preserves bytes, statuses and count slots across padded chunks.

    The couple reads the single's outside value at its point.
    """
    monkeypatch.setattr(composition_queries, "_READ_STORED_OUTSIDE_VALUES", False)
    model = _sharing_control_model(state_count=state_count, constrained=constrained)
    source = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(source, _PolicyQueryKernel)
    definitions = dict(source.definitions)
    root = definitions[source.root]
    kernels = []
    for reuse in (False, True):
        definitions[source.root] = root if reuse else replace(root, own_raw=None)
        kernels.append(
            build_policy_query_kernel(
                root_context=source.root_context,
                period=0,
                definitions=definitions.copy(),
                root=source.root,
                chunk_size=chunk_size,
            ).with_fixed_params(fixed_flat_params=source.fixed_flat_params)
        )
    outputs = []
    for reuse, kernel in zip((False, True), kernels, strict=True):
        function = cast("_QueryChunks", kernel.programs["_chunks"].function)
        assert bool(function.raw_sources) is reuse
        padded = ((state_count + chunk_size - 1) // chunk_size) * chunk_size
        run = jax.jit(function) if compiled else function
        arguments = _sharing_chunk_arguments(kernel=kernel)
        with jax.disable_jit(disable=not compiled):
            output = run(
                bank=jnp.zeros((padded, 1)),
                states=cast("Mapping[str, FloatND | IntND]", arguments["states"]),
                query_arguments=cast(
                    "Mapping[str, Mapping[str, PytreeValue]]",
                    arguments["query_arguments"],
                ),
            )
        outputs.append(jax.block_until_ready(output))
    for control, shared in zip(
        jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1]), strict=True
    ):
        assert control.shape == shared.shape
        assert control.dtype == shared.dtype
        assert np.asarray(control).tobytes() == np.asarray(shared).tobytes()
    expected = np.asarray(
        [5.0, 7.0] if state_count == 2 else [5.0, 6.0, 7.0], dtype=outputs[1][0].dtype
    )[:, None]
    if constrained:
        expected[0] = -np.inf
    assert np.asarray(outputs[1][0][:state_count]).tobytes() == expected.tobytes()
    assert np.all(np.isneginf(outputs[1][0][state_count:]))
    np.testing.assert_array_equal(outputs[1][1], np.ones(len(source.definitions), bool))
    assert len(outputs[1][2]) == len(source.definitions)


@pytest.mark.parametrize(
    "changed", ["projection", "guard", "context", "bindings", "role", "provenance"]
)
def test_static_single_sharing_requires_exact_provenance(changed: str) -> None:
    """Numerically equal helpers and different binding contexts retain separate work."""

    model = _household(
        newlywed=True, enable_jit=True, source_only=True, outside_lag=True
    )
    kernel = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    definitions = dict(kernel.definitions)
    root = definitions[kernel.root]
    assert root.own_raw is not None
    raw = definitions[root.own_raw]
    if changed == "projection":
        raw = replace(raw, read=replace(raw.read, projection={"wealth": _wealth}))
    elif changed == "guard":
        raw = replace(raw, available=_positive_wealth)
    elif changed == "context":
        raw = replace(raw, context=replace(raw.context))
    elif changed == "bindings":
        raw = replace(
            raw,
            read=replace(
                raw.read, operand_paths={"extra": ("single_f", "utility__level")}
            ),
        )
    elif changed == "role":
        raw = replace(raw, read=replace(raw.read, role="f"))
    else:
        definitions[kernel.root] = replace(root, own_raw=None)
    definitions[root.own_raw] = raw
    assert not _own_raw_evaluation_sources(
        root=kernel.root, definitions=definitions, period=0
    )


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
@pytest.mark.parametrize("guarded", [False, True])
def test_static_single_sharing_retains_guard_and_projection_fallback(
    *,
    guarded: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guarded or remapped outside option keeps its own support and mapped value.

    The couple reads the single's outside value at its point.
    """
    monkeypatch.setattr(composition_queries, "_READ_STORED_OUTSIDE_VALUES", False)
    model = _sharing_control_model(guarded=guarded)
    source = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(source, _PolicyQueryKernel)
    definitions = dict(source.definitions)
    root = definitions[source.root]
    assert root.own_raw is not None
    if not guarded:
        own = definitions[root.own_raw]
        definitions[root.own_raw] = replace(
            own, read=replace(own.read, projection={"wealth": _reversed_wealth})
        )
    kernel = build_policy_query_kernel(
        root_context=source.root_context,
        period=0,
        definitions=definitions,
        root=source.root,
        chunk_size=2,
    ).with_fixed_params(fixed_flat_params=source.fixed_flat_params)
    function = cast("_QueryChunks", kernel.programs["_chunks"].function)
    assert not function.raw_sources
    arguments = _sharing_chunk_arguments(kernel=kernel)
    bank, statuses, counts = jax.jit(function)(bank=jnp.zeros((2, 1)), **arguments)
    np.testing.assert_array_equal(bank, np.asarray([[5.0], [7.0]], dtype=bank.dtype))
    np.testing.assert_array_equal(statuses, np.ones(len(definitions), bool))
    assert len(counts) == len(definitions)
    own_core = function.cores[root.own_raw]
    own = jax.jit(own_core)(
        **cast("Mapping[str, Mapping[str, PytreeValue]]", arguments["query_arguments"])[
            root.own_raw
        ],
        _lcm_query_source={"wealth": jnp.asarray([0.0, 1.0])},
        _lcm_query_live=jnp.asarray([True, True]),
    )
    np.testing.assert_array_equal(
        own[0][..., 0], [-np.inf, 7.0] if guarded else [7.0, 5.0]
    )
    np.testing.assert_array_equal(own[1], [-1, 0] if guarded else [0, 0])
    np.testing.assert_array_equal(own[2], [False, True] if guarded else [True, True])
    assert bool(own[3])
    assert own[4].shape == (0,)
    assert own[4].dtype == jnp.int32


@pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="representative"
)
@pytest.mark.parametrize(("state_count", "chunk_size"), [(3, 2)])
def test_static_single_sharing_reads_changed_params_and_preserves_jvp(
    *, state_count: int, chunk_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full and padded loops retain fresh parameters and forward/reverse directions.

    The couple reads the single's outside value at its point.
    """
    monkeypatch.setattr(composition_queries, "_READ_STORED_OUTSIDE_VALUES", False)
    model = _sharing_control_model(state_count=state_count, parameterized=True)
    source = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(source, _PolicyQueryKernel)
    kernel = build_policy_query_kernel(
        root_context=source.root_context,
        period=0,
        definitions=source.definitions,
        root=source.root,
        chunk_size=chunk_size,
    ).with_fixed_params(fixed_flat_params=source.fixed_flat_params)
    function = jax.jit(kernel.programs["_chunks"].function)
    padded = ((state_count + chunk_size - 1) // chunk_size) * chunk_size

    def values(level: FloatND) -> FloatND:
        return function(
            bank=jnp.zeros((padded, 1)),
            **_sharing_chunk_arguments(kernel=kernel, level=level),
        )[0][:state_count, 0]

    def total_value(level: FloatND) -> FloatND:
        return jnp.sum(values(level))

    baseline = values(jnp.asarray(5.0))
    changed, tangent = jax.jvp(values, (jnp.asarray(8.0),), (jnp.asarray(1.0),))
    fresh = (
        _sharing_control_model(state_count=state_count, parameterized=True)
        .solve(
            params={"single_f": {"utility": {"level": 8.0}}},
            log_level="off",
            runtime_checks=False,
        )
        .values[0]["single_f"][..., 0]
    )
    increments = np.linspace(0.0, 2.0, state_count, dtype=baseline.dtype)
    np.testing.assert_array_equal(baseline, 5.0 + increments)
    np.testing.assert_array_equal(changed, 8.0 + increments)
    assert np.asarray(changed).tobytes() == np.asarray(fresh).tobytes()
    np.testing.assert_array_equal(tangent, np.ones(state_count, dtype=changed.dtype))
    reverse = jax.jit(jax.grad(total_value))(jnp.asarray(8.0))
    np.testing.assert_array_equal(reverse, np.asarray(state_count, dtype=changed.dtype))


_NODES = np.array([1.0, 2.0, 3.0])


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    pair: ScalarInt
    alone_f: ScalarInt
    alone_m: ScalarInt


def _no_flow(wage: ContinuousState) -> FloatND:
    return 0.0 * wage


def _move_cost(move: DiscreteAction) -> FloatND:
    return -6.5 * move


def _keep(wage: ContinuousState) -> ContinuousState:
    return wage


def _moved_wage(*, wage: ContinuousState, move: DiscreteAction) -> ContinuousState:
    return jnp.where(move == 0, wage, 1.0)


def _wife(wage: ContinuousState) -> FloatND:
    return (wage - 2.0) ** 2


def _husband(wage: ContinuousState) -> FloatND:
    return 10.0 + 0.0 * wage


def _outside_f(wage: ContinuousState) -> FloatND:
    return 0.5 + 0.0 * wage


def _outside_m(wage: ContinuousState) -> FloatND:
    return wage


def _bend(wage: ContinuousState) -> ContinuousState:
    """Keep wages 1 and 3 and send wage 2 to 4."""
    return wage + 2.0 * (wage - 1.0) * (3.0 - wage)


def _plus_five(wage: ContinuousState) -> ContinuousState:
    return wage + 5.0


def _not_at_two(wage: ContinuousState) -> BoolND:
    return (wage - 2.0) ** 2 > 0.25


def _jolted(*, wage: ContinuousState, jolt: FloatND) -> ContinuousState:
    """Keep the wage at jolt 0; at jolt 1 send wage 2 to 4."""
    return wage + 2.0 * jolt * (wage - 1.0) * (3.0 - wage)


def _bonus_mean(wage: ContinuousState) -> FloatND:
    return wage


def _outside_m_with_bonus(*, wage: ContinuousState, bonus: FloatND) -> FloatND:
    return wage + bonus


def _household_model(
    *,
    rule: OffGrid,
    with_move: bool = False,
    enable_jit: bool = True,
    projection_m: Callable[..., ContinuousState] = _bend,
    projection_f: Callable[..., ContinuousState] = _keep,
    available: Callable[..., BoolND] | None = None,
    pair_shocks: tuple[str, ...] = (),
    alone_m_shocks: tuple[str, ...] = (),
    shocks: dict[str, ShockGroup] | None = None,
    outside_f: Callable[..., FloatND] = _outside_f,
    outside_m: Callable[..., FloatND] = _outside_m,
    initial_nodes: InitialNodes | None = None,
) -> Model:
    """Build a couple whose husband, on leaving, lands beyond his destination grid.

    A source household carries its wage into `pair`, whose wage grid has nodes 1, 2 and
    3. In `pair` the wife's payoff `(wage - 2)**2` beats her outside option of 0.5 at
    nodes 1 and 3 but not at 2, so at wage 2 the household dissolves. The husband's
    composition action projection keeps wages 1 and 3 and sends wage 2 to 4, beyond the
    top node of his destination `alone_m`, whose grid declares `rule`; there his payoff
    is the wage itself. The source puts all weight on the husband. Its optional move
    costs 6.5 and carries the household at wage 1 instead, where both spouses stay and
    the husband gets 10.
    """
    grid = IrregSpacedGrid(points=_NODES)
    source_utility = _move_cost if with_move else _no_flow
    return Model(
        regimes={
            "source": Regime(
                states={"wage": grid},
                actions={"move": DiscreteGrid(values=jnp.array([0, 1]))}
                if with_move
                else {},
                state_transitions={"wage": _moved_wage if with_move else _keep},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": source_utility, "m": source_utility},
                        objective=ParetoObjective(weights={"f": 0.0, "m": 1.0}),
                    )
                },
            ),
            "pair": Regime(
                states={"wage": grid},
                functions={
                    "utility": CollectiveUtility(utilities={"f": _wife, "m": _husband})
                },
                shocks=pair_shocks,
                composition_action=CompositionAction(
                    destinations={
                        "f": Destination(
                            regime="alone_f", projection={"wage": projection_f}
                        ),
                        "m": Destination(
                            regime="alone_m", projection={"wage": projection_m}
                        ),
                    },
                    rule=PCOnActions(tie="leave"),
                    available=available,
                ),
            ),
            "alone_f": Regime(states={"wage": grid}, functions={"utility": outside_f}),
            "alone_m": Regime(
                states={"wage": IrregSpacedGrid(points=_NODES, off_grid=rule)},
                functions={"utility": outside_m},
                shocks=alone_m_shocks,
            ),
        },
        edges={"source": {"pair": 0}},
        shocks={} if shocks is None else shocks,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes=InitialNodes(by_age={0: "source"})
        if initial_nodes is None
        else initial_nodes,
        enable_jit=enable_jit,
    )


_PARAMS = {
    "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "pair": {},
    "alone_f": {},
    "alone_m": {},
}


def _reference(*, rule: OffGrid, with_move: bool) -> tuple[int, float, str, float]:
    """Enumerate the source's moves at wage 2: move, value, final regime and wage.

    Without a move the household reaches `pair` at wage 2 and dissolves, and the
    husband lands at 4 or, capped by `clip_state`, at 3. The move reaches `pair` at
    wage 1, where both spouses stay.
    """
    landing = 3.0 if rule == "clip_state" else 4.0
    candidates = [(0, landing, "alone_m", landing)]
    if with_move:
        candidates.append((1, 10.0 - 6.5, "pair", 1.0))
    return max(candidates, key=lambda candidate: candidate[1])


def _solve_and_simulate(
    *, model: Model, own_members: tuple[str, ...] = ("m",)
) -> tuple[np.ndarray, pd.DataFrame]:
    solution = model.solve(params=_PARAMS, log_level="off")
    n_rows = len(own_members)
    result = model.simulate(
        params=_PARAMS,
        solution=solution,
        initial_conditions={
            "wage": jnp.full(n_rows, 2.0),
            "age": jnp.zeros(n_rows),
            "regime_id": jnp.full(n_rows, model.regime_names_to_ids["source"]),
            "own_member": jnp.asarray(
                [model.member_names_to_ids[name] for name in own_members]
            ),
        },
        seed=0,
        log_level="off",
    )
    panel = result.to_dataframe(use_categorical_codes=True)
    return np.asarray(solution.values[0]["source"]), panel


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("rule", ["extrapolate", "clip_state", "clip_lookup"])
@pytest.mark.parametrize("with_move", [True], ids=["move"])
@pytest.mark.parametrize("enable_jit", [True], ids=["jit"])
def test_composition_action_is_priced_and_realised_under_the_destination_rule(
    *, rule: OffGrid, with_move: bool, enable_jit: bool
) -> None:
    """The source's move and value, and the husband's landing, match the reference.

    Under `clip_state` the dissolution is worth 3, so with a move the source pays
    6.5 to keep the household together at value 3.5; otherwise the husband lands at
    4 and the source keeps its wage.
    """
    values, panel = _solve_and_simulate(
        model=_household_model(rule=rule, with_move=with_move, enable_jit=enable_jit)
    )
    first = panel.loc[panel["period"] == 0].iloc[0]
    last = panel.loc[panel["period"] == 1].iloc[0]
    move, value, regime, wage = _reference(rule=rule, with_move=with_move)
    assert (
        int(first["move"]) if with_move else 0,
        str(last["regime_name"]),
        round(float(values[1, 1]), DECIMAL_PRECISION),
        round(float(last["wage"]), DECIMAL_PRECISION),
    ) == (move, regime, value, wage)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_clip_state_prices_leaving_at_the_value_of_the_realised_landing() -> None:
    """The pair's husband value at wage 2 is his `alone_m` payoff at his landing.

    `alone_m` pays the wage itself, so the price equals the realised landing.
    """
    model = _household_model(rule="clip_state")
    solution = model.solve(params=_PARAMS, log_level="off")
    _, panel = _solve_and_simulate(model=model)
    landing = float(panel.loc[panel["period"] == 1, "wage"].iloc[0])
    np.testing.assert_array_almost_equal(
        solution.values[1]["pair"][1, 1], landing, decimal=DECIMAL_PRECISION
    )


def _stored_read_and_price(*, rule: OffGrid) -> tuple[list[bool], float]:
    """Whether `pair` reads `alone_m` from its stored table, and the price at 2.

    `alone_m` is also a start in period 1, so the solve tabulates its values there.
    """
    model = _household_model(
        rule=rule, initial_nodes=InitialNodes(by_age={0: "source", 1: "alone_m"})
    )
    kernel = cast(
        "_PolicyQueryKernel", model._regimes["pair"].solution.period_kernels[1]
    )
    stored = [
        node.outside_value is not None
        for node in kernel.definitions.values()
        if node.read.target == "alone_m"
    ]
    solution = model.solve(params=_PARAMS, log_level="off")
    return stored, round(float(solution.values[1]["pair"][1, 1]), DECIMAL_PRECISION)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    ("rule", "expected"),
    [("extrapolate", ([False], 4.0)), ("clip_state", ([True], 3.0))],
)
def test_capped_landings_are_read_from_the_stored_destination_table(
    *, rule: OffGrid, expected: tuple[list[bool], float]
) -> None:
    """Capped, every landing is a grid node, so the read gathers stored values.

    Uncapped, wage 2 lands at 4, off the grid, and the read is evaluated at that
    point instead. Either way the price is the destination's value at the landing
    the simulation realises.
    """
    assert _stored_read_and_price(rule=rule) == expected


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_raise_refuses_an_active_off_grid_composition_action_read() -> None:
    """At wage 2 the husband's outside read lands at 4, beyond `alone_m`'s grid."""
    with pytest.raises(
        OffGridStateError,
        match=r"regime='pair', target='alone_m', period=1, state='wage', "
        r"phase='solve'",
    ):
        _household_model(rule="raise").solve(params=_PARAMS, log_level="off")


def _worse_than_staying(wage: ContinuousState) -> FloatND:
    return -1.0 + 0.0 * wage


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_raise_refuses_an_off_grid_read_the_household_does_not_act_on() -> None:
    """The wife always stays and so does the husband, yet his read at 4 is priced."""
    model = _household_model(rule="raise", outside_f=_worse_than_staying)
    with pytest.raises(OffGridStateError, match=r"target='alone_m'"):
        model.solve(params=_PARAMS, log_level="off")


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_raise_ignores_an_off_grid_read_where_leaving_is_unavailable() -> None:
    """Leaving is closed at wage 2, so the couple stays and nothing is read."""
    model = _household_model(rule="raise", available=_not_at_two)
    solution = model.solve(params=_PARAMS, log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[1]["pair"][1], [0.0, 10.0], decimal=DECIMAL_PRECISION
    )


def _jolt_shocks(*, weights: tuple[float, float]) -> dict[str, ShockGroup]:
    return {
        "jolts": ShockGroup(
            shocks={"jolt": Normal(mu=0.0, sigma=1.0)},
            nodes=UserNodes(
                nodes=np.asarray([[0.0], [1.0]]), weights=np.asarray(weights)
            ),
        )
    }


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_raise_ignores_an_off_grid_read_that_carries_no_mass() -> None:
    """Only the jolt node of weight 0 sends the husband off the grid."""
    model = _household_model(
        rule="raise",
        projection_m=_jolted,
        pair_shocks=("jolts",),
        shocks=_jolt_shocks(weights=(1.0, 0.0)),
    )
    solution = model.solve(params=_PARAMS, log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[1]["pair"][1], [0.5, 2.0], decimal=DECIMAL_PRECISION
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_raise_refuses_an_off_grid_read_that_carries_mass() -> None:
    """The same read with half its mass on the off-grid jolt node fails."""
    model = _household_model(
        rule="raise",
        projection_m=_jolted,
        pair_shocks=("jolts",),
        shocks=_jolt_shocks(weights=(0.5, 0.5)),
    )
    with pytest.raises(OffGridStateError, match=r"target='alone_m'"):
        model.solve(params=_PARAMS, log_level="off")


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    ("rule", "expected"), [("extrapolate", 8.0), ("clip_state", 6.0)]
)
def test_hidden_destination_law_reads_the_landing_the_composition_action_prices(
    *, rule: OffGrid, expected: float
) -> None:
    """A bonus with mean equal to the wage makes `alone_m` worth twice the wage.

    The pair does not observe the bonus, so its read integrates the bonus's law at
    the husband's landing: 2 * 4 uncapped, 2 * 3 capped. A law read at the uncapped
    landing beside a payoff at the capped one would give 7.
    """
    model = _household_model(
        rule=rule,
        alone_m_shocks=("bonuses",),
        shocks={
            "bonuses": ShockGroup(
                shocks={"bonus": Normal(mu=_bonus_mean, sigma=1.0)},
                nodes=Product(bonus=GaussHermite(2)),
            )
        },
        outside_m=_outside_m_with_bonus,
    )
    solution = model.solve(params=_PARAMS, log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[1]["pair"][1, 1], expected, decimal=DECIMAL_PRECISION
    )


@categorical(ordered=False)
class FormationId:
    single: ScalarInt
    incoming_m: ScalarInt
    couple: ScalarInt
    alone_f: ScalarInt
    alone_m: ScalarInt


def _single_stay(wage: ContinuousState) -> FloatND:
    return 0.0 * wage


def _incoming_stay() -> float:
    return 2.0


def _couple_f(wage: ContinuousState) -> FloatND:
    return 1.0 + 0.0 * wage


def _couple_m(wage: ContinuousState) -> FloatND:
    return 3.5 + 0.0 * wage


def _formation_model(*, rule: Literal["extrapolate", "clip_state"]) -> Model:
    """A single woman may form a couple by consent with an incoming man.

    In the couple the husband stays only where his payoff of 3.5 beats his outside
    value in `alone_m`, read at the landing his dissolution composition action projects.
    """
    grid = IrregSpacedGrid(points=_NODES)
    return Model(
        regimes={
            "single": Regime(
                states={"wage": grid},
                functions={"utility": _single_stay},
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="couple", role="f", projection={"wage": _keep}
                        )
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="incoming_m", role="m", projection={}
                        )
                    ),
                ),
            ),
            "incoming_m": Regime(functions={"utility": _incoming_stay}),
            "couple": Regime(
                states={"wage": grid},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _couple_f, "m": _couple_m}
                    )
                },
                composition_action=CompositionAction(
                    destinations={
                        "f": Destination(regime="alone_f", projection={"wage": _keep}),
                        "m": Destination(regime="alone_m", projection={"wage": _bend}),
                    },
                    rule=PCOnActions(),
                ),
            ),
            "alone_f": Regime(states={"wage": grid}, functions={"utility": _outside_f}),
            "alone_m": Regime(
                states={"wage": IrregSpacedGrid(points=_NODES, off_grid=rule)},
                functions={"utility": _outside_m},
            ),
        },
        n_periods=1,
        regime_id_class=FormationId,
        initial_nodes=InitialNodes(by_period={0: "single"}),
        edges={},
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    ("rule", "expected"),
    [("extrapolate", [1.0, 0.0, 1.0]), ("clip_state", [1.0, 1.0, 1.0])],
)
def test_consent_formation_prices_the_couple_at_the_capped_dissolution_landing(
    *, rule: Literal["extrapolate", "clip_state"], expected: list[float]
) -> None:
    """At wage 2 the husband's outside value is 4 uncapped and 3 capped.

    Uncapped he would leave, the couple keeps no action and the single woman stays
    single at 0. Capped he stays, both consent, and she gets her couple payoff 1.
    """
    solution = _formation_model(rule=rule).solve(params={}, log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[0]["single"][..., 0], expected, decimal=DECIMAL_PRECISION
    )
