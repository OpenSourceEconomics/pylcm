"""Public exits compare actual mapped common policies, with independent anchors."""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import lcm
from _lcm.current_iid import CurrentIIDDrawPlan
from _lcm.execution.core_program import CoreBuildContext
from _lcm.exit_queries import _compose, _QueryGraphBuilder, bind_exit_query_graph
from _lcm.policy_queries import QueryKind, ScalarPolicyQueries, _PolicyQueryKernel
from _lcm.solution.fingerprint import _semantic_fingerprint
from _lcm.typing import PytreeValue
from lcm import (
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidParamsError,
    RegimeInitializationError,
    ShockDeclarationError,
    UnsupportedOperationError,
)
from lcm.exits import ExitRule
from lcm.shocks import Discrete, Normal, PhysicalValues, ShockGroup, UserNodes
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.transition import fixed_transition
from lcm.typing import BoolND, FloatND, IntND, Period, ScalarInt, UserFunction


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
                exits={
                    "marry": Exit(
                        targets={
                            "self": ExitTarget(
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
                    )
                },
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
                exits={
                    "divorce": Exit(
                        targets={
                            "f": ExitTarget(
                                regime="single_f",
                                projection={
                                    "wealth": _school_projection
                                    if outside_lag
                                    else _wealth
                                },
                                utility_adjustment=_cost,
                            ),
                            "m": ExitTarget(
                                regime="single_m",
                                projection={
                                    "wealth": _school_projection
                                    if outside_lag
                                    else _wealth
                                },
                                utility_adjustment=_cost,
                            ),
                        },
                        rule="unilateral",
                    )
                },
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
                "exits": {
                    "divorce": {
                        "targets": {
                            "f": {"utility_adjustment": {"cost": 0.0}},
                            "m": {"utility_adjustment": {"cost": 2.0}},
                        }
                    }
                }
            }
        },
    )


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
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


def test_public_query_only_targets_need_contexts_without_full_banks() -> None:
    """Synthetic spouse/target policies retain Q/F without stored target Full."""
    model = _household(newlywed=True, enable_jit=True, source_only=True)
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(result.values[0]["single_f"], 6.0)
    assert set(result.values[0]) == {"single_f"}
    assert model._regimes["couple"].active_periods == ()
    assert model._regimes["single_m"].active_periods == ()
    assert model.graph.visited_nodes == frozenset({(0, "single_f")})


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


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("fixed", [False, True])
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


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("fixed", [False, True])
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
    state_conditioned: bool = False,
) -> Model:
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    return Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                functions={"utility": _stay_with_offer if shared else _stay},
                shocks=("offers",) if shared else (),
                exits={
                    "move": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="single_m",
                                projection={
                                    "wealth": _opposite if flipped else _wealth
                                },
                                utility_adjustment=_fee,
                            )
                        },
                        rule="unilateral",
                        observes=("offer",) if observed else (),
                    )
                },
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
                        probs=_offer_probs
                        if shared or state_conditioned
                        else np.asarray([0.5, 0.5]),
                    )
                },
            )
        },
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
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
    np.testing.assert_allclose(result.values[0]["single_f"], [0.75, 1.5])


def test_shared_offer_refuses_a_projection_that_changes_its_cdf() -> None:
    with pytest.raises(ShockDeclarationError, match="effective identity is unproved"):
        _offer_model(observed=False, shared=True, flipped=True)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
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


def _settlement(wealth: FloatND) -> FloatND:
    return 2 * wealth - 1


def _unsafe_scale(wealth: FloatND) -> FloatND:
    return jnp.sqrt(wealth)


def _signal(*, eps: FloatND, wealth: FloatND) -> FloatND:
    return eps + 0 * wealth


def test_unavailable_observer_guards_unsafe_mapped_law() -> None:
    """At source0 the target law is sqrt(-1), behind a false source guard."""
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    model = Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                functions={"utility": _stay},
                exits={
                    "move": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="single_m", projection={"wealth": _settlement}
                            )
                        },
                        rule="unilateral",
                        observes=("eps",),
                        available=_positive,
                    )
                },
            ),
            "single_m": Regime(
                states={"wealth": wealth},
                functions={"utility": _signal},
                shocks=("signal",),
            ),
        },
        shocks={
            "signal": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=_unsafe_scale)},
                nodes=UserNodes(
                    nodes=np.asarray([[-1.0], [1.0]]), weights=np.asarray([0.5, 0.5])
                ),
            )
        },
        regime_id_class=_Singles,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(result.values[0]["single_f"], [0.5, 0.75])
    kernel = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    scalar = ScalarPolicyQueries(
        definitions=kernel.definitions, root=kernel.root, period=0
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

    def price(wealth: FloatND) -> FloatND:
        return scalar(
            states={"wealth": wealth},
            arguments=cast("Mapping[str, Mapping[str, PytreeValue]]", arguments),
        )[0]

    @jax.jit
    def prices(points: FloatND) -> tuple[FloatND, FloatND]:
        return jax.lax.map(jax.value_and_grad(price), points)

    values, derivatives = prices(jnp.asarray([0.0, 1.0]))
    np.testing.assert_allclose(values, [0.5, 0.75])
    np.testing.assert_allclose(derivatives, [0.0, 0.5])


def _taste_target(*, action: IntND, wealth: FloatND) -> FloatND:
    return jnp.log(3.0) * action + 0 * wealth


def _taste_stay(wealth: FloatND) -> FloatND:
    return 1.2 + 0 * wealth


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_hidden_target_taste_is_priced_at_logsum_then_drawn_when_acting() -> None:
    """log(4)>1.2>log(3): a hard target max would make the wrong exit decision."""
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    model = Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                functions={"utility": _taste_stay},
                exits={
                    "move": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="single_m", projection={"wealth": _wealth}
                            )
                        },
                        rule="unilateral",
                    )
                },
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
    definitions, root = bind_exit_query_graph(
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


def _source_taste(*, action: IntND, consumption: FloatND, wealth: FloatND) -> FloatND:
    return 2 * action - 0.25 * (1 - consumption) ** 2 + 0 * wealth


def _outside15(wealth: FloatND) -> FloatND:
    return 1.5 + 0 * wealth


def _outside3(wealth: FloatND) -> FloatND:
    return 3.0 + 0 * wealth


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("fallback", [False, True])
def test_source_full_taste_preserves_dissolution_fallback(*, fallback: bool) -> None:
    """Dissolution cannot acquire a taste-based escape from surviving kept stays."""
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    model = Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                actions={
                    "action": DiscreteGrid(values=jnp.asarray([0, 1])),
                    "consumption": LinSpacedGrid(start=0, stop=1, n_points=2),
                },
                functions={"utility": _source_taste},
                taste_shocks=ExtremeValueTasteShocks(),
                exits={
                    "move": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="single_m", projection={"wealth": _wealth}
                            )
                        },
                        rule="unilateral",
                    )
                },
            ),
            "single_m": Regime(
                states={"wealth": wealth},
                functions={"utility": _outside3 if fallback else _outside15},
            ),
        },
        regime_id_class=_Singles,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        fixed_params={"single_f": {"taste_shocks": {"scale": 1.0}}},
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(result.values[0]["single_f"], 3.0 if fallback else 2.0)
    kernel = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    scalar = ScalarPolicyQueries(
        definitions=kernel.definitions, root=kernel.root, period=0
    )
    arguments = dict(
        scalar.arguments(
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
    )
    for seed in (0, 1, 2, 3):
        key = jax.random.key(seed)
        call_arguments = {
            **arguments,
            kernel.root: {**arguments[kernel.root], "taste_shock_key": key},
        }
        value, action, support, valid, landings = jax.jit(scalar)(
            states={"wealth": jnp.asarray(0.0)}, arguments=call_arguments
        )
        assert landings.dtype == jnp.int32
        np.testing.assert_array_equal(landings, np.zeros(landings.shape, np.int32))
        # Reserve exit tasteID2 even when its escape is economically inadmissible.
        # The surviving discrete stay optimizes consumption to global stayID3.
        roster = jnp.asarray(
            [-jnp.inf, -jnp.inf, 3.0] if fallback else [-jnp.inf, 2.0, -jnp.inf]
        )
        selected = int(jnp.argmax(roster + jax.random.gumbel(key, (3,))))
        assert int(action) == (3 if selected == 1 else 4)
        np.testing.assert_allclose(value, 2.0 if selected == 1 else 3.0)
        assert bool(support)
        assert bool(valid)


def _joint_value(*, action: IntND, x: FloatND, y: FloatND, wealth: FloatND) -> FloatND:
    return action * (x + y) + 0 * wealth


def _quarter(wealth: FloatND) -> FloatND:
    return 0.25 + 0 * wealth


def _joint_information_model(
    *,
    observed: bool,
    correlated: bool,
    deterministic: bool,
    complete: bool = False,
    runtime: bool = False,
) -> Model:
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    return Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                functions={"utility": _quarter},
                exits={
                    "move": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="single_m", projection={"wealth": _wealth}
                            )
                        },
                        rule="unilateral",
                        observes=("x", "y") if complete else ("x",) if observed else (),
                    )
                },
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
                shocks={
                    "x": Normal(mu=0, sigma=0 if deterministic else 1),
                    "y": Normal(mu=0, sigma=1),
                },
                **(
                    {"correlation": None}
                    if runtime
                    else {"correlation": ((1.0, 0.5), (0.5, 1.0))}
                    if correlated
                    else {}
                ),
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
    result = _joint_information_model(
        observed=False, correlated=correlated, deterministic=False
    ).solve(params={}, log_level="off", runtime_checks=False)
    # Rows are (-1,-1),(1,1). Correlation transforms y=.5*x+sqrt(.75)*base_y.
    expected = 0.5 * (1.5 + np.sqrt(0.75)) if correlated else 1.0
    np.testing.assert_allclose(result.values[0]["single_f"], expected)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_independent_observation_projects_hidden_rows_without_empirical_filtering() -> (
    None
):
    """At observed x=-1/1, integrating original y rows gives prices0/1."""
    result = _joint_information_model(
        observed=True, correlated=False, deterministic=False
    ).solve(params={}, log_level="off", runtime_checks=False)
    # Source averages max(.25,0) and max(.25,1), rather than matching row IDs.
    np.testing.assert_allclose(result.values[0]["single_f"], 0.625)


@pytest.mark.parametrize("correlated", [False, True])
def test_deterministic_observation_does_not_reveal_a_hidden_base(
    *, correlated: bool
) -> None:
    """A constant x=0 never reveals the retained address underlying that output."""
    result = _joint_information_model(
        observed=True, correlated=correlated, deterministic=True
    ).solve(params={}, log_level="off", runtime_checks=False)
    expected = 0.5 * (0.5 + np.sqrt(0.75)) if correlated else 0.5
    np.testing.assert_allclose(result.values[0]["single_f"], expected)


def test_partial_correlated_observation_refuses_mandatorily() -> None:
    """An informative x with hidden correlated y needs deferred conditioning."""
    model = _joint_information_model(
        observed=True, correlated=True, deterministic=False
    )
    with pytest.raises(ExecutionPlanningError, match="unresolved numerical status"):
        model.solve(params={}, log_level="off", runtime_checks=False)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_runtime_observation_checks_the_actual_admitted_correlation_each_call() -> None:
    """The support marker stays engine-owned as actual identity changes per call."""
    model = _joint_information_model(
        observed=True, correlated=False, deterministic=False, runtime=True
    )
    result = model.solve(
        params={"joint": {"corr": np.eye(2)}},
        log_level="off",
        runtime_checks=False,
    )
    np.testing.assert_allclose(result.values[0]["single_f"], 0.625)
    with pytest.raises(ExecutionPlanningError, match="unresolved numerical status"):
        model.solve(
            params={"joint": {"corr": np.array([[1.0, 0.5], [0.5, 1.0]])}},
            log_level="off",
            runtime_checks=False,
        )


def test_complete_correlated_observation_reuses_its_actual_joint_realization() -> None:
    """The source sees both coordinates, so no conditional Gaussian rule is needed."""
    result = _joint_information_model(
        observed=True, correlated=True, deterministic=False, complete=True
    ).solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(
        result.values[0]["single_f"], 0.125 + 0.5 * (1.5 + np.sqrt(0.75))
    )


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_nested_outside_projection_reads_the_already_mapped_marital_history() -> None:
    """Newlywed lag0 exposes schooling7; incumbent lag1 outside remains5."""
    model = _household(newlywed=True, enable_jit=True, outside_lag=True)
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    # At lag0 neither couple action gives both members their schooling outside7.
    # A root source must therefore stay, rather than price the lag1 policy.
    np.testing.assert_allclose(result.values[0]["single_f"], [5.0, 7.0])
    couple = model._regimes["couple"]
    by_lag = np.moveaxis(
        np.asarray(result.values[0]["couple"]),
        couple.solution.state_names.index("married_last_period"),
        0,
    )
    np.testing.assert_allclose(by_lag[0], np.broadcast_to([7.0, 7.0], by_lag[0].shape))
    np.testing.assert_allclose(by_lag[1], np.broadcast_to([20.0, 4.0], by_lag[1].shape))


@pytest.mark.parametrize("different_context", [False, True])
def test_target_only_observation_has_an_unambiguous_producing_cdf(
    *, different_context: bool
) -> None:
    """A shared observed category cannot be assigned two mapped probability laws."""
    base = _offer_model(observed=True, state_conditioned=True)
    source = base.user_regimes["single_f"]
    rule = _StrictRule()
    second = Exit(
        targets={
            "self": ExitTarget(
                regime="single_m",
                projection={"wealth": _opposite if different_context else _wealth},
            )
        },
        rule=rule,
        observes=("offer",),
    )
    regimes = {
        **base.user_regimes,
        "single_f": replace(
            source,
            exits={"move": replace(source.exits["move"], rule=rule), "another": second},
        ),
    }
    if different_context:
        with pytest.raises(RegimeInitializationError, match="producing law contexts"):
            Model(
                regimes=regimes,
                shocks=base.shocks,
                regime_id_class=_Singles,
                n_periods=1,
                edges={},
                initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
                execution_config=ExecutionConfig(device_memory_bytes=None),
            )
    else:
        result = Model(
            regimes=regimes,
            shocks=base.shocks,
            regime_id_class=_Singles,
            n_periods=1,
            edges={},
            initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
            execution_config=ExecutionConfig(device_memory_bytes=None),
        ).solve(params={}, log_level="off", runtime_checks=False)
        # The first exit pays -.25; the second pays no adjustment.
        np.testing.assert_allclose(result.values[0]["single_f"], [0.8, 1.7])


def _one() -> float:
    return 1.0


def _ten() -> float:
    return 10.0


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_query_only_target_retains_its_genuine_next_period_value_read() -> None:
    """Current Part is transient; its future continuation is a stored Full bank."""
    model = Model(
        regimes={
            "single_f": Regime(
                functions={"utility": _one},
                exits={
                    "move": Exit(
                        targets={"self": ExitTarget(regime="single_m", projection={})},
                        rule="unilateral",
                    )
                },
            ),
            "single_m": Regime(functions={"utility": _one}),
            "couple": Regime(functions={"utility": _ten}),
        },
        regime_id_class=_Household,
        n_periods=2,
        edges={"single_m": {"couple": lcm.Periods(values=(0,))}},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    result = model.solve(
        params={"discount_factor": 0.9}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(result.values[0]["single_f"], 10.0)
    np.testing.assert_allclose(result.values[1]["couple"], 10.0)
    assert set(result.values[0]) == {"single_f"}
    assert set(result.values[1]) == {"couple"}
    assert model._regimes["single_m"].active_periods == ()


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_null_shared_category_keeps_the_query_boundary_probability_derivative() -> None:
    """The null solve category retains its finite target policy without 0/0."""
    base = _offer_model(observed=False, shared=True)
    model = Model(
        regimes=base.user_regimes,
        shocks={"offers": ShockGroup(shocks={"offer": Discrete(categories=_Offer)})},
        regime_id_class=_Singles,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    result = model.solve(
        params={"offers": {"offer__probs": np.asarray([1.0, 0.0])}},
        log_level="off",
        runtime_checks=False,
    )
    np.testing.assert_allclose(result.values[0]["single_f"], 0.5)
    kernel = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    scalar = ScalarPolicyQueries(
        definitions=kernel.definitions, root=kernel.root, period=0
    )

    def price(probability: FloatND) -> FloatND:
        arguments = scalar.arguments(
            context=CoreBuildContext(
                state_action_space=kernel.root_context.state_action_space,
                next_regime_to_V_arr={},
                next_regime_to_continuation={},
                flat_params=MappingProxyType(
                    {
                        "offers": MappingProxyType(
                            {"offer__probs": jnp.stack([1 - probability, probability])}
                        )
                    }
                ),
                period=0,
                ages=kernel.root_context.ages,
            ),
            fixed_flat_params=kernel.fixed_flat_params,
        )
        return scalar(
            states={"wealth": jnp.asarray(0.0)},
            arguments=cast("Mapping[str, Mapping[str, PytreeValue]]", arguments),
        )[0]

    value, derivative = jax.jit(jax.value_and_grad(price))(jnp.asarray(0.0))
    np.testing.assert_allclose(value, 0.5)
    # Prices are .5 and 1.75 after source choice; dp*(1.75-.5) survives at p=0.
    np.testing.assert_allclose(derivative, 1.25)


def _unused_helper(toll: FloatND) -> FloatND:
    return toll


def _ignored_formation_adjustment(*, unused_helper: FloatND, fee: FloatND) -> FloatND:
    return unused_helper + fee


def _reporter(report_toll: FloatND) -> FloatND:
    return report_toll


@pytest.mark.parametrize("preserve", [False, True])
def test_raw_counterparty_omits_unread_formation_params(*, preserve: bool) -> None:
    """A synthetic Raw/Part spouse keeps declarations, but never calls Consent."""
    base = _household(newlywed=True, enable_jit=True, source_only=True)
    spouse = base.user_regimes["single_m"]
    ignored = Exit(
        targets={
            "self": ExitTarget(
                regime="couple",
                role="m",
                projection={"wealth": _wealth, "married_last_period": _newlywed},
                utility_adjustment=_ignored_formation_adjustment,
            )
        },
        rule=Consent(
            counterparty=Counterparty(
                regime="single_f", role="f", projection={"wealth": _wealth}
            )
        ),
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
                exits={"marry": ignored},
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
    assert "exits" not in template
    assert "unused_helper" not in template
    if preserve:
        assert template["reporter"] == {"report_toll": "FloatND"}
    else:
        assert "reporter" not in template
    assert "marry" in model.user_regimes["single_m"].exits
    assert model._model_structure_fingerprint != base._model_structure_fingerprint
    result = model.solve(
        params={"single_m": {"reporter": {"report_toll": 0.0}}} if preserve else {},
        log_level="off",
        runtime_checks=False,
    )
    np.testing.assert_allclose(result.values[0]["single_f"], 6.0)


class _StrictRule:
    reads_outside_values: ClassVar[bool] = True

    def __call__(
        self,
        *,
        Q_self: FloatND,
        exit_payoffs: Mapping[str, Mapping[str, FloatND]],
        availability: Mapping[str, BoolND],
        alternative_ids: IntND,
    ) -> lcm.RuleResult:
        outside = jnp.stack(
            [
                jnp.where(availability[name], value["self"], -jnp.inf)
                for name, value in exit_payoffs.items()
            ]
        )
        stay = Q_self.reshape(-1)
        kept = stay > jnp.max(outside)
        stay_id = jnp.argmax(jnp.where(kept, stay, -jnp.inf))
        exit_id = stay.size + jnp.argmax(outside)
        return lcm.RuleResult(
            action=alternative_ids[jnp.where(jnp.any(kept), stay_id, exit_id)]
        )


def _strict_stay(*, action: IntND, wealth: FloatND) -> FloatND:
    return action + 0 * wealth


def _rule_model(
    *,
    rule: ExitRule,
    second_rule: ExitRule | None = None,
    source_taste: bool = False,
    target_taste: bool = False,
    available: UserFunction | None = None,
) -> Model:
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    target = ExitTarget(regime="single_m", projection={"wealth": _wealth})
    exits = {"move": Exit(targets={"self": target}, rule=rule, available=available)}
    if second_rule is not None:
        exits["another"] = Exit(targets={"self": target}, rule=second_rule)
    return Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _strict_stay},
                exits=exits,
                taste_shocks=ExtremeValueTasteShocks() if source_taste else None,
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
        fixed_params={
            **({"single_f": {"taste_shocks": {"scale": 1.0}}} if source_taste else {}),
            **({"single_m": {"taste_shocks": {"scale": 1.0}}} if target_taste else {}),
        },
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _single_tie(wealth: FloatND) -> FloatND:
    return 1.0 + 0 * wealth


def test_strict_custom_tie_has_empty_participating_policy_and_full_exit() -> None:
    """Strict source choice rejects the tied stay; Raw still chooses stayID1."""
    model = _rule_model(rule=_StrictRule())
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
        definitions, root = bind_exit_query_graph(
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


def test_shared_custom_rule_is_called_over_the_global_alternative_roster() -> None:
    """Multiple exits share one authored selector and one stable ID space."""
    rule = _StrictRule()
    result = _rule_model(rule=rule, second_rule=rule).solve(
        params={}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(result.values[0]["single_f"], 1.0)
    with pytest.raises(RegimeInitializationError, match="same authored rule object"):
        _rule_model(rule=_StrictRule(), second_rule=_StrictRule())


def test_custom_source_refuses_its_unsupported_ev1_integration() -> None:
    """A general selector cannot be silently replaced by canonical logsum."""
    with pytest.raises(
        UnsupportedOperationError, match="arbitrary custom exit selector"
    ):
        _rule_model(rule=_StrictRule(), source_taste=True)
    result = _rule_model(rule=_StrictRule(), target_taste=True).solve(
        params={}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(result.values[0]["single_f"], np.log(4.0))


@dataclass(frozen=True, kw_only=True)
class _ReturnedRule:
    action: PytreeValue
    reads_outside_values: ClassVar[bool] = False

    def __call__(self) -> lcm.RuleResult:
        # Deliberately malformed public payloads exercise the native boundary.
        return lcm.RuleResult(action=cast("int", self.action))


def _never_available() -> bool:
    return False


def _vector_available() -> BoolND:
    return jnp.asarray([False, True])


@pytest.mark.parametrize(
    ("action", "error", "message"),
    [
        (np.int64(2**32), BeartypeCallHintParamViolation, "RuleResult.__init__"),
        (
            np.asarray(2**32, dtype=np.int64),
            BeartypeCallHintParamViolation,
            "RuleResult.__init__",
        ),
        (np.int64(7), BeartypeCallHintParamViolation, "RuleResult.__init__"),
        (1.0, BeartypeCallHintParamViolation, "RuleResult.__init__"),
        (True, ValueError, "scalar integer"),
        (np.asarray([0]), BeartypeCallHintParamViolation, "RuleResult.__init__"),
        (2**32, ValueError, "known global alternative"),
        (7, ValueError, "known global alternative"),
    ],
)
def test_custom_rule_validates_host_ids_before_active_dtype_narrowing(
    *, action: PytreeValue, error: type[Exception], message: str
) -> None:
    """A host int64 ID cannot wrap to an admitted stay when x64 is disabled."""
    with pytest.raises(error, match=message):
        _rule_model(rule=_ReturnedRule(action=action)).solve(
            params={}, log_level="off", runtime_checks=False
        )


@dataclass(frozen=True)
class _ReturnedVectorRule:
    reads_outside_values: ClassVar[bool] = False

    def __call__(self) -> lcm.RuleResult:
        return lcm.RuleResult(action=jnp.asarray([0], dtype=jnp.int32))


def test_custom_rule_refuses_a_supported_integer_array_with_vector_shape() -> None:
    with pytest.raises(ValueError, match="scalar integer"):
        _rule_model(rule=_ReturnedVectorRule()).solve(
            params={}, log_level="off", runtime_checks=False
        )


def test_custom_rule_cannot_select_an_unavailable_exit() -> None:
    """A declared global ID still requires current support and availability."""
    model = _rule_model(rule=_ReturnedRule(action=2), available=_never_available)
    with pytest.raises(ExecutionPlanningError, match="unresolved numerical status"):
        model.solve(params={}, log_level="off", runtime_checks=False)


def test_public_availability_accepts_python_scalar_but_refuses_vector_reduction() -> (
    None
):
    """False is a scalar guard; a vector cannot silently become all(vector)."""
    result = _rule_model(rule="unilateral", available=_never_available).solve(
        params={}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(result.values[0]["single_f"], 1.0)
    with pytest.raises(ValueError, match="scalar boolean"):
        _rule_model(rule="unilateral", available=_vector_available).solve(
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
    formation = replace(source.exits["marry"], observes=("offer",) if observed else ())
    couple = base.user_regimes["couple"]
    regimes = {
        **base.user_regimes,
        "single_f": replace(source, exits={"marry": formation}),
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


def test_statically_false_exit_does_not_require_an_inactive_target_context() -> None:
    """A proof of false guards the same missing context admitted by graph validation."""
    model = Model(
        regimes={
            "single_f": Regime(
                functions={"utility": _one},
                exits={
                    "move": Exit(
                        targets={"self": ExitTarget(regime="single_m", projection={})},
                        rule="unilateral",
                        available=_never_available,
                    )
                },
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
    declaration = source.exits["marry"]
    formation = replace(
        declaration,
        available=_positive,
        targets={
            "self": replace(
                declaration.targets["self"],
                projection={"wealth": _settlement, "married_last_period": _newlywed},
            )
        },
    )
    couple = base.user_regimes["couple"]
    model = Model(
        regimes={
            **base.user_regimes,
            "single_f": replace(source, exits={"marry": formation}),
            "couple": replace(
                couple,
                exits={
                    "divorce": replace(
                        couple.exits["divorce"], available=_unsafe_available
                    )
                },
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
    np.testing.assert_allclose(result.values[0]["single_f"], [5.0, 6.0])


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("reverse", [False, True])
def test_observed_aliases_share_the_active_producing_context(*, reverse: bool) -> None:
    """An inactive first alias cannot hide a signal required by an active exit."""
    base = _offer_model(observed=True, state_conditioned=True)
    source = base.user_regimes["single_f"]
    rule = _StrictRule()
    model_exits = {
        "first": replace(source.exits["move"], available=_positive, rule=rule),
        "second": Exit(
            targets={
                "self": ExitTarget(regime="single_m", projection={"wealth": _wealth})
            },
            rule=rule,
            observes=("offer",),
        ),
    }
    if reverse:
        model_exits = dict(reversed(tuple(model_exits.items())))
    model = Model(
        regimes={
            **base.user_regimes,
            "single_f": replace(source, exits=model_exits),
        },
        shocks=base.shocks,
        regime_id_class=_Singles,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(result.values[0]["single_f"], [0.8, 1.7])


def _poison_settlement(wealth: FloatND) -> FloatND:
    return jnp.sqrt(wealth - 1)


def _alias_signal(*, eps: FloatND, settlement: FloatND, wealth: FloatND) -> FloatND:
    return eps + settlement + 0 * wealth


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("both_inactive", [False, True])
def test_observer_aliases_skip_unrelated_inactive_settlements(
    *, both_inactive: bool
) -> None:
    """The canonical alias's unsafe extra state never feeds its physical law."""
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    rule = _StrictRule()
    model = Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                functions={"utility": _stay},
                exits={
                    "first": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="single_m",
                                projection={
                                    "wealth": _wealth,
                                    "settlement": _poison_settlement,
                                },
                            )
                        },
                        rule=rule,
                        observes=("eps",),
                        available=_positive,
                    ),
                    "second": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="single_m",
                                projection={"wealth": _wealth, "settlement": _zero},
                            )
                        },
                        rule=rule,
                        observes=("eps",),
                        available=_positive if both_inactive else None,
                    ),
                },
            ),
            "single_m": Regime(
                states={"wealth": wealth, "settlement": wealth},
                functions={"utility": _alias_signal},
                shocks=("signal",),
            ),
        },
        shocks={
            "signal": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=1)},
                nodes=UserNodes(
                    nodes=np.asarray([[-1.0], [1.0]]),
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
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(
        result.values[0]["single_f"], [0.5 if both_inactive else 0.75, 0.75]
    )


def _formation_f(wealth: FloatND) -> FloatND:
    return 3.0 + 0 * wealth


def _formation_m(wealth: FloatND) -> FloatND:
    return 1.0 + 0 * wealth


def _zero(wealth: FloatND) -> FloatND:
    return 0.0 * wealth


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
def test_source_taste_competes_with_an_eligible_consent_formation() -> None:
    """A strict eligible formation joins the source's discrete-combination roster."""
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    model = Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                actions={
                    "action": DiscreteGrid(values=jnp.asarray([0, 1])),
                    "consumption": LinSpacedGrid(start=0, stop=1, n_points=2),
                },
                functions={"utility": _source_taste},
                taste_shocks=ExtremeValueTasteShocks(),
                exits={
                    "marry": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="couple",
                                role="f",
                                projection={"wealth": _wealth},
                            )
                        },
                        rule=Consent(
                            counterparty=Counterparty(
                                regime="single_m",
                                role="m",
                                projection={"wealth": _wealth},
                            )
                        ),
                    )
                },
            ),
            "single_m": Regime(states={"wealth": wealth}, functions={"utility": _zero}),
            "couple": Regime(
                states={"wealth": wealth},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _formation_f, "m": _formation_m}
                    )
                },
            ),
        },
        regime_id_class=_Household,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        fixed_params={"single_f": {"taste_shocks": {"scale": 1.0}}},
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    # Consent uses existing Raw logsum(0,2)<3 and counterparty Raw0<1.
    expected_price = np.log(np.exp(0.0) + np.exp(2.0) + np.exp(3.0))
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(result.values[0]["single_f"], expected_price)
    kernel = model._regimes["single_f"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    scalar = ScalarPolicyQueries(
        definitions=kernel.definitions, root=kernel.root, period=0
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
    literal_routes = set()
    for seed in range(16):
        key = jax.random.key(seed)
        selected = int(
            jnp.argmax(jnp.asarray([0.0, 2.0, 3.0]) + jax.random.gumbel(key, (3,)))
        )
        literal_routes.add(selected == 2)
        value, action, support, valid, landings = jax.jit(scalar)(
            states={"wealth": jnp.asarray(0.0)},
            arguments={
                **arguments,
                kernel.root: {**arguments[kernel.root], "taste_shock_key": key},
            },
        )
        assert landings.dtype == jnp.int32
        np.testing.assert_array_equal(landings, np.zeros(landings.shape, np.int32))
        assert int(action) == (1, 3, 4)[selected]
        np.testing.assert_allclose(value, (0.0, 2.0, 3.0)[selected])
        assert bool(support)
        assert bool(valid)
    assert literal_routes == {False, True}


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("n_points", [64, 65])
def test_custom_outside_choice_ignores_padded_query_lanes(*, n_points: int) -> None:
    """A declared outside choice keeps every live source value on a partial chunk."""
    wealth = LinSpacedGrid(start=0.0, stop=1.0, n_points=n_points)
    model = Model(
        regimes={
            "single_f": Regime(
                states={"wealth": wealth},
                actions={
                    "action": DiscreteGrid(values=np.asarray([0], dtype=np.int32))
                },
                functions={"utility": _strict_stay},
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="single_m", projection={"wealth": _wealth}
                            )
                        },
                        rule=_ReturnedRule(action=1),
                    )
                },
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
        execution_config=ExecutionConfig(device_memory_bytes=None),
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
        result.values[0]["single_f"], np.full((n_points,), 5.0)
    )


def _ancestor_settlement(*, wealth: FloatND, ancestor_shift: FloatND) -> FloatND:
    return wealth + ancestor_shift


def _descendant_settlement(descendant_shift: FloatND) -> FloatND:
    return descendant_shift


def _ancestor_available(*, wealth: FloatND, ancestor_threshold: FloatND) -> BoolND:
    return wealth >= ancestor_threshold


def _information_wealth(*, wealth: FloatND, middle_draw: FloatND) -> FloatND:
    return 10.0 + wealth + middle_draw


def _information_benchmark(*, wealth: FloatND, leaf_draw: FloatND) -> FloatND:
    return wealth + leaf_draw


def _information_stay(root_draw: FloatND) -> FloatND:
    return -100.0 + root_draw


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("fixed", [False, True])
def test_nested_information_uses_each_ancestors_settlement_operands(
    *, fixed: bool
) -> None:
    """A constant descendant map cannot supply its ancestor's shift or guard."""
    wealth = LinSpacedGrid(start=0, stop=12, n_points=13)
    regimes = {
        "single_f": Regime(
            states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
            functions={"utility": _information_stay},
            shocks=("root_noise",),
            exits={
                "move": Exit(
                    targets={
                        "self": ExitTarget(
                            regime="couple",
                            projection={"wealth": _ancestor_settlement},
                        )
                    },
                    available=_ancestor_available,
                    rule="unilateral",
                )
            },
        ),
        "couple": Regime(
            states={"wealth": wealth},
            functions={"utility": _information_wealth},
            shocks=("middle_noise",),
            exits={
                "move": Exit(
                    targets={
                        "self": ExitTarget(
                            regime="single_m",
                            projection={"wealth": _descendant_settlement},
                        )
                    },
                    rule="unilateral",
                )
            },
        ),
        "single_m": Regime(
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
                regime_id_class=_Household,
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
            if node.read.target == "single_m"
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
        # The benchmark hides its mean-zero shock, so its value is descendant.
        # A Participating middle policy must beat that benchmark at every hidden
        # middle draw. It cannot exit to the benchmark on the root's behalf.
        middle = 10.0 + source_wealth[:, None] + shift + np.asarray([-1.0, 1.0])
        support = np.all(middle >= descendant, axis=1)
        expected = np.where(
            (source_wealth >= threshold) & support, middle.mean(axis=1), -100.0
        )
        np.testing.assert_allclose(solution.values[0]["single_f"], expected)
        assert np.all(np.isfinite(solution.values[0]["single_f"]))


def test_leaf_exit_projection_does_not_capture_a_dag() -> None:
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


def test_exit_projection_keeps_computed_inputs_and_parameter_aliases() -> None:
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


def test_callable_exit_projection_keeps_fingerprint_admission() -> None:
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
