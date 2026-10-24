"""Current categorical laws use current state and one physical category per row."""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from lcm import InitialNodes, Model, Regime, categorical
from lcm.shocks import Discrete, Product, ShockGroup
from lcm.typing import BoolND, DiscreteAction, DiscreteState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt


@categorical(ordered=True)
class _Outcome:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _TwoRegimeId:
    alive: ScalarInt
    last: ScalarInt


def _code_payoff(*, eps: ScalarInt, eps_code: ScalarInt) -> FloatND:
    return 2.0 * eps + eps_code


@pytest.mark.parametrize("enable_jit", [False, True])
def test_discrete_shock_code_is_the_physical_category(*, enable_jit: bool) -> None:
    """Bare and explicit code reads consume the same discrete realization."""
    model = Model(
        regimes={
            "alive": Regime(functions={"utility": _code_payoff}, shocks=("innovation",))
        },
        n_periods=1,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Discrete(categories=_Outcome, probs=(0.25, 0.75))},
                nodes=Product(),
            )
        },
        enable_jit=enable_jit,
    )
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        solution.values[0]["alive"], 2.25, decimal=DECIMAL_PRECISION
    )
    result = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(4, dtype=jnp.int32),
            "regime_id": jnp.zeros(4, dtype=jnp.int32),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(result["value"], 3 * result["eps"])


def _continuous_code_payoff(eps_code: ScalarInt) -> FloatND:
    return 1.0 * eps_code


def test_continuous_physical_shock_has_no_code_alias() -> None:
    """A continuous physical draw never exposes a quadrature-node position."""
    from lcm.exceptions import ShockDeclarationError  # noqa: PLC0415
    from lcm.shocks import GaussHermite, Normal  # noqa: PLC0415

    with pytest.raises(ShockDeclarationError, match=r"eps_code.*continuous"):
        Model(
            regimes={
                "alive": Regime(
                    functions={"utility": _continuous_code_payoff},
                    shocks=("innovation",),
                )
            },
            n_periods=1,
            regime_id_class=_RegimeId,
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={},
            shocks={
                "innovation": ShockGroup(
                    shocks={"eps": Normal(mu=0, sigma=1)},
                    nodes=Product(eps=GaussHermite(n=3)),
                )
            },
        )


def _impute_status(wealth: FloatND) -> FloatND:
    return 10.0 + 10.0 * wealth


def _status_location(status: FloatND) -> FloatND:
    return status / 40


def _status_code_location(status_code: ScalarInt) -> FloatND:
    return 0.25 + 0.25 * status_code


def _shock_payoff(eps: FloatND) -> FloatND:
    return 1.0 * eps


def _fixed_wealth(wealth: FloatND) -> FloatND:
    return wealth


def _fixed_status(status: FloatND) -> FloatND:
    return status


def _impute_status_code(wealth: FloatND) -> ScalarInt:
    return jnp.where(wealth == 0, jnp.int32(0), jnp.int32(1))


def _class_location(status: ScalarInt) -> FloatND:
    return 0.25 + 0.25 * status


def _status_probabilities(status: FloatND) -> FloatND:
    high = _status_location(status)
    return jnp.stack((1 - high, high))


def _class_probabilities(status: ScalarInt) -> FloatND:
    high = _class_location(status)
    return jnp.stack((1 - high, high))


def _status_code_probabilities(status_code: ScalarInt) -> FloatND:
    high = _status_code_location(status_code)
    return jnp.stack((1 - high, high))


def _flip_status(status: FloatND) -> FloatND:
    return 30.0 - status


def _flip_status_code(status: ScalarInt) -> ScalarInt:
    return jnp.int32(1) - status


@pytest.mark.parametrize(
    "mode", ["value", "code", "lookup", "discrete_value", "discrete_code"]
)
@pytest.mark.parametrize("value_grid", [False, True])
def test_carried_categorical_law_uses_its_canonical_domain(
    *, mode: str, value_grid: bool
) -> None:
    """Solve imputations and actual carried categories share the declared codec."""
    from lcm import (  # noqa: PLC0415
        DiscreteGrid,
        LinSpacedGrid,
        Periods,
        Phased,
        StateConditioned,
    )
    from lcm.shocks import GaussHermite, Normal  # noqa: PLC0415

    location = {
        "value": _status_location if value_grid else _class_location,
        "code": _status_code_location,
        "lookup": StateConditioned(
            on="status",
            by={"10.0": 0.25, "20.0": 0.5}
            if value_grid
            else {"low": 0.25, "high": 0.5},
        ),
        "discrete_value": _status_probabilities if value_grid else _class_probabilities,
        "discrete_code": _status_code_probabilities,
    }[mode]
    model = Model(
        regimes={
            "alive": Regime(
                states={
                    "wealth": LinSpacedGrid(start=0, stop=1, n_points=2),
                    "status": Phased(
                        solve=_impute_status if value_grid else _impute_status_code,
                        simulate=DiscreteGrid(values=np.array([10.0, 20.0]))
                        if value_grid
                        else DiscreteGrid(_Outcome),
                    ),
                },
                functions={"utility": _shock_payoff},
                state_transitions={
                    "wealth": _fixed_wealth,
                    "status": _flip_status if value_grid else _flip_status_code,
                },
                shocks=("innovation",),
            ),
            "last": Regime(
                states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _fixed_wealth},
            ),
        },
        n_periods=3,
        regime_id_class=_TwoRegimeId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"alive": Periods(values=(0,)), "last": Periods(values=(1,))}},
        shocks={
            "innovation": ShockGroup(
                shocks={
                    "eps": Discrete(categories=_Outcome, probs=location)
                    if mode.startswith("discrete")
                    else Normal(mu=location, sigma=0)
                },
                nodes=Product()
                if mode.startswith("discrete")
                else Product(eps=GaussHermite(n=1)),
            )
        },
    )
    params = {"discount_factor": 0.0}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    assert solution.values[0]["alive"].shape == (2,)
    assert_array_almost_equal(
        solution.values[0]["alive"], [0.25, 0.5], decimal=DECIMAL_PRECISION
    )
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.zeros(2, dtype=jnp.int32),
            "wealth": jnp.zeros(2),
            "status": jnp.array([20.0, 10.0])
            if value_grid
            else jnp.array([1, 0], dtype=jnp.int32),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    for period in (0, 1):
        rows = result.query("period == @period").sort_values("subject_id")
        codes = np.array([1, 0]) if period == 0 else np.array([0, 1])
        np.testing.assert_array_equal(rows["status"], codes)
        probabilities = 0.25 + 0.25 * codes
        if mode.startswith("discrete"):
            import jax  # noqa: PLC0415

            from _lcm.simulation.random_address import (  # noqa: PLC0415
                RandomAddress,
                draw_row_keys,
            )

            keys = draw_row_keys(
                key=jax.random.key(47),
                address_words=np.asarray(
                    RandomAddress(kind="shock", name="eps").words(), dtype=np.uint32
                ),
                row_high=np.zeros(2, dtype=np.uint32),
                row_low=np.arange(2, dtype=np.uint32),
                period=period,
            )
            uniforms = np.asarray([jax.random.uniform(key) for key in keys])
            np.testing.assert_array_equal(rows["eps"], uniforms >= 1 - probabilities)
        else:
            assert_array_almost_equal(
                rows["eps"], probabilities, decimal=DECIMAL_PRECISION
            )
        np.testing.assert_array_equal(rows["value"], rows["eps"])


def _current_probabilities(*, state: FloatND, period: ScalarInt, p: FloatND) -> FloatND:
    high = p + state / 8 + period / 8
    return jnp.stack((1 - high, high))


def _next_state(state: FloatND) -> FloatND:
    return 2.0 - state


@pytest.mark.parametrize("width", [1, 3])
def test_categorical_law_uses_current_date_and_original_uniform(width: int) -> None:
    """Use current-state probabilities and original-row uniforms at each decision."""
    import jax  # noqa: PLC0415

    from _lcm.simulation.random_address import (  # noqa: PLC0415
        RandomAddress,
        draw_row_keys,
    )
    from lcm import ExecutionConfig, LinSpacedGrid, Periods  # noqa: PLC0415

    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _shock_payoff},
                state_transitions={"state": _next_state},
                shocks=("innovation",),
            ),
            "last": Regime(
                functions={"utility": _shock_payoff}, shocks=("innovation",)
            ),
        },
        states={"state": LinSpacedGrid(start=0, stop=2, n_points=3)},
        n_periods=2,
        regime_id_class=_TwoRegimeId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"last": Periods(values=(0,))}},
        shocks={
            "innovation": ShockGroup(
                shocks={
                    "eps": Discrete(categories=_Outcome, probs=_current_probabilities)
                },
                nodes=Product(),
            )
        },
        execution_config=ExecutionConfig(axis_widths={"subject": width, "cell": width}),
    )
    params = {"discount_factor": 0.0, "innovation": {"p": 0.25}}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    for period, name in enumerate(("alive", "last")):
        assert_array_almost_equal(
            solution.values[period][name],
            0.25 + np.arange(3) / 8 + period / 8,
            decimal=DECIMAL_PRECISION,
        )
    initial_state = jnp.array([0.0, 2.0, 1.0, 0.0])
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(4, dtype=jnp.int32),
            "regime_id": jnp.zeros(4, dtype=jnp.int32),
            "state": initial_state,
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    for period in (0, 1):
        rows = result.query("period == @period").sort_values("subject_id")
        state = initial_state if period == 0 else 2 - initial_state
        keys = draw_row_keys(
            key=jax.random.key(47),
            address_words=np.asarray(
                RandomAddress(kind="shock", name="eps").words(), dtype=np.uint32
            ),
            row_high=np.zeros(4, dtype=np.uint32),
            row_low=np.arange(4, dtype=np.uint32),
            period=period,
        )
        uniforms = jnp.stack(
            tuple(jax.random.uniform(key, dtype=state.dtype) for key in keys)
        )
        expected = uniforms >= 1 - (0.25 + state / 8 + period / 8)
        np.testing.assert_array_equal(rows["eps"], expected)
        np.testing.assert_array_equal(rows["value"], expected)
        np.testing.assert_array_equal(rows["state"], state)


def _param_probabilities(p: FloatND) -> FloatND:
    return jnp.stack((1 - p, p))


@pytest.mark.parametrize("probability", [0.0, 0.375, 1.0])
def test_categorical_probability_derivative_keeps_finite_null_payload(
    probability: float,
) -> None:
    """Retain dp times a finite null payoff in native forward and reverse AD."""
    import jax  # noqa: PLC0415

    from _lcm.current_iid import CURRENT_IID_ARGUMENT  # noqa: PLC0415
    from tests.solution.test_current_iid_grid_search import (  # noqa: PLC0415
        _scalar_core,
    )

    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _shock_payoff}, shocks=("innovation",)
            )
        },
        n_periods=1,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={
                    "eps": Discrete(categories=_Outcome, probs=_param_probabilities)
                },
                nodes=Product(),
            )
        },
    )
    core = _scalar_core(model=model, params={"innovation": {"p": probability}})

    def value(p: FloatND) -> FloatND:
        return core.function(
            **{
                **core.arguments,
                CURRENT_IID_ARGUMENT: {"innovation": {"p": p}},
            }
        )[0]

    p = jnp.asarray(probability)
    actual, tangent = jax.jit(lambda x: jax.jvp(value, (x,), (jnp.ones_like(x),)))(p)
    reverse = jax.jit(jax.grad(value))(p)
    assert_array_almost_equal(actual, probability, decimal=DECIMAL_PRECISION)
    assert_array_almost_equal(tangent, 1.0, decimal=DECIMAL_PRECISION)
    assert_array_almost_equal(reverse, 1.0, decimal=DECIMAL_PRECISION)


def _impute_invalid_status(wealth: FloatND) -> FloatND:
    return wealth * 0 + 15.0


def _saturating_status_code(status_code: ScalarInt) -> FloatND:
    return jnp.where(status_code == 0, 0.25, 0.5)


def test_invalid_carried_code_cannot_be_hidden_by_a_finite_law() -> None:
    """A code-reading law cannot launder an invalid solve imputation into a value."""
    from lcm import DiscreteGrid, LinSpacedGrid, Periods, Phased  # noqa: PLC0415
    from lcm.exceptions import ShockDeclarationError  # noqa: PLC0415
    from lcm.shocks import GaussHermite, Normal  # noqa: PLC0415

    model = Model(
        regimes={
            "alive": Regime(
                states={
                    "wealth": LinSpacedGrid(start=0, stop=1, n_points=2),
                    "status": Phased(
                        solve=_impute_invalid_status,
                        simulate=DiscreteGrid(values=np.array([10.0, 20.0])),
                    ),
                },
                functions={"utility": _shock_payoff},
                state_transitions={"wealth": _fixed_wealth, "status": _fixed_status},
                shocks=("innovation",),
            ),
            "last": Regime(
                states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _fixed_wealth},
            ),
        },
        n_periods=2,
        regime_id_class=_TwoRegimeId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"last": Periods(values=(0,))}},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=_saturating_status_code, sigma=0)},
                nodes=Product(eps=GaussHermite(n=1)),
            )
        },
    )
    with pytest.raises(ShockDeclarationError, match="invalid or unresolved"):
        model.solve(
            params={"discount_factor": 0.0}, log_level="off", runtime_checks=False
        )


@pytest.mark.parametrize("code", [-1, 2])
def test_conditioned_lookup_cannot_wrap_an_invalid_code(code: int) -> None:
    """A conditioning code outside its declared domain never selects a category."""
    from lcm import DiscreteGrid, StateConditioned  # noqa: PLC0415
    from lcm.shocks import GaussHermite, Normal  # noqa: PLC0415

    group = ShockGroup(
        shocks={
            "eps": Normal(
                mu=StateConditioned(on="status", by={"low": 0.25, "high": 0.5}), sigma=0
            )
        },
        nodes=Product(eps=GaussHermite(n=1)),
    ).bind_conditioning(state_grids={"status": DiscreteGrid(_Outcome)})
    realised = group.transform(
        base_nodes=jnp.zeros((1, 1)),
        inputs={"status": jnp.asarray(code, dtype=jnp.int32)},
    )
    assert not bool(realised.valid)


def _ordinary_code_helper(shift: FloatND) -> FloatND:
    return shift


def _unread_ordinary_helper(shift: FloatND) -> FloatND:  # noqa: ARG001 - must remain unread
    raise AssertionError("Unrelated ordinary helper must not run inside a shock law.")


def _from_ordinary_code_helper(scale_code: FloatND) -> FloatND:
    return scale_code


def test_code_suffix_on_an_ordinary_helper_is_not_a_state_alias() -> None:
    """Only declared categorical state aliases participate in law normalization."""
    from lcm.shocks import GaussHermite, Normal  # noqa: PLC0415

    model = Model(
        regimes={
            "alive": Regime(
                functions={
                    "utility": _shock_payoff,
                    "scale_code": _ordinary_code_helper,
                    "scale": _unread_ordinary_helper,
                },
                output_only_functions=("scale",),
                shocks=("innovation",),
            )
        },
        n_periods=1,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=_from_ordinary_code_helper, sigma=0)},
                nodes=Product(eps=GaussHermite(n=1)),
            )
        },
    )
    solution = model.solve(
        params={"innovation": {"shift": 2.0}, "alive": {"scale": {"shift": 99.0}}},
        log_level="off",
        runtime_checks=False,
    )
    assert_array_almost_equal(
        solution.values[0]["alive"], 2.0, decimal=DECIMAL_PRECISION
    )


def _solve_marker(eps_code: ScalarInt) -> FloatND:
    return 2.0 * eps_code


def _simulate_marker(eps_code: ScalarInt) -> FloatND:
    return 3.0 * eps_code


def _coded_choice_utility(
    *, eps: DiscreteState, marker: FloatND, action: DiscreteAction
) -> FloatND:
    return 10.0 * eps + marker + action


def _matches_code(*, action: DiscreteAction, eps_code: ScalarInt) -> BoolND:
    return action == eps_code


def _wealth_from_code(eps_code: ScalarInt) -> FloatND:
    return 1.0 * eps_code


def test_category_alias_is_shared_by_the_entire_native_flow() -> None:
    """Utility, phase helpers, feasibility, transitions and reports share one code."""
    from lcm import DiscreteGrid, LinSpacedGrid, Periods, Phased  # noqa: PLC0415

    model = Model(
        regimes={
            "alive": Regime(
                actions={"action": DiscreteGrid(_Outcome)},
                functions={
                    "utility": _coded_choice_utility,
                    "marker": Phased(solve=_solve_marker, simulate=_simulate_marker),
                    "reported": _wealth_from_code,
                },
                constraints={"matches": _matches_code},
                output_only_functions=("reported",),
                state_transitions={"wealth": _wealth_from_code},
                shocks=("innovation",),
            ),
            "last": Regime(
                states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _fixed_wealth},
            ),
        },
        n_periods=2,
        regime_id_class=_TwoRegimeId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"last": Periods(values=(0,))}},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Discrete(categories=_Outcome, probs=(0.25, 0.75))},
                nodes=Product(),
            )
        },
    )
    params = {"discount_factor": 0.0}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        solution.values[0]["alive"], 9.75, decimal=DECIMAL_PRECISION
    )
    panel = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(4, dtype=jnp.int32),
            "regime_id": jnp.zeros(4, dtype=jnp.int32),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True, additional_targets=["reported"])
    source = panel.query("period == 0").sort_values("subject_id")
    destination = panel.query("period == 1").sort_values("subject_id")
    np.testing.assert_array_equal(source["action"], source["eps"])
    np.testing.assert_array_equal(source["value"], 14 * source["eps"])
    np.testing.assert_array_equal(source["reported"], source["eps"])
    np.testing.assert_array_equal(destination["wealth"], source["eps"])


def _shock_plus_scale(*, eps: FloatND, scale: FloatND) -> FloatND:
    return eps + scale


def test_declared_helper_takes_precedence_over_a_continuous_code_spelling() -> None:
    """Retain a real scale_code helper beside a continuous scale state."""
    from lcm import LinSpacedGrid  # noqa: PLC0415
    from lcm.shocks import GaussHermite, Normal  # noqa: PLC0415

    model = Model(
        regimes={
            "alive": Regime(
                states={"scale": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={
                    "utility": _shock_plus_scale,
                    "scale_code": _ordinary_code_helper,
                },
                shocks=("innovation",),
            )
        },
        n_periods=1,
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=_from_ordinary_code_helper, sigma=0)},
                nodes=Product(eps=GaussHermite(n=1)),
            )
        },
    )
    solution = model.solve(
        params={"innovation": {"shift": 2.0}}, log_level="off", runtime_checks=False
    )
    assert_array_almost_equal(
        solution.values[0]["alive"], [2.0, 3.0], decimal=DECIMAL_PRECISION
    )
