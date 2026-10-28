"""Current physical innovations precede native singleton action selection."""

import dataclasses
from dataclasses import replace
from types import MappingProxyType
from typing import Unpack

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from _lcm import shock_nodes
from _lcm.current_iid import CURRENT_IID_ARGUMENT
from _lcm.execution.core_program import (
    CoreBuildContext,
    core_program_graph,
    materialize_core_program,
)
from _lcm.params.edges import regime_kernel_params
from _lcm.processes.base import standard_normal
from _lcm.simulation import chunk_admission as admission
from _lcm.simulation.chunk_planning import (
    SimulationChunkProfile,
    plan_simulation_chunks,
)
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.program_arguments import current_iid_arguments
from _lcm.simulation.program_types import SUBJECT_WIDTH_KEYWORD
from _lcm.simulation.random_address import RandomAddress, draw_row_keys
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.zero_safe import factored_weighted_term
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    LinSpacedGrid,
    LogNormalIIDProcess,
    Model,
    NormalIIDProcess,
    PeriodRange,
    Periods,
    Phased,
    Regime,
    UniformIIDProcess,
    categorical,
    post_decision_lower_bound,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidNameError,
    InvalidParamsError,
    InvalidSimulationInputError,
    ShockDeclarationError,
    UnsupportedOperationError,
)
from lcm.result import SimulationResult
from lcm.shocks import (
    Discrete,
    EqualSpaced,
    EqualSpacedUniform,
    GaussHermite,
    LogNormal,
    MonteCarlo,
    Normal,
    Product,
    ShockGroup,
    Uniform,
    UserNodes,
)
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import ContinuousState, DiscreteAction, FloatND, ScalarInt, UserParams
from tests.conftest import DECIMAL_PRECISION
from tests.constraints import test_dynamic_lower_bound as bound
from tests.simulation import test_aot_same_period_refs as reference
from tests.simulation._callback_types import ChunkProfileInputs
from tests.solution import test_decoupled
from tests.solution import test_gated_edge_stateless_target_solve as gated
from tests.solution import test_invariant_blocking_scalar as invariant
from tests.solution._iid_shock_oracles import finite_backward
from tests.test_models import dcegm_paper_twin


@categorical(ordered=False)
class _RegimeId:
    first: ScalarInt
    second: ScalarInt
    last: ScalarInt


@categorical(ordered=True)
class _Action:
    off: ScalarInt
    on: ScalarInt


def _utility(*, wealth: ContinuousState, action: DiscreteAction, eps: float) -> FloatND:
    return wealth + 2 * action + eps * (2 * action - 1)


def _next_wealth(action: DiscreteAction) -> FloatND:
    return 1.0 * action


def _model(
    *, ev1: bool, enable_jit: bool, width: int = 1, budgeted: bool = False
) -> Model:
    regimes = {}
    for name in ("first", "second", "last"):
        regimes[name] = Regime(
            states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
            actions={"action": DiscreteGrid(category_class=_Action)},
            functions={"utility": _utility},
            state_transitions={} if name == "last" else {"wealth": _next_wealth},
            shocks=("innovation",),
            taste_shocks=ExtremeValueTasteShocks() if ev1 else None,
        )
    return Model(
        regimes=regimes,
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=1)},
                nodes=UserNodes(
                    nodes=np.array([[-1.0], [0.0], [1.0]]),
                    weights=np.array([0.25, 0.5, 0.25]),
                ),
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "first"},
        edges={"first": {"second": 0}, "second": {"last": 1}},
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(
            axis_widths={"cell": width} | ({} if ev1 else {"action_product": width}),
            device_memory_bytes=2**26 if budgeted else None,
        ),
    )


@pytest.mark.parametrize(
    ("enable_jit", "budgeted"), [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("ev1", [False, True])
@pytest.mark.parametrize("width", [1, 3])
def test_public_solve_integrates_the_nodewise_decision(
    *, enable_jit: bool, ev1: bool, width: int, budgeted: bool
) -> None:
    """Stored real-state values match literal independent backward induction."""
    model = _model(ev1=ev1, enable_jit=enable_jit, width=width, budgeted=budgeted)
    params = {"discount_factor": 0.5}
    if ev1:
        params.update(
            {
                name: {"taste_shocks": {"scale": 1.0}}
                for name in ("first", "second", "last")
            }
        )
    result = model.solve(params=params, log_level="off", runtime_checks=False)
    expected, _ = finite_backward(mode="ev1" if ev1 else "hard")
    for period, name in enumerate(("first", "second", "last")):
        assert_array_almost_equal(
            np.asarray(result.values[period][name]),
            [float(expected[period, wealth][0]) for wealth in (0, 1)],
            decimal=DECIMAL_PRECISION,
        )


@categorical(ordered=False)
class _TerminalId:
    alive: ScalarInt


@categorical(ordered=False)
class _TwoId:
    alive: ScalarInt
    last: ScalarInt


def _squared_shock(eps: FloatND) -> FloatND:
    return eps * eps


def _runtime_scale_model(*, fixed_params: UserParams = MappingProxyType({})) -> Model:
    return Model(
        regimes={
            "alive": Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            )
        },
        n_periods=1,
        fixed_params=fixed_params,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=None)},
                nodes=UserNodes(
                    nodes=np.array([[-1.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )


def test_runtime_scale_has_a_shared_group_namespace() -> None:
    """Changing a group-local scale changes a scalar stored expectation."""
    model = _runtime_scale_model()
    assert model.get_params_template()["innovation"] == {"eps__sigma": "float"}
    for scale in (0.0, 2.0, 3.0):
        result = model.solve(
            params={"innovation": {"eps__sigma": scale}},
            log_level="off",
            runtime_checks=False,
        )
        assert_array_almost_equal(
            np.asarray(result.values[0]["alive"]),
            scale * scale,
            decimal=DECIMAL_PRECISION,
        )


def test_invalid_runtime_scale_cannot_publish_with_checks_off() -> None:
    """Mandatory law integrity rejects negative scale independently of logging."""

    with pytest.raises(ShockDeclarationError, match=r"invalid or unresolved"):
        _runtime_scale_model().solve(
            params={"innovation": {"eps__sigma": -1.0}},
            log_level="off",
            runtime_checks=False,
        )


def test_fixed_group_slots_are_bound_and_cannot_be_overridden() -> None:
    """Fixed current-law parameters retain their identity outside regime slots."""

    model = _runtime_scale_model(fixed_params={"innovation": {"eps__sigma": 2.0}})
    assert model.get_params_template()["innovation"] == {}
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        np.asarray(result.values[0]["alive"]), 4.0, decimal=DECIMAL_PRECISION
    )
    with pytest.raises(InvalidParamsError, match=r"fixed"):
        model.solve(params={"innovation": {"eps__sigma": 3.0}}, log_level="off")


def test_fixed_group_values_change_model_identity() -> None:
    """Fixed physical-law declarations are part of durable model meaning."""
    first = _runtime_scale_model(fixed_params={"innovation": {"eps__sigma": 2.0}})
    second = _runtime_scale_model(fixed_params={"innovation": {"eps__sigma": 3.0}})
    assert first._model_structure_fingerprint != second._model_structure_fingerprint


def _location_from_taste_slot(taste_shocks__scale: float) -> FloatND:
    return jnp.asarray(taste_shocks__scale)


def test_group_slot_does_not_inherit_regime_taste_scale_validation() -> None:
    """A negative location parameter is unrelated to an EV1 scale slot."""
    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=_location_from_taste_slot, sigma=0)},
                nodes=UserNodes(nodes=np.array([[0.0]]), weights=np.array([1.0])),
            )
        },
    )
    result = model.solve(
        params={"innovation": {"taste_shocks__scale": -2.0}},
        log_level="off",
        runtime_checks=False,
    )
    assert_array_almost_equal(
        np.asarray(result.values[0]["alive"]), 4.0, decimal=DECIMAL_PRECISION
    )


def _group_location(mu: FloatND) -> FloatND:
    return mu


def _wealth_plus_shock(*, wealth: ContinuousState, eps: float) -> FloatND:
    return wealth + eps


def test_fixed_group_root_cannot_bind_a_same_named_legacy_state() -> None:
    """Group roots stay separate before process-law binding and graph pruning."""
    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _wealth_plus_shock},
                shocks=("wealth",),
                states={
                    "wealth": NormalIIDProcess(
                        n_points=2, gauss_hermite=True, mu=None, sigma=1.0
                    )
                },
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        fixed_params={"wealth": {"mu": 2.0}},
        shocks={
            "wealth": ShockGroup(
                shocks={"eps": Normal(mu=_group_location, sigma=0)},
                nodes=UserNodes(nodes=np.array([[0.0]]), weights=np.array([1.0])),
            )
        },
    )
    assert "mu" in model.get_params_template()["alive"]["wealth"]
    result = model.solve(
        params={"alive": {"wealth": {"mu": 0.0}}}, log_level="off", runtime_checks=False
    )
    assert_array_almost_equal(
        np.asarray(result.values[0]["alive"]), [1.0, 3.0], decimal=DECIMAL_PRECISION
    )


def _state_scale(*, wealth: ContinuousState, scale: float) -> FloatND:
    return scale * (1.0 + wealth)


def _state_law_model(*, broadcast: bool) -> Model:
    states = {"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)}
    return Model(
        states=states if broadcast else {},
        regimes={
            "alive": Regime(
                states={} if broadcast else states,
                functions={"utility": _squared_shock},
                shocks=("innovation",),
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=_state_scale)},
                nodes=UserNodes(
                    nodes=np.array([[-1.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )


@pytest.mark.parametrize("broadcast", [False, True])
def test_current_state_used_only_by_law_is_retained(*, broadcast: bool) -> None:
    """A law is evaluated at each current wealth cell before integrating."""
    result = _state_law_model(broadcast=broadcast).solve(
        params={"innovation": {"scale": 2.0}},
        log_level="off",
        runtime_checks=False,
    )
    assert_array_almost_equal(
        np.asarray(result.values[0]["alive"]), [4.0, 16.0], decimal=DECIMAL_PRECISION
    )


def _report_only(scale: float) -> FloatND:  # noqa: ARG001 - intentionally unread helper
    raise AssertionError("An unrelated report must not run while evaluating a law.")


def _fixed_offer_probabilities(*, base_share: FloatND, law_shift: FloatND) -> FloatND:
    high = base_share + law_shift
    return jnp.stack((1 - high, high))


def _offer_share(*, base_share: FloatND, law_shift: FloatND) -> FloatND:
    return base_share / 2 + law_shift


def _helper_offer_probabilities(offer_share: FloatND) -> FloatND:
    return jnp.stack((1 - offer_share, offer_share))


def _offer_utility(*, eps: ScalarInt, base_share: FloatND) -> FloatND:
    return 4 * eps + base_share


def _fixed_offer_model(*, helper: bool, fixed_params: UserParams) -> Model:
    return Model(
        regimes={
            "alive": Regime(
                functions={"utility": _offer_utility}
                | ({"offer_share": _offer_share} if helper else {}),
                shocks=("innovation",),
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        fixed_params=fixed_params,
        shocks={
            "innovation": ShockGroup(
                shocks={
                    "eps": Discrete(
                        categories=_Action,
                        probs=_helper_offer_probabilities
                        if helper
                        else _fixed_offer_probabilities,
                    )
                }
            )
        },
    )


@pytest.mark.parametrize("helper", [False, True])
def test_fixed_callable_law_dependencies_share_ordinary_broadcasts(
    *, helper: bool
) -> None:
    """A fixed leaf serves both the current law DAG and ordinary utility."""
    model = _fixed_offer_model(
        helper=helper, fixed_params={"base_share": 0.25, "law_shift": 0.125}
    )
    assert model.get_params_template()["innovation"] == {}
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        np.asarray(result.values[0]["alive"]),
        1.25 if helper else 1.75,
        decimal=DECIMAL_PRECISION,
    )


def test_fixed_callable_law_does_not_consume_unused_parameters() -> None:
    """A current-law dependency does not license unrelated fixed leaves."""
    with pytest.raises(InvalidParamsError, match="unused_probability"):
        _fixed_offer_model(
            helper=True,
            fixed_params={
                "base_share": 0.25,
                "law_shift": 0.125,
                "unused_probability": 0.5,
            },
        )


def test_fixed_callable_law_rejects_ambiguous_binding() -> None:
    """Model and group fixed values cannot both specify a callable leaf."""
    with pytest.raises(InvalidNameError, match=r"Ambiguous.*law_shift"):
        _fixed_offer_model(
            helper=True,
            fixed_params={
                "base_share": 0.25,
                "law_shift": 0.125,
                "innovation": {"law_shift": 0.5},
            },
        )


def _adjusted_scale(*, wealth: ContinuousState, scale: float) -> FloatND:
    return scale * (1.0 + wealth)


def _scale_from_adjusted(adjusted: FloatND) -> FloatND:
    return adjusted


@pytest.mark.parametrize("helper", [False, True])
def test_only_law_helper_closure_uses_group_slots(*, helper: bool) -> None:
    """Report helpers keep ordinary slots; exclusive law helpers use group slots."""
    functions = {"utility": _squared_shock}
    functions.update(
        {"adjusted": _adjusted_scale} if helper else {"report": _report_only}
    )
    model = Model(
        regimes={
            "alive": Regime(
                states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions=functions,
                shocks=("innovation",),
                output_only_functions=() if helper else ("report",),
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={
                    "eps": Normal(
                        mu=0, sigma=_scale_from_adjusted if helper else _state_scale
                    )
                },
                nodes=UserNodes(
                    nodes=np.array([[-1.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )
    params = {"innovation": {"scale": 2.0}}
    if helper:
        assert model.get_params_template()["alive"].get("adjusted", {}) == {}
    else:
        assert "scale" in model.get_params_template()["alive"]["report"]
        params["alive"] = {"report": {"scale": 99.0}}
    result = model.solve(params=params, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        np.asarray(result.values[0]["alive"]), [4.0, 16.0], decimal=DECIMAL_PRECISION
    )


def test_fixed_literal_law_cannot_accept_unknown_fixed_slot() -> None:
    """A fixed literal law cannot silently swallow a misspelled fixed leaf."""

    with pytest.raises(InvalidParamsError, match=r"shock group|Shock group"):  # noqa: PT012 - constructor or invocation may reject
        model = Model(
            regimes={
                "alive": Regime(
                    functions={"utility": _squared_shock}, shocks=("innovation",)
                )
            },
            n_periods=1,
            regime_id_class=_TerminalId,
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={},
            fixed_params={"innovation": {"eps__sigm": 2.0}},
            shocks={
                "innovation": ShockGroup(
                    shocks={"eps": Normal(mu=0, sigma=1)},
                    nodes=UserNodes(nodes=np.array([[0.0]]), weights=np.array([1.0])),
                )
            },
        )
        model.solve(params={}, log_level="off", runtime_checks=False)


def _physical_scale(health: float) -> FloatND:
    return jnp.asarray(health)


def _code_scale(health_code: ScalarInt) -> FloatND:
    return jnp.asarray(10.0 + health_code)


@pytest.mark.parametrize("read_code", [False, True])
def test_law_callables_observe_value_grid_boundary(*, read_code: bool) -> None:
    """Bare categorical reads decode values, explicit code reads retain codes."""
    model = Model(
        regimes={
            "alive": Regime(
                states={"health": DiscreteGrid(values=np.array([10.0, 20.0]))},
                functions={"utility": _squared_shock},
                shocks=("innovation",),
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={
                    "eps": Normal(
                        mu=0, sigma=_code_scale if read_code else _physical_scale
                    )
                },
                nodes=UserNodes(
                    nodes=np.array([[-1.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        np.asarray(result.values[0]["alive"]),
        [100.0, 121.0 if read_code else 400.0],
        decimal=DECIMAL_PRECISION,
    )


def test_period_rule_uses_period_seed_coordinate() -> None:
    """The native solve uses exactly the declared period-tagged MC support."""

    group = ShockGroup(
        shocks={"eps": Normal(mu=0, sigma=1)}, nodes=MonteCarlo(n=7, seed=19)
    )
    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={"innovation": group},
    )
    nodes = group.resolve_nodes(
        group_name="innovation", age=0, coordinate_kind="period"
    )
    expected = np.dot(nodes.weights, nodes.base_nodes[:, 0] ** 2)
    result = model.solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        np.asarray(result.values[0]["alive"]), expected, decimal=DECIMAL_PRECISION
    )


def _shock_cross_product(*, first: FloatND, second: FloatND) -> FloatND:
    return first * second


def _correlated_model() -> Model:
    return Model(
        regimes={
            "alive": Regime(
                functions={"utility": _shock_cross_product}, shocks=("joint",)
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "joint": ShockGroup(
                shocks={
                    "first": Normal(mu=0, sigma=1),
                    "second": Normal(mu=0, sigma=1),
                },
                correlation=None,
                nodes=UserNodes(
                    nodes=np.array(
                        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
                    ),
                    weights=np.full(4, 0.25),
                ),
            )
        },
    )


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("rho", [0.0, 0.6, 1.0, -1.0])
def test_runtime_correlation_is_a_dynamic_admitted_native_operand(rho: float) -> None:
    """Interior and exact singular supports use the supplied physical covariance."""
    model = _correlated_model()
    for correlation in (rho, -rho):
        result = model.solve(
            params={
                "joint": {"corr": np.array([[1.0, correlation], [correlation, 1.0]])}
            },
            log_level="off",
            runtime_checks=False,
        )
        assert_array_almost_equal(
            np.asarray(result.values[0]["alive"]),
            correlation,
            decimal=DECIMAL_PRECISION,
        )


def test_invalid_runtime_correlation_is_rejected_before_native_dispatch() -> None:
    """An indefinite matrix cannot enter the admitted support-mask kernel."""

    with pytest.raises(
        ShockDeclarationError, match=r"correlation|corr|PSD|semidefinite"
    ):
        _correlated_model().solve(
            params={"joint": {"corr": np.array([[1.0, 2.0], [2.0, 1.0]])}},
            log_level="off",
            runtime_checks=False,
        )


def test_raw_indefinite_correlation_cannot_hide_in_active_rounding() -> None:
    """Raw host PSD admission precedes conversion to the active precision."""

    with pytest.raises(ShockDeclarationError, match=r"positive semidefinite"):
        _correlated_model().solve(
            params={
                "joint": {
                    "corr": np.array(
                        [[1.0, 1.0 + 1e-9], [1.0 + 1e-9, 1.0]], dtype=np.float64
                    )
                }
            },
            log_level="off",
            runtime_checks=False,
        )


def _scalar_core(*, model: Model, params: UserParams):
    """Materialize the same declared core and operands used by public solve."""

    solution = model._regimes["alive"].solution
    flat = model._process_params(params)
    context = CoreBuildContext(
        state_action_space=solution.state_action_space(
            regime_params=regime_kernel_params(flat, regime_name="alive")
        ),
        next_regime_to_V_arr={},
        next_regime_to_continuation={},
        flat_params=flat,
        period=0,
        ages=model._time,
    )
    return materialize_core_program(
        program=core_program_graph(kernel=solution.period_kernels[0])["main"],
        context=context,
    )


def test_native_value_consumes_runtime_correlation_derivative() -> None:
    """The admitted mask stays fixed while the SPD correlation remains traced."""

    core = _scalar_core(
        model=_correlated_model(), params={"joint": {"corr": np.eye(2)}}
    )

    def value(rho):
        groups = core.arguments[CURRENT_IID_ARGUMENT]
        group = {**groups["joint"], "corr": jnp.array([[1.0, rho], [rho, 1.0]])}
        return core.function(
            **{**core.arguments, CURRENT_IID_ARGUMENT: {"joint": group}}
        )[0]

    actual, derivative = jax.jit(jax.value_and_grad(value))(0.37)
    np.testing.assert_allclose(actual, 0.37, rtol=1e-5, atol=0)
    np.testing.assert_allclose(derivative, 1.0, rtol=1e-5, atol=0)


def test_native_node_banks_are_declared_dynamic_operands() -> None:
    """Workspace and residency admission can see the complete fixed support banks."""
    core = _scalar_core(
        model=_runtime_scale_model(), params={"innovation": {"eps__sigma": 1.0}}
    )
    leaves = jax.tree_util.tree_leaves(core.arguments)
    for expected in (np.array([[-1.0], [1.0]]), np.array([0.5, 0.5])):
        assert any(
            hasattr(leaf, "shape")
            and leaf.shape == expected.shape
            and np.array_equal(np.asarray(leaf), expected)
            for leaf in leaves
        )


def test_resolved_rule_algorithm_is_part_of_native_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal law/rule declarations cannot hide a different resolved algorithm seal."""

    first = _runtime_scale_model()
    resolve = shock_nodes.resolve_nodes

    def changed_algorithm(**kwargs):
        return replace(resolve(**kwargs), fingerprint="different-resolved-algorithm")

    monkeypatch.setattr(shock_nodes, "resolve_nodes", changed_algorithm)
    second = _runtime_scale_model()
    assert first._model_structure_fingerprint != second._model_structure_fingerprint


@pytest.mark.parametrize("factor", [-1.0, np.nan])
def test_factored_nonfinite_terms_preserve_sign_and_poison(factor: float) -> None:
    """The shared weighted primitive cannot erase a signed or invalid factor."""

    actual = jax.jit(factored_weighted_term)(
        factors=jnp.array([[factor]]), values=jnp.array([np.inf])
    )
    if np.isnan(factor):
        assert np.isnan(actual[0])
    else:
        assert np.isneginf(actual[0])


@categorical(ordered=False)
class _Outcome:
    common: ScalarInt
    rare: ScalarInt


def _rare_value(*, first: ScalarInt, second: ScalarInt, payoff: float) -> FloatND:
    return jnp.where((first == 1) & (second == 1), payoff, 0.0)


def _rare_infinite_value(*, first: ScalarInt, second: ScalarInt) -> FloatND:
    return jnp.where((first == 1) & (second == 1), -jnp.inf, 0.0)


def _rare_model(*, infinite: bool = False) -> Model:

    return Model(
        regimes={
            "alive": Regime(
                functions={
                    "utility": _rare_infinite_value if infinite else _rare_value
                },
                shocks=("joint",),
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "joint": ShockGroup(
                shocks={
                    "first": Discrete(categories=_Outcome),
                    "second": Discrete(categories=_Outcome),
                }
            )
        },
    )


@pytest.mark.parametrize("infinite", [False, True])
def test_joint_rare_factors_survive_until_the_economic_value(*, infinite: bool) -> None:
    """A positive joint event is retained even when its rounded weight is zero."""
    probability = 1e-200 if jax.config.x64_enabled else 1e-25
    payoff = 1e300 if jax.config.x64_enabled else 1e30
    result = _rare_model(infinite=infinite).solve(
        params={
            "joint": {
                "first__probs": np.array([1.0, probability]),
                "second__probs": np.array([1.0, probability]),
            },
            **({} if infinite else {"alive": {"utility": {"payoff": payoff}}}),
        },
        log_level="off",
        runtime_checks=False,
    )
    actual = np.asarray(result.values[0]["alive"])
    if infinite:
        assert np.isneginf(actual)
    else:
        # Multiply the independent host oracle in an order that remains in range.
        np.testing.assert_allclose(
            actual, (probability * payoff) * probability, rtol=1e-5, atol=0
        )


def test_zero_probability_retains_native_weight_derivative() -> None:
    """A null primal event still contributes dweight times its finite payoff."""

    params = {
        "joint": {
            "first__probs": np.array([1.0, 0.0]),
            "second__probs": np.array([0.7, 0.3]),
        },
        "alive": {"utility": {"payoff": 2.0}},
    }
    core = _scalar_core(model=_rare_model(), params=params)

    def value(probability):
        groups = core.arguments[CURRENT_IID_ARGUMENT]
        group = {
            **groups["joint"],
            "first__probs": jnp.array([1.0 - probability, probability]),
        }
        return core.function(
            **{**core.arguments, CURRENT_IID_ARGUMENT: {"joint": group}}
        )[0]

    actual, derivative = jax.jit(jax.value_and_grad(value))(0.0)
    assert actual == 0.0
    np.testing.assert_allclose(derivative, 0.6, rtol=1e-5, atol=0)


def test_native_rare_probability_derivative_meets_its_finite_payoff() -> None:
    """A probability slope forms the other factor with payoff before rounding."""

    probability = 1e-200 if jax.config.x64_enabled else 1e-25
    payoff = 1e300 if jax.config.x64_enabled else 1e30
    core = _scalar_core(
        model=_rare_model(),
        params={
            "joint": {
                "first__probs": np.array([1.0, probability]),
                "second__probs": np.array([1.0, probability]),
            },
            "alive": {"utility": {"payoff": payoff}},
        },
    )

    def value(p):
        group = {
            **core.arguments[CURRENT_IID_ARGUMENT]["joint"],
            "first__probs": jnp.array([1.0, p]),
        }
        return core.function(
            **{**core.arguments, CURRENT_IID_ARGUMENT: {"joint": group}}
        )[0]

    actual, derivative = jax.jit(jax.value_and_grad(value))(probability)
    np.testing.assert_allclose(
        actual, probability * payoff * probability, rtol=1e-5, atol=0
    )
    np.testing.assert_allclose(derivative, probability * payoff, rtol=1e-5, atol=0)


def test_native_value_cotangent_range_is_explicit(record_property) -> None:
    """Record the precision limit without certifying an extreme AD consumer."""
    probability = 1e-160 if jax.config.x64_enabled else 1e-22
    multiplier = 1e300 if jax.config.x64_enabled else 1e30
    core = _scalar_core(
        model=_rare_model(),
        params={
            "joint": {
                "first__probs": np.array([1.0, probability]),
                "second__probs": np.array([1.0, probability]),
            },
            "alive": {"utility": {"payoff": multiplier}},
        },
    )

    def value(theta):
        return core.function(
            **{**core.arguments, "utility__payoff": theta * multiplier}
        )[0]

    primal, forward = jax.jit(
        lambda theta: jax.jvp(value, (theta,), (jnp.ones_like(theta),))
    )(jnp.asarray(1.0))
    reverse = jax.jit(jax.grad(value))(1.0)
    expected = probability * multiplier * probability
    np.testing.assert_allclose(primal, expected, rtol=1e-5, atol=0)
    for name, actual in (("forward", forward), ("reverse", reverse)):
        record_property(f"{name}_extreme_value_sensitivity", float(actual))
        # Active value cotangents can flush or round in their subnormal range.
        assert float(actual) == 0.0 or np.isclose(actual, expected, rtol=0.1, atol=0)
    record_property("analytic_extreme_value_sensitivity", expected)
    record_property("extreme_value_sensitivity_certified", value=False)


@pytest.mark.parametrize("mu", [-1e4, 0.0])
def test_native_lognormal_requires_represented_positive_support(mu: float) -> None:
    """A finite zero from exp underflow is unresolved, rather than a lognormal draw."""

    group = ShockGroup(
        shocks={"eps": LogNormal(mu=mu, sigma=0)},
        nodes=UserNodes(nodes=np.array([[0.0]]), weights=np.array([1.0])),
    )
    realised = group.realise(nodes=group.resolve_nodes(group_name="innovation", age=0))
    if mu < 0:
        with pytest.raises(ShockDeclarationError):
            realised.raise_if_invalid()
    else:
        realised.raise_if_invalid()
    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={"innovation": group},
    )
    if mu < 0:
        with pytest.raises(ShockDeclarationError, match=r"invalid or unresolved"):
            model.solve(params={}, log_level="off", runtime_checks=False)
    else:
        result = model.solve(params={}, log_level="off", runtime_checks=False)
        np.testing.assert_allclose(result.values[0]["alive"], 1.0)


def test_public_simulation_replays_fresh_addressed_physical_draws() -> None:
    """Continuous physical rows are unrelated to the finite solve support."""

    model = _runtime_scale_model()
    params = {"innovation": {"eps__sigma": 2.0}}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    simulated = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(7, dtype=jnp.int32),
            "regime_id": jnp.zeros(7, dtype=jnp.int32),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    )
    keys = draw_row_keys(
        key=jax.random.key(47, impl="threefry2x32"),
        address_words=np.asarray(
            RandomAddress(kind="shock", name="eps").words(), dtype=np.uint32
        ),
        row_high=np.zeros(7, dtype=np.uint32),
        row_low=np.arange(7, dtype=np.uint32),
        period=0,
    )
    physical = 2.0 * jax.vmap(standard_normal)(keys)
    np.testing.assert_allclose(
        simulated.to_dataframe()["value"], physical**2, rtol=1e-5, atol=0
    )


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("width", [1, 2, 7])
def test_rbg_physical_replay_matches_scalar_addressed_draws(
    *, uniform: bool, width: int
) -> None:
    """Every key keeps its scalar draw under native subject tiling."""

    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        execution_config=ExecutionConfig(axis_widths={"subject": width}),
        shocks={
            "innovation": ShockGroup(
                shocks={
                    "eps": Uniform(start=0, stop=1)
                    if uniform
                    else Normal(mu=0, sigma=1)
                },
                nodes=UserNodes(
                    nodes=np.array([[0.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )
    root = jax.random.key(47, impl="rbg")
    keys = draw_row_keys(
        key=root,
        address_words=np.asarray(
            RandomAddress(kind="shock", name="eps").words(), dtype=np.uint32
        ),
        row_high=np.zeros(7, dtype=np.uint32),
        row_low=np.arange(7, dtype=np.uint32),
        period=0,
    )
    regime = model._regimes["alive"]
    program = regime.simulation.programs.decision[0]
    output = jax.jit(program.function, static_argnames=(SUBJECT_WIDTH_KEYWORD,))(
        **{
            **current_iid_arguments(
                regime=regime,
                flat_params={},
                keys={"eps": keys},
                overrides=MappingProxyType({}),
            ),
            "period": jnp.int32(0),
            "age": jnp.int32(0),
            "next_regime_to_V_arr": {},
            SUBJECT_WIDTH_KEYWORD: width,
        }
    )
    expected = np.asarray(
        [
            jax.random.uniform(key=key[0], dtype=jnp.asarray(0.0).dtype)
            if uniform
            else standard_normal(key[0])
            for key in keys
        ]
    )
    np.testing.assert_array_equal(output[2]["eps"], expected)


def test_public_current_iid_preserves_ordinary_seed_admission() -> None:

    with pytest.raises(InvalidSimulationInputError, match=r"threefry2x32"):
        _runtime_scale_model().simulate(
            params={"innovation": {"eps__sigma": 1.0}},
            initial_conditions={
                "period": jnp.zeros(1, dtype=jnp.int32),
                "regime_id": jnp.zeros(1, dtype=jnp.int32),
            },
            seed=jax.random.key(47, impl="rbg"),
            log_level="off",
            runtime_checks=False,
        )


def _next_wealth_with_current_draw(*, action: DiscreteAction, eps: float) -> FloatND:
    return action + eps


def test_replay_reuses_physical_draw_for_decision_flow_and_transition() -> None:
    """The current innovation is shared through its action and persistent-state law."""

    model = Model(
        regimes={
            name: Regime(
                states={"wealth": LinSpacedGrid(start=-100, stop=100, n_points=2)},
                actions={"action": DiscreteGrid(category_class=_Action)},
                functions={"utility": _utility},
                shocks=("innovation",),
                state_transitions={"wealth": _next_wealth_with_current_draw}
                if name == "alive"
                else {},
            )
            for name in ("alive", "last")
        },
        n_periods=2,
        regime_id_class=_TwoId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"last": Periods(values=(0,))}},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=1)},
                nodes=UserNodes(
                    nodes=np.array([[-1.0], [0.0], [1.0]]),
                    weights=np.array([0.25, 0.5, 0.25]),
                ),
            )
        },
    )
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(solution.values[0]["alive"], [-96.5, 103.5])
    simulated = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(7, dtype=jnp.int32),
            "regime_id": jnp.zeros(7, dtype=jnp.int32),
            "wealth": jnp.zeros(7),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(additional_targets=["utility"], use_categorical_codes=True)
    first = simulated.query("period == 0")
    last = simulated.query("period == 1")
    expected_action = (first["eps"].to_numpy() > -1.25).astype(int)
    np.testing.assert_array_equal(first["action"], expected_action)
    np.testing.assert_allclose(
        first["utility"],
        2 * expected_action + first["eps"] * (2 * expected_action - 1),
        rtol=1e-5,
        atol=0,
    )
    np.testing.assert_allclose(
        last["wealth"], first["eps"].to_numpy() + expected_action, rtol=1e-5, atol=0
    )
    expected_value = np.where(
        expected_action, 3.5 + 1.5 * first["eps"], 1.0 - 0.5 * first["eps"]
    )
    np.testing.assert_allclose(first["value"], expected_value, rtol=1e-5, atol=0)


def test_all_period_node_banks_participate_in_native_budget_admission() -> None:
    """An individually small core cannot hide the model's retained support banks."""

    def model(n_periods):
        regimes = {
            "alive": Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            )
        }
        if n_periods > 1:
            regimes["last"] = Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            )
        return Model(
            regimes=regimes,
            n_periods=n_periods,
            regime_id_class=_TerminalId if n_periods == 1 else _TwoId,
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={}
            if n_periods == 1
            else {
                "alive": {
                    "alive": PeriodRange(start=0, exclusive_stop=n_periods - 2),
                    "last": Periods(values=(n_periods - 2,)),
                }
            },
            execution_config=ExecutionConfig(device_memory_bytes=131072),
            shocks={
                "innovation": ShockGroup(
                    shocks={"eps": Normal(mu=0, sigma=1)},
                    nodes=MonteCarlo(n=1024, seed=19),
                )
            },
        )

    one = model(1).solve(params={}, log_level="off", runtime_checks=False)
    assert np.isfinite(one.values[0]["alive"])
    with pytest.raises(ExecutionPlanningError, match=r"resident|budget|memory"):
        model(16).solve(
            params={"discount_factor": 0.5}, log_level="off", runtime_checks=False
        )


def test_signed_ancestry_cannot_enter_the_native_decision() -> None:
    """Positive concrete weights do not erase a rule's signed provenance."""

    with pytest.raises(ShockDeclarationError, match=r"Signed|signed"):
        Model(
            regimes={
                "alive": Regime(
                    functions={"utility": _squared_shock}, shocks=("innovation",)
                )
            },
            n_periods=1,
            regime_id_class=_TerminalId,
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={},
            shocks={
                "innovation": ShockGroup(
                    shocks={"eps": Normal(mu=0, sigma=1)},
                    nodes=UserNodes(
                        nodes=np.array([[-1.0], [1.0]]),
                        weights=np.array([0.5, 0.5]),
                        signed=True,
                    ),
                )
            },
        )


def test_saved_replay_retains_its_physical_innovations(tmp_path) -> None:
    """A restored simulation has the exact physical data used by its native decision."""

    model = _runtime_scale_model()
    simulated = model.simulate(
        params={"innovation": {"eps__sigma": 2.0}},
        initial_conditions={
            "period": jnp.zeros(3, dtype=jnp.int32),
            "regime_id": jnp.zeros(3, dtype=jnp.int32),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    )
    simulated.save(directory=tmp_path / "physical-replay")
    restored = SimulationResult.load(directory=tmp_path / "physical-replay")
    np.testing.assert_array_equal(
        restored.to_dataframe()["eps"], simulated.to_dataframe()["eps"]
    )


@pytest.mark.parametrize("family", ["dcegm", "decoupled"])
def test_new_iid_refuses_uncomposed_native_families(family: str) -> None:
    """Legacy admitted families retain their routes; new IID requires composition."""

    if family == "dcegm":
        base = dcegm_paper_twin.get_model(solver="dcegm")
        regime_id_class = dcegm_paper_twin.TwinRegimeId
        # A dated DC-EGM source is call-owned, refused at its declaration shape.
        message = r"requires original one-state one-action DCEGM"
    else:
        base = test_decoupled._model()
        regime_id_class = test_decoupled.RegimeId
        message = r"Current IID"
    with pytest.raises(UnsupportedOperationError, match=message):
        Model(
            regimes={
                name: regime.replace(shocks=("innovation",))
                for name, regime in base.user_regimes.items()
            },
            ages=base.ages,
            n_periods=base.n_periods if base.ages is None else None,
            regime_id_class=regime_id_class,
            initial_nodes=base.initial_nodes,
            edges=base.edges,
            shocks={
                "innovation": ShockGroup(
                    shocks={"new_iid": Normal(mu=0, sigma=0)},
                    nodes=UserNodes(nodes=np.array([[0.0]]), weights=np.array([1.0])),
                )
            },
        )


@pytest.mark.parametrize("simulate", [False, True])
@pytest.mark.parametrize("callable_law", [False, True])
def test_native_scalar_law_rejects_batched_marginals(
    *, simulate: bool, callable_law: bool
) -> None:
    """Native cells cannot silently interpret a parameter batch as node rows."""

    model = _runtime_scale_model() if not callable_law else _vector_scale_model()
    params = {
        "innovation": {
            "scales" if callable_law else "eps__sigma": jnp.asarray([1.0, 2.0])
        }
    }
    with pytest.raises(ShockDeclarationError, match=r"innovation.*eps.*scalar"):  # noqa: PT012 - exercise both public dispatch routes
        if simulate:
            model.simulate(
                params=params,
                initial_conditions={
                    "period": jnp.zeros(1, dtype=jnp.int32),
                    "regime_id": jnp.zeros(1, dtype=jnp.int32),
                },
                seed=47,
                log_level="off",
                runtime_checks=False,
            )
        else:
            model.solve(params=params, log_level="off", runtime_checks=False)


def _vector_scale(*, scales: FloatND) -> FloatND:
    return scales


def _vector_scale_model() -> Model:
    return Model(
        regimes={
            "alive": Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=_vector_scale)},
                nodes=UserNodes(
                    nodes=np.array([[-1.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )


def _lookup_scale(*, scales: FloatND) -> FloatND:
    return scales[1]


def test_native_scalar_law_allows_vector_helper_inputs() -> None:
    model = _vector_scale_model()
    group = ShockGroup(
        shocks={"eps": Normal(mu=0, sigma=_lookup_scale)},
        nodes=model.shocks["innovation"].nodes,
    )
    model = Model(
        regimes=model.user_regimes,
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=model.initial_nodes,
        edges={},
        shocks={"innovation": group},
    )
    params = {"innovation": {"scales": jnp.asarray([1.0, 2.0])}}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(solution.values[0]["alive"], 4.0)
    simulated = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(1, dtype=jnp.int32),
            "regime_id": jnp.zeros(1, dtype=jnp.int32),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    )
    assert np.isfinite(simulated.to_dataframe()["eps"]).all()


def _impute_years(*, wealth: ContinuousState) -> FloatND:
    return wealth * 0.0


def _solve_scale(*, wealth: ContinuousState) -> FloatND:
    return 1.0 + wealth


def _actual_scale(*, years: ContinuousState) -> FloatND:
    return 1.0 + years


def _from_scale_helper(*, scale_helper: FloatND) -> FloatND:
    return scale_helper


def _fixed_years(*, years: ContinuousState) -> FloatND:
    return years


def _fixed_wealth(*, wealth: ContinuousState) -> FloatND:
    return wealth


@pytest.mark.parametrize("budgeted", [False, True])
def test_current_law_replay_uses_actual_carried_state(*, budgeted: bool) -> None:
    """Simulate-phase law helpers see actual carried rows, never solve imputations."""

    model = Model(
        regimes={
            "alive": Regime(
                states={
                    "wealth": LinSpacedGrid(start=0, stop=1, n_points=2),
                    "years": Phased(
                        solve=_impute_years,
                        simulate=LinSpacedGrid(start=0, stop=10, n_points=2),
                    ),
                },
                functions={
                    "utility": _squared_shock,
                    "scale_helper": Phased(solve=_solve_scale, simulate=_actual_scale),
                },
                state_transitions={"wealth": _fixed_wealth, "years": _fixed_years},
                shocks=("innovation",),
            ),
            "last": Regime(
                states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _solve_scale},
            ),
        },
        n_periods=2,
        regime_id_class=_TwoId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"last": Periods(values=(0,))}},
        execution_config=ExecutionConfig(
            device_memory_bytes=2**26, axis_widths={"subject": 1}
        )
        if budgeted
        else ExecutionConfig(),
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=_from_scale_helper)},
                nodes=UserNodes(
                    nodes=np.array([[-1.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )
    params = {"discount_factor": 0.0}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(solution.values[0]["alive"], [1.0, 4.0])
    actual_years = jnp.asarray([2.0, 5.0])
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.zeros(2, dtype=jnp.int32),
            "wealth": jnp.zeros(2),
            "years": actual_years,
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe()
    keys = draw_row_keys(
        key=jax.random.key(47),
        address_words=np.asarray(
            RandomAddress(kind="shock", name="eps").words(), dtype=np.uint32
        ),
        row_high=np.zeros(2, dtype=np.uint32),
        row_low=np.arange(2, dtype=np.uint32),
        period=0,
    )
    expected = (1 + actual_years) * jax.vmap(standard_normal)(keys)
    np.testing.assert_allclose(
        result.query("period == 0")["eps"], expected, rtol=1e-5, atol=0
    )


def test_forward_inventory_reserves_physical_bank_and_actual_producers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The profile reserves physical rows and admitted key/status operations."""

    profiles: list[SimulationChunkProfile] = []
    original = admission.profile_simulation_chunk

    def observe(**call: Unpack[ChunkProfileInputs]) -> SimulationChunkProfile:
        profile = original(**call)
        profiles.append(profile)
        return profile

    monkeypatch.setattr(admission, "profile_simulation_chunk", observe)
    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        execution_config=ExecutionConfig(
            device_memory_bytes=2**27, axis_widths={"subject": 32}
        ),
        shocks={
            "innovation": ShockGroup(
                shocks={
                    name: Normal(mu=0, sigma=0)
                    for name in ("eps", *(f"aux{i}" for i in range(15)))
                },
                nodes=UserNodes(nodes=np.zeros((1, 16)), weights=np.ones(1)),
            )
        },
    )
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    result = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(32, dtype=jnp.int32),
            "regime_id": jnp.zeros(32, dtype=jnp.int32),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    )
    assert profiles
    record_bytes = sum(
        leaf.nbytes
        for period in result.raw_results.values()
        for record in period.values()
        for leaf in jax.tree.leaves(
            tuple(getattr(record, field.name) for field in dataclasses.fields(record))
        )
    )
    profile = profiles[-1]
    assert sum(profile.output_reservation.values()) >= record_bytes
    names = [stage.name for stage in profile.stages]
    assert names.count("draw_window_keys") == 16
    assert "_current_iid_invalid" in names
    # A controlled omission must change the real profile's strict admission result.
    live = measure_buffer_footprint(tree=())
    devices = tuple(profile.output_reservation)
    full = plan_simulation_chunks(
        candidates=(32,),
        profile_candidate=lambda *, n_subjects: {profile.n_subjects: profile}[
            n_subjects
        ],
        live=live,
        budget_bytes=2**27,
        devices=devices,
    )
    budget = max(full.required_bytes.values()) - 1
    with pytest.raises(ExecutionPlanningError, match=r"chunk"):
        plan_simulation_chunks(
            candidates=(32,),
            profile_candidate=lambda *, n_subjects: {profile.n_subjects: profile}[
                n_subjects
            ],
            live=live,
            budget_bytes=budget,
            devices=devices,
        )
    physical_bytes = sum(
        value.nbytes
        for period in result.raw_results.values()
        for record in period.values()
        for value in record.shocks.values()
    )
    omitted = replace(
        profile,
        output_reservation=MappingProxyType(
            {
                device: amount - physical_bytes // len(devices)
                for device, amount in profile.output_reservation.items()
            }
        ),
    )
    control = plan_simulation_chunks(
        candidates=(32,),
        profile_candidate=lambda *, n_subjects: {omitted.n_subjects: omitted}[
            n_subjects
        ],
        live=live,
        budget_bytes=budget,
        devices=devices,
    )
    assert max(control.required_bytes.values()) <= budget


def test_solve_law_reads_carried_imputation_without_a_grid_axis() -> None:

    model = Model(
        regimes={
            "alive": Regime(
                states={
                    "wealth": LinSpacedGrid(start=0, stop=1, n_points=2),
                    "years": Phased(
                        solve=_impute_years,
                        simulate=LinSpacedGrid(start=0, stop=10, n_points=2),
                    ),
                },
                functions={"utility": _wealth_plus_shock},
                state_transitions={"wealth": _fixed_wealth, "years": _fixed_years},
                shocks=("innovation",),
            ),
            "last": Regime(
                states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _solve_scale},
            ),
        },
        n_periods=2,
        regime_id_class=_TwoId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"last": Periods(values=(0,))}},
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=_actual_scale, sigma=0)},
                nodes=UserNodes(nodes=np.array([[0.0]]), weights=np.ones(1)),
            )
        },
    )
    solution = model.solve(
        params={"discount_factor": 0.0}, log_level="off", runtime_checks=False
    )
    np.testing.assert_allclose(solution.values[0]["alive"], [1.0, 2.0])
    assert solution.values[0]["alive"].shape == (2,)


def test_new_iid_policy_query_is_explicitly_deferred() -> None:
    """Unobserved current-IID policy queries require the later query semantics."""
    model = _runtime_scale_model()
    params = {"innovation": {"eps__sigma": 2.0}}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    with pytest.raises(UnsupportedOperationError, match=r"Current IID"):
        model.lookup_policy(
            params=params, solution=solution, period=0, regime_name="alive", states={}
        )


@pytest.mark.parametrize("automatic_solve", [False, True])
def test_runtime_correlation_mask_upload_uses_entry_admission(
    *, monkeypatch: pytest.MonkeyPatch, automatic_solve: bool
) -> None:

    base = _correlated_model()
    model = Model(
        regimes=base.user_regimes,
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=base.initial_nodes,
        edges={},
        shocks=base.shocks,
        execution_config=ExecutionConfig(device_memory_bytes=2**26),
    )
    calls: list[str] = []
    original = SimulationEntryAllocations.__call__

    # keyword-only-exempt: library-callback=monkeypatch.__call__
    def record(
        self: SimulationEntryAllocations,
        *,
        value: np.ndarray | jax.Array,
        dtype: np.dtype,
        name: str,
    ) -> jax.Array:
        calls.append(name)
        if name.endswith("_lcm_corr_zero_pivots"):
            assert isinstance(value, np.ndarray)
            assert value.dtype == np.dtype(np.bool_)
        return original(self, value=value, dtype=dtype, name=name)

    monkeypatch.setattr(SimulationEntryAllocations, "__call__", record)
    solution = (
        None
        if automatic_solve
        else model.solve(
            params={"joint": {"corr": np.array([[1.0, 0.3], [0.3, 1.0]])}},
            log_level="off",
            runtime_checks=False,
        )
    )
    model.simulate(
        solution=solution,
        params={"joint": {"corr": np.array([[1.0, 0.3], [0.3, 1.0]])}},
        initial_conditions={
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.zeros(2, dtype=jnp.int32),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    )
    assert any(name.endswith("_lcm_corr_zero_pivots") for name in calls)


def test_nonproduct_joint_rows_are_not_reconstructed_from_marginals() -> None:
    base = _correlated_model()
    group = ShockGroup(
        shocks={"first": Normal(mu=0, sigma=1), "second": Normal(mu=0, sigma=1)},
        nodes=UserNodes(
            nodes=np.array([[-1.0, -2.0], [1.0, 2.0]]), weights=np.array([0.5, 0.5])
        ),
    )
    model = Model(
        regimes=base.user_regimes,
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=base.initial_nodes,
        edges={},
        shocks={"joint": group},
    )
    np.testing.assert_allclose(
        model.solve(params={}, log_level="off", runtime_checks=False).values[0][
            "alive"
        ],
        2.0,
    )


@pytest.mark.parametrize("poison", [np.nan, -np.inf])
def test_null_joint_events_do_not_poison_native_values(poison: float) -> None:
    def utility(*, first: ScalarInt, second: ScalarInt) -> FloatND:
        return jnp.where((first == 1) & (second == 1), poison, 4.0)

    base = _rare_model()
    model = Model(
        regimes={
            "alive": base.user_regimes["alive"].replace(functions={"utility": utility})
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=base.initial_nodes,
        edges={},
        shocks=base.shocks,
    )
    value = model.solve(
        params={
            "joint": {
                "first__probs": np.array([1.0, 0.0]),
                "second__probs": np.array([0.0, 1.0]),
            }
        },
        log_level="off",
        runtime_checks=False,
    ).values[0]["alive"]
    np.testing.assert_allclose(value, 4.0)


@pytest.mark.parametrize(
    "channel",
    [
        "co-map",
        "invariant blocks",
        "bound",
        "gated source",
        "gated open target",
        "gated fallback",
        "same-period",
    ],
)
def test_current_iid_refuses_actual_uncomposed_reader_intersections(
    channel: str,
) -> None:

    if channel in {"co-map", "invariant blocks"}:
        base = invariant._model(
            blocked=channel == "invariant blocks",
            enable_jit=False,
            device_memory_bytes=None,
        )
        regime_id_class = invariant.RegimeId
        target = "working"
        execution = (
            ExecutionConfig(
                devices=(0,), sharded_states=("pref_type",), device_memory_bytes=None
            )
            if channel == "co-map"
            else ExecutionConfig(
                devices=(0,),
                invariant_block_widths={"pref_type": 1},
                device_memory_bytes=None,
            )
        )
    elif channel == "bound":
        base = bound._model(
            declaration=post_decision_lower_bound(
                margin=bound.LIQUID, lower=bound.lower
            )
        )
        regime_id_class = bound.RegimeId
        target = "done"
        execution = ExecutionConfig(device_memory_bytes=None)
    elif channel.startswith("gated"):
        base = gated._build_model(gate=gated._closed_gate, enable_jit=False)
        regime_id_class = gated.RegimeId
        target = {
            "gated source": "src",
            "gated open target": "stateless_target",
            "gated fallback": "stateless_fallback",
        }[channel]
        execution = ExecutionConfig(device_memory_bytes=None)
    else:
        base = reference._make_participation_model()
        regime_id_class = reference.ParticipationRegimeId
        target = "single_f"
        execution = ExecutionConfig(device_memory_bytes=None)
    expected = "gated value channel" if channel.startswith("gated") else channel
    with pytest.raises(UnsupportedOperationError, match=f"Current IID.*{expected}"):
        Model(
            states={"pref_type": base.user_regimes["working"].states["pref_type"]}
            if channel == "co-map"
            else {},
            regimes={
                name: regime.replace(
                    states={
                        key: value
                        for key, value in regime.states.items()
                        if channel != "co-map" or key != "pref_type"
                    },
                    shocks=("innovation",) if name == target else regime.shocks,
                )
                for name, regime in base.user_regimes.items()
            },
            ages=base.ages,
            n_periods=base.n_periods if base.ages is None else None,
            regime_id_class=regime_id_class,
            initial_nodes=base.initial_nodes,
            edges=base.edges,
            execution_config=execution,
            enable_jit=base.enable_jit,
            shocks={
                "innovation": ShockGroup(
                    shocks={"new_iid": Normal(mu=0, sigma=0)},
                    nodes=UserNodes(nodes=np.array([[0.0]]), weights=np.ones(1)),
                )
            },
        )


def _active_diagonal_value(
    *, first: FloatND, second: FloatND, third: FloatND
) -> FloatND:
    return jnp.where((first == 0) & (second == 0), third, 0.0)


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_exact_positive_correlation_cannot_publish_a_collapsed_active_pivot(
    *, runtime: bool, compiled: bool
) -> None:
    """A rounded active pivot is either retained positive or explicitly unresolved."""
    near = np.nextafter(np.float32(-0.5), np.float32(0.0))
    matrix = np.array(
        [[1.0, 0.5, 0.5], [0.5, 1.0, near], [0.5, near, 1.0]], dtype=np.float32
    )
    group = ShockGroup(
        shocks={name: Normal(mu=0, sigma=1) for name in ("first", "second", "third")},
        correlation=None if runtime else matrix,
        nodes=UserNodes(nodes=np.eye(3), weights=np.full(3, 1 / 3)),
    )
    assert group.validate_runtime_correlation(correlation=matrix) == (
        False,
        False,
        False,
    )
    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _active_diagonal_value}, shocks=("joint",)
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={"joint": group},
        enable_jit=compiled,
        execution_config=ExecutionConfig()
        if compiled
        else ExecutionConfig(device_memory_bytes=None),
    )
    try:
        result = model.solve(
            params={"joint": {"corr": matrix}} if runtime else {},
            log_level="off",
            runtime_checks=False,
        )
    except ShockDeclarationError:
        return
    assert float(result.values[0]["alive"]) > 0


@pytest.mark.coverage(backends=("cpu", "gpu-small", "gpu-large"), precisions="both")
@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_public_group_correlation_cannot_label_a_collapsed_pivot_valid(
    *, runtime: bool, compiled: bool
) -> None:
    near = np.nextafter(np.float32(-0.5), np.float32(0.0))
    matrix = np.array(
        [[1.0, 0.5, 0.5], [0.5, 1.0, near], [0.5, near, 1.0]], dtype=np.float32
    )
    names = ("first", "second", "third")
    group = ShockGroup(
        shocks={name: Normal(mu=0, sigma=1) for name in names},
        correlation=None if runtime else matrix,
        nodes=UserNodes(nodes=np.eye(3), weights=np.full(3, 1 / 3)),
    )
    if runtime:
        group = group.bind_correlation(correlation=matrix)

    def evaluate():
        return group.transform(
            base_nodes=jnp.eye(3),
            params={"corr": jnp.asarray(matrix)} if runtime else {},
        )

    result = jax.jit(evaluate)() if compiled else evaluate()
    diagonal = np.diag(np.column_stack([result.values[name] for name in names]))
    assert not bool(result.valid) or np.all(np.isfinite(diagonal) & (diagonal > 0))


@categorical(ordered=False)
class _IslandId:
    src: ScalarInt
    stateless_target: ScalarInt
    stateless_fallback: ScalarInt
    iid_island: ScalarInt


def test_current_iid_preserves_an_unrelated_legacy_gated_component() -> None:
    base = gated._build_model(gate=gated._closed_gate, enable_jit=False)
    model = Model(
        regimes={
            **base.user_regimes,
            "iid_island": Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            ),
        },
        ages=base.ages,
        regime_id_class=_IslandId,
        initial_nodes={0: ("src", "iid_island")},
        edges=base.edges,
        enable_jit=False,
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=2, sigma=0)},
                nodes=UserNodes(nodes=np.array([[0.0]]), weights=np.ones(1)),
            )
        },
    )
    result = model.solve(
        params={"src": {"koopmans_aggregator": {"discount_factor": gated._BETA}}},
        log_level="off",
        runtime_checks=False,
    )
    np.testing.assert_allclose(
        result.values[0]["src"], gated._BETA * gated._FALLBACK_VALUE
    )
    np.testing.assert_allclose(result.values[0]["iid_island"], 4.0)


def _legacy_with_new(*, eps: FloatND, new_iid: FloatND) -> FloatND:
    return eps**2 + new_iid**2


def test_current_iid_refuses_uncomposed_legacy_fold_axis() -> None:
    with pytest.raises(UnsupportedOperationError, match=r"Current IID.*legacy folded"):
        Model(
            regimes={
                "alive": Regime(
                    states={
                        "eps": NormalIIDProcess(
                            n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, fold=True
                        )
                    },
                    functions={"utility": _legacy_with_new},
                    shocks=("innovation",),
                )
            },
            n_periods=1,
            regime_id_class=_TerminalId,
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={},
            shocks={
                "innovation": ShockGroup(
                    shocks={"new_iid": Normal(mu=0, sigma=0)},
                    nodes=UserNodes(nodes=np.array([[0.0]]), weights=np.ones(1)),
                )
            },
        )


@pytest.mark.parametrize(
    "law_name", ["normal-hermite", "normal-spaced", "lognormal", "uniform"]
)
def test_affine_legacy_fold_rule_preserves_ordinary_values(law_name: str) -> None:
    if law_name == "uniform":
        legacy = UniformIIDProcess(n_points=5, start=-1.0, stop=3.0, fold=True)
        law = Uniform(start=-1, stop=3)
        rule = EqualSpacedUniform(n=5)
    else:
        gauss_hermite = law_name != "normal-spaced"
        legacy = (LogNormalIIDProcess if law_name == "lognormal" else NormalIIDProcess)(
            n_points=5,
            gauss_hermite=gauss_hermite,
            mu=0.3,
            sigma=1.2,
            n_std=None if gauss_hermite else 2.5,
            fold=True,
        )
        law = (LogNormal if law_name == "lognormal" else Normal)(mu=0.3, sigma=1.2)
        rule = GaussHermite(n=5) if gauss_hermite else EqualSpaced(n=5, n_std=2.5)
    reference = Model(
        regimes={
            "alive": Regime(
                states={"eps": legacy}, functions={"utility": _squared_shock}
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
    ).solve(params={}, log_level="off", runtime_checks=False)
    candidate = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _squared_shock}, shocks=("innovation",)
            )
        },
        shocks={"innovation": ShockGroup(shocks={"eps": law}, nodes=Product(eps=rule))},
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
    ).solve(params={}, log_level="off", runtime_checks=False)
    assert_array_almost_equal(
        candidate.values[0]["alive"],
        reference.values[0]["alive"],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.requires(min_devices=2, device="cpu")
def test_current_iid_refuses_uncomposed_action_partitions() -> None:
    if len(jax.devices()) < 2:
        pytest.skip("Requires two visible CPU devices.")
    with pytest.raises(
        UnsupportedOperationError, match=r"Current IID.*action partitions"
    ):
        Model(
            states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2)},
            regimes={
                "alive": Regime(
                    actions={"action": DiscreteGrid(category_class=_Action)},
                    functions={"utility": _utility},
                    state_transitions={"wealth": _next_wealth},
                    shocks=("innovation",),
                ),
                "last": Regime(functions={"utility": _solve_scale}),
            },
            n_periods=2,
            regime_id_class=_TwoId,
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={"alive": {"last": Periods(values=(0,))}},
            execution_config=ExecutionConfig(
                devices=(0, 1), action_partitions={"alive": 2}
            ),
            shocks={
                "innovation": ShockGroup(
                    shocks={"eps": Normal(mu=0, sigma=1)},
                    nodes=UserNodes(
                        nodes=np.array([[-1.0], [1.0]]), weights=np.full(2, 0.5)
                    ),
                ),
            },
        )


def _sharded_flow(*, wealth: FloatND, eps: FloatND) -> FloatND:
    return wealth + eps**2


@pytest.mark.requires(min_devices=2, device="cpu")
def test_current_iid_preserves_native_state_and_subject_sharding() -> None:
    if len(jax.devices()) < 2:
        pytest.skip("Requires two visible CPU devices.")
    model = Model(
        states={"wealth": LinSpacedGrid(start=0, stop=5, n_points=6)},
        regimes={
            "alive": Regime(
                functions={"utility": _sharded_flow}, shocks=("innovation",)
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        execution_config=ExecutionConfig(
            devices=(0, 1), sharded_states=("wealth",), device_memory_bytes=2**27
        ),
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=None)},
                nodes=UserNodes(
                    nodes=np.array([[-1.0], [1.0]]), weights=np.full(2, 0.5)
                ),
            )
        },
    )
    params = {"innovation": {"eps__sigma": 1.0}}
    solution = model.solve(params=params, log_level="off", runtime_checks=False)
    np.testing.assert_allclose(solution.values[0]["alive"], np.arange(6) + 1.0)
    assert len(solution.values[0]["alive"].sharding.device_set) == 2
    replay = model.simulate(
        params=params,
        initial_conditions={
            "period": jnp.zeros(6, dtype=jnp.int32),
            "regime_id": jnp.zeros(6, dtype=jnp.int32),
            "wealth": jnp.arange(6, dtype=jnp.asarray(0.0).dtype),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe()
    np.testing.assert_allclose(
        replay["value"],
        replay["wealth"] + replay["eps"] ** 2,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=0,
    )


def _rounded_physical_utility(*, eps: FloatND) -> FloatND:
    return eps - 1e20


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("width", [1, 3])
def test_ordinary_native_consumers_read_the_published_physical_bits(
    *, compiled: bool, width: int
) -> None:
    """Cancellation after a rounded law cannot recover its hidden base coordinate."""
    model = Model(
        regimes={
            "alive": Regime(
                functions={"utility": _rounded_physical_utility}, shocks=("innovation",)
            )
        },
        n_periods=1,
        regime_id_class=_TerminalId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        enable_jit=compiled,
        execution_config=ExecutionConfig(axis_widths={"subject": width}),
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=1e20, sigma=1)},
                nodes=UserNodes(nodes=np.array([[0.5]]), weights=np.ones(1)),
            )
        },
    )
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_array_equal(solution.values[0]["alive"], 0.0)
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(3, dtype=jnp.int32),
            "regime_id": jnp.zeros(3, dtype=jnp.int32),
        },
        seed=47,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe()
    physical = frame["eps"].to_numpy()
    np.testing.assert_array_equal(
        frame["value"], physical - np.asarray(1e20, dtype=physical.dtype)
    )


def test_native_scale_sensitivity_consumes_the_physical_law() -> None:
    core = _scalar_core(
        model=_runtime_scale_model(), params={"innovation": {"eps__sigma": 2.0}}
    )

    def value(scale: FloatND) -> FloatND:
        groups = core.arguments[CURRENT_IID_ARGUMENT]
        return core.function(
            **{
                **core.arguments,
                CURRENT_IID_ARGUMENT: {
                    "innovation": {**groups["innovation"], "eps__sigma": scale}
                },
            }
        )[0]

    actual, derivative = jax.jit(jax.value_and_grad(value))(2.0)
    np.testing.assert_allclose(actual, 4.0)
    np.testing.assert_allclose(derivative, 4.0)
