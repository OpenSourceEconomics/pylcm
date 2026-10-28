"""Original NB ride reconstruction with the declared joint Epstein-Zin recursion."""

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm import nbegm_step
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_case_u_replay import (
    CASE_U_FUTURE_REPLAY,
    CaseUReplayChildren,
)
from _lcm.egm.current_iid_nbegm import NBCaseUProgram, NBCaseURidePoint
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.execution.core_program import CoreBuildContext, CoreProgram
from _lcm.solution.current_iid_case_u_native import (
    CaseUExactOutput,
    CaseURideExactFunction,
)
from _lcm.solution.current_iid_nbegm import NativeNBCaseUPeriodKernel
from _lcm.solution.feasibility import BoundedResult
from _lcm.solution.generation import CanonicalModelGeneration
from _lcm.typing import (
    EconFunctionArg,
    FlatEdgeParams,
    FlatRegimeParams,
    QualifiedName,
    ReferenceName,
)
from lcm import (
    AgeGrid,
    CESAggregator,
    ConsumptionSavingsRegime,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    Model,
    PowerMean,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import IncompatibleSolutionError
from lcm.persistence import load_solution
from lcm.shocks import PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import EGM_CONTINUATION, ArtifactRef, ResultRetention
from lcm.solvers import NBEGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    Float1D,
    FloatND,
    IntND,
    RegimeName,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_CONSUMPTION_SCALE = 15.0 / 4.0
_WEALTH = 1.0 + _CONSUMPTION_SCALE


@categorical(ordered=False)
class _Regimes:
    child: ScalarInt
    done: ScalarInt
    other: ScalarInt


def _flow(
    *, consumption: ContinuousAction, q: ContinuousState, eps: FloatND
) -> FloatND:
    return jnp.exp(q) * consumption ** (1.0 + eps)


def _scaled_flow(
    *,
    consumption: ContinuousAction,
    q: ContinuousState,
    eps: FloatND,
    log_scale: FloatND,
) -> FloatND:
    return jnp.exp(q + log_scale) * consumption ** (1.0 + eps)


def _resources(*, wealth: ContinuousState, eps: FloatND) -> FloatND:
    return wealth + eps * _CONSUMPTION_SCALE


def _landing(savings: FloatND) -> ContinuousState:
    return savings


def _first(wealth: ContinuousState) -> FloatND:
    return wealth + 2.0


def _second(wealth: ContinuousState) -> FloatND:
    return wealth + 4.0


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _params() -> UserParams:
    return {
        "child": {
            "koopmans_aggregator": {
                "discount_factor": 0.5,
                "intertemporal_elasticity_of_substitution": 1.0,
            },
            "certainty_equivalent": {"risk_aversion": 1.0},
        }
    }


def _first_order_params(
    *,
    log_scale: float = 0.0,
    discount: float = 0.5,
    risk: float = 1.0,
    stop: float = 1.0,
) -> UserParams:
    return {
        "child": {
            "utility": {"log_scale": log_scale},
            "koopmans_aggregator": {
                "discount_factor": discount,
                "intertemporal_elasticity_of_substitution": 1.0,
            },
            "certainty_equivalent": {"risk_aversion": risk},
        },
        "income": {"eps__start": 0.0, "eps__stop": stop},
    }


def _model(*, first_order: bool = False) -> Model:
    declarations = _actual_declarations()
    original = declarations.regimes["child"]
    assert isinstance(original, ConsumptionSavingsRegime)
    savings = IrregSpacedGrid(points=np.asarray([0.25, 0.5, 1.0, 2.0, 4.0, 6.0]))
    child = replace(
        original,
        states={
            "wealth": IrregSpacedGrid(points=np.asarray([_WEALTH, 5.0])),
            "q": IrregSpacedGrid(points=np.asarray([0.0, 1.0])),
        },
        actions={"consumption": IrregSpacedGrid(points=np.asarray([0.125, 9.0]))},
        functions={
            "utility": _scaled_flow if first_order else _flow,
            "resources": _resources,
            "savings": original.functions["savings"],
        },
        state_transitions={
            "wealth": {"done": _landing, "other": _landing},
            "q": fixed_transition("q"),
        },
        certainty_equivalent=PowerMean(),
        koopmans_aggregator=CESAggregator(),
        solver=NBEGM(savings_grid=savings),
    )
    return Model._from_call_owned_params(
        declarations=replace(
            declarations,
            regimes={
                "child": child,
                "done": Regime(
                    states={"wealth": savings}, functions={"utility": _first}
                ),
                "other": Regime(
                    states={"wealth": savings}, functions={"utility": _second}
                ),
            },
            ages=AgeGrid(exact_values=(0, 1)),
            initial_nodes=InitialNodes(by_age={0: "child"}),
            regime_id_class=_Regimes,
            edges={
                "child": Transition(
                    targets={"done": (0,), "other": (0,)},
                    law={
                        "done": StochasticTransition(func=_half),
                        "other": StochasticTransition(func=_half),
                    },
                )
            },
            execution_config=ExecutionConfig(device_memory_bytes=None),
            shocks={
                "income": ShockGroup(
                    shocks={
                        "eps": Uniform()
                        if first_order
                        else Uniform(start=0.0, stop=1.0)
                    },
                    nodes=UserNodes(
                        nodes=np.asarray([[0.0], [1.0]]),
                        weights=np.asarray([0.5, 0.5]),
                    ),
                )
            },
        ),
        params=_first_order_params() if first_order else _params(),
    )


def _conditional_value(*, q: float, shock: float) -> float:
    consumption = (1.0 + shock) * _CONSUMPTION_SCALE
    return math.sqrt(math.exp(q) * consumption ** (1.0 + shock) * math.sqrt(15.0))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("inspect_projection", [False, True])
def test_original_nbegm_ez_u_model_keeps_joint_ce_and_physical_candidates(
    *, inspect_projection: bool
) -> None:
    """The original savings node one solves every represented and supplied frame."""
    model = _model()
    params = _params()
    solution = (
        model.solve(
            params=params,
            log_level="off",
            retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
        )
        if inspect_projection
        else model.solve(params=params, log_level="off")
    )
    ref = ArtifactRef(period=0, regime="child", key=PROJECTED_EGM_CONTINUATION)
    expected = np.asarray(
        [
            np.mean([_conditional_value(q=q, shock=z) for z in (0.0, 1.0)])
            for q in (0.0, 1.0)
        ]
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["child"][0], expected, decimal=DECIMAL_PRECISION
    )
    assert ArtifactRef(period=0, regime="child", key=CASE_U_FUTURE_REPLAY) in (
        solution.replay_artifacts
    )
    if inspect_projection:
        stored = solution.retained_continuations[ref]
        assert isinstance(stored, ProjectedEGMCarry)
        for row in range(expected.size):
            length = int(stored.valid_length[row])
            positions = np.flatnonzero(np.asarray(stored.grid[row, :length]) == _WEALTH)
            assert positions.size == 1
            slot = positions[0]
            np.testing.assert_almost_equal(
                stored.value[row, slot], expected[row], decimal=DECIMAL_PRECISION
            )
            np.testing.assert_almost_equal(
                stored.economic_marginal[row, slot],
                expected[row] / (2.0 * _CONSUMPTION_SCALE),
                decimal=DECIMAL_PRECISION,
            )
    else:
        assert ref not in solution.retained_continuations
    shocks = np.asarray([0.0, 0.5, 1.0], dtype=canonical_float_dtype())
    q = np.asarray([0.0, 1.0, 0.25])
    simulated = model.simulate(
        params=params,
        solution=solution,
        seed=23,
        log_level="off",
        initial_conditions={
            "wealth": jnp.full(shocks.shape, _WEALTH),
            "q": jnp.asarray(q),
            "age": jnp.zeros(shocks.shape),
            "regime_id": jnp.full(
                shocks.shape, model.regime_names_to_ids["child"], jnp.int32
            ),
            "eps": PhysicalValues(values=shocks),
        },
    )
    original = simulated.raw_results["child"][0]
    np.testing.assert_array_almost_equal(
        original.actions["consumption"],
        (1.0 + shocks) * _CONSUMPTION_SCALE,
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        original.V_arr,
        np.asarray(
            [
                _conditional_value(q=state, shock=shock)
                for state, shock in zip(q, shocks, strict=True)
            ]
        ),
        decimal=DECIMAL_PRECISION,
    )
    for target in ("done", "other"):
        landing = simulated.raw_results.get(target, {}).get(1)
        if landing is not None:
            np.testing.assert_array_almost_equal(
                landing.states["wealth"],
                np.ones_like(landing.states["wealth"]),
                decimal=DECIMAL_PRECISION,
            )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_recursive_nbegm_u_retains_original_transform_and_aggregator() -> None:
    """Equal transforms and equivalent callables cannot replace actual source owners."""
    model = _model()
    kernel = model._regimes["child"].solution.period_kernels[0]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    point = kernel.point
    assert isinstance(point, NBCaseURidePoint)
    assert isinstance(point.certainty_equivalent, PowerMean)
    assert point.certainty_equivalent is kernel.context.certainty_equivalent
    assert point.koopmans_aggregator is kernel.context.koopmans_aggregator
    assert point.koopmans_aggregator is not None
    with pytest.raises(TypeError, match="Model-owned declarations"):
        replace(
            kernel,
            point=replace(
                point, certainty_equivalent=replace(point.certainty_equivalent)
            ),
        )
    with pytest.raises(TypeError, match="Model-owned declarations"):
        replace(
            kernel,
            point=replace(
                point, koopmans_aggregator=partial(point.koopmans_aggregator)
            ),
        )


def _array_namespace(
    *,
    flat: Mapping[RegimeName, FlatRegimeParams | FlatEdgeParams],
    name: RegimeName,
) -> MappingProxyType[QualifiedName, FloatND | IntND | BoolND]:
    """Keep the actual canonical array leaves without inventing parameter slots."""
    arrays: dict[QualifiedName, FloatND | IntND | BoolND] = {}
    for key, value in flat[name].items():
        assert isinstance(value, jax.Array)
        arrays[key] = value
    return MappingProxyType(arrays)


@pytest.fixture(scope="module")
def first_order_call() -> tuple[
    CanonicalModelGeneration, CoreProgram, CoreBuildContext, int, Float1D
]:
    """Acquire the real generation, original core and actually solved child rows."""
    model = _model(first_order=True)
    params = _first_order_params()
    solution = model.solve(
        params=params,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    flat, generation = model._prepare_call(params=params)
    execution = generation.execution
    kernel = execution._regimes["child"].solution.period_kernels[0]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    program = kernel.program
    assert isinstance(program.function, CaseURideExactFunction)
    assert program.function.point.pieces is kernel.point
    assert isinstance(kernel.point, NBCaseURidePoint)
    assert kernel.point.certainty_equivalent is kernel.context.certainty_equivalent
    assert kernel.point.koopmans_aggregator is kernel.context.koopmans_aggregator
    children = {}
    for target in ("done", "other"):
        child = solution.retained_continuations[
            ArtifactRef(period=1, regime=target, key=EGM_CONTINUATION)
        ]
        assert isinstance(child, EGMCarry)
        children[target] = child
    context = CoreBuildContext(
        state_action_space=kernel.context.state_action_space,
        next_regime_to_V_arr={
            target: solution.values[1][target] for target in children
        },
        next_regime_to_continuation=MappingProxyType(children),
        flat_params=flat,
        period=0,
        ages=kernel.context.ages,
    )
    actual = program.argument_builder(context)
    assert not any(
        isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(actual)
    )
    produced = program.function(**actual)
    assert isinstance(produced, BoundedResult)
    assert isinstance(produced.output, CaseUExactOutput)
    for status in produced.query_status or ():
        assert not any(bool(jnp.any(flag)) for flag in status)
    for status in produced.bound_status:
        assert not any(bool(jnp.any(flag)) for flag in status)
    stored = solution.retained_continuations[
        ArtifactRef(period=0, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(stored, ProjectedEGMCarry)
    positions = np.flatnonzero(
        np.asarray(stored.grid[0, : int(stored.valid_length[0])]) == _WEALTH
    )
    assert positions.size == 1
    slot = int(positions[0])
    expected_value = np.mean(
        [_conditional_value(q=0.0, shock=shock) for shock in (0.0, 1.0)]
    )
    expected = jnp.asarray(
        [expected_value, expected_value / (2.0 * _CONSUMPTION_SCALE)]
    )
    np.testing.assert_array_almost_equal(
        [
            produced.output.value[0, 0],
            produced.output.projected.economic_marginal[0, slot],
        ],
        expected,
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_almost_equal(
        produced.output.value[0, 0],
        solution.values[0]["child"][0, 0],
        decimal=DECIMAL_PRECISION,
    )
    # Concrete parameter calls, not tracers, establish this fixed-support owner.
    for changed in (
        _first_order_params(log_scale=0.125),
        _first_order_params(discount=0.625),
        _first_order_params(risk=1.125),
        _first_order_params(stop=1.125),
    ):
        _changed_flat, changed_generation = model._prepare_call(params=changed)
        assert changed_generation is generation
    return generation, program, context, slot, expected


def _represented_value_direction(  # noqa: PLR0915
    *,
    program: CoreProgram,
    context: CoreBuildContext,
    shock: float,
    direction: Literal["discount", "risk"],
) -> float:
    """Differentiate literal endpoint equations at the actual native branch.

    The spy supplies only original primal geometry/owner. Endpoint derivatives
    below come from the independently declared nu(s), Euler and fixed-corner
    equations, never from the production JVP being tested.
    """
    assert isinstance(program.function, CaseURideExactFunction)
    original = program.function.point
    assert isinstance(original, NBCaseUProgram)
    assert isinstance(original.pieces, NBCaseURidePoint)
    arguments = program.argument_builder(context)
    savings = arguments["_lcm_case_u_savings_nodes"]
    support = arguments["_lcm_nbegm_source_grid"]
    assert isinstance(savings, jax.Array)
    assert isinstance(support, jax.Array)
    point = replace(original.pieces.with_savings(nodes=savings), source_grid=support)
    supplied = arguments["next_regime_to_continuation"]
    assert isinstance(supplied, Mapping)
    children: dict[RegimeName, EGMCarry] = {}
    for target in point.targets:
        child = supplied[target]
        assert isinstance(child, EGMCarry)
        children[target] = child
    pool: dict[ReferenceName, EconFunctionArg] = {}
    for name, value in arguments.items():
        if not name.startswith("_") and name != "next_regime_to_continuation":
            assert isinstance(value, jax.Array)
            pool[name] = value
    q = pool["q"]
    assert isinstance(q, jax.Array)
    pool["q"] = q[0]
    native_envelope = nbegm_step.envelope_at_query
    with patch.object(
        nbegm_step, "envelope_at_query", wraps=native_envelope
    ) as observed:
        selected = point.select(
            liquid=jnp.asarray(_WEALTH),
            physical={"eps": jnp.asarray(shock)},
            pool=pool,
            children=children,
        )
        assert observed.call_count == 1
        assert observed.call_args is not None
        recorded = observed.call_args.kwargs
    assert nbegm_step.envelope_at_query is native_envelope
    assert not any(
        bool(jnp.any(flag))
        for flag in jax.tree.leaves((selected.bound_status, selected.query_status))
    )
    bank = {}
    for name in ("endog_grid", "value", "policy", "candidate_savings", "x_query"):
        operand = recorded[name]
        assert isinstance(operand, jax.Array)
        bank[name] = operand
    owner = int(selected.output.original.native_owner)
    count = bank["endog_grid"].size
    assert 0 <= owner < 2 * count - 1
    left = owner if owner < count - 1 else owner - count + 1
    right = left + 1 if owner < count - 1 else left
    alpha = 1.0 + shock

    def endpoint(index: int) -> tuple[float, float]:
        saving = float(bank["candidate_savings"][index])
        consumption = float(bank["policy"][index])
        value = float(bank["value"][index])
        nu = math.sqrt((saving + 2.0) * (saving + 4.0))
        log_ratio = math.log((saving + 4.0) / (saving + 2.0))
        derivative = (
            value * math.log(nu / consumption**alpha)
            if direction == "discount"
            else -value * log_ratio**2 / 16.0
        )
        coordinate = 0.0
        if index < savings.size:
            coordinate = (
                -4.0 * consumption
                if direction == "discount"
                else -consumption * log_ratio / (2.0 * saving + 6.0)
            )
            derivative += value * alpha / (2.0 * consumption) * coordinate
        return derivative, coordinate

    dv0, dx0 = endpoint(left)
    dv1, dx1 = endpoint(right)
    x0, x1 = float(bank["endog_grid"][left]), float(bank["endog_grid"][right])
    if x0 == x1:
        return dv0
    fraction = (float(bank["x_query"][0]) - x0) / (x1 - x0)
    slope = (float(bank["value"][right]) - float(bank["value"][left])) / (x1 - x0)
    return (
        (1.0 - fraction) * dv0
        + fraction * dv1
        - slope * ((1.0 - fraction) * dx0 + fraction * dx1)
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("direction", ["log_scale", "discount", "risk", "stop"])
def test_original_nbegm_ez_model_runtime_parameter_directions(
    *,
    first_order_call: tuple[
        CanonicalModelGeneration, CoreProgram, CoreBuildContext, int, Float1D
    ],
    direction: Literal["log_scale", "discount", "risk", "stop"],
) -> None:
    """Differentiate the actual Model core against independent economic equations."""
    generation, program, context, slot, expected = first_order_call
    kernel = generation.execution._regimes["child"].solution.period_kernels[0]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    assert kernel.program is program
    flat = context.flat_params
    child = _array_namespace(flat=flat, name="child")
    income = _array_namespace(flat=flat, name="income")
    leaf_names = (
        "utility__log_scale",
        "koopmans_aggregator__discount_factor",
        "certainty_equivalent__risk_aversion",
    )
    primal = jnp.stack([*(child[name] for name in leaf_names), income["eps__stop"]])
    np.testing.assert_array_equal(primal, [0.0, 0.5, 1.0, 1.0])

    def evaluate(parameters: Float1D) -> Float1D:
        actual_flat = MappingProxyType(
            {
                **flat,
                "child": MappingProxyType(
                    {
                        **child,
                        **{
                            name: parameters[index]
                            for index, name in enumerate(leaf_names)
                        },
                    }
                ),
                "income": MappingProxyType({**income, "eps__stop": parameters[3]}),
            }
        )
        actual = program.argument_builder(replace(context, flat_params=actual_flat))
        result = program.function(**actual)
        assert isinstance(result, BoundedResult)
        assert isinstance(result.output, CaseUExactOutput)
        return jnp.stack(
            (
                result.output.value[0, 0],
                result.output.projected.economic_marginal[0, slot],
            )
        )

    tangent = (
        jnp.zeros_like(primal)
        .at[("log_scale", "discount", "risk", "stop").index(direction)]
        .set(1.0)
    )
    values = [_conditional_value(q=0.0, shock=z) for z in (0.0, 1.0)]
    if direction == "log_scale":
        expected_direction = np.asarray(expected) * 0.5
    elif direction == "discount":
        expected_direction = np.asarray(
            [
                np.mean(
                    [
                        value
                        * math.log(
                            math.sqrt(15.0)
                            / ((1.0 + z) * _CONSUMPTION_SCALE) ** (1.0 + z)
                        )
                        for z, value in zip((0.0, 1.0), values, strict=True)
                    ]
                )
            ]
        )
    elif direction == "risk":
        expected_direction = np.asarray(
            [-float(expected[0]) * math.log(5.0 / 3.0) ** 2 / 16.0]
        )
    else:
        derivative = values[1] * (math.log(2.0 * _CONSUMPTION_SCALE) + 1.0) / 4.0
        expected_direction = np.asarray(
            [derivative, derivative / (2.0 * _CONSUMPTION_SCALE)]
        )
    if direction in ("discount", "risk"):
        continuous_optimum = expected_direction.copy()
        expected_direction = np.asarray(
            [
                np.mean(
                    [
                        _represented_value_direction(
                            program=program,
                            context=context,
                            shock=z,
                            direction=direction,
                        )
                        for z in (0.0, 1.0)
                    ]
                )
            ]
        )
        print(  # noqa: T201 - separate scientific representation reference
            "PYLCM607_EZ_DIRECTION_REFERENCE "
            + json.dumps(
                {
                    "direction": direction,
                    "continuous_optimum": float(continuous_optimum[0]),
                    "represented_native_branch": float(expected_direction[0]),
                    "representation_difference": float(
                        expected_direction[0] - continuous_optimum[0]
                    ),
                },
                sort_keys=True,
            )
        )
    for apply in (evaluate, jax.jit(evaluate)):
        value, derivative = jax.jvp(apply, (primal,), (tangent,))
        np.testing.assert_array_almost_equal(value, expected, decimal=DECIMAL_PRECISION)
        # Discount/risk compare the independently differentiated actual native
        # branch. The ideal continuous envelope response is reported separately;
        # marginal cross-derivatives have a separate acceptance obligation.
        actual_direction = (
            derivative[:1] if direction in ("discount", "risk") else derivative
        )
        np.testing.assert_array_almost_equal(
            actual_direction, expected_direction, decimal=DECIMAL_PRECISION
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nbegm_ez_default_archive_keeps_recursive_physical_replay(  # noqa: PLR0915
    *, tmp_path: Path
) -> None:
    """A fresh Model replays on/off-node frames from the original joint CE future."""
    model = _model()
    params = _params()
    solution = model.solve(params=params, log_level="off")
    assert solution.metadata.retention is ResultRetention.VALUES_AND_REPLAY
    assert ArtifactRef(
        period=0, regime="child", key=PROJECTED_EGM_CONTINUATION
    ) not in (solution.retained_continuations)
    future_ref = ArtifactRef(period=0, regime="child", key=CASE_U_FUTURE_REPLAY)
    future = solution.replay_artifacts[future_ref]
    assert isinstance(future, CaseUReplayChildren)
    assert future.child_names == ("done", "other")
    shocks = np.asarray([0.0, 0.5, 1.0], dtype=canonical_float_dtype())
    q = np.asarray([0.0, 0.25, 1.0])
    initial = {
        "wealth": jnp.full(shocks.shape, _WEALTH),
        "q": jnp.asarray(q),
        "age": jnp.zeros(shocks.shape),
        "regime_id": jnp.full(
            shocks.shape, model.regime_names_to_ids["child"], jnp.int32
        ),
        "eps": PhysicalValues(values=shocks),
    }
    original = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        seed=23,
        log_level="off",
    )
    path = tmp_path / "recursive-nb-u.lcm"
    solution.save(path=path)
    loaded = load_solution(path=path)
    with pytest.raises(IncompatibleSolutionError, match="model-authoritative template"):
        _ = loaded.replay_artifacts[future_ref]
    fresh = _model()
    old_kernel = model._regimes["child"].solution.period_kernels[0]
    kernel = fresh._regimes["child"].solution.period_kernels[0]
    assert isinstance(old_kernel, NativeNBCaseUPeriodKernel)
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    assert isinstance(kernel.point, NBCaseURidePoint)
    assert isinstance(old_kernel.point, NBCaseURidePoint)
    assert kernel.point.certainty_equivalent is kernel.context.certainty_equivalent
    assert (
        kernel.point.certainty_equivalent is not old_kernel.point.certainty_equivalent
    )
    assert kernel.point.koopmans_aggregator is kernel.context.koopmans_aggregator
    assert isinstance(kernel.program.function, CaseURideExactFunction)
    assert kernel.program.function.point.pieces is kernel.point
    replayed = fresh.simulate(
        params=params,
        solution=loaded,
        initial_conditions=initial,
        seed=23,
        log_level="off",
    )
    loaded_future = loaded.replay_artifacts[future_ref]
    assert isinstance(loaded_future, CaseUReplayChildren)
    assert (
        (loaded_future.source, loaded_future.period)
        == (
            future.source,
            future.period,
        )
        == ("child", 0)
    )
    assert loaded_future.child_names == future.child_names
    for before, after in zip(
        future.child_values, loaded_future.child_values, strict=True
    ):
        assert jax.tree.structure(before) == jax.tree.structure(after)
        for actual, expected in zip(
            jax.tree.leaves(after), jax.tree.leaves(before), strict=True
        ):
            assert (actual.shape, actual.dtype) == (expected.shape, expected.dtype)
            np.testing.assert_array_equal(actual, expected)
    expected_value = np.asarray(
        [
            _conditional_value(q=state, shock=shock)
            for state, shock in zip(q, shocks, strict=True)
        ]
    )
    for result in (original, replayed):
        child = result.raw_results["child"][0]
        np.testing.assert_array_almost_equal(
            child.actions["consumption"],
            (1.0 + shocks) * _CONSUMPTION_SCALE,
            decimal=DECIMAL_PRECISION,
        )
        np.testing.assert_array_almost_equal(
            child.V_arr,
            expected_value,
            decimal=DECIMAL_PRECISION,
        )
    np.testing.assert_array_equal(
        replayed.raw_results["child"][0].actions["consumption"],
        original.raw_results["child"][0].actions["consumption"],
    )
    np.testing.assert_array_equal(
        replayed.raw_results["child"][0].V_arr,
        original.raw_results["child"][0].V_arr,
    )
    for target in ("done", "other"):
        before = original.raw_results.get(target, {}).get(1)
        after = replayed.raw_results.get(target, {}).get(1)
        assert (before is None) == (after is None)
        if before is not None and after is not None:
            np.testing.assert_array_almost_equal(
                after.states["wealth"],
                np.ones_like(after.states["wealth"]),
                decimal=DECIMAL_PRECISION,
            )
            np.testing.assert_array_equal(
                after.states["wealth"], before.states["wealth"]
            )
            np.testing.assert_array_equal(after.V_arr, before.V_arr)
