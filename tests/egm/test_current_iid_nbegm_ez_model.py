"""Original NB ride reconstruction with the declared joint Epstein-Zin recursion."""

import math
from dataclasses import replace
from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_case_u_replay import CASE_U_FUTURE_REPLAY
from _lcm.egm.current_iid_nbegm import NBCaseURidePoint
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.solution.current_iid_nbegm import NativeNBCaseUPeriodKernel
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
from lcm.shocks import PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import ArtifactRef, ResultRetention
from lcm.solvers import NBEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt, UserParams
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


def _model() -> Model:
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
            "utility": _flow,
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
                    shocks={"eps": Uniform(start=0.0, stop=1.0)},
                    nodes=UserNodes(
                        nodes=np.asarray([[0.0], [1.0]]),
                        weights=np.asarray([0.5, 0.5]),
                    ),
                )
            },
        ),
        params=_params(),
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
    shocks = np.asarray([0.0, 0.5, 1.0])
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
