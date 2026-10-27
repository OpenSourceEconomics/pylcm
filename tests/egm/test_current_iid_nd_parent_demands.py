"""Genuine ordinary-parent Euler demands on a current-NB child's passive rows."""

import math
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidValueFunctionError
from lcm.shocks import ShockGroup, Uniform, UserNodes
from lcm.solver_api import ArtifactRef, ArtifactStore
from lcm.solvers import DCEGM, NBEGM, ExactEnvelope
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations
from tests.egm.test_current_iid_nbegm_ride_model import (
    _flow,
    _inverse,
    _next_q,
    _resources,
    _terminal,
    _Work,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class _Regimes:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


def _parent_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _parent_inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _parent_resources(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _parent_wealth(*, savings: FloatND) -> ContinuousState:
    return savings + 0.75


def _parent_q(*, landing_q: FloatND) -> ContinuousState:
    return landing_q


def _params(*, landing_q: float) -> UserParams:
    return {
        "discount_factor": 1.0,
        "parent": {"next_q": {"landing_q": landing_q}},
        "child": {
            "utility": {"work_penalty": 1.25},
            "resources": {"work_gain": 1.5},
            "next_wealth": {"slope": 1.0},
            "next_q": {"ride_bonus": 0.25},
        },
    }


def _model(*, landing_q: float) -> Model:
    original = _actual_declarations()
    scalar = original.regimes["child"]
    assert isinstance(scalar, ConsumptionSavingsRegime)
    # Child optima for both work branches at declared and demanded liquid
    # coordinates are actual saving nodes; no continuous-optimum surrogate.
    saving_grid = IrregSpacedGrid(
        points=np.asarray(
            [
                0.25,
                1.0 / 3.0,
                0.5,
                0.75,
                5.0 / 6.0,
                1.0,
                1.125,
                1.25,
                1.5,
                1.75,
                1.875,
                2.25,
                2.625,
                3.0,
                3.75,
                4.5,
            ]
        )
    )
    parent = replace(
        scalar,
        states={"wealth": IrregSpacedGrid(points=np.asarray([2.5, 3.0]))},
        actions={"consumption": IrregSpacedGrid(points=np.asarray([1.5, 2.75]))},
        functions={
            "utility": _parent_utility,
            "inverse_marginal_utility": _parent_inverse,
            "resources": _parent_resources,
            "savings": scalar.functions["savings"],
        },
        state_transitions={
            "wealth": {"child": _parent_wealth},
            "q": {"child": _parent_q},
        },
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.asarray([0.25, 1.5, 3.0])),
            envelope=ExactEnvelope(),
        ),
        shocks=(),
    )
    child = replace(
        scalar,
        states={
            "wealth": IrregSpacedGrid(points=np.asarray([1.0, 3.75])),
            "q": IrregSpacedGrid(points=np.asarray([0.0, 1.0])),
        },
        actions={
            "consumption": IrregSpacedGrid(points=np.asarray([0.1, 4.0])),
            "work": DiscreteGrid(_Work),
        },
        functions={
            "utility": _flow,
            "inverse_marginal_utility": _inverse,
            "resources": _resources,
            "savings": scalar.functions["savings"],
        },
        state_transitions={
            **scalar.state_transitions,
            "q": {"done": _next_q},
        },
        solver=NBEGM(savings_grid=saving_grid),
    )
    done = Regime(
        states={
            "wealth": saving_grid,
            "q": IrregSpacedGrid(points=np.asarray([0.0, 0.25, 1.0, 1.25])),
        },
        functions={"utility": _terminal},
    )
    return Model._from_call_owned_params(
        declarations=replace(
            original,
            regimes={"parent": parent, "child": child, "done": done},
            ages=AgeGrid(exact_values=(0, 1, 2)),
            initial_nodes=InitialNodes(by_age={0: "parent"}),
            edges={"parent": {"child": (0,)}, "child": {"done": (1,)}},
            regime_id_class=_Regimes,
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
        params=_params(landing_q=landing_q),
    )


def _child_value(*, wealth: float, q: float) -> float:
    """Integrate after the literal original work choice, at exact Euler nodes."""
    conditional = []
    for shock in (0.0, 1.0):
        values = []
        for work in (0, 1):
            cash = wealth + 1.5 * work
            saving = max(0.25, cash / (2.0 + shock))
            consumption = cash - saving
            values.append(
                (1.0 + shock) * math.log(consumption) + math.log(saving) + q - work
            )
        conditional.append(max(values))
    return 0.5 * sum(conditional)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("landing_q", [0.0, 0.25])
def test_ordinary_parent_consumes_exact_nd_current_u_demands(
    *, landing_q: float
) -> None:
    """At parent wealth 3, original c=1.5 lands at child (2.25, q)."""
    model = _model(landing_q=0.0)
    params = _params(landing_q=landing_q)
    solution = model.solve(params=params, log_level="off")
    # Both physical shocks choose work=1 at child wealth 2.25. Their weighted
    # liquid marginal is (1/1.875 + 2/2.5)/2 = 2/3, so parent c=1.5 is an
    # original Euler candidate and saving=1.5, not an off-bank optimizer.
    expected = math.log(1.5) + _child_value(wealth=2.25, q=landing_q)
    np.testing.assert_almost_equal(
        solution.values[0]["parent"][1], expected, decimal=DECIMAL_PRECISION
    )
    ref = ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    carry = solution.retained_continuations[ref]
    assert isinstance(carry, ProjectedEGMCarry)
    assert carry.grid.ndim == 2
    # 2.25 was not a declared child wealth coordinate. Every active passive
    # stencil row must receive it under its own complete original identity.
    for row, q in enumerate((0.0, 1.0)):
        if landing_q == 0.0 and row == 1:
            continue
        length = int(carry.valid_length[row])
        positions = np.flatnonzero(np.asarray(carry.grid[row, :length]) == 2.25)
        assert positions.size == 1
        slot = positions[0]
        np.testing.assert_almost_equal(
            carry.value[row, slot],
            _child_value(wealth=2.25, q=q),
            decimal=DECIMAL_PRECISION,
        )
        np.testing.assert_almost_equal(
            carry.economic_marginal[row, slot],
            2.0 / 3.0,
            decimal=DECIMAL_PRECISION,
        )
    result = model.simulate(
        params=params,
        solution=solution,
        seed=31,
        log_level="off",
        initial_conditions={
            "wealth": jnp.asarray([3.0]),
            "age": jnp.asarray([0.0]),
            "regime_id": jnp.full((1,), model.regime_names_to_ids["parent"], jnp.int32),
        },
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["parent"][0].actions["consumption"],
        [1.5],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["child"][1].states["wealth"],
        [2.25],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["child"][1].states["q"],
        [landing_q],
        decimal=DECIMAL_PRECISION,
    )

    if landing_q == 0.0:
        # The untouched upper passive row has zero interpolation weight, but
        # its explicit source-integrity flag must never become a zero mask.
        bad = replace(carry, valid=carry.valid.at[1].set(False))
        supplied = replace(
            solution,
            retained_continuations=ArtifactStore(
                {**dict(solution.retained_continuations), ref: bad}
            ),
        )
        with pytest.raises(InvalidValueFunctionError, match="projected continuation"):
            model.simulate(
                params=params,
                solution=supplied,
                seed=31,
                log_level="off",
                initial_conditions={
                    "wealth": jnp.asarray([3.0]),
                    "age": jnp.asarray([0.0]),
                    "regime_id": jnp.full(
                        (1,), model.regime_names_to_ids["parent"], jnp.int32
                    ),
                },
            )
