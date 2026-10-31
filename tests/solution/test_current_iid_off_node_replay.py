"""EGM consumers of a current-IID shock replay continuously at off-node draws.

A shock group read by an EGM regime is solved on its node axis. A realised draw
between the quadrature nodes still has to replay the continuous Euler policy and
publish the literal economics, not the policy at the nearest node or on the declared
action grid. The child below consumes from current resources `R = w + eps` with
utility `log(c)` and saves into a terminal regime valued `log(s)`; with a unit
discount factor every interior optimum is `c = s = R / 2`.
"""

from pathlib import Path
from typing import TypedDict

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry
from _lcm.egm.interp import interp_on_padded_grid
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    InitialNodes,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
)
from lcm.persistence import load_solution
from lcm.result import SimulationResult
from lcm.shocks import (
    Normal,
    PhysicalValues,
    ShockGroup,
    Uniform,
    UserNodes,
)
from lcm.solver_api import (
    EGM_CONTINUATION,
    ArtifactRef,
    ResultRetention,
    SolutionResult,
)
from lcm.solvers import DCEGM, NBEGM, ExactEnvelope
from lcm.transition import ModelEdges
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserParams,
)
from tests.conftest import (
    DECIMAL_PRECISION,
    EXACT_KERNEL_SKIP_REASON,
    assert_agrees_to_ulp,
)

_DCEGM_CORNER_VALUE_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "DC-EGM node-state route reads the borrowing-corner value from interpolated "
        "rows, not the literal corner (reported gap)"
    ),
)


@categorical(ordered=False)
class _RegimeId:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class _ChildRegimeId:
    child: ScalarInt
    done: ScalarInt


def _utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _resources(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _current_resources(*, wealth: ContinuousState, eps: FloatND) -> FloatND:
    return wealth + eps


def _doubled_current_resources(*, wealth: ContinuousState, eps: FloatND) -> FloatND:
    return 2 * wealth + eps


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _landing(
    *, savings: FloatND, slope: FloatND, period: ScalarInt, age: FloatND | ScalarInt
) -> ContinuousState:
    return slope * savings + 0.5 + 0.125 * period + 0.0625 * age


def _identity(*, savings: FloatND, slope: FloatND) -> ContinuousState:
    return slope * savings


def _terminal(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


_MARGIN = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)
_FUNCTIONS = {
    "utility": _utility,
    "resources": _resources,
    "savings": _savings,
    "inverse_marginal_utility": _inverse,
}


class _Declarations(TypedDict):
    """The `Model` arguments a test edits before constructing it."""

    regimes: dict[str, Regime]
    ages: AgeGrid
    initial_nodes: InitialNodes
    edges: ModelEdges
    regime_id_class: type[_RegimeId]
    shocks: dict[str, ShockGroup]


def _dc_declarations(
    *,
    child_wealth: np.ndarray,
    child_savings: np.ndarray,
    parent_savings: np.ndarray,
    nodes: tuple[float, float],
    parent_grid_replay: bool,
) -> _Declarations:
    """Model arguments: a DC-EGM parent landing on a child that reads the draw."""
    savings = IrregSpacedGrid(points=child_savings)
    parent = ConsumptionSavingsRegime(
        states={"wealth": IrregSpacedGrid(points=np.array([1.0, 4.0]))},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.25, 2.0]))},
        functions=_FUNCTIONS,
        state_transitions={"wealth": {"child": _landing}},
        liquid=_MARGIN,
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=parent_savings),
            envelope=ExactEnvelope(),
            replay="grid" if parent_grid_replay else "candidate",
        ),
    )
    child = ConsumptionSavingsRegime(
        states={"wealth": IrregSpacedGrid(points=child_wealth)},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.25, 2.0]))},
        functions={**_FUNCTIONS, "resources": _current_resources},
        state_transitions={"wealth": {"done": _identity}},
        liquid=_MARGIN,
        solver=DCEGM(savings_grid=savings, envelope=ExactEnvelope()),
        shocks=("income",),
    )
    return {
        "regimes": {
            "parent": parent,
            "child": child,
            "done": Regime(
                states={"wealth": savings}, functions={"utility": _terminal}
            ),
        },
        "ages": AgeGrid(exact_values=(0, 1, 2)),
        "initial_nodes": InitialNodes(by_age={0: "parent"}),
        "edges": {"parent": {"child": (0,)}, "child": {"done": (1,)}},
        "regime_id_class": _RegimeId,
        "shocks": {
            "income": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.array([[nodes[0]], [nodes[1]]]),
                    weights=np.array([0.5, 0.5]),
                ),
            )
        },
    }


def _interior_declarations() -> _Declarations:
    """Every conditional optimum `w/2`, `(w+1)/2` queried below is a saving node.

    Parent saving at most four lands at 8.5; the child grid {1, 12.5} extends past
    that landing, so every queried wealth lies off the child grid.
    """
    return _dc_declarations(
        child_wealth=np.array([1.0, 12.5]),
        child_savings=np.array(
            [
                0.25,
                0.5,
                0.625,
                0.75,
                1.0,
                1.125,
                1.25,
                1.375,
                1.5,
                1.75,
                1.875,
                2.0,
                2.25,
                2.5,
                4.25,
                4.75,
                6.25,
                6.375,
                6.75,
                6.875,
                12.5,
                12.75,
            ]
        ),
        parent_savings=np.array([0.5, 0.75, 1.0, 4.0]),
        nodes=(0.0, 1.0),
        parent_grid_replay=False,
    )


def _interior_model() -> Model:
    return Model(**_interior_declarations())


def _params() -> UserParams:
    return {
        "discount_factor": 1.0,
        "parent": {"next_wealth": {"slope": 2.0}},
        "child": {"next_wealth": {"slope": 1.0}},
    }


def _solve(*, model: Model) -> SolutionResult:
    return model.solve(params=_params(), log_level="warning", max_compilation_workers=1)


def _integrated_read(
    *, result: SolutionResult, wealth: float, nodes: tuple[float, ...]
) -> tuple[float, float, bool]:
    """The child's continuation the parent integrates, read at an off-grid wealth.

    A realised draw is a state of the node-state child, so its row axis is the
    income node axis, kept when the solve retains every persistable artifact.

    The retained carry holds one row per income node, each over current resources;
    the integrated value and economic marginal weight the node rows equally.
    """
    carry = result.retained_continuations[
        ArtifactRef(period=1, regime="child", key=EGM_CONTINUATION)
    ]
    assert isinstance(carry, EGMCarry)
    assert carry.endog_grid.shape[0] == len(nodes)
    dtype = carry.endog_grid.dtype
    values, marginals = [], []
    for index, node in enumerate(nodes):
        query = jnp.asarray(wealth + node, dtype=dtype)
        values.append(
            interp_on_padded_grid(
                x_query=query,
                xp=carry.endog_grid[index],
                fp=carry.value[index],
                fp_slopes=carry.marginal_utility[index],
            )
        )
        marginals.append(
            interp_on_padded_grid(
                x_query=query,
                xp=carry.endog_grid[index],
                fp=carry.marginal_utility[index],
            )
        )
    value, marginal = float(np.mean(values)), float(np.mean(marginals))
    return value, marginal, bool(np.isfinite(value) and np.isfinite(marginal))


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_row_matches_the_literal_economics_off_the_child_grid() -> None:
    """Integrated W and economic M at off-grid wealth match the closed form.

    At w in {1.5, 2.5, 3.5} both conditional log/log optima (w/2 and (w+1)/2)
    lie on the declared saving grid, so the grid-conditional solution is
    W = log(w/2) + log((w+1)/2) and M = 1/w + 1/(w+1). The child grid is {1, 12.5}.
    """
    result = _interior_model().solve(
        params=_params(),
        log_level="warning",
        max_compilation_workers=1,
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    eps = float(np.finfo(np.dtype(canonical_float_dtype())).eps)
    for wealth in (1.5, 2.5, 3.5):
        read_value, read_marginal, valid = _integrated_read(
            result=result, wealth=wealth, nodes=(0.0, 1.0)
        )
        assert valid
        value = np.log(wealth / 2.0) + np.log((wealth + 1.0) / 2.0)
        marginal = 1.0 / wealth + 1.0 / (wealth + 1.0)
        assert abs(read_value - value) / max(1.0, abs(value)) <= max(1e-6, 64 * eps)
        assert abs(read_marginal - marginal) / max(1.0, marginal) <= max(
            1e-4, 8 * np.sqrt(eps)
        )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_replay_matches_the_literal_policy_at_off_node_shocks() -> None:
    """Physical replay at drawn, off-node shocks follows the closed-form policy.

    Every Euler point of this log/log child lies on c = R/2, so its exact
    envelope is c = (w + eps)/2 for any realized resources whose half lies on
    the saving-grid span. Drawn shocks are not quadrature nodes {0, 1}; the
    next state equals the saving R - c because the law's slope is one.
    """
    model = _interior_model()
    params = _params()
    result = model.solve(params=params, log_level="warning", max_compilation_workers=1)
    n_subjects = 8
    frame = model.simulate(
        params=params,
        solution=result,
        seed=19,
        log_level="warning",
        initial_conditions={
            "regime_id": np.zeros(n_subjects, dtype=int),
            "age": np.zeros(n_subjects, dtype=int),
            "wealth": np.linspace(2.5, 4.0, n_subjects, dtype=canonical_float_dtype()),
        },
    )
    child = frame.raw_results["child"][1]
    done = frame.raw_results["done"][2]
    wealth = np.asarray(child.states["wealth"])
    shock = np.asarray(child.shocks["eps"])
    consumption = np.asarray(child.actions["consumption"])
    assert not np.isin(shock, [0.0, 1.0]).any()
    resources = wealth + shock
    assert (resources / 2.0 >= 0.25).all()
    assert (resources / 2.0 <= 12.75).all()
    tolerance = max(1e-4, 8 * np.sqrt(np.finfo(consumption.dtype).eps))
    np.testing.assert_allclose(consumption, resources / 2.0, rtol=0, atol=tolerance)
    np.testing.assert_allclose(
        np.asarray(done.states["wealth"]),
        resources - consumption,
        rtol=0,
        atol=tolerance,
    )


# A saving floor `L = 1`: below resources `2L` the child saves exactly `L` and
# consumes `R - L`, which is positive on the child domain because `w - 0.75 > L`.
_BORROWING_LIMIT = 1.0
_LOW, _HIGH = -0.75, 1.0
_KINK = 2.0 * _BORROWING_LIMIT
# Parent wealth 2.875 and 3.5 consume 2.0 and land on child wealth 2.25 and 3.5.
_PARENT_WEALTH = np.array([2.875, 2.875, 3.5, 3.5])
_CHILD_SHOCK = np.array([_LOW, _HIGH, _LOW, _HIGH])


def _corner_model() -> Model:
    """The interior economics with a saving floor of one and a child domain from 2.

    Every interior saving queried below, `R / 2`, is a grid node. At the child
    domain's lower end the low node's resources are 1.25, strictly above the floor.
    The parent replays on its declared consumption grid, so it consumes exactly 2.0
    and lands the child on the corner.
    """
    return Model(
        **_dc_declarations(
            child_wealth=np.array([2.0, 12.5]),
            child_savings=np.array(
                [
                    1.0,
                    1.125,
                    1.25,
                    1.375,
                    1.5,
                    1.625,
                    1.75,
                    1.875,
                    2.0,
                    2.25,
                    2.5,
                    4.25,
                    4.75,
                    6.25,
                    6.375,
                    6.75,
                    6.875,
                    12.5,
                    12.75,
                ]
            ),
            parent_savings=np.array([0.75, 1.0, 4.0]),
            nodes=(_LOW, _HIGH),
            parent_grid_replay=True,
        )
    )


def _corner_consumption(resources: np.ndarray) -> np.ndarray:
    return np.where(resources >= _KINK, resources / 2.0, resources - _BORROWING_LIMIT)


def _corner_value(resources: np.ndarray) -> np.ndarray:
    consumption = _corner_consumption(resources)
    return np.log(consumption) + np.log(resources - consumption)


def _corner_marginal(resources: np.ndarray) -> np.ndarray:
    return 1.0 / _corner_consumption(resources)


def _value_budget(*, dtype: np.dtype) -> float:
    return max(1e-6, 64 * float(np.finfo(dtype).eps))


def _simulate_corner(*, model: Model, solution: SolutionResult) -> SimulationResult:
    dtype = canonical_float_dtype()
    return model.simulate(
        params=_params(),
        solution=solution,
        seed=19,
        log_level="warning",
        initial_conditions={
            "regime_id": np.zeros(_PARENT_WEALTH.size, dtype=int),
            "age": np.zeros(_PARENT_WEALTH.size, dtype=int),
            "wealth": _PARENT_WEALTH.astype(dtype),
        },
        shock_paths={
            "eps": PhysicalValues(
                values=np.tile(_CHILD_SHOCK[:, None], 3).astype(dtype)
            )
        },
    )


@_DCEGM_CORNER_VALUE_XFAIL
@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_corner_row_matches_the_closed_form_off_the_child_grid() -> None:
    """W and economic M mix a corner node and an interior node literally.

    At w in {2.25, 2.5} the low node's resources lie below the kink 2, so it
    saves the floor; the high node is interior. At w = 3.5 both are interior.
    """
    result = _corner_model().solve(
        params=_params(),
        log_level="warning",
        max_compilation_workers=1,
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    eps = float(np.finfo(np.dtype(canonical_float_dtype())).eps)
    for wealth, low_at_corner in ((2.25, True), (2.5, True), (3.5, False)):
        resources = np.array([wealth + _LOW, wealth + _HIGH])
        assert ((resources < _KINK) == [low_at_corner, False]).all()
        read_value, read_marginal, valid = _integrated_read(
            result=result, wealth=wealth, nodes=(_LOW, _HIGH)
        )
        assert valid
        value = float(np.mean(_corner_value(resources)))
        marginal = float(np.mean(_corner_marginal(resources)))
        assert abs(read_value - value) / max(1.0, abs(value)) <= _value_budget(
            dtype=np.dtype(canonical_float_dtype())
        )
        assert abs(read_marginal - marginal) / max(1.0, marginal) <= max(
            1e-4, 8 * np.sqrt(eps)
        )


@_DCEGM_CORNER_VALUE_XFAIL
@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_replay_saves_the_floor_exactly_at_the_corner() -> None:
    """A low draw at child wealth 2.25 saves the floor; the others are interior.

    The decision value is the realized utility plus the terminal log saving,
    which associates the replayed action with its own value.
    """
    model = _corner_model()
    frame = _simulate_corner(model=model, solution=_solve(model=model))
    np.testing.assert_array_equal(
        frame.raw_results["parent"][0].actions["consumption"], 2.0
    )
    child = frame.raw_results["child"][1]
    wealth = np.asarray(child.states["wealth"])
    np.testing.assert_array_equal(wealth, 2.0 * (_PARENT_WEALTH - 2.0) + 0.5)
    np.testing.assert_array_equal(np.asarray(child.shocks["eps"]), _CHILD_SHOCK)
    resources = wealth + _CHILD_SHOCK
    corner = resources < _KINK
    np.testing.assert_array_equal(corner, [True, False, False, False])
    consumption = np.asarray(child.actions["consumption"])
    tolerance = max(1e-4, 8 * np.sqrt(np.finfo(consumption.dtype).eps))
    np.testing.assert_allclose(
        consumption, _corner_consumption(resources), rtol=0, atol=tolerance
    )
    saving = np.asarray(frame.raw_results["done"][2].states["wealth"])
    np.testing.assert_array_equal(saving[corner], _BORROWING_LIMIT)
    np.testing.assert_allclose(
        saving, resources - _corner_consumption(resources), rtol=0, atol=tolerance
    )
    value = np.asarray(child.V_arr)
    expected = _corner_value(resources)
    np.testing.assert_array_less(
        np.abs(value - expected) / np.maximum(1.0, np.abs(expected)),
        _value_budget(dtype=value.dtype),
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_corner_archive_replays_the_frame_up_to_rounding(
    *, tmp_path: Path
) -> None:
    """A fresh Model replays the saved corner solution to the frame up to rounding.

    The archive holds the node-integrated value, which replay broadcasts along the
    node axis, while the in-memory result keeps its per-node values. Every column
    but `value` is therefore identical, and `value` agrees to within a few ULP at
    the column's scale.
    """
    model = _corner_model()
    result = _solve(model=model)
    expected = _simulate_corner(model=model, solution=result).to_dataframe()
    path = tmp_path / "case-r-corner.lcm"
    result.save(path=path)
    loaded = load_solution(path=path)
    assert loaded._engine_view is None
    replay = _simulate_corner(model=_corner_model(), solution=loaded).to_dataframe()
    pd.testing.assert_frame_equal(
        replay.drop(columns="value"), expected.drop(columns="value"), check_exact=True
    )
    expected_value = expected["value"].to_numpy()
    assert_agrees_to_ulp(
        got=replay["value"].to_numpy(),
        expected=expected_value,
        n_ulp=4,
        operand_magnitude=float(np.abs(expected_value).max()),
    )


_NB_SAVINGS = np.array(
    [0.25, 0.375, 0.5, 0.6875, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
)


def _nb_model(*, resource_scale: int) -> Model:
    """An NB-EGM child whose resources `scale * w + eps` read a uniform draw."""
    savings = IrregSpacedGrid(points=_NB_SAVINGS)
    child = ConsumptionSavingsRegime(
        states={"wealth": IrregSpacedGrid(points=np.array([0.375, 1.0, 2.0]))},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.1, 4.0]))},
        functions={
            **_FUNCTIONS,
            "resources": _current_resources
            if resource_scale == 1
            else _doubled_current_resources,
        },
        state_transitions={"wealth": {"done": _identity}},
        liquid=_MARGIN,
        solver=NBEGM(savings_grid=savings, probe_schedule="never"),
        shocks=("income",),
    )
    return Model(
        regimes={
            "child": child,
            "done": Regime(
                states={"wealth": savings}, functions={"utility": _terminal}
            ),
        },
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={1: "child"}),
        edges={"child": {"done": (1,)}},
        regime_id_class=_ChildRegimeId,
        shocks={
            "income": ShockGroup(
                shocks={"eps": Uniform(start=0.0, stop=1.0)},
                nodes=UserNodes(
                    nodes=jnp.asarray([[0.0], [1.0]]), weights=jnp.asarray([0.5, 0.5])
                ),
            )
        },
    )


def _nb_oracle(
    *, wealth: float, shock: float, resource_scale: int
) -> tuple[float, float, float, float]:
    cash = resource_scale * wealth + shock
    saving = max(0.25, cash / 2)
    consumption = cash - saving
    return (
        float(np.log(consumption) + np.log(saving)),
        resource_scale / consumption,
        consumption,
        saving,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("resource_scale", [1, 2])
def test_original_nbegm_r_model_retains_one_bank_and_replays_offnode(
    *, resource_scale: int
) -> None:
    """Integrated values match the oracle; an off-node draw replays it, corner too."""
    model = _nb_model(resource_scale=resource_scale)
    params = {"discount_factor": 1.0, "child": {"next_wealth": {"slope": 1.0}}}
    solution = model.solve(params=params, log_level="off")
    for index, wealth in enumerate((0.375, 1.0, 2.0)):
        reference = np.mean(
            [
                _nb_oracle(wealth=wealth, shock=z, resource_scale=resource_scale)[:2]
                for z in (0.0, 1.0)
            ],
            axis=0,
        )
        np.testing.assert_almost_equal(
            solution.values[1]["child"][index], reference[0], decimal=DECIMAL_PRECISION
        )

    wealth = np.asarray([0.375, 1.0, 1.0, 2.0])
    physical = np.asarray([0.0, 0.0, 0.5, 1.0], dtype=canonical_float_dtype())
    simulated = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.asarray(wealth),
            "age": jnp.ones((4,)),
            "regime_id": jnp.full((4,), model.regime_names_to_ids["child"], jnp.int32),
            "eps": PhysicalValues(values=physical),
        },
        log_level="off",
        seed=29,
    )
    actual = simulated.raw_results["child"][1]
    expected = np.asarray(
        [
            _nb_oracle(wealth=w, shock=z, resource_scale=resource_scale)[2]
            for w, z in zip(wealth, physical, strict=True)
        ]
    )
    np.testing.assert_array_almost_equal(
        actual.actions["consumption"], expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(actual.shocks["eps"], physical)
    np.testing.assert_array_almost_equal(
        simulated.raw_results["done"][2].states["wealth"],
        resource_scale * wealth + physical - expected,
        decimal=DECIMAL_PRECISION,
    )
    assert float(actual.actions["consumption"][2]) not in (0.1, 4.0)
