"""Scalar DC-R row with a reachable borrowing corner, through the public path.

The child consumes from current resources `R = w + eps` with utility `log(c)`,
saves `s = R - c >= L` and the terminal regime values `log(s)`. With a unit
discount factor the Euler equation `c = s` gives the interior policy
`c = s = R / 2` whenever that saving is at least the borrowing limit, i.e.
`R >= 2L`. Below that the household saves exactly `L` and consumes `c = R - L`,
which is positive on the child domain because `w - 0.75 > L` there. The
borrowing limit is the first node of the child's saving grid, `L = 1`.
"""

from dataclasses import fields, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.current_iid_balanced import BalancedProjectedEGMCarry
from _lcm.egm.current_iid_projection import _read_projected
from _lcm.egm.current_iid_resource_query import read_resources_candidate
from _lcm.egm.current_iid_resources_artifact import restore_resources_row
from _lcm.egm.projected_continuation import PROJECTED_EGM_CONTINUATION
from _lcm.egm.step_core import OwnedResourcesRow
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.solution.current_iid_case_r import CASE_R_RESOURCES_REPLAY
from lcm import IrregSpacedGrid, Model, Regime
from lcm.model import _ModelConstructorInputs
from lcm.persistence import load_solution
from lcm.result import SimulationResult
from lcm.shocks import Normal, PhysicalValues, ShockGroup, UserNodes
from lcm.solver_api import ArtifactRef, SolutionResult
from lcm.solvers import DCEGM, ExactEnvelope
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_call_owned_generation import _declarations, _params
from tests.egm.test_current_iid_parent_demands import _terminal

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_BORROWING_LIMIT = 1.0
_LOW, _HIGH = -0.75, 1.0
# Resources below which the interior saving would breach the borrowing limit.
_KINK = 2.0 * _BORROWING_LIMIT

# Parent wealth 2.875 and 3.5 consume 2.0 and land on child wealth 2.25 and 3.5.
_PARENT_WEALTH = np.array([2.875, 2.875, 3.5, 3.5])
_CHILD_SHOCK = np.array([_LOW, _HIGH, _LOW, _HIGH])


def _corner_declarations() -> _ModelConstructorInputs:
    """P1 economics with a saving floor of one and a child domain from 2.

    Every interior saving queried below, `R / 2`, is a grid node, so the
    grid-conditional reference is the literal closed form. At the child domain's
    lower end the low node's resources are 1.25, strictly above the floor.
    Parent saving starts at 0.75, landing at 2.0.
    """
    base = _declarations()
    savings = IrregSpacedGrid(
        points=np.array(
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
        )
    )
    parent = base.regimes["parent"]
    child = base.regimes["child"]
    regimes = {
        "parent": replace(
            parent,
            solver=DCEGM(
                savings_grid=IrregSpacedGrid(points=np.array([0.75, 1.0, 4.0])),
                envelope=ExactEnvelope(),
            ),
        ),
        "child": replace(
            child,
            states={"wealth": IrregSpacedGrid(points=np.array([2.0, 12.5]))},
            solver=DCEGM(savings_grid=savings, envelope=ExactEnvelope()),
        ),
        "done": Regime(states={"wealth": savings}, functions={"utility": _terminal}),
    }
    shocks = {
        "income": ShockGroup(
            shocks={"eps": Normal(mu=0.0, sigma=1.0)},
            nodes=UserNodes(
                nodes=np.array([[_LOW], [_HIGH]]), weights=np.array([0.5, 0.5])
            ),
        )
    }
    return replace(base, regimes=regimes, shocks=shocks)


def _model() -> Model:
    declarations = _corner_declarations()
    return Model(
        **{
            field.name: getattr(declarations, field.name)
            for field in fields(declarations)
            if field.init
        }
    )


def _solve(*, model: Model) -> SolutionResult:
    return model.solve(params=_params(), log_level="warning", max_compilation_workers=1)


def _consumption(resources: np.ndarray) -> np.ndarray:
    return np.where(resources >= _KINK, resources / 2.0, resources - _BORROWING_LIMIT)


def _value(resources: np.ndarray) -> np.ndarray:
    consumption = _consumption(resources)
    return np.log(consumption) + np.log(resources - consumption)


def _marginal(resources: np.ndarray) -> np.ndarray:
    return 1.0 / _consumption(resources)


def _value_budget(*, dtype: np.dtype) -> float:
    return max(1e-6, 64 * float(np.finfo(dtype).eps))


def _simulate(*, model: Model, solution: SolutionResult) -> SimulationResult:
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


def _solved_child_row(*, result: SolutionResult) -> tuple[OwnedResourcesRow, int]:
    """The child's saved Case-R row under its model-owned replay schema."""
    view = result._engine_view
    assert isinstance(view, OwnedSolutionView)
    assert view.generation is not None
    spec = view.generation.execution._regimes["child"].simulation.case_r_replay_spec
    assert spec is not None
    declaration = spec.periods[1]
    row = restore_resources_row(
        payload=result.replay_artifacts[
            ArtifactRef(period=1, regime="child", key=CASE_R_RESOURCES_REPLAY)
        ],
        tree=declaration.row_tree,
        leaf_specs=declaration.row_leaf_specs,
    )
    assert type(row) is OwnedResourcesRow
    return row, declaration.owner_capacity


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_corner_row_matches_the_closed_form_off_the_child_grid() -> None:
    """W and economic M mix a corner node and an interior node literally.

    At w in {2.25, 2.5} the low node's resources lie below the kink 2, so it
    saves the floor; the high node is interior. At w = 3.5 both are interior.
    """
    result = _solve(model=_model())
    carry = result.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(carry, BalancedProjectedEGMCarry)
    dtype = carry.grid.dtype
    eps = float(jnp.finfo(dtype).eps)
    for wealth, low_at_corner in ((2.25, True), (2.5, True), (3.5, False)):
        resources = np.array([wealth + _LOW, wealth + _HIGH])
        assert ((resources < _KINK) == [low_at_corner, False]).all()
        read = _read_projected(carry=carry, query=jnp.asarray(wealth, dtype=dtype))
        assert bool(read.valid)
        value = float(np.mean(_value(resources)))
        marginal = float(np.mean(_marginal(resources)))
        assert abs(float(read.value) - value) / max(1.0, abs(value)) <= (
            _value_budget(dtype=dtype)
        )
        assert abs(float(read.economic_marginal) - marginal) / max(
            1.0, marginal
        ) <= max(1e-4, 8 * np.sqrt(eps))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_r_corner_selection_associates_the_literal_corner() -> None:
    """Below the kink the replay selects the literal corner with its own value.

    The selected action is exactly `R - L`, the saving exactly `L`, and the value
    is `log(R - L) + log(L)`, flagged as the borrowing corner without a native
    row owner. At the kink both branches name the same action and value. Above
    it the interior row candidate is selected.
    """
    result = _solve(model=_model())
    row, owner_capacity = _solved_child_row(result=result)
    dtype = row.envelope.grid.dtype
    budget = _value_budget(dtype=dtype)
    for resources in (1.25, 1.5, 1.75, 2.0, 2.75, 3.25):
        read = read_resources_candidate(
            row=row,
            resources=jnp.asarray(resources, dtype=dtype),
            utility_of_action=jnp.log,
            owner_capacity=owner_capacity,
        )
        assert bool(read.valid)
        assert bool(read.feasible)
        value = float(_value(np.asarray(resources)))
        assert abs(float(read.value) - value) / max(1.0, abs(value)) <= budget
        if resources < _KINK:
            assert bool(read.corner)
            assert int(read.native_owner) == -1
            assert float(read.policy) == resources - _BORROWING_LIMIT
            assert float(read.savings) == _BORROWING_LIMIT
        elif resources == _KINK:
            assert float(read.policy) == resources - _BORROWING_LIMIT
            assert float(read.savings) == _BORROWING_LIMIT
        else:
            assert not bool(read.corner)
            assert int(read.native_owner) >= 0
            np.testing.assert_allclose(
                float(read.policy),
                resources / 2.0,
                rtol=0,
                atol=max(1e-4, 8 * np.sqrt(np.finfo(dtype).eps)),
            )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_replay_saves_the_floor_exactly_at_the_corner() -> None:
    """A low draw at child wealth 2.25 saves the floor; the others are interior.

    The decision value is the realized utility plus the terminal log saving,
    which associates the replayed action with its own value.
    """
    model = _model()
    frame = _simulate(model=model, solution=_solve(model=model))
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
        consumption, _consumption(resources), rtol=0, atol=tolerance
    )
    saving = np.asarray(frame.raw_results["done"][2].states["wealth"])
    np.testing.assert_array_equal(saving[corner], _BORROWING_LIMIT)
    np.testing.assert_allclose(
        saving, resources - _consumption(resources), rtol=0, atol=tolerance
    )
    value = np.asarray(child.V_arr)
    expected = _value(resources)
    np.testing.assert_array_less(
        np.abs(value - expected) / np.maximum(1.0, np.abs(expected)),
        _value_budget(dtype=value.dtype),
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_corner_archive_replays_the_frame_exactly(
    *, tmp_path: Path
) -> None:
    """A fresh Model replays the saved corner row to the identical frame."""
    model = _model()
    result = _solve(model=model)
    expected = _simulate(model=model, solution=result).to_dataframe()
    path = tmp_path / "case-r-corner.lcm"
    result.save(path=path)
    loaded = load_solution(path=path)
    assert loaded._engine_view is None
    replay = _simulate(model=_model(), solution=loaded).to_dataframe()
    pd.testing.assert_frame_equal(replay, expected, check_exact=True)
