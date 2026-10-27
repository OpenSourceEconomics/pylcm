"""Actual enclosing parent reads of retained saturated-U economics and integrity."""

import re
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_case_u_balanced import BalancedCaseUCarry
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.typing import FootprintTree
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    InitialNodes,
    IrregSpacedGrid,
    Model,
    categorical,
)
from lcm.exceptions import (
    IncompatibleSolutionError,
    InvalidSimulationInputError,
    InvalidValueFunctionError,
)
from lcm.persistence import load_solution
from lcm.solver_api import (
    EGM_CONTINUATION,
    ArtifactRef,
    ArtifactStore,
    OmissionReason,
    PersistencePolicy,
    ResultRetention,
    SolutionResult,
)
from lcm.solvers import DCEGM, ExactEnvelope
from lcm.typing import ContinuousAction, FloatND, IntND, ScalarInt
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_balanced import _declarations, _next_wealth

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class _Regimes:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


def _utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _parent_model() -> Model:
    """Build the original parent/child economic declarations."""
    declarations = _declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    child = replace(
        child,
        states={"wealth": IrregSpacedGrid(points=np.array([2.75, 3.875]))},
    )
    parent = replace(
        child,
        states={"wealth": IrregSpacedGrid(points=np.array([3.75, 3.875]))},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.25, 1.0]))},
        functions={
            **child.functions,
            "utility": _utility,
            "inverse_marginal_utility": _inverse,
        },
        state_transitions={"wealth": {"child": _next_wealth}},
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([2.75, 2.8125, 2.875, 3.875])),
            envelope=ExactEnvelope(),
            n_constrained_points=2,
        ),
        shocks=(),
    )
    declarations = replace(
        declarations,
        regimes={"parent": parent, **declarations.regimes, "child": child},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={0: "parent"}),
        edges={"parent": {"child": (0,)}, "child": {"done": (1,)}},
        regime_id_class=_Regimes,
    )
    return Model._from_call_owned_params(
        declarations=declarations, params={"discount_factor": 1.0}
    )


@pytest.fixture(scope="module")
def parent_solution() -> tuple[Model, SolutionResult]:
    """Solve the actual parent with its required default child retention."""
    model = _parent_model()
    solution = model.solve(
        params={"discount_factor": 1.0},
        log_level="off",
    )
    return model, solution


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_required_enclosing_parent_forward_consumes_nonexact_u_guard(
    *, parent_solution: tuple[Model, SolutionResult]
) -> None:
    """At off-grid parent wealth 3.8, c=1 and next wealth 2.8 is a nonexact U read."""
    model, solution = parent_solution
    carry = solution.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(carry, ProjectedEGMCarry)
    ref = ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    assert solution.metadata.retention is ResultRetention.VALUES_AND_REPLAY
    route = solution.metadata.replay_routes["parent"]
    assert route is not None
    assert solution.metadata.artifact_descriptors[ref].required_for == frozenset(
        {route}
    )
    assert (
        solution.metadata.artifact_descriptors[ref].persistence
        is PersistencePolicy.MODEL_VERIFIABLE
    )
    unrelated = ArtifactRef(period=2, regime="done", key=EGM_CONTINUATION)
    assert unrelated not in solution.retained_continuations
    assert solution.omissions[unrelated] is OmissionReason.NOT_REQUESTED
    query = jnp.asarray(2.8)
    assert not bool(jnp.any(carry.grid[: int(carry.valid_length)] == query))
    _, generation = model._prepare_call(params={"discount_factor": 1.0})
    program = generation.execution._regimes[
        "parent"
    ].simulation.programs.forward_decision[0]
    assert any(
        read.target.regime == "child"
        and read.target.artifact_key == PROJECTED_EGM_CONTINUATION
        and read.target.leaf_path[:1] == ("guard",)
        for read in program.requirements.value_reads
    ), "Enclosing parent forward lacks its projected child guard operands."
    result = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": jnp.asarray([3.8]),
            "age": jnp.asarray([0.0]),
            "regime_id": jnp.asarray(
                [model.regime_names_to_ids["parent"]], dtype=jnp.int32
            ),
        },
        solution=solution,
        seed=23,
        log_level="off",
    )
    record = result.raw_results["parent"][0]
    np.testing.assert_array_almost_equal(
        record.actions["consumption"], [1.0], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["child"][1].states["wealth"],
        [query],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        record.V_arr, [query + np.log(2.0) - 1.5], decimal=DECIMAL_PRECISION
    )


def _initial(*, model: Model) -> dict[str, FloatND | IntND]:
    return {
        "wealth": jnp.asarray([3.8]),
        "age": jnp.asarray([0.0]),
        "regime_id": jnp.asarray(
            [model.regime_names_to_ids["parent"]], dtype=jnp.int32
        ),
    }


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_parent_forward_refuses_missing_retained_child(
    *, parent_solution: tuple[Model, SolutionResult]
) -> None:
    """A genuine result's missing original continuation cannot become a V read."""
    model, solution = parent_solution
    ref = ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    supplied = replace(
        solution,
        retained_continuations=ArtifactStore(
            {
                key: value
                for key, value in solution.retained_continuations.items()
                if key != ref
            }
        ),
    )
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params={"discount_factor": 1.0},
            initial_conditions=_initial(model=model),
            solution=supplied,
            seed=23,
            log_level="off",
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_parent_forward_checks_original_stored_guard_clock(
    *, parent_solution: tuple[Model, SolutionResult]
) -> None:
    """Finite stored values do not conceal a foreign producing source/date."""
    model, solution = parent_solution
    ref = ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    carry = solution.retained_continuations[ref]
    assert isinstance(carry, BalancedCaseUCarry)
    assert "period" in carry.guard.pool_names
    foreign_clock = replace(
        carry,
        guard=replace(
            carry.guard,
            pool_values=tuple(
                jnp.int32(0) if name == "period" else value
                for name, value in zip(
                    carry.guard.pool_names, carry.guard.pool_values, strict=True
                )
            ),
        ),
    )
    supplied = replace(
        solution,
        retained_continuations=ArtifactStore(
            {**dict(solution.retained_continuations), ref: foreign_clock}
        ),
    )
    with pytest.raises(InvalidValueFunctionError, match="projected continuation"):
        model.simulate(
            params={"discount_factor": 1.0},
            initial_conditions=_initial(model=model),
            solution=supplied,
            seed=23,
            log_level="off",
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_parent_forward_rejects_unrequested_omission_for_required_child(
    *, parent_solution: tuple[Model, SolutionResult]
) -> None:
    """Result metadata cannot discard the actual default forward child demand."""
    model, solution = parent_solution
    ref = ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    supplied = replace(
        solution,
        retained_continuations=ArtifactStore(
            {
                key: value
                for key, value in solution.retained_continuations.items()
                if key != ref
            }
        ),
        omissions=MappingProxyType(
            {**solution.omissions, ref: OmissionReason.NOT_REQUESTED}
        ),
    )
    with pytest.raises(InvalidSimulationInputError, match="required selected"):
        model.simulate(
            params={"discount_factor": 1.0},
            initial_conditions=_initial(model=model),
            solution=supplied,
            seed=23,
            log_level="off",
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_values_only_parent_omits_projected_forward_child(
    *, parent_solution: tuple[Model, SolutionResult]
) -> None:
    """Values-only output does not promise the retained child needed by forward."""
    model, _ = parent_solution
    solution = model.solve(
        params={"discount_factor": 1.0},
        log_level="off",
        retention=ResultRetention.VALUES,
    )
    ref = ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    assert ref not in solution.retained_continuations
    assert solution.omissions[ref] is OmissionReason.NOT_REQUESTED
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params={"discount_factor": 1.0},
            initial_conditions=_initial(model=model),
            solution=solution,
            seed=23,
            log_level="off",
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_parent_u_guard_has_closed_detached_artifact_storage(
    *, parent_solution: tuple[Model, SolutionResult]
) -> None:
    """Every stored guard leaf survives detached snapshot in original bank order."""
    from lcm._solver_api.authority import (  # noqa: PLC0415
        _snapshot_artifact_template_once,
    )

    _, solution = parent_solution
    carry = solution.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(carry, BalancedCaseUCarry)
    guard = carry.guard
    views = (
        guard.pool,
        guard.children,
        guard.group_params,
        guard.node_operands,
        guard.indices,
        guard.savings_nodes,
        guard.borrowing_limit,
        guard.floor_status,
        guard.child_nodes,
        guard.lower,
        guard.upper,
    )
    for stored, original in zip(
        jax.tree.leaves(guard), jax.tree.leaves(views), strict=True
    ):
        assert stored is original
    snapshot, containers = _snapshot_artifact_template_once(
        template=carry, payload_runtime_type=BalancedCaseUCarry
    )
    detached = snapshot.payload
    assert isinstance(detached, BalancedCaseUCarry)
    assert MappingProxyType not in containers.values()
    assert jax.tree.structure(detached) == jax.tree.structure(carry)
    assert tuple(detached.leaves()) == tuple(carry.leaves())
    assert tuple(detached.guard.pool) == tuple(guard.pool)
    assert tuple(detached.guard.children) == tuple(guard.children)
    assert tuple(detached.guard.group_params) == tuple(guard.group_params)
    assert tuple(detached.guard.node_operands) == tuple(guard.node_operands)
    for expected, actual in zip(
        jax.tree.leaves(carry), jax.tree.leaves(detached), strict=True
    ):
        assert isinstance(expected, jax.Array)
        assert isinstance(actual, jax.Array)
        assert actual is not expected
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    for path, original in carry.leaves().items():
        current: FootprintTree = carry
        for key in carry.input_path(leaf_path=path):
            if isinstance(key, int):
                assert isinstance(current, tuple)
                current = current[key]
            else:
                current = getattr(current, key)
        assert current is original


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_default_projected_guard_archive_reaches_fresh_parent_forward(
    *, parent_solution: tuple[Model, SolutionResult], tmp_path: Path
) -> None:
    """A fresh canonical model consumes the archived default child's real guard."""
    _, solution = parent_solution
    ref = ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    route = solution.metadata.replay_routes["parent"]
    assert route is not None
    assert solution.metadata.artifact_descriptors[ref].required_for == frozenset(
        {route}
    )
    path = tmp_path / "projected-parent.lcm"
    solution.save(path=path)
    loaded = load_solution(path=path)
    assert loaded.metadata.artifact_descriptors[ref].required_for == frozenset({route})
    original = solution.retained_continuations[ref]
    assert isinstance(original, BalancedCaseUCarry)
    with pytest.raises(
        IncompatibleSolutionError,
        match="^"
        + re.escape(
            "Persisted artifact 'pylcm.egm.pre_shock_continuation' version 1 at "
            "period=1, regime='child' is a plugin-defined PyTree. Install the "
            "matching plugin and replay it through Model.simulate(), which "
            "supplies the model-authoritative template."
        )
        + "$",
    ):
        loaded.retained_continuations[ref]
    fresh = _parent_model()
    result = fresh.simulate(
        params={"discount_factor": 1.0},
        initial_conditions=_initial(model=fresh),
        solution=loaded,
        seed=23,
        log_level="off",
    )
    restored = loaded.retained_continuations[ref]
    assert isinstance(restored, BalancedCaseUCarry)
    assert jax.tree.structure(restored) == jax.tree.structure(original)
    for wanted, actual in zip(
        jax.tree.leaves(original), jax.tree.leaves(restored), strict=True
    ):
        np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_array_almost_equal(
        result.raw_results["parent"][0].actions["consumption"],
        [1.0],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["parent"][0].V_arr,
        [2.8 + np.log(2.0) - 1.5],
        decimal=DECIMAL_PRECISION,
    )
