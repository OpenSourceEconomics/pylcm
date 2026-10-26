"""Genuine U Model publication retains PWA economics and original forward choices."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.projected_child import ProjectedChildReader
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.solution.current_iid_case_u_native import acquire_installed_case_u_date
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
from lcm.model import _ModelConstructorInputs
from lcm.shocks import PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import ArtifactRef, ResultRetention, SolutionResult
from lcm.solvers import DCEGM, ExactEnvelope
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class _Regimes:
    child: ScalarInt
    done: ScalarInt


def _utility(*, consumption: ContinuousAction, eps: FloatND) -> FloatND:
    return (1.0 + eps) * jnp.log(consumption)


def _inverse(*, marginal_continuation: FloatND, eps: FloatND) -> FloatND:
    return (1.0 + eps) / marginal_continuation


def _resources(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _next_wealth(*, savings: FloatND) -> FloatND:
    return savings


def _aliased_next_wealth(*, savings: FloatND, resources: FloatND) -> FloatND:
    # Genuine candidate production changes with the current resources alias.
    return savings + 0.125 * resources


def _terminal(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _coordinate(savings: FloatND) -> FloatND:
    return savings


def _declarations(
    *, lower: float = 2.75, capacity: int = 256, alias: bool = False
) -> _ModelConstructorInputs:
    savings = IrregSpacedGrid(
        points=np.array(
            [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.875, 2.0, 2.75, 2.875, 3.875]
        )
    )
    child = ConsumptionSavingsRegime(
        states={"wealth": IrregSpacedGrid(points=np.array([lower, lower + 0.125]))},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.25, 2.0]))},
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        functions={
            "utility": _utility,
            "inverse_marginal_utility": _inverse,
            "resources": _resources,
            "savings": _savings,
        },
        state_transitions={
            "wealth": {"done": _aliased_next_wealth if alias else _next_wealth}
        },
        solver=DCEGM(
            savings_grid=savings,
            envelope=ExactEnvelope(),
            n_constrained_points=2,
            current_iid_projection_capacity=capacity,
        ),
        shocks=("income",),
    )
    return _ModelConstructorInputs(
        regimes={
            "child": child,
            "done": Regime(
                states={
                    "wealth": IrregSpacedGrid(
                        points=np.array(
                            [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 3.875, 4.5]
                        )
                    )
                },
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(exact_values=(0, 1)),
        initial_nodes=InitialNodes(by_age={0: "child"}),
        edges={"child": {"done": (0,)}},
        regime_id_class=_Regimes,
        shocks={
            "income": ShockGroup(
                shocks={"eps": Uniform(start=0.0, stop=1.0)},
                nodes=UserNodes(
                    nodes=np.array([[0.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )


def _model(*, lower: float = 2.75, capacity: int = 256, alias: bool = False) -> Model:
    return Model._from_call_owned_params(
        declarations=_declarations(lower=lower, capacity=capacity, alias=alias),
        params={"discount_factor": 1.0},
    )


def _solved(*, model: Model) -> SolutionResult:
    return model.solve(
        params={"discount_factor": 1.0},
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )


def _carry(*, solution: SolutionResult) -> ProjectedEGMCarry:
    result = solution.retained_continuations[
        ArtifactRef(period=0, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(result, ProjectedEGMCarry)
    return result


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_genuine_u_model_publishes_nonexact_pwa_and_reconstructs_original_policy() -> (
    None
):
    """Linear future gives c=alpha, W=m+log(2)-1.5, and economic marginal one."""
    model = _model()
    solution = _solved(model=model)
    carry = _carry(solution=solution)
    # Genuine RED on the exact-only predecessor, before any new-module import.
    assert bool(jnp.any(carry.link_valid)), "Original U publisher has no smooth link."
    flat, generation = model._prepare_call(params={"discount_factor": 1.0})
    del flat
    prepared = acquire_installed_case_u_date(
        generation=generation, regime_name="child", period=0
    )
    assert prepared.date.guard_program is not None
    from _lcm.egm.current_iid_case_u_balanced import (  # noqa: PLC0415
        BalancedCaseUCarry,
        SaturatedCaseUProgram,
        _select,
    )

    assert isinstance(carry, BalancedCaseUCarry)
    program = prepared.date.guard_program
    assert isinstance(program, SaturatedCaseUProgram)
    nodes = program.nodes(guard=carry.guard)
    physical = {name: value[0] for name, value in nodes.physical.items()}
    tangent = {name: value[0] for name, value in nodes.physical_tangent.items()}
    constructed, saturated = program.conditional(
        guard=carry.guard, physical=physical, tangent=tangent
    )
    assert bool(saturated)
    assert constructed.owned_envelope is not None
    # The exact original row/corner tie belongs to U's corner, never R's row.
    at_corner = program.solver(guard=carry.guard).node(
        liquid=constructed.first_endogenous_point,
        physical=physical,
        physical_tangent=tangent,
    )
    _, corner, local_valid = _select(
        node=constructed,
        row=constructed.owned_envelope,
        resources=constructed.first_endogenous_point,
    )
    assert bool(local_valid)
    assert bool(corner)
    assert bool(at_corner.output.corner)
    reader = ProjectedChildReader.bind_date(
        date=prepared.date, target="child", carry=carry, next_liquid=_coordinate
    )
    index = int(jnp.argmax(carry.link_valid))
    query = (carry.grid[index] + carry.grid[index + 1]) / 2
    assert not bool(jnp.any(carry.grid[: int(carry.valid_length)] == query))
    selected = program.solver(guard=carry.guard).node(
        liquid=query, physical=physical, physical_tangent=tangent
    )
    owner, corner, local_valid = _select(
        node=constructed, row=constructed.owned_envelope, resources=query
    )
    assert bool(local_valid)
    np.testing.assert_array_equal(selected.output.native_owner, owner)
    np.testing.assert_array_equal(selected.output.side_id, 0)
    np.testing.assert_array_equal(selected.output.corner, corner)
    result = reader.read_coordinate(query=query, jacobian=jnp.ones_like(query))
    assert result.query_status is not None
    assert not any(bool(jnp.any(jnp.stack(status))) for status in result.query_status)
    np.testing.assert_array_almost_equal(
        result.output[0], query + np.log(2.0) - 1.5, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.output[1], 1.0, decimal=DECIMAL_PRECISION
    )
    # The acquired PWA supplies levels; the guard never substitutes reference levels.
    shifted = replace(carry, value=carry.value + 0.125)
    shifted_read = replace(reader, carry=shifted).read_coordinate(
        query=query, jacobian=jnp.ones_like(query)
    )
    np.testing.assert_array_almost_equal(
        shifted_read.output[0] - result.output[0], 0.125, decimal=DECIMAL_PRECISION
    )
    simulated = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": jnp.asarray([2.8, 2.8, 2.8]),
            "age": jnp.zeros((3,)),
            "regime_id": jnp.full(
                (3,), model.regime_names_to_ids["child"], dtype=jnp.int32
            ),
            "eps": PhysicalValues(values=np.array([0.0, 0.5, 1.0])),
        },
        solution=solution,
        seed=19,
        log_level="off",
    )
    np.testing.assert_array_almost_equal(
        simulated.raw_results["child"][0].actions["consumption"],
        [1.0, 1.5, 2.0],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("alias", [False, True])
def test_small_capacity_or_resources_alias_keeps_original_exact_u(
    *, alias: bool
) -> None:
    """A real future-law resources alias must not gain fixed-bank authority."""
    model = _model(capacity=256 if alias else 8, alias=alias)
    _, generation = model._prepare_call(params={"discount_factor": 1.0})
    prepared = acquire_installed_case_u_date(
        generation=generation, regime_name="child", period=0
    )
    assert prepared.date.guard_program is None
    carry = _carry(solution=_solved(model=model))
    assert bool(carry.valid)
    assert not bool(jnp.any(carry.link_valid))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_unsaturated_u_retains_exact_records_without_smooth_authority() -> None:
    """The same economic class below its first Euler point remains exact-only."""
    model = _model(lower=0.75)
    solution = _solved(model=model)
    carry = _carry(solution=solution)
    assert bool(carry.valid)
    assert not bool(jnp.any(carry.link_valid))
    _, generation = model._prepare_call(params={"discount_factor": 1.0})
    prepared = acquire_installed_case_u_date(
        generation=generation, regime_name="child", period=0
    )
    reader = ProjectedChildReader.bind_date(
        date=prepared.date, target="child", carry=carry, next_liquid=_coordinate
    )
    exact = reader.read_coordinate(
        query=carry.grid[0], jacobian=jnp.ones_like(carry.grid[0])
    )
    missing = reader.read_coordinate(query=jnp.asarray(0.8), jacobian=jnp.asarray(1.0))
    assert exact.query_status is not None
    assert missing.query_status is not None
    assert not bool(jnp.any(jnp.stack(exact.query_status[-1])))
    assert bool(missing.query_status[-1].coverage_missing)
