"""A genuine nested NB model maximizes both margins before current expectation."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.current_iid_case_u_replay import CASE_U_FUTURE_REPLAY, CaseUReplayChildren
from _lcm.egm.current_iid_nnbegm import _ORIGINAL_NNBEGM_METHODS
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from lcm import Model, NestedConsumptionSavingsRegime
from lcm.exceptions import UnsupportedOperationError
from lcm.model import _ModelConstructorInputs
from lcm.shocks import PhysicalValues
from lcm.solver_api import ArtifactRef, ResultRetention
from lcm.solvers import NBEGM, NEGM, NNBEGM, FiniteOuterGrid
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_negm_model import _declarations, _oracle, _params

_REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nnbegm_u_selects_inner_nb_before_outer_choice() -> None:
    """Keeper and adjuster use their original NB solves at every physical frame."""
    declarations = _declarations()
    original = declarations.regimes["child"]
    assert isinstance(original, NestedConsumptionSavingsRegime)
    assert isinstance(original.solver, NEGM)
    solver = NNBEGM(
        inner=NBEGM(
            savings_grid=original.solver.inner.savings_grid, jump_read="bridged"
        ),
        outer_search=FiniteOuterGrid(grid=original.solver.outer_grid),
    )
    child = replace(original, solver=solver)
    assert isinstance(child.solver, NNBEGM)
    assert isinstance(child.solver.inner, NBEGM)
    model = Model._from_call_owned_params(
        declarations=replace(
            declarations, regimes={**declarations.regimes, "child": child}
        ),
        params=_params(),
    )
    solution = model.solve(
        params=_params(),
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    reference = np.mean(
        [_oracle(wealth=2.0, housing=1.0, shock=z)[:2] for z in (0.0, 3.0)], axis=0
    )
    assert _oracle(wealth=2.0, housing=1.0, shock=0.0)[6] == 0
    assert _oracle(wealth=2.0, housing=1.0, shock=3.0)[6] == 1
    np.testing.assert_almost_equal(
        solution.values[1]["child"][1, 1], reference[0], decimal=DECIMAL_PRECISION
    )
    stored = solution.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(stored, ProjectedEGMCarry)
    length = int(stored.valid_length[1])
    slots = np.flatnonzero(np.asarray(stored.grid[1, :length]) == 2.0)
    assert slots.size == 1
    np.testing.assert_almost_equal(
        stored.economic_marginal[1, slots[0]], reference[1], decimal=DECIMAL_PRECISION
    )
    physical = np.asarray([0.0, 2.0, 3.0], dtype=canonical_float_dtype())
    result = model.simulate(
        params=_params(),
        solution=solution,
        log_level="off",
        seed=31,
        initial_conditions={
            "wealth": jnp.full(physical.shape, 2.0),
            "housing": jnp.ones(physical.shape),
            "age": jnp.ones(physical.shape),
            "regime_id": jnp.full(
                physical.shape, model.regime_names_to_ids["child"], jnp.int32
            ),
            "eps": PhysicalValues(values=physical),
        },
    )
    actual = result.raw_results["child"][1]
    expected = np.asarray([_oracle(wealth=2.0, housing=1.0, shock=z) for z in physical])
    np.testing.assert_array_almost_equal(
        actual.actions["consumption"], expected[:, 2], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        actual.actions["investment"], expected[:, 4], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(actual.shocks["eps"], physical)
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["wealth"],
        expected[:, 3],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["housing"],
        expected[:, 5],
        decimal=DECIMAL_PRECISION,
    )
    assert float(actual.actions["investment"][2]) not in (-2.0, 0.0, 2.0)
    assert float(actual.actions["consumption"][1]) not in (0.1, 5.0)
    future = solution.replay_artifacts[
        ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    ]
    assert isinstance(future, CaseUReplayChildren)
    _assert_original_nnbegm_banks(model=model, future=future)


def _assert_original_nnbegm_banks(  # noqa: PLR0915 - original bank/tuple and custody controls
    *, model: Model, future: CaseUReplayChildren
) -> None:
    """The actual nested NB bank owns every retained associated query record."""
    from types import MappingProxyType, MethodType  # noqa: PLC0415

    from _lcm.certainty_equivalent import (  # noqa: PLC0415
        LinearExpectation,
        PowerMean,
    )
    from _lcm.egm.current_iid_case_u_replay import CaseUReplayChildren  # noqa: PLC0415
    from _lcm.egm.current_iid_nnbegm import NNBEGMCaseUPoint  # noqa: PLC0415
    from _lcm.solution.continuation_target import (  # noqa: PLC0415
        union_fixed_params,
        union_free_params,
    )
    from _lcm.solution.current_iid_case_u_native import (  # noqa: PLC0415
        CaseUExactArguments,
    )
    from _lcm.solution.current_iid_negm import (  # noqa: PLC0415
        NativeNEGMCaseUPeriodKernel,
    )
    from _lcm.solution.model_authority import (  # noqa: PLC0415
        _is_installed_negm_u_continuation,
    )
    from _lcm.solution.nbegm import (  # noqa: PLC0415
        _NBEGMQueryIntervalEvaluators,
        _NBEGMTiledCore,
        _RideAlongNBEGMPeriodKernel,
    )
    from _lcm.solution.nnbegm import _FiniteNNBEGMPeriodKernel  # noqa: PLC0415

    assert isinstance(future, CaseUReplayChildren)
    regime = model._regimes["child"]
    kernel = regime.solution.period_kernels[1]
    assert isinstance(kernel, NativeNEGMCaseUPeriodKernel)
    point = kernel.point
    assert isinstance(point, NNBEGMCaseUPoint)
    assert type(point.owner) is _FiniteNNBEGMPeriodKernel
    assert point.plan is kernel.context.current_iid_plans[1]
    for owner, inner in (
        (point.owner.keeper_kernel, point.keeper),
        (point.owner.adjuster_kernel, point.adjuster),
    ):
        assert type(owner) is _RideAlongNBEGMPeriodKernel
        assert inner.kernel is owner
        assert type(inner.core) is _NBEGMTiledCore
        assert owner.core_programs()["main"].function is inner.core
        assert inner.recipe.source_identity is inner.core.schedule_spec.sources
        assert type(point.context.certainty_equivalent) is LinearExpectation
        assert inner.continuation_plan.certainty_equivalent is None
        assert inner.continuation_plan.risk_aversion_param_name is None
        assert inner.has_original_linear_plan(context=point.context)
    arguments = kernel.program.argument_builder
    assert isinstance(arguments, CaseUExactArguments)
    flat, _generation = model._prepare_call(params=_params())
    pool = {
        **union_fixed_params(
            fixed_flat_params=arguments.point.fixed_flat_params,
            regime_name="child",
            transition_target_names=("done",),
        ),
        **union_free_params(
            flat_params=flat, regime_name="child", transition_target_names=("done",)
        ),
        "period": jnp.int32(1),
        "age": kernel.context.ages.values[1],
        "housing": jnp.asarray(1.0),
    }
    physical = {"eps": jnp.asarray(3.0)}
    children = MappingProxyType(dict(future.children))
    liquid = jnp.asarray(2.0)
    banks = point.construct(
        liquid=liquid, physical=physical, pool=pool, children=children
    )
    assert len(banks) == point.outer_nodes.shape[0] + 1
    for constructed in banks:
        bank = constructed.output
        live = np.asarray(bank.grid)[np.isfinite(bank.grid)]
        np.testing.assert_array_equal(live, np.unique(np.r_[point.source_grid, liquid]))
        assert bank.grid.shape[0] == point.source_grid.shape[0] + 1
        assert float(bank.grid[bank.requested_index]) == float(liquid)
        assert bool(bank.support_valid)
    actual = point.read_constructed(
        liquid=liquid, physical=physical, pool=pool, banks=banks
    )
    selected, investment, outer_index = actual.output
    assert bool(selected.valid)
    assert int(outer_index) == 1
    bank = banks[int(outer_index)].output
    slot = int(bank.requested_index)
    np.testing.assert_array_equal(
        [selected.native_owner, selected.policy, selected.savings],
        [
            bank.row.owner[slot],
            bank.row.policy[slot],
            bank.row.query_data.savings[slot],
        ],
    )
    np.testing.assert_array_almost_equal(
        [
            selected.value,
            selected.marginal,
            selected.policy,
            selected.savings,
            investment,
        ],
        _oracle(wealth=2.0, housing=1.0, shock=3.0)[:5],
        decimal=DECIMAL_PRECISION,
    )
    native_candidates = point.savings_nodes.shape[0] * (
        1 + 2 * point.source_grid.shape[0]
    )
    first_invalid_owner = 2 * native_candidates - 1
    bad = replace(
        banks[1],
        output=bank._replace(
            row=bank.row._replace(
                owner=bank.row.owner.at[slot].set(first_invalid_owner)
            )
        ),
    )
    corrupted = point.read_constructed(
        liquid=liquid, physical=physical, pool=pool, banks=(banks[0], bad, *banks[2:])
    )
    assert not bool(corrupted.output[0].valid)
    bad = replace(
        banks[1],
        output=bank._replace(
            row=bank.row._replace(
                query_data=replace(
                    bank.row.query_data, floor_used=bank.row.query_data.floor_used + 1
                )
            )
        ),
    )
    assert not bool(
        point.read_constructed(
            liquid=liquid,
            physical=physical,
            pool=pool,
            banks=(banks[0], bad, *banks[2:]),
        )
        .output[0]
        .valid
    )
    # Both physical and durable/liquid coordinates are reconstructed from their
    # actual frame, including coordinates absent from the original source grids.
    for wealth, housing, shock in ((2.5, 0.5, 3.0), (2.0, 1.0, 2.0), (0.01, 1.0, 3.0)):
        actual = point.select(
            liquid=jnp.asarray(wealth),
            physical={"eps": jnp.asarray(shock)},
            pool={**pool, "housing": jnp.asarray(housing)},
            children=children,
        ).output
        assert bool(actual[0].valid)
        np.testing.assert_array_almost_equal(
            [
                actual[0].value,
                actual[0].marginal,
                actual[0].policy,
                actual[0].savings,
                actual[1],
            ],
            _oracle(wealth=wealth, housing=housing, shock=shock)[:5],
            decimal=DECIMAL_PRECISION,
        )
    with pytest.raises(TypeError, match="original finite owner"):
        replace(point, plan=replace(point.plan))
    nonlinear = replace(point.context, certainty_equivalent=PowerMean())
    assert not point.keeper.has_original_linear_plan(context=nonlinear)
    with pytest.raises(
        UnsupportedOperationError, match="actual liquid/durable ordinary cells"
    ):
        replace(point, context=nonlinear)
    with pytest.raises(TypeError, match="original finite owner"):
        replace(point, keeper=point.adjuster)
    with pytest.raises(TypeError, match="original numerical owner"):
        replace(
            point.keeper,
            core=replace(
                point.keeper.core,
                continuation_plan=point.adjuster.continuation_plan,
            ),
        )
    with pytest.raises(TypeError, match="original declared sources"):
        replace(point.keeper, recipe=replace(point.keeper.recipe, equal_above=(True,)))
    with pytest.raises(TypeError, match="original declared sources"):
        replace(
            point.keeper,
            recipe=replace(
                point.keeper.recipe,
                input_names=point.keeper.recipe.input_names | {"foreign_input"},
            ),
        )
    preimages = point.keeper.recipe.evaluate_preimages
    assert isinstance(preimages, MethodType)
    evaluator = preimages.__self__
    assert isinstance(evaluator, _NBEGMQueryIntervalEvaluators)
    foreign = replace(evaluator, liquid_name="foreign_liquid")
    with pytest.raises(TypeError, match="original declared sources"):
        replace(
            point.keeper,
            recipe=replace(
                point.keeper.recipe,
                evaluate_preimages=foreign.preimages,
                evaluate_sides=foreign.sides,
            ),
        )
    template = kernel.template.output.projected
    assert _is_installed_negm_u_continuation(
        regime=regime,
        regime_name="child",
        period=1,
        spec=kernel.spec,
        template=template,
    )
    assert not _is_installed_negm_u_continuation(
        regime=regime,
        regime_name="child",
        period=1,
        spec=replace(kernel.spec),
        template=template,
    )
    assert not _is_installed_negm_u_continuation(
        regime=regime,
        regime_name="child",
        period=1,
        spec=kernel.spec,
        template=replace(template),
    )
    # Each original inner branch has already bound its actual durable cell;
    # all associated corner channels must match the scalar ordinary query.
    frame = {**pool, **physical, point.euler_state_name: liquid}
    targets = (point.keeper_target(**frame), *tuple(point.outer_nodes))
    for position, (target, constructed) in enumerate(zip(targets, banks, strict=True)):
        profile, _integrity = point._profile(
            inner=point.keeper if position == 0 else point.adjuster,
            bank=constructed.output,
            actual={**frame, point.owner.outer_post_decision: target},
            liquid=liquid,
        )
        assert profile.right.value.shape == ()
        assert profile.left.value.shape == ()
        for channel in profile.corner:
            assert channel.shape == ()
    replay = regime.simulation.case_u_replay_spec
    assert replay is not None
    assert replay.periods[1].negm_point is point
    assert regime.simulation.replay_route is replay
    assert regime.simulation.egm_policy_read is None
    # At the exact closed upper node, the native economic derivative is distinct
    # from the represented value's constant right-extension germ.
    upper_liquid = point.source_grid[-1]
    upper_banks = point.construct(
        liquid=upper_liquid, physical=physical, pool=pool, children=children
    )
    upper, _investment, upper_index = point.read_constructed(
        liquid=upper_liquid, physical=physical, pool=pool, banks=upper_banks
    ).output
    assert bool(upper.valid)
    assert bool(upper.feasible)
    assert not bool(upper.corner)
    upper_bank = upper_banks[int(upper_index)].output
    upper_slot = int(upper_bank.requested_index)
    assert float(upper_bank.grid[upper_slot]) == float(upper_liquid)
    original_row = upper_bank.row
    original_marginal = original_row.marginal[upper_slot]
    assert float(original_marginal) > 0.0
    selected_inner = point.keeper if int(upper_index) == 0 else point.adjuster
    selected_target = (
        point.keeper_target(
            **{**pool, **physical, point.euler_state_name: upper_liquid}
        )
        if int(upper_index) == 0
        else point.outer_nodes[int(upper_index) - 1]
    )
    profile, integrity = point._profile(
        inner=selected_inner,
        bank=upper_bank,
        actual={
            **pool,
            **physical,
            point.euler_state_name: upper_liquid,
            point.owner.outer_post_decision: selected_target,
        },
        liquid=upper_liquid,
    )
    assert bool(integrity)
    assert bool(profile.right.supported)
    assert bool(profile.right.dead)
    assert bool(jnp.isposinf(upper_bank.grid[-1]))
    np.testing.assert_array_equal(
        [upper.native_owner, upper.side_id],
        [original_row.owner[upper_slot], original_row.query_data.side_id[upper_slot]],
    )
    np.testing.assert_array_almost_equal(
        [upper.value, upper.policy, upper.savings],
        [
            original_row.value[upper_slot],
            original_row.policy[upper_slot],
            original_row.query_data.savings[upper_slot],
        ],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_almost_equal(
        profile.right.marginal, original_marginal, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_almost_equal(
        upper.marginal, original_marginal, decimal=DECIMAL_PRECISION
    )


def _nnbegm_declarations(*, budget: int | None) -> _ModelConstructorInputs:
    """The default nested NB solver over the NEGM fixture's grids."""
    declarations = _declarations()
    original = declarations.regimes["child"]
    assert isinstance(original, NestedConsumptionSavingsRegime)
    assert isinstance(original.solver, NEGM)
    solver = NNBEGM(
        inner=NBEGM(
            savings_grid=original.solver.inner.savings_grid, jump_read="bridged"
        ),
        outer_search=FiniteOuterGrid(grid=original.solver.outer_grid),
    )
    return replace(
        declarations,
        regimes={**declarations.regimes, "child": replace(original, solver=solver)},
        execution_config=replace(
            declarations.execution_config, device_memory_bytes=budget
        ),
    )


_FRESH_BUDGETED_SOLVE = """
from tests.egm.test_current_iid_call_owned_generation import (
    _public_model_from_declarations,
)
from tests.egm.test_current_iid_negm_model import _params
from tests.egm.test_current_iid_nnbegm_model import _nnbegm_declarations

model = _public_model_from_declarations(
    declarations=_nnbegm_declarations(budget=1 << 28)
)
model.solve(params=_params(), log_level="off")
"""


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_default_nnbegm_u_budgeted_solve_needs_no_exact_kernel_registration() -> None:
    """An ordinary nested NB source solves in a process that never loaded the FFI."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _FRESH_BUDGETED_SOLVE],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr[-4000:]


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("budget", [None, 1 << 28], ids=["unbudgeted", "budgeted"])
def test_public_nnbegm_u_default_archive_replays_original_physics(
    *, tmp_path: Path, budget: int | None
) -> None:
    """Default retention delivers original nested NB physics to a fresh Model."""
    from pandas.testing import assert_frame_equal  # noqa: PLC0415

    from _lcm.solution.artifacts import OwnedSolutionView  # noqa: PLC0415
    from lcm.persistence import load_solution  # noqa: PLC0415
    from tests.egm.test_current_iid_call_owned_generation import (  # noqa: PLC0415
        _public_model_from_declarations,
    )

    declarations = _nnbegm_declarations(budget=budget)
    params = _params()
    model = _public_model_from_declarations(declarations=declarations)
    solution = model.solve(params=params, log_level="off")
    assert solution.metadata.retention is ResultRetention.VALUES_AND_REPLAY
    view = solution._engine_view
    assert isinstance(view, OwnedSolutionView)
    generation = view.generation
    assert generation is not None
    assert generation.execution is not model
    assert model._generation_cache._cached[generation.key] is generation
    reference = np.mean(
        [_oracle(wealth=2.0, housing=1.0, shock=z)[:2] for z in (0.0, 3.0)], axis=0
    )
    assert _oracle(wealth=2.0, housing=1.0, shock=0.0)[6] == 0
    assert _oracle(wealth=2.0, housing=1.0, shock=3.0)[6] == 1
    np.testing.assert_almost_equal(
        solution.values[1]["child"][1, 1], reference[0], decimal=DECIMAL_PRECISION
    )
    future = solution.replay_artifacts[
        ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    ]
    assert isinstance(future, CaseUReplayChildren)
    assert future.child_names == ("done",)
    physical = np.asarray([0.0, 2.0, 3.0], dtype=canonical_float_dtype())
    initial_conditions = {
        "wealth": jnp.full(physical.shape, 2.0),
        "housing": jnp.ones(physical.shape),
        "age": jnp.ones(physical.shape),
        "regime_id": jnp.full(
            physical.shape, model.regime_names_to_ids["child"], jnp.int32
        ),
        "eps": PhysicalValues(values=physical),
    }
    result = model.simulate(
        params=params,
        solution=solution,
        log_level="off",
        seed=31,
        initial_conditions=initial_conditions,
    )
    actual = result.raw_results["child"][1]
    expected = np.asarray([_oracle(wealth=2.0, housing=1.0, shock=z) for z in physical])
    np.testing.assert_array_almost_equal(
        actual.actions["consumption"], expected[:, 2], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        actual.actions["investment"], expected[:, 4], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(actual.shocks["eps"], physical)
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["wealth"],
        expected[:, 3],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["housing"],
        expected[:, 5],
        decimal=DECIMAL_PRECISION,
    )
    assert float(actual.actions["investment"][2]) not in (-2.0, 0.0, 2.0)
    assert float(actual.actions["consumption"][1]) not in (0.1, 5.0)
    _assert_original_nnbegm_banks(model=generation.execution, future=future)
    path = tmp_path / "public-nnbegm-default.lcm"
    solution.save(path=path)
    loaded = load_solution(path=path)
    assert loaded._engine_view is None
    fresh = _public_model_from_declarations(declarations=declarations)
    replay = fresh.simulate(
        params=params,
        solution=loaded,
        log_level="off",
        seed=31,
        initial_conditions=initial_conditions,
    )
    fresh_generation = fresh._generation_cache._cached[generation.key]
    assert fresh_generation.key == generation.key
    assert fresh_generation is not generation
    assert fresh_generation.execution is not fresh
    assert fresh_generation.execution is not generation.execution
    assert_frame_equal(replay.to_dataframe(), result.to_dataframe(), check_exact=True)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nnbegm_shape_capture_keeps_both_banks_and_refuses_changed_engine(  # noqa: PLR0915 - one real prepared source with independent before-effect detectors
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Original arrays stay owned; changed methods, getters and init cannot run."""
    from types import MappingProxyType  # noqa: PLC0415

    import jax  # noqa: PLC0415

    from _lcm.egm.current_iid_negm_program import NEGMCaseUArguments  # noqa: PLC0415
    from _lcm.egm.current_iid_nnbegm import (  # noqa: PLC0415
        NNBEGMCaseUPoint,
        NNBEGMInnerSource,
    )
    from _lcm.execution.core_program import CoreBuildContext  # noqa: PLC0415
    from _lcm.solution.current_iid_case_u_build import (  # noqa: PLC0415
        guard_case_u_shape,
    )
    from _lcm.solution.current_iid_case_u_native import (  # noqa: PLC0415
        CaseUExactArguments,
    )
    from _lcm.solution.current_iid_negm import (  # noqa: PLC0415
        NativeNEGMCaseUPeriodKernel,
    )
    from _lcm.solution.nbegm import _NBEGMCellContinuation  # noqa: PLC0415
    from _lcm.typing import PytreeValue  # noqa: PLC0415
    from lcm.exceptions import ExecutionPlanningError  # noqa: PLC0415
    from tests.egm.test_current_iid_call_owned_generation import (  # noqa: PLC0415
        _public_model_from_declarations,
    )
    from tests.egm.test_current_iid_generation_materialization import (  # noqa: PLC0415
        _owner,
    )

    declarations = _declarations()
    original = declarations.regimes["child"]
    assert isinstance(original, NestedConsumptionSavingsRegime)
    assert isinstance(original.solver, NEGM)
    child = replace(
        original,
        solver=NNBEGM(
            inner=NBEGM(
                savings_grid=original.solver.inner.savings_grid, jump_read="bridged"
            ),
            outer_search=FiniteOuterGrid(grid=original.solver.outer_grid),
        ),
    )
    declarations = replace(
        declarations,
        regimes={**declarations.regimes, "child": child},
        execution_config=replace(
            declarations.execution_config, device_memory_bytes=1 << 28
        ),
    )
    model = _public_model_from_declarations(declarations=declarations)
    flat, generation = model._prepare_call(params=_params())
    kernel = generation.execution._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNEGMCaseUPeriodKernel)
    point = kernel.point
    assert isinstance(point, NNBEGMCaseUPoint)
    context = CoreBuildContext(
        state_action_space=kernel.context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation=kernel.future.template().children,
        flat_params=flat,
        period=1,
        ages=kernel.context.ages,
    )
    owner = _owner(roots=generation.live_roots())
    owner.budget_bytes = 1 << 28
    try:
        held = guard_case_u_shape(
            function=kernel.program.function,
            context=context,
            fixed_flat_params=MappingProxyType({}),
            materializer=owner,
        )
        for original_array in (
            point.plan.indices,
            point.source_grid,
            point.outer_nodes,
            point.keeper.core.envelope_build.bind_cell.action_upper,
            point.adjuster.core.envelope_build.bind_cell.action_upper,
            point.keeper.savings_nodes,
            point.adjuster.savings_nodes,
        ):
            assert any(value is original_array for value in held)
            assert any(
                value is original_array
                for value in jax.tree.leaves(owner.solve_input_roots())
            )
        clock_owner = _owner(roots=generation.live_roots())
        clock_owner.budget_bytes = 0
        try:
            arguments = kernel.program.argument_builder
            assert isinstance(arguments, CaseUExactArguments)
            assert isinstance(arguments.point, NEGMCaseUArguments)
            # The budgeted cold build already admitted this clock; refuse a fresh one.
            assert arguments.point.clock_values is not None
            unadmitted = replace(arguments.point, clock_values=None)
            with pytest.raises(ExecutionPlanningError):
                unadmitted.prepare(context=context, materializer=clock_owner)
            assert not clock_owner._pending
        finally:
            clock_owner.close()
        effects: list[bool] = []

        def changed_point(self: NNBEGMCaseUPoint) -> None:
            assert self is point
            effects.append(True)
            raise AssertionError("Changed NN point ran")

        with pytest.raises(AssertionError, match="Changed NN point ran"):
            changed_point(point)
        effects.clear()
        with monkeypatch.context() as patch:
            patch.setattr(NNBEGMCaseUPoint, "__post_init__", changed_point)
            with pytest.raises(
                TypeError, match="Original NNBEGM shape method authority"
            ):
                guard_case_u_shape(
                    function=kernel.program.function,
                    context=context,
                    fixed_flat_params=MappingProxyType({}),
                    materializer=owner,
                )
            assert not effects

        def changed_getter(self: NNBEGMInnerSource) -> None:
            assert self is point.keeper
            effects.append(True)
            raise AssertionError("Changed NN getter ran")

        with monkeypatch.context() as patch:
            patch.setattr(NNBEGMInnerSource, "savings_nodes", property(changed_getter))
            with pytest.raises(AssertionError, match="Changed NN getter ran"):
                _ = point.keeper.savings_nodes
            effects.clear()
            with pytest.raises(
                TypeError, match="Original NNBEGM shape property authority"
            ):
                guard_case_u_shape(
                    function=kernel.program.function,
                    context=context,
                    fixed_flat_params=MappingProxyType({}),
                    materializer=owner,
                )
            assert not effects

        from _lcm.solution import nbegm as original_nb_module  # noqa: PLC0415

        class ChangedCellSolver:
            def __init__(self, **kwargs: PytreeValue) -> None:
                del kwargs
                effects.append(True)
                raise AssertionError("Changed NN constructor class ran")

        original_class = original_nb_module._NBEGMCellSolver
        ChangedCellSolver.__module__ = original_class.__module__
        ChangedCellSolver.__qualname__ = original_class.__qualname__
        with monkeypatch.context() as patch:
            patch.setattr(original_nb_module, "_NBEGMCellSolver", ChangedCellSolver)
            assert vars(original_nb_module)["_NBEGMCellSolver"] is ChangedCellSolver
            with pytest.raises(
                AssertionError, match="Changed NN constructor class ran"
            ):
                ChangedCellSolver()
            effects.clear()
            with pytest.raises(
                TypeError, match="Original NNBEGM constructor class binding"
            ):
                guard_case_u_shape(
                    function=kernel.program.function,
                    context=context,
                    fixed_flat_params=MappingProxyType({}),
                    materializer=owner,
                )
            assert not effects

        def changed_init(self: _NBEGMCellContinuation, **kwargs: PytreeValue) -> None:
            del self, kwargs
            effects.append(True)
            raise AssertionError("Changed NN constructor ran")

        with monkeypatch.context() as patch:
            patch.setattr(_NBEGMCellContinuation, "__init__", changed_init)
            with pytest.raises(AssertionError, match="Changed NN constructor ran"):
                _NBEGMCellContinuation(
                    kwargs={"wealth": point.source_grid},
                    carry=MappingProxyType(dict(kernel.future.template().children)),
                    comap_bindings=MappingProxyType({}),
                    savings_grid=point.savings_nodes,
                    continuation_plan=point.keeper.core.continuation_plan,
                    statics=point.keeper.core.statics,
                    regime_name="child",
                    cliff_candidates=False,
                    schedule_spec=point.keeper.core.schedule_spec,
                )
            effects.clear()
            with pytest.raises(
                TypeError, match="Original NNBEGM shape method authority"
            ):
                guard_case_u_shape(
                    function=kernel.program.function,
                    context=context,
                    fixed_flat_params=MappingProxyType({}),
                    materializer=owner,
                )
            assert not effects
    finally:
        owner.close()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("branch", ["keeper", "adjuster", "plan-alias"])
def test_nnbegm_shape_capture_refuses_nested_host_or_plan_alias(*, branch: str) -> None:
    """Neither genuine tiled branch can hide host data in a numerical callback."""
    from types import MappingProxyType  # noqa: PLC0415

    from _lcm.egm.current_iid_nnbegm import NNBEGMCaseUPoint  # noqa: PLC0415
    from _lcm.execution.core_program import CoreBuildContext  # noqa: PLC0415
    from _lcm.solution.current_iid_case_u_build import (  # noqa: PLC0415
        guard_case_u_shape,
    )
    from _lcm.solution.current_iid_case_u_native import (  # noqa: PLC0415
        CaseURideExactFunction,
    )
    from _lcm.solution.current_iid_negm import (  # noqa: PLC0415
        NativeNEGMCaseUPeriodKernel,
    )
    from lcm.typing import FloatND  # noqa: PLC0415
    from tests.egm.test_current_iid_call_owned_generation import (  # noqa: PLC0415
        _public_model_from_declarations,
    )
    from tests.egm.test_current_iid_case_u_constructor_admission import (  # noqa: PLC0415
        _consume_user_plan,
    )
    from tests.egm.test_current_iid_generation_materialization import (  # noqa: PLC0415
        _owner,
    )

    declarations = _declarations()
    original = declarations.regimes["child"]
    assert isinstance(original, NestedConsumptionSavingsRegime)
    assert isinstance(original.solver, NEGM)
    child = replace(
        original,
        solver=NNBEGM(
            inner=NBEGM(
                savings_grid=original.solver.inner.savings_grid, jump_read="bridged"
            ),
            outer_search=FiniteOuterGrid(grid=original.solver.outer_grid),
        ),
    )
    declarations = replace(
        declarations,
        regimes={**declarations.regimes, "child": child},
        execution_config=replace(
            declarations.execution_config, device_memory_bytes=1 << 28
        ),
    )
    model = _public_model_from_declarations(declarations=declarations)
    flat, generation = model._prepare_call(params=_params())
    kernel = generation.execution._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNEGMCaseUPeriodKernel)
    point = kernel.point
    assert isinstance(point, NNBEGMCaseUPoint)
    hidden = np.asarray([2.0])
    original_plan = point.plan

    def host_capture(*, consumption: FloatND, eps: FloatND) -> FloatND:
        return (1 + eps) * jnp.log(consumption) + hidden[0]

    def plan_capture(*, consumption: FloatND, eps: FloatND) -> FloatND:
        return _consume_user_plan(
            plan=original_plan, value=(1 + eps) * jnp.log(consumption)
        )

    inner = point.adjuster if branch == "adjuster" else point.keeper
    spec = replace(
        inner.core.schedule_spec,
        utility_dag=plan_capture if branch == "plan-alias" else host_capture,
    )
    core = replace(
        inner.core,
        schedule_spec=spec,
        envelope_build=replace(
            inner.core.envelope_build,
            bind_cell=replace(inner.core.envelope_build.bind_cell, schedule_spec=spec),
        ),
    )
    changed_kernel = replace(
        inner.kernel,
        _core_programs=MappingProxyType(
            {
                **inner.kernel._core_programs,
                "main": replace(inner.kernel._core_programs["main"], function=core),
            }
        ),
    )
    inner = replace(inner, core=core, kernel=changed_kernel)
    changed_owner = replace(
        point.owner,
        keeper_kernel=point.owner.keeper_kernel
        if branch == "adjuster"
        else changed_kernel,
        adjuster_kernel=changed_kernel
        if branch == "adjuster"
        else point.owner.adjuster_kernel,
    )
    point = replace(
        point,
        owner=changed_owner,
        keeper=point.keeper if branch == "adjuster" else inner,
        adjuster=inner if branch == "adjuster" else point.adjuster,
    )
    function = kernel.program.function
    assert isinstance(function, CaseURideExactFunction)
    function = replace(function, point=replace(function.point, pieces=point))
    context = CoreBuildContext(
        state_action_space=kernel.context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation=kernel.future.template().children,
        flat_params=flat,
        period=1,
        ages=kernel.context.ages,
    )
    owner = _owner(roots=generation.live_roots())
    owner.budget_bytes = 1 << 28
    try:
        with pytest.raises(
            (UnsupportedOperationError, TypeError),
            match=r"host-array capture|direct object dependency .*CurrentIIDPlan",
        ):
            guard_case_u_shape(
                function=function,
                context=context,
                fixed_flat_params=MappingProxyType({}),
                materializer=owner,
            )
        assert not owner._pending
    finally:
        owner.close()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nnbegm_method_snapshot_excludes_beartype_sizeof_cache() -> None:
    """The original-method snapshot holds only Python engine code."""
    assert all(row[1] != "__sizeof__" for row in _ORIGINAL_NNBEGM_METHODS)
    assert all(row[5] is row[4].__code__ for row in _ORIGINAL_NNBEGM_METHODS)
