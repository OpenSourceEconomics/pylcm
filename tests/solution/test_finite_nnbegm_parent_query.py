"""Read the finite nested child through a genuine EGM-family parent."""

from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import Unpack

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import ContinuationPlan, bind_continuation
from _lcm.egm.nbegm_query import NBEGMQuery
from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.execution.core_program import (
    CoreBuildContext,
    core_program_graph,
    materialize_core_program,
    select_programs,
)
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.solution import backward_induction
from _lcm.solution.feasibility import (
    BoundedResult,
    ContinuationCountedKernelOutput,
    ContinuationReadCounts,
    FeasibilityKernelOutput,
)
from _lcm.solution.kernel_output import ConsumedKernelOutput
from _lcm.solution.nbegm import (
    _NBEGMNestedKernelOutput,
    _RideAlongNBEGMPeriodKernel,
)
from _lcm.solution.negm import _with_outer_post_decision
from _lcm.solution.nnbegm import _NNBEGMPeriodKernel
from _lcm.typing import EconFunctionKwargs
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    InitialNodes,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    categorical,
    post_decision_lower_bound,
)
from lcm.solver_api import EGM_CONTINUATION, SIMULATION_POLICY, KernelOutput
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution import test_nested_dynamic_borrowing as original
from tests.solution._nbegm_direct_oracle import ride_along_kernel

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_PARAMS = {
    "parent": {"koopmans_aggregator": {"discount_factor": 0.5}},
    "alive": {"koopmans_aggregator": {"discount_factor": 0.0}},
}


@categorical(ordered=False)
class RegimeId:
    parent: ScalarInt
    alive: ScalarInt
    done: ScalarInt


def _parent_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _landing_liquid(*, savings: FloatND) -> ContinuousState:
    return 2.0 * savings + 0.125


def _landing_stock(*, illiquid: ContinuousState) -> ContinuousState:
    return 0.5 + illiquid / 2.0


def _model() -> Model:
    """Prepend one positive-discount NB ride parent to the nested child."""
    seed = original._model(family="nnbegm", lower=original.current_floor)
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    parent = ConsumptionSavingsRegime(
        koopmans_aggregator=seed.user_regimes["alive"].koopmans_aggregator,
        certainty_equivalent=seed.user_regimes["alive"].certainty_equivalent,
        states={"liquid": original._LIQUID, "illiquid": original._STOCK},
        actions={"consumption": original._CONSUMPTION},
        functions={
            "utility": _parent_utility,
            "resources": original.resources_before_outer_cost,
            "savings": original.savings,
            "inverse_marginal_utility": original.inverse_marginal_utility,
        },
        constraints={"borrowing": post_decision_lower_bound(margin=margin, lower=0.0)},
        state_transitions={"liquid": _landing_liquid, "illiquid": _landing_stock},
        liquid=margin,
        solver=original.NBEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.125, 0.25, 8.0]))
        ),
    )
    return Model(
        regimes={"parent": parent, **seed.user_regimes},
        ages=AgeGrid(exact_values=(0, 10, 20)),
        edges={"parent": {"alive": (0,)}, "alive": {"done": (10,)}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
    )


def _create_child_model() -> Model:
    """Start the original nested child at its declared age and period."""
    seed = original._model(family="nnbegm", lower=original.current_floor)
    model = Model(
        regimes=seed.user_regimes,
        ages=AgeGrid(exact_values=(0, 10, 20)),
        edges={"alive": {"done": (10,)}},
        initial_nodes=InitialNodes(by_age={10: "alive"}),
        regime_id_class=original.RegimeId,
    )
    assert model._regimes["alive"].active_periods == (1,)
    assert model._regimes["done"].active_periods == (2,)
    return model


def _assert_parent_query_authority(
    *, model: Model, plan: ContinuationPlan
) -> ContinuationPlan:
    """Qualify the parent reader's exact finite native-liquid child authority."""
    assert plan.query_targets == ("alive",)
    read = plan.child_reads["alive"]
    query = read.query
    assert isinstance(query, NBEGMQuery)
    authority = model._regimes["alive"].solution.query_authority
    assert query.model.authority is authority
    assert authority is not None
    assert authority.coordinate_kind == "liquid"
    assert authority.outer_mode == "finite_max"
    assert authority.outer_capacity == 4
    assert authority.finite_outer is not None
    assert authority.finite_outer.outer_nodes == (0.0, 1.0, 2.0)
    assert authority.finite_outer.outer_state == "illiquid"
    assert authority.finite_outer.outer_post_decision == "new_illiquid"
    assert authority.row_topology is not None
    assert authority.row_topology.kind == "nb_budget_interval_run"
    assert authority.state_names == ("illiquid",)
    assert authority.action_names == ()
    assert authority.target_periods == (1,)
    assert read.n_outer_candidates == 4
    assert read.passive_state_names == ("illiquid",)
    np.testing.assert_array_equal(read.passive_grids[0], [0.0, 1.0, 2.0])
    return plan


def _assert_finite_inner_role_authorities(*, kernel: _NNBEGMPeriodKernel) -> None:
    """Qualify the actual keeper and adjuster canonical source pools."""
    model_authority = kernel.query_model
    assert model_authority is not None
    roles = model_authority.finite_inner_models
    assert roles is not None
    assert tuple(roles) == ("keeper", "adjuster")
    for role in ("keeper", "adjuster"):
        derived = roles[role]
        assert derived.authority.outer_mode == "none"
        assert derived.authority.coordinate_kind == "liquid"
        assert derived.authority.interval_source_identity == ()
        assert derived.intervals.source_identity == ()
        assert derived.intervals.equal_above == ()
        assert derived.intervals_by_period[1].source_identity == ()
        assert derived.functions_by_period is not None
        pool = derived.functions_by_period[1]
        assert ("new_illiquid" in pool) == (role == "keeper")


def _assert_finite_query_publication(
    *, model: Model, child: EGMCarry, outputs: tuple[KernelOutput, ...]
) -> None:
    """Keep all published query leaves, templates and transported identities exact."""
    data = child.query_data_for_transport()
    assert data is not None
    assert data.liquid_corner_data is None
    assert data.corner_continuation is not None
    np.testing.assert_array_equal(data.floor_used, [0.0, 0.25, 0.5])
    np.testing.assert_array_equal(
        data.corner_continuation,
        np.asarray(
            [[0.0, 0.0, 1.0, 2.0], [1.25, 0.25, 1.25, 2.25], [2.5, 0.5, 1.5, 2.5]]
        )[..., None],
    )
    np.testing.assert_array_equal(data.corner_interval_id, np.zeros((3, 4, 1)))
    np.testing.assert_array_equal(data.corner_side_id, np.zeros((3, 4, 1)))
    assert data.savings.shape == (3, 4, 8)
    assert data.link_valid.shape == (3, 4, 7)
    for index, output in enumerate(outputs):
        carry = output.continuations[EGM_CONTINUATION]
        assert isinstance(carry, EGMCarry)
        native = carry.query_data_for_transport()
        assert native is not None
        assert data.leaves().keys() == native.leaves().keys()
        for path, source in native.leaves().items():
            actual = data.leaves()[path]
            if path == ("floor_used",):
                np.testing.assert_array_equal(actual, source)
            else:
                np.testing.assert_array_equal(actual[:, index], source)
    solution = model._regimes["alive"].solution
    template = solution.continuation_template
    assert isinstance(template, EGMCarry)
    assert template.leaves().keys() == child.leaves().keys()
    for path, leaf in child.leaves().items():
        assert template.leaves()[path].shape == leaf.shape
        assert template.leaves()[path].dtype == leaf.dtype
    leaves, tree = jax.tree.flatten(child)
    restored = jax.tree.unflatten(tree, leaves)
    assert isinstance(restored, EGMCarry)
    assert restored.leaves().keys() == child.leaves().keys()
    for path, leaf in child.leaves().items():
        assert restored.leaves()[path] is leaf


def _assert_finite_replay_output_roles(
    *, kernel: _NNBEGMPeriodKernel, outputs: tuple[KernelOutput, ...]
) -> None:
    """Match actual inner publication roles to retained finite candidate leaves."""
    for role, inner, output in zip(
        ("keeper", "adjuster"),
        (kernel.keeper_kernel, kernel.adjuster_kernel),
        (outputs[0], outputs[1]),
        strict=True,
    ):
        assert inner.retain_nested_feasibility is True
        programs = inner.core_programs()
        assert "replay" in programs
        assert f"{role}:replay" in kernel.core_programs()
        carry = output.continuations[EGM_CONTINUATION]
        assert isinstance(carry, EGMCarry)
        for program in programs.values():
            bounded = program.output_roles
            assert isinstance(bounded, BoundedResult)
            role_carry = bounded.output[0][1]
            assert isinstance(role_carry, EGMCarry)
            role_leaves = {
                tuple(key.name for key in path): leaf
                for path, leaf in jax.tree_util.tree_flatten_with_path(role_carry)[0]
            }
            assert role_leaves.keys() == carry.leaves().keys()
            for path, leaf in carry.leaves().items():
                declared = role_leaves[path]
                assert isinstance(declared, StateAxesLeading)
                assert declared.state_names == (
                    () if path == ("taste_shock_scale",) else ("illiquid",)
                )
                assert declared.n_free_leading_axes == 0
                if declared.shape is not None:
                    assert declared.shape == leaf.shape
                if declared.dtype is not None:
                    assert declared.dtype == leaf.dtype


def test_parent_reads_off_grid_finite_nb_value_and_associated_marginal() -> None:
    """Fold actual finite winners before blending their passive-node tuples."""
    model = _model()
    kernel, context = ride_along_kernel(
        model=model, params=_PARAMS, regime_name="parent", period=0
    )
    assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
    assert kernel.statics.ride_names == ("illiquid",)
    assert kernel.source_period == 0
    assert model._regimes["parent"].active_periods == (0,)
    program = kernel.core_programs()["main"]
    plan = _assert_parent_query_authority(model=model, plan=kernel.continuation_plan)
    arguments = materialize_core_program(
        program=program, context=CoreBuildContext(**context)
    ).arguments
    carries = arguments["next_regime_to_continuation"]
    assert isinstance(carries, Mapping)
    child = carries["alive"]
    assert isinstance(child, EGMCarry)
    assert child.query_data_for_transport() is not None
    pool: EconFunctionKwargs = {}
    for name, value in arguments.items():
        if name in {"next_regime_to_continuation", "liquid", "illiquid"}:
            continue
        assert isinstance(value, jax.Array | float | int)
        pool[name] = value
    passive = arguments["illiquid"]
    assert isinstance(passive, jax.Array)
    np.testing.assert_array_equal(passive, [0.0, 1.0, 2.0])
    pool["illiquid"] = passive[0]
    np.testing.assert_array_equal(_landing_stock(illiquid=passive[0]), 0.5)
    continuation = bind_continuation(
        plan=plan,
        combo_pool=pool,
        next_regime_to_continuation=MappingProxyType({"alive": child}),
        dtype=child.value.dtype,
    )
    # The ordinary source rows use the owned Hermite value and their separately
    # linear paired marginal. The three explicit midpoint formulae include both.
    expected = np.asarray(
        [
            [np.log(9.0 / 4.0) / 2.0 + np.log(5.0) / 4.0 + 1.0 / 320.0, 161.0 / 90.0],
            [
                np.log(11.0 / 4.0) / 2.0
                + (np.log(3.0) + np.log(4.0)) / 4.0
                + 1.0 / 192.0,
                125.0 / 66.0,
            ],
            [
                np.log(13.0 / 4.0) / 2.0
                + (np.log(4.0) + np.log(5.0)) / 4.0
                + 1.0 / 320.0,
                197.0 / 130.0,
            ],
        ]
    )
    pairs = []
    for saving in (0.0, 0.125, 0.25):
        result = continuation(jnp.asarray(saving))
        assert isinstance(result, BoundedResult)
        assert result.bound_status == ()
        assert result.query_status is not None
        assert len(result.query_status) == 1
        np.testing.assert_array_equal(result.query_status[0], [False, False, False])
        pairs.append(jnp.stack(result.output))
    actual = jnp.stack(pairs)
    np.testing.assert_array_almost_equal(actual, expected, decimal=DECIMAL_PRECISION)


def test_finite_nb_publication_transports_all_query_leaves_and_replay(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish named finite rows, held intervals and keeper-first replay together."""
    model = _create_child_model()
    published: list[KernelOutput] = []
    consume = backward_induction.consume_kernel_output

    def observe(
        *, output: KernelOutput, **kwargs: Unpack[original._ConsumeArgs]
    ) -> ConsumedKernelOutput:
        if kwargs["regime_name"] == "alive":
            published.append(output)
        return consume(output=output, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(backward_induction, "consume_kernel_output", observe)
        kernel, context = ride_along_kernel(
            model=model,
            params={"alive": _PARAMS["alive"]},
            regime_name="alive",
            period=1,
        )
    assert len(published) == 1
    assert isinstance(kernel, _NNBEGMPeriodKernel)
    _assert_finite_inner_role_authorities(kernel=kernel)
    keeper = original._inner(kernel=kernel.keeper_kernel, context=context)
    assert isinstance(keeper, _NBEGMNestedKernelOutput)
    assert keeper.prepared_source_floor is not None
    programs = select_programs(
        graph=core_program_graph(kernel=kernel.adjuster_kernel), retain_replay=True
    )
    adjuster_cores = {
        name: partial(
            program.function,
            **{axis.width_keyword: axis.extent for axis in program.requirements.axes},
        )
        for name, program in programs.items()
    }
    outputs = (
        keeper,
        *(
            kernel.adjuster_kernel(
                compiled_cores=adjuster_cores,
                state_action_space=context["state_action_space"],
                next_regime_to_V_arr=context["next_regime_to_V_arr"],
                next_regime_to_continuation=context["next_regime_to_continuation"],
                flat_params=_with_outer_post_decision(
                    flat_params=context["flat_params"],
                    regime_name="alive",
                    outer_post_decision="new_illiquid",
                    value=jnp.asarray(node),
                ),
                period=context["period"],
                ages=context["ages"],
                logger=original._LOGGER,
                prepared_source_floor=keeper.prepared_source_floor,
            )
            for node in (0.0, 1.0, 2.0)
        ),
    )
    original._assert_published_bank(
        published=published[0], outputs=outputs, family="nnbegm"
    )
    original._assert_candidate_economics(outputs=outputs, dynamic=True)
    child = published[0].continuations[EGM_CONTINUATION]
    assert isinstance(child, EGMCarry)
    _assert_finite_query_publication(model=model, child=child, outputs=outputs)
    _assert_finite_replay_output_roles(kernel=kernel, outputs=outputs)
    policy = published[0].replay[SIMULATION_POLICY]
    assert isinstance(policy, NNBEGMSimPolicy)
    assert policy.state_names == ("liquid", "illiquid")
    assert policy.n_keeper_candidates == 1
    assert policy.discrete_action_names == ()
    assert policy.candidate_discrete_actions is None
    np.testing.assert_array_equal(policy.outer_grid_values, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        policy.candidate_value,
        np.stack([np.asarray(output.value) for output in outputs]),
    )
    assert np.isnan(policy.candidate_inner_action[3, 0, 0])
    assert np.isneginf(policy.candidate_value[3, 0, 0])
    assert np.isnan(policy.candidate_outer_target[3, 0, 0])
    np.testing.assert_array_equal(
        policy.candidate_outer_target[:, 0, :2],
        [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [np.nan, 2.0]],
    )
    at_zero = np.asarray(policy.candidate_value)[:, 0, :2]
    np.testing.assert_array_equal(at_zero[0, 1], at_zero[3, 1])
    np.testing.assert_array_equal(np.argmax(at_zero, axis=0), [2, 0])
    np.testing.assert_array_almost_equal(
        np.max(at_zero, axis=0), np.log([1.5, 2.0]), decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        policy.candidate_inner_action[[0, 3], 0, 1],
        [1.0, 0.5],
        decimal=DECIMAL_PRECISION,
    )


def test_finite_nb_checked_programs_count_terminal_reads_by_producer() -> None:
    """Checked reads of a literal terminal child count against its state grid."""
    model = _create_child_model()
    kernel, context = ride_along_kernel(
        model=model,
        params={"alive": _PARAMS["alive"]},
        regime_name="alive",
        period=1,
    )
    assert isinstance(kernel, _NNBEGMPeriodKernel)
    # The actual child continues only to the ordinary terminal producer. Its
    # installed borrowing floor does not turn that terminal into a query target.
    for inner in (kernel.keeper_kernel, kernel.adjuster_kernel):
        assert inner.continuation_plan.query_targets == ()
        for program in core_program_graph(kernel=inner).values():
            assert program.checked_variant is not None
    graph = core_program_graph(kernel=kernel)
    outputs: list[KernelOutput] = []
    for collect in (False, True):
        programs = select_programs(
            graph=graph,
            retain_replay=True,
            collect_continuation_read_counts=collect,
        )
        outputs.append(
            kernel(
                compiled_cores={
                    name: program.function for name, program in programs.items()
                },
                **context,
                logger=original._LOGGER,
            )
        )
    ordinary, checked = outputs
    np.testing.assert_array_equal(ordinary.value, checked.value)
    ordinary_carry = ordinary.continuations[EGM_CONTINUATION]
    checked_carry = checked.continuations[EGM_CONTINUATION]
    assert isinstance(ordinary_carry, EGMCarry)
    assert isinstance(checked_carry, EGMCarry)
    assert ordinary_carry.leaves().keys() == checked_carry.leaves().keys()
    for path, leaf in ordinary_carry.leaves().items():
        np.testing.assert_array_equal(leaf, checked_carry.leaves()[path])
    ordinary_counts = (
        ordinary.continuation_read_counts
        if isinstance(
            ordinary, FeasibilityKernelOutput | ContinuationCountedKernelOutput
        )
        else None
    )
    assert ordinary_counts is None
    counts = (
        checked.continuation_read_counts
        if isinstance(
            checked, FeasibilityKernelOutput | ContinuationCountedKernelOutput
        )
        else None
    )
    assert isinstance(counts, ContinuationReadCounts)
    # The terminal publishes its value on its own state grid, so every carried
    # node is support. Each parent illiquid node (3) reads it once per savings
    # node (6) for each of the keeper and the three outer adjuster candidates
    # (4); the landing stock is an illiquid node, so one row is active per read.
    # No saving lies below the terminal's first liquid node, 0.0.
    n_illiquid, n_savings, n_candidates = 3, 6, 4
    np.testing.assert_array_equal(counts, [0, n_illiquid * n_savings * n_candidates])
    for count in counts:
        assert count.shape == ()
        assert count.dtype == jnp.int32
    for role, inner in (
        ("keeper", kernel.keeper_kernel),
        ("adjuster", kernel.adjuster_kernel),
    ):
        for name in core_program_graph(kernel=inner):
            assert graph[f"{role}:{name}"].checked_variant is not None
