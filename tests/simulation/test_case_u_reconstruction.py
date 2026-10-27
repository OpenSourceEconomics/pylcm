"""Genuine installed-U solve results reconstruct original realised-node actions."""

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_case_u import CaseUPointSolver
from _lcm.egm.current_iid_case_u_replay import (
    CASE_U_FUTURE_REPLAY,
    CaseUReplayChildren,
    CaseUReplayInput,
)
from _lcm.simulation.chunk_admission import prepare_simulation_chunks
from _lcm.simulation.compile import bind_simulation_runtime
from _lcm.simulation.current_iid_overrides import CurrentIIDOverrides
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.simulate import simulate
from _lcm.utils.logging import get_logger
from lcm import (
    ConsumptionSavingsRegime,
    ExecutionConfig,
    InitialNodes,
    Model,
    categorical,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidSimulationInputError,
    InvalidValueFunctionError,
)
from lcm.persistence import load_solution
from lcm.shocks import BaseInnovations, PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import ArtifactRef, ArtifactStore
from lcm.solvers import DCEGM
from lcm.typing import ContinuousAction, FloatND, ScalarInt, UserParams
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations
from tests.egm.test_current_iid_parent_demands import _resources
from tests.simulation._callback_types import ChunkPreparation

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class _ReplayRegime:
    child: ScalarInt
    done: ScalarInt


def _utility(*, consumption: ContinuousAction, eps: FloatND) -> FloatND:
    return (1.0 + eps) * jnp.log(consumption)


def _inverse(*, marginal_continuation: FloatND, eps: FloatND) -> FloatND:
    return (1.0 + eps) / marginal_continuation


def _params() -> UserParams:
    return {"discount_factor": 1.0, "child": {"next_wealth": {"slope": 1.0}}}


def _model(*, subject_width: int = 1) -> Model:
    declarations = _actual_declarations()
    original = declarations.regimes["child"]
    assert isinstance(original, ConsumptionSavingsRegime)
    assert isinstance(original.solver, DCEGM)
    child = replace(
        original,
        solver=replace(original.solver, current_iid_projection_capacity=8),
        functions={
            **original.functions,
            "utility": _utility,
            "inverse_marginal_utility": _inverse,
            "resources": _resources,
        },
    )
    declarations = replace(
        declarations,
        regimes={"child": child, "done": declarations.regimes["done"]},
        initial_nodes=InitialNodes(by_age={1: "child"}),
        edges={"child": {"done": (1,)}},
        regime_id_class=_ReplayRegime,
        execution_config=ExecutionConfig(
            device_memory_bytes=None, axis_widths={"subject": subject_width}
        ),
        shocks={
            "income": ShockGroup(
                shocks={"eps": Uniform(start=0.0, stop=1.0)},
                nodes=UserNodes(
                    nodes=np.array([[0.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )
    return Model._from_call_owned_params(declarations=declarations, params=_params())


def _initial(*, model: Model) -> dict[str, jax.Array]:
    return {
        "wealth": jnp.asarray([1.0, 12.5, 1.5, 1.75]),
        "age": jnp.ones((4,)),
        "regime_id": jnp.full((4,), model.regime_names_to_ids["child"], jnp.int32),
    }


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("kind", ["physical", "base"])
def test_model_u_reconstructs_original_candidates_at_nodes_and_offgrid(
    kind: str,
) -> None:
    """No projected policy is read; off-node frames alter the original Euler bank."""
    model = _model()
    solution = model.solve(params=_params(), log_level="off")
    frame = np.asarray([0.0, 0.0, 1.0, 0.5])
    initial = _initial(model=model)
    tagged = (
        PhysicalValues(values=frame)
        if kind == "physical"
        else BaseInnovations(values=frame)
    )
    result = model.simulate(
        params=_params(),
        initial_conditions={**initial, "eps": tagged},
        solution=solution,
        log_level="off",
        seed=17,
    )
    record = result.raw_results["child"][1]
    np.testing.assert_array_equal(record.shocks["eps"], frame)
    # alpha*log(c)+log(m-c): each Euler row has m=(1+alpha)*s
    # and c=alpha*s. The original selector interpolates policy linearly, so this
    # policy formula is exact also between those same original row nodes.
    alpha = 1 + frame
    expected = np.asarray(initial["wealth"]) * alpha / (1 + alpha)
    np.testing.assert_array_almost_equal(
        record.actions["consumption"], expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["wealth"],
        np.asarray(initial["wealth"]) - expected,
        decimal=DECIMAL_PRECISION,
    )
    payload = solution.replay_artifacts[
        ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    ]
    assert isinstance(payload, CaseUReplayChildren)
    flat, generation = model._prepare_call(params=_params())
    regime = generation.execution._regimes["child"]
    spec = regime.simulation.case_u_replay_spec
    assert spec is not None
    declaration = spec.periods[1]
    from _lcm.solution.continuation_target import (  # noqa: PLC0415
        union_fixed_params,
        union_free_params,
    )

    pool = {
        **union_fixed_params(
            fixed_flat_params=declaration.fixed_flat_params,
            regime_name="child",
            transition_target_names=declaration.transition_target_names,
        ),
        **union_free_params(
            flat_params=flat,
            regime_name="child",
            transition_target_names=declaration.transition_target_names,
        ),
        "period": jnp.int32(1),
        "age": jnp.asarray(1.0),
    }
    assert declaration.pieces is not None
    original = CaseUPointSolver(
        pieces=declaration.pieces,
        pool=pool,
        children=MappingProxyType(dict(payload.children)),
        savings_point_width=declaration.savings_point_width,
        stochastic_node_width=declaration.stochastic_node_width,
        envelope_cell_width=declaration.envelope_cell_width,
    )
    for index in range(4):
        selected = original.node(
            liquid=initial["wealth"][index],
            physical={"eps": record.shocks["eps"][index]},
            physical_tangent={"eps": jnp.asarray(0.0)},
        )
        assert bool(selected.output.valid & selected.output.feasible)
        assert selected.query_status is not None
        assert not any(
            bool(jnp.any(jnp.stack(status))) for status in selected.query_status
        )
        if index < 3:
            np.testing.assert_array_equal(
                selected.output.policy, record.actions["consumption"][index]
            )
        else:
            np.testing.assert_almost_equal(
                selected.output.policy,
                record.actions["consumption"][index],
                decimal=DECIMAL_PRECISION,
            )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_u_unsupplied_frames_preserve_original_actor_addresses_across_tiles() -> None:
    outputs = []
    for width in (1, 3):
        model = _model(subject_width=width)
        initial = _initial(model=model)
        result = model.simulate(
            params=_params(),
            initial_conditions=initial,
            log_level="off",
            seed=19,
        )
        record = result.raw_results["child"][1]
        physical = np.asarray(record.shocks["eps"])
        expected = np.asarray(initial["wealth"]) * (1 + physical) / (2 + physical)
        np.testing.assert_array_almost_equal(
            record.actions["consumption"], expected, decimal=DECIMAL_PRECISION
        )
        np.testing.assert_array_almost_equal(
            result.raw_results["done"][2].states["wealth"],
            np.asarray(initial["wealth"]) - expected,
            decimal=DECIMAL_PRECISION,
        )
        outputs.append(record)
    np.testing.assert_array_equal(outputs[0].shocks["eps"], outputs[1].shocks["eps"])
    # Compiler topologies preserve action ownership, with numerical tolerances:
    # docs/development/iid-exits-redesign.md, simulation reproducibility contract.
    np.testing.assert_array_almost_equal(
        outputs[0].actions["consumption"],
        outputs[1].actions["consumption"],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_u_missing_or_wrong_date_future_artifacts_refuse_before_forward() -> None:
    model = _model()
    solution = model.solve(params=_params(), log_level="off")
    ref = ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    payload = solution.replay_artifacts[ref]
    assert isinstance(payload, CaseUReplayChildren)
    flat, generation = model._prepare_call(params=_params())
    del flat
    spec = generation.execution._regimes["child"].simulation.case_u_replay_spec
    assert spec is not None
    with pytest.raises(InvalidSimulationInputError, match="producer/date/key"):
        CaseUReplayInput(payload=replace(payload, period=2)).require(
            spec=spec, period=1
        )
    # A public result boundary must account for the required replay key.
    missing = replace(solution, replay_artifacts=ArtifactStore())
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=_params(),
            initial_conditions=_initial(model=model),
            solution=missing,
            log_level="off",
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_u_corrupt_future_economic_marginal_cannot_replay_an_action() -> None:
    model = _model()
    solution = model.solve(params=_params(), log_level="off")
    ref = ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    payload = solution.replay_artifacts[ref]
    assert isinstance(payload, CaseUReplayChildren)
    child = payload.children["done"]
    corrupted = replace(
        child, marginal_utility=jnp.full_like(child.marginal_utility, jnp.nan)
    )
    assert payload.child_names == ("done",)
    bad = replace(payload, child_values=(corrupted,))
    entries = {**dict(solution.replay_artifacts), ref: bad}
    invalid_solution = replace(solution, replay_artifacts=ArtifactStore(entries))
    with pytest.raises(
        (ExecutionPlanningError, InvalidSimulationInputError, InvalidValueFunctionError)
    ):
        model.simulate(
            params=_params(),
            initial_conditions={
                **_initial(model=model),
                "eps": PhysicalValues(values=np.asarray([0.0, 0.0, 1.0, 0.5])),
            },
            solution=invalid_solution,
            log_level="off",
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("budget", [1, 1 << 28])
def test_actual_u_forward_subphase_admits_complete_program_or_refuses_budget(
    *, budget: int
) -> None:
    """Budget the real Model-built forward program, separately from cold building."""
    model = _model()
    solution = model.solve(params=_params(), log_level="off")
    flat, generation = model._prepare_call(params=_params())
    actual_model = generation.execution
    ref = ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    payload = solution.replay_artifacts[ref]
    assert isinstance(payload, CaseUReplayChildren)
    replay = CaseUReplayInput(payload=payload)
    spec = actual_model._regimes["child"].simulation.case_u_replay_spec
    assert spec is not None
    replay.require(spec=spec, period=1)
    # Rebind only the genuine forward subphase's existing execution owner;
    # the separate cold Model allocation frontier is not granted admission.
    execution = replace(
        actual_model._execution,
        device_memory_bytes=budget,
        requested_device_memory_bytes=budget,
        budget_source="explicit",
    )
    regimes = bind_simulation_runtime(
        regimes=actual_model._regimes, execution=execution, enable_jit=True
    )
    initial = _initial(model=model)
    numeric_values = {}
    for period, cells in solution.values.items():
        arrays = {}
        for name, value in cells.items():
            assert isinstance(value, jax.Array)
            arrays[name] = value
        numeric_values[period] = MappingProxyType(arrays)
    values = MappingProxyType(numeric_values)
    inputs = {1: {"child": replay}}
    overrides = CurrentIIDOverrides(
        entries={"eps": PhysicalValues(values=np.asarray([0.0, 0.0, 1.0, 0.5]))},
        paths={},
        starts=np.ones((4,), dtype=np.int32),
    )
    footprint = measure_buffer_footprint(tree=(values, payload, initial, flat))
    arguments: ChunkPreparation = {
        "regimes": regimes,
        "flat_params": flat,
        "values": values,
        "flags": {},
        "ages": actual_model._time,
        "initial_conditions": initial,
        "regime_names_to_ids": model.regime_names_to_ids,
        "original_population": 4,
        "retained_footprint": footprint,
        "independent_taste": False,
        "log_level": "off",
        "current_iid_replay_inputs": inputs,
        "current_iid_override_schema": overrides.schema,
        "max_compilation_workers": 1,
    }
    if budget == 1:
        with pytest.raises(ExecutionPlanningError):
            prepare_simulation_chunks(**arguments)
        return
    prepared = prepare_simulation_chunks(**arguments)
    result = simulate(
        regimes=regimes,
        flat_params=flat,
        initial_conditions=initial,
        regime_names_to_ids=model.regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=values,
        ages=actual_model._time,
        simulation_output_dtypes=actual_model.simulation_output_dtypes,
        seed=17,
        current_iid_replay_inputs=inputs,
        current_iid_overrides=overrides,
        retained_footprint=footprint,
        prepared_chunks=prepared,
        subject_batch_size=prepared.plan.profile.n_subjects,
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["child"][1].actions["consumption"],
        [0.5, 6.25, 1.0, 1.05],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_u_durable_solution_reconstructs_original_candidates_in_a_fresh_model(
    *, tmp_path: Path
) -> None:
    """Archive actual future inputs and authenticate them against a fresh producer."""
    model = _model()
    params = _params()
    solution = model.solve(params=params, log_level="off")
    ref = ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    original = solution.replay_artifacts[ref]
    assert isinstance(original, CaseUReplayChildren)
    path = tmp_path / "case-u.lcm"
    solution.save(path=path)
    loaded = load_solution(path=path)
    assert loaded._engine_view is None
    fresh_model = _model()
    physical = np.asarray([0.0, 0.0, 1.0, 0.5])
    initial = {
        **_initial(model=fresh_model),
        "eps": PhysicalValues(values=physical),
    }
    outputs = []
    for producer, result in ((model, solution), (fresh_model, loaded)):
        outputs.append(
            producer.simulate(
                params=params,
                solution=result,
                initial_conditions=initial,
                seed=17,
                log_level="off",
            ).raw_results
        )
    restored = loaded.replay_artifacts[ref]
    assert isinstance(restored, CaseUReplayChildren)
    assert restored.source == original.source
    assert restored.period == original.period
    assert jax.tree.structure(restored) == jax.tree.structure(original)
    for expected, actual in zip(
        jax.tree.leaves(original), jax.tree.leaves(restored), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    expected_policy = np.asarray(initial["wealth"]) * (1 + physical) / (2 + physical)
    np.testing.assert_array_almost_equal(
        outputs[1]["child"][1].actions["consumption"],
        expected_policy,
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(
        outputs[1]["child"][1].actions["consumption"],
        outputs[0]["child"][1].actions["consumption"],
    )
    np.testing.assert_array_equal(
        outputs[1]["child"][1].shocks["eps"],
        outputs[0]["child"][1].shocks["eps"],
    )
    np.testing.assert_array_equal(
        outputs[1]["done"][2].states["wealth"],
        outputs[0]["done"][2].states["wealth"],
    )
