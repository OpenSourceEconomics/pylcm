import dataclasses
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import jax
import numpy as np
import pandas as pd
import pytest
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from _lcm.execution.value_transfer import (
    ValueArtifactKind,
    ValueInputChannel,
    ValueTransferKind,
)
from _lcm.grids import categorical
from _lcm.grids.continuous import LinSpacedGrid
from _lcm.grids.discrete import DiscreteGrid
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.solution import backward_induction
from _lcm.solution.v_topology import (
    _build_zero_V_arr,
    _get_regime_V_shapes_and_shardings,
)
from _lcm.utils.logging import v_array_has_inf, v_array_has_nan
from lcm import CollectiveUtility, LinearAggregator, LinearExpectation, fixed_transition
from lcm.ages import AgeGrid
from lcm.exceptions import PyLCMError, RegimeInitializationError
from lcm.model import Model
from lcm.regime import Regime as UserRegime
from lcm.result import SimulationResult
from lcm.typing import ScalarInt

# Run these tests on a four-CPU-device topology. The pin only applies in a
# process whose JAX backends are not yet initialized (a serial run importing
# this module early); otherwise the tests skip. The device-count update is
# attempted FIRST because it is the one that raises after initialization —
# this keeps the pin atomic. The reverse order would flip the default
# platform to CPU (that update succeeds at any time) and then skip, leaving
# every later model build in the process compiled for CPU while arrays from
# earlier accelerator computations stay committed to their device.
try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _PYTEST_PARALLEL = False
except RuntimeError:
    _PYTEST_PARALLEL = True

_skip_pytest_parallel = pytest.mark.skipif(
    _PYTEST_PARALLEL, reason="Can't set num cpus in pytest paralellel"
)


def test_importing_this_module_after_jax_init_leaves_the_platform_unpinned():
    """The CPU-topology pin is atomic: all of it applies, or none of it.

    Once any JAX backend is initialized, this module's four-CPU-device pin
    can no longer apply and its tests skip. The platform default must then
    stay untouched: a partially applied pin (platform flipped to CPU, device
    count unchanged) would silently retarget every model built later in the
    process onto the CPU, while arrays produced by earlier accelerator
    computations stay committed to their device — a sharding mismatch at the
    first compiled call that mixes them.
    """
    code = (
        "import jax; jax.numpy.zeros(1); "
        "import tests.test_distributed; "
        "print(repr(jax.config.read('jax_platform_name')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "''"


def _make_correct_distributed_model(
    *,
    n_subjects: int | None = None,
    distributed: bool = True,
    distribute_type2: bool | None = None,
    retirement_reads_type2: bool = True,
) -> Model:
    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt

    if distribute_type2 is False:

        @categorical(ordered=True)
        class Type:
            lowest: ScalarInt
            low: ScalarInt
            high: ScalarInt
            highest: ScalarInt

    else:

        @categorical(ordered=True)
        class Type:
            low: ScalarInt
            high: ScalarInt

    working_life = UserRegime(
        functions={
            "utility": lambda wealth, consumption, type1, type2: (
                (jnp.log(consumption) + wealth * 0.001) * type1 * type2
            ),
        },
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=100,
                n_points=10,
            ),
        },
        state_transitions={
            "wealth": lambda wealth, consumption: wealth - consumption,
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(
            age >= 4, RegimeId.retirement, RegimeId.working_life
        ),
        active=lambda age: age < 5,
    )

    def retirement_utility_with_type2(*, wealth, type1, type2):
        return (wealth * 0.5) * type1 * type2

    def retirement_utility_without_type2(*, wealth, type1):
        return (wealth * 0.5) * type1

    retirement_utility = (
        retirement_utility_with_type2
        if retirement_reads_type2
        else retirement_utility_without_type2
    )

    retirement = UserRegime(
        transition=None,
        functions={"utility": retirement_utility},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
        active=lambda age: age >= 5,
    )

    return Model(
        regimes={"working_life": working_life, "retirement": retirement},
        ages=AgeGrid(start=0, stop=5, step="Y"),
        regime_id_class=RegimeId,
        states={
            "type1": DiscreteGrid(category_class=Type, distributed=distributed),
            "type2": DiscreteGrid(
                category_class=Type,
                distributed=(
                    distributed if distribute_type2 is None else distribute_type2
                ),
            ),
        },
        state_transitions={
            "type1": fixed_transition("type1"),
            "type2": fixed_transition("type2"),
        },
        n_subjects=n_subjects,
    )


def _make_one_axis_collective_model(*, distributed: bool) -> Model:
    """Tiny collective GridSearch model with one four-device fixed-state axis."""

    @categorical(ordered=False)
    class RegimeId:
        working: ScalarInt
        retired: ScalarInt

    @categorical(ordered=True)
    class Type:
        lowest: ScalarInt
        low: ScalarInt
        high: ScalarInt
        highest: ScalarInt

    def utility_f(*, wealth, consumption, type1):
        return jnp.log(consumption) + 0.01 * wealth + 0.1 * type1

    def utility_m(*, wealth, consumption, type1):
        return 1.1 * jnp.log(consumption) + 0.02 * wealth + 0.2 * type1

    def terminal_f(*, wealth, type1):
        return 0.5 * wealth + 0.1 * type1

    def terminal_m(*, wealth, type1):
        return 0.4 * wealth + 0.2 * type1

    def next_regime(age):
        del age
        return RegimeId.retired

    return Model(
        regimes={
            "working": UserRegime(
                transition=next_regime,
                active=lambda age: age < 1,
                states={"wealth": LinSpacedGrid(start=1, stop=4, n_points=4)},
                state_transitions={
                    "wealth": lambda wealth, consumption: wealth - consumption
                },
                actions={"consumption": LinSpacedGrid(start=0.5, stop=1, n_points=2)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": utility_f, "m": utility_m}
                    )
                },
            ),
            "retired": UserRegime(
                transition=None,
                active=lambda age: age >= 1,
                states={"wealth": LinSpacedGrid(start=1, stop=4, n_points=4)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": terminal_f, "m": terminal_m}
                    )
                },
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
        states={"type1": DiscreteGrid(category_class=Type, distributed=distributed)},
        state_transitions={"type1": fixed_transition("type1")},
    )


@pytest.fixture
def correct_distributed_model():
    return _make_correct_distributed_model()


@pytest.fixture
def wrong_distributed_model():
    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt

    @categorical(ordered=True)
    class Type:
        low: ScalarInt
        medium: ScalarInt
        high: ScalarInt

    working_life = UserRegime(
        functions={
            "utility": lambda wealth, consumption, type1, type2: (
                (jnp.log(consumption) + wealth * 0.001) * type1 * type2
            ),
        },
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=100,
                n_points=10,
            ),
        },
        state_transitions={
            "wealth": lambda wealth, consumption: wealth - consumption,
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(
            age >= 4, RegimeId.retirement, RegimeId.working_life
        ),
        active=lambda age: age < 5,
    )

    retirement = UserRegime(
        transition=None,
        functions={
            "utility": lambda wealth, type1, type2: (wealth * 0.5) * type1 * type2
        },
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
        active=lambda age: age >= 5,
    )

    return Model(
        regimes={"working_life": working_life, "retirement": retirement},
        ages=AgeGrid(start=0, stop=5, step="Y"),
        regime_id_class=RegimeId,
        states={
            "type1": DiscreteGrid(category_class=Type, distributed=True),
            "type2": DiscreteGrid(category_class=Type, distributed=True),
        },
        state_transitions={
            "type1": fixed_transition("type1"),
            "type2": fixed_transition("type2"),
        },
    )


@_skip_pytest_parallel
def test_solution_running_on_multiple_cpus(correct_distributed_model):
    """Test that distribution over multiple CPU's works for solution."""

    period_to_regime_to_V_arr = correct_distributed_model.solve(
        log_level="debug",
        params={"discount_factor": 0.95},
    )

    assert period_to_regime_to_V_arr[0]["working_life"].sharding.num_devices == 4


@_skip_pytest_parallel
def test_distributed_solve_matches_single_device_per_type():
    """A sharded solve yields the same value function as the single-device solve.

    Distributing the permanent type axes across devices changes only where the
    continuation-value interpolation runs, never its result: every (type1, type2)
    slice of the sharded V-array is identical to the same slice solved on one
    device. This pins the per-type-local read of the continuation V — a wrong
    local index would corrupt some types' values.
    """
    params = {"discount_factor": 0.95}
    sharded = _make_correct_distributed_model(distributed=True).solve(
        log_level="debug", params=params
    )
    single = _make_correct_distributed_model(distributed=False).solve(
        log_level="debug", params=params
    )

    for period, regime_to_V_arr in sharded.items():
        for regime_name, V_arr in regime_to_V_arr.items():
            np.testing.assert_array_equal(
                np.asarray(V_arr), np.asarray(single[period][regime_name])
            )


def _compiled_solve_kernel_hlo(*, model: Model, regime_name: str, period: int) -> str:
    """Lower and compile a regime's period kernel exactly as backward induction does.

    Reproduces the AOT lowering args (sharded states, sharded continuation-V
    template, flat params) so the optimized HLO reflects the real solve, then
    returns its text for collective inspection.
    """
    flat_params = model._process_params({"discount_factor": 0.95})
    regimes = model._regimes
    topology = _get_regime_V_shapes_and_shardings(
        regimes=regimes, flat_params=flat_params
    )
    next_regime_to_V_arr = MappingProxyType(
        {name: _build_zero_V_arr(topology=topo) for name, topo in topology.items()}
    )
    regime = regimes[regime_name]
    period_kernel = regime.solution.period_kernels[period]
    # Lower every shared core the period kernel carries exactly as backward
    # induction does (a brute regime carries the single `"main"` core); the
    # continuation V enters through `build_lower_args`, so the optimized HLO
    # reflects the real sharded read.
    texts: list[str] = []
    for core_key, core in period_kernel.cores().items():
        lower_args = period_kernel.build_lower_args(
            core_key=core_key,
            state_action_space=regime.solution.state_action_space(
                regime_params=flat_params[regime_name]
            ),
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=MappingProxyType({}),
            flat_params=flat_params,
            period=period,
            ages=model.ages,
        )
        text = jax.jit(core).lower(**lower_args).compile().as_text()
        assert text is not None
        texts.append(text)
    hlo = "\n".join(texts)
    assert hlo
    return hlo


@_skip_pytest_parallel
def test_distributed_solve_kernel_does_not_all_gather_continuation_v():
    """The backward-induction kernel reads only its device-local continuation V.

    `type1`/`type2` never transition, so a regime's continuation depends only on
    its own type slice of the next-period V-array. Sharded on those axes, the
    interpolation must run device-locally — the optimized kernel contains no
    `all-gather` collective assembling the full continuation V on every device.
    """
    model = _make_correct_distributed_model(distributed=True)
    hlo = _compiled_solve_kernel_hlo(model=model, regime_name="working_life", period=0)
    assert "all-gather" not in hlo


@_skip_pytest_parallel
def test_planned_solve_rejects_a_replicated_output_injected_after_kernel_call(
    *, monkeypatch
):
    """A planned V cannot be repaired after the GridSearch adapter returns.

    The injection happens after the adapter's full-output assertion.  The
    generic publication seam therefore checks the actual ``KernelResult`` too
    and fails closed instead of silently device-putting it into place.
    """
    model = _make_correct_distributed_model(distribute_type2=False)
    original_run_period_kernel = backward_induction._run_period_kernel
    replicated_outputs: list[str] = []

    def emit_replicated_V(**kwargs):
        result = original_run_period_kernel(**kwargs)
        if isinstance(result.V_arr.sharding, NamedSharding):
            replicated_outputs.append(kwargs["regime"].name)
            return dataclasses.replace(
                result,
                V_arr=jax.device_put(
                    result.V_arr,
                    NamedSharding(result.V_arr.sharding.mesh, PartitionSpec()),
                ),
            )
        return result

    monkeypatch.setattr(backward_induction, "_run_period_kernel", emit_replicated_V)

    with pytest.raises(AssertionError, match="post-run repair is not permitted"):
        model.solve(
            log_level="off",
            params={"discount_factor": 0.95},
        )

    assert replicated_outputs, "no kernel output was replicated; test is inert"


@_skip_pytest_parallel
def test_grid_search_aot_output_layout_is_native_local_and_deduplicated(monkeypatch):
    """Real AOT GridSearch outputs are born sharded without repair or collectives."""
    model = _make_correct_distributed_model(distribute_type2=False)
    working_kernels = model._regimes["working_life"].solution.period_kernels
    assert working_kernels
    assert all(
        getattr(kernel, "streamed_core", None) is not None
        for kernel in working_kernels.values()
    )
    single = _make_correct_distributed_model(
        distributed=False, distribute_type2=False
    ).solve(log_level="off", params={"discount_factor": 0.95})
    captured = []
    original_attach = backward_induction._attach_resolved_output_layout

    def capture_planned_core(**kwargs):
        core = original_attach(**kwargs)
        if hasattr(core, "layout"):
            captured.append(core)
        return core

    def fail_repair(**kwargs):
        raise AssertionError(f"planned output reached repair: {kwargs}")

    monkeypatch.setattr(
        backward_induction, "_attach_resolved_output_layout", capture_planned_core
    )
    monkeypatch.setattr(
        backward_induction, "_repair_unplanned_kernel_value", fail_repair
    )

    distributed = model.solve(log_level="off", params={"discount_factor": 0.95})

    assert captured
    assert len({id(core.compiled) for core in captured}) < len(captured)
    transfers = tuple(
        transfer for core in captured for transfer in core.input_transfer_plan
    )
    assert transfers
    assert {transfer.kind for transfer in transfers} == {
        ValueTransferKind.ALIGNED_LOCAL
    }
    assert len({transfer.source for transfer in transfers}) == len(transfers)
    for transfer in transfers:
        assert transfer.target.kind is ValueArtifactKind.REGIME_VALUE
        assert transfer.target.period == transfer.source.source_period + 1
        assert transfer.source.path[0] == transfer.target.regime
        assert isinstance(transfer.stored_sharding, NamedSharding)
        assert transfer.source_sharding == transfer.stored_sharding
    for core in captured:
        assert core.compiled.output_shardings == core.layout.out_shardings
    hlo = "\n".join(core.compiled.as_text().lower() for core in captured)
    for collective in (
        "all-gather",
        "all-reduce",
        "all-to-all",
        "collective-permute",
        "reduce-scatter",
    ):
        assert collective not in hlo

    assert model._regimes["working_life"].solution.state_names == (
        "type1",
        "type2",
        "wealth",
    )
    for period, regime_to_value in distributed.items():
        for regime_name, value in regime_to_value.items():
            assert isinstance(value.sharding, NamedSharding)
            assert value.sharding.spec == PartitionSpec("type1", None, None)
            np.testing.assert_array_equal(value, single[period][regime_name])


@_skip_pytest_parallel
def test_grid_search_same_mesh_rank_specific_value_input_stays_aligned(monkeypatch):
    """A target keeps its own rank-specific spec on a shared execution mesh."""
    model = _make_correct_distributed_model(
        distribute_type2=False,
        retirement_reads_type2=False,
    )
    captured = []
    original_attach = backward_induction._attach_resolved_output_layout

    def capture_planned_core(**kwargs):
        core = original_attach(**kwargs)
        if hasattr(core, "layout"):
            captured.append(core)
        return core

    monkeypatch.setattr(
        backward_induction, "_attach_resolved_output_layout", capture_planned_core
    )
    params = {"discount_factor": 0.95}
    distributed = model.solve(log_level="off", params=params)

    assert model._regimes["working_life"].solution.state_names == (
        "type1",
        "type2",
        "wealth",
    )
    assert model._regimes["retirement"].solution.state_names == ("type1", "wealth")
    matching = [
        (core, transfer)
        for core in captured
        for transfer in core.input_transfer_plan
        if transfer.source.source_period == 4
        and transfer.source.source_regime == "working_life"
        and transfer.target.period == 5
        and transfer.target.regime == "retirement"
    ]
    assert len(matching) == 1
    core, transfer = matching[0]
    assert transfer.kind is ValueTransferKind.ALIGNED_LOCAL

    target_sharding = distributed[5]["retirement"].sharding
    source_output_sharding = distributed[4]["working_life"].sharding
    assert isinstance(target_sharding, NamedSharding)
    assert isinstance(source_output_sharding, NamedSharding)
    assert target_sharding.mesh == source_output_sharding.mesh
    assert target_sharding.spec == PartitionSpec("type1", None)
    assert source_output_sharding.spec == PartitionSpec("type1", None, None)
    assert transfer.stored_sharding == target_sharding
    assert transfer.source_sharding == target_sharding

    single = _make_correct_distributed_model(
        distributed=False,
        distribute_type2=False,
        retirement_reads_type2=False,
    ).solve(log_level="off", params=params)
    for period, regime_to_value in distributed.items():
        for regime_name, value in regime_to_value.items():
            np.testing.assert_array_equal(value, single[period][regime_name])

    hlo = core.compiled.as_text().lower()
    for collective in (
        "all-gather",
        "all-reduce",
        "all-to-all",
        "collective-permute",
        "reduce-scatter",
    ):
        assert collective not in hlo


@_skip_pytest_parallel
def test_value_transfer_rejects_named_target_to_single_device_source():
    """The planner fails closed if model-construction invariants are bypassed."""
    # Distributed states are model-level declarations, and construction rejects a
    # nonterminal source that prunes one. A valid model therefore cannot produce this
    # transfer direction; the private resolver still refuses it explicitly.
    target_sharding = NamedSharding(
        jax.make_mesh((4,), ("type1",)),
        PartitionSpec("type1"),
    )
    source_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    with pytest.raises(
        ValueError,
        match="NamedSharding -> SingleDeviceSharding",
    ):
        backward_induction._resolve_value_transfer_layout(
            stored_sharding=target_sharding,
            source_execution_sharding=source_sharding,
        )


@_skip_pytest_parallel
def test_collective_grid_search_value_and_dissolution_have_planned_state_layout():
    """Collective V keeps its stakeholder axis replicated; D has state rank.

    Values agree across differently fused device topologies to dtype-scale rounding;
    dissolution and layout remain exact.
    """
    distributed_model = _make_one_axis_collective_model(distributed=True)
    working_kernels = distributed_model._regimes["working"].solution.period_kernels
    assert working_kernels
    assert all(
        getattr(kernel, "streamed_core", None) is not None
        for kernel in working_kernels.values()
    )
    single_model = _make_one_axis_collective_model(distributed=False)

    distributed_values, distributed_flags = distributed_model.solve(
        log_level="off",
        params={"discount_factor": 0.95},
        return_dissolution_flags=True,
    )
    single_values, single_flags = single_model.solve(
        log_level="off",
        params={"discount_factor": 0.95},
        return_dissolution_flags=True,
    )

    for period, regime_to_value in distributed_values.items():
        for regime_name, value in regime_to_value.items():
            assert value.shape == (4, 4, 2)
            # `_RegimeSharding.V_arr_sharding` names only state axes: the
            # replicated stakeholder entry is intentionally omitted.
            assert isinstance(value.sharding, NamedSharding)
            assert value.sharding.spec == PartitionSpec("type1", None)
            reference = single_values[period][regime_name]
            eps = np.finfo(np.asarray(value).dtype).eps
            np.testing.assert_allclose(value, reference, rtol=2 * eps, atol=2 * eps)
            dissolution = distributed_flags[period][regime_name]
            assert dissolution.shape == (4, 4)
            assert isinstance(dissolution.sharding, NamedSharding)
            assert dissolution.sharding.spec == PartitionSpec("type1", None)
            np.testing.assert_array_equal(
                dissolution, single_flags[period][regime_name]
            )


@_skip_pytest_parallel
def test_solve_returns_eagerly_materialised_V_arrs(correct_distributed_model):
    """Every V_arr shard is materialised before `solve()` returns.

    Backward induction must drain the device-side compute graph before
    the simulate phase consumes the V_arrs, so V stays sharded but no
    pending kernels leak from solve to simulate.
    """
    period_to_regime_to_V_arr = correct_distributed_model.solve(
        log_level="debug",
        params={"discount_factor": 0.95},
    )
    for regime_to_V_arr in period_to_regime_to_V_arr.values():
        for V_arr in regime_to_V_arr.values():
            for shard in V_arr.addressable_shards:
                assert shard.data.is_ready()


@_skip_pytest_parallel
def test_simulate_returns_eagerly_materialised_V_arrs(correct_distributed_model):
    """Every V_arr in the `SimulationResult` is materialised before `simulate()`
    returns.

    Forward simulation must drain its lazy compute graph before returning so
    downstream consumers (`to_dataframe`, `save`, anything that reads from
    `raw_results`) start with concrete arrays rather than pending kernels.
    """
    res = correct_distributed_model.simulate(
        log_level="debug",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(36, 0),
            "wealth": jnp.full(36, 100.0),
            "type1": jnp.full(36, 1),
            "type2": jnp.full(36, 1),
            "regime_id": jnp.zeros(36, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )
    for regime_period_data in res._raw_results.values():
        for period_data in regime_period_data.values():
            for shard in period_data.V_arr.addressable_shards:
                assert shard.data.is_ready()


@_skip_pytest_parallel
def test_simulation_running_on_multiple_cpus(correct_distributed_model):
    """Test that distribution over multiple CPU's works for simulation."""

    res = correct_distributed_model.simulate(
        log_level="debug",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(36, 0),
            "wealth": jnp.full(36, 100.0),
            "type1": jnp.full(36, 1),
            "type2": jnp.full(36, 1),
            "regime_id": jnp.zeros(36, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )

    assert res._raw_results["working_life"][2].states["type1"].sharding.num_devices == 4
    assert res._raw_results["working_life"][2].states["type2"].sharding.num_devices == 4
    assert (
        res._raw_results["working_life"][2].states["wealth"].sharding.num_devices == 4
    )


@_skip_pytest_parallel
def test_save_load_preserves_sharding_and_dataframe(
    *, correct_distributed_model, tmp_path
):
    """`save` / `load` round-trip preserves per-shard data and DataFrame output.

    Arrays must travel through the on-disk format without an implicit
    gather: each shard is written and restored on the same device mesh,
    and the `to_dataframe()` projection is byte-identical to the
    in-memory result.
    """
    res = correct_distributed_model.simulate(
        log_level="debug",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(36, 0),
            "wealth": jnp.full(36, 100.0),
            "type1": jnp.full(36, 1),
            "type2": jnp.full(36, 1),
            "regime_id": jnp.zeros(36, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )

    save_dir = tmp_path / "result"
    res.save(directory=save_dir)
    loaded = SimulationResult.load(directory=save_dir)

    for period, regime_dict in res._period_to_regime_to_V_arr.items():
        for regime_name, V_arr in regime_dict.items():
            loaded_V = loaded._period_to_regime_to_V_arr[period][regime_name]
            assert loaded_V.sharding.num_devices == V_arr.sharding.num_devices
            for original_shard, loaded_shard in zip(
                V_arr.addressable_shards,
                loaded_V.addressable_shards,
                strict=True,
            ):
                assert loaded_shard.data.shape == original_shard.data.shape

    pd.testing.assert_frame_equal(loaded.to_dataframe(), res.to_dataframe())


@_skip_pytest_parallel
def test_aot_compiled_simulation_running_on_multiple_cpus():
    """AOT-compiled simulate functions run on multi-device-sharded inputs.

    Setting `n_subjects` makes the first matching `simulate(...)` AOT-compile
    every simulate function for that batch shape. With distributed grids the
    runtime state and value-function arrays are device-sharded, so the
    compiled programs must be lowered with shardings matching what runtime
    dispatches rather than single-device defaults.
    """
    model = _make_correct_distributed_model(n_subjects=36)

    res = model.simulate(
        log_level="debug",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(36, 0),
            "wealth": jnp.full(36, 100.0),
            "type1": jnp.full(36, 1),
            "type2": jnp.full(36, 1),
            "regime_id": jnp.zeros(36, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )

    assert (
        res._raw_results["working_life"][2].states["wealth"].sharding.num_devices == 4
    )


@_skip_pytest_parallel
def test_solution_error_if_grid_product_exceeds_devices(wrong_distributed_model):
    """Solve raises when the product of distributed grid sizes exceeds devices."""

    with pytest.raises(PyLCMError, match="must equal the number"):
        wrong_distributed_model.solve(
            log_level="debug",
            params={"discount_factor": 0.95},
        )


@_skip_pytest_parallel
def test_simulation_pads_non_device_multiple_subject_count(correct_distributed_model):
    """A subject count that is not a multiple of the device count simulates cleanly.

    Distributed grids shard subjects across devices, which needs the leading axis to
    divide evenly. pylcm pads internally (duplicating the last subject up to the next
    device multiple) and trims the pad rows back out, so 5 subjects on 4 devices
    yields a result with exactly 5 subjects.
    """
    result = correct_distributed_model.simulate(
        log_level="debug",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(5, 0),
            "wealth": jnp.full(5, 100.0),
            "type1": jnp.full(5, 1),
            "type2": jnp.full(5, 1),
            "regime_id": jnp.zeros(5, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )

    assert result.n_subjects == 5
    assert result.to_dataframe()["subject_id"].nunique() == 5


@_skip_pytest_parallel
@pytest.mark.parametrize("subject_batch_size", [3, 4])
def test_distributed_simulation_with_subject_batching_matches_single_pass(
    *,
    correct_distributed_model,
    subject_batch_size,
):
    """Chunked simulate under distributed grids equals the single-pass result.

    The value-function arrays stay sharded across the devices throughout; each
    subject chunk is placed onto the subject mesh axis before its period loop, so
    chunking bounds the per-chunk device workspace without gathering anything onto
    one device. A batch size that does not divide the device count (3 on 4
    devices) is rounded up to the next device multiple.
    """
    initial_conditions = {
        "age": jnp.full(8, 0),
        "wealth": jnp.linspace(50.0, 120.0, 8),
        "type1": jnp.ones(8, dtype=jnp.int32),
        "type2": jnp.ones(8, dtype=jnp.int32),
        "regime_id": jnp.zeros(8, dtype=jnp.int32),
    }
    single_pass = correct_distributed_model.simulate(
        log_level="off",
        params={"discount_factor": 0.95},
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=12345,
    )
    chunked = correct_distributed_model.simulate(
        log_level="off",
        params={"discount_factor": 0.95},
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=12345,
        subject_batch_size=subject_batch_size,
    )
    pd.testing.assert_frame_equal(chunked.to_dataframe(), single_pass.to_dataframe())


@_skip_pytest_parallel
def test_distributed_aot_simulation_pads_subjects_to_a_chunk_multiple():
    """A chunk size that does not divide the subject count simulates cleanly.

    Under distributed grids the chunk size is rounded up to a device multiple
    and every chunk must match the AOT-compiled shape, so the subject axis is
    padded up to a chunk multiple (duplicating the last subject) and the pad
    rows are trimmed back out — the result holds exactly the real subjects and
    equals the single-pass result.
    """
    model = _make_correct_distributed_model(n_subjects=12)
    initial_conditions = {
        "age": jnp.full(12, 0),
        "wealth": jnp.linspace(50.0, 120.0, 12),
        "type1": jnp.ones(12, dtype=jnp.int32),
        "type2": jnp.ones(12, dtype=jnp.int32),
        "regime_id": jnp.zeros(12, dtype=jnp.int32),
    }
    single_pass = model.simulate(
        log_level="off",
        params={"discount_factor": 0.95},
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=12345,
    )
    chunked = model.simulate(
        log_level="off",
        params={"discount_factor": 0.95},
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        seed=12345,
        subject_batch_size=8,
    )
    assert chunked.n_subjects == 12
    pd.testing.assert_frame_equal(chunked.to_dataframe(), single_pass.to_dataframe())


def _make_partially_distributed_model(*, distributed: bool) -> Model:
    """Model where one regime has distributed grids and the other does not."""

    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt

    @categorical(ordered=True)
    class Type:
        low: ScalarInt
        high: ScalarInt

    working_life = UserRegime(
        functions={
            "utility": lambda wealth, consumption, type1, type2: (
                (jnp.log(consumption) + wealth * 0.001) * type1 * type2
            ),
        },
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
        state_transitions={
            "wealth": lambda wealth, consumption: wealth - consumption,
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(
            age >= 4, RegimeId.retirement, RegimeId.working_life
        ),
        active=lambda age: age < 5,
    )

    retirement = UserRegime(
        transition=None,
        functions={"utility": lambda wealth: wealth * 0.5},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        active=lambda age: age >= 5,
    )

    return Model(
        regimes={"working_life": working_life, "retirement": retirement},
        ages=AgeGrid(start=0, stop=5, step="Y"),
        regime_id_class=RegimeId,
        states={
            "type1": DiscreteGrid(category_class=Type, distributed=distributed),
            "type2": DiscreteGrid(category_class=Type, distributed=distributed),
        },
        state_transitions={
            "type1": fixed_transition("type1"),
            "type2": fixed_transition("type2"),
        },
    )


@pytest.fixture
def partially_distributed_model():
    """Return the mixed-layout model used by solve and transfer tests."""
    return _make_partially_distributed_model(distributed=True)


@_skip_pytest_parallel
def test_solve_with_partial_distribution_returns_correct_shardings(
    *,
    partially_distributed_model,
    monkeypatch,
):
    """A single-device target is copied as a replicated source-mesh input."""
    captured = []
    original_attach = backward_induction._attach_resolved_output_layout

    def capture_planned_core(**kwargs):
        core = original_attach(**kwargs)
        if hasattr(core, "layout"):
            captured.append(core)
        return core

    monkeypatch.setattr(
        backward_induction, "_attach_resolved_output_layout", capture_planned_core
    )
    params = {"discount_factor": 0.95}
    distributed = partially_distributed_model.solve(
        log_level="debug",
        params=params,
    )

    assert distributed[0]["working_life"].sharding.num_devices == 4
    assert distributed[5]["retirement"].sharding.num_devices == 1
    period_four_copies = [
        transfer
        for core in captured
        for transfer in core.input_transfer_plan
        if transfer.kind is ValueTransferKind.COPY_TO_SOURCE_LAYOUT
        and transfer.source.source_period == 4
        and transfer.source.source_regime == "working_life"
        and transfer.target.period == 5
        and transfer.target.regime == "retirement"
    ]
    assert len(period_four_copies) == 1
    transfer = period_four_copies[0]
    assert transfer.target.kind is ValueArtifactKind.REGIME_VALUE
    assert transfer.target.target_regime is None
    assert transfer.source.core_key == "main"
    assert transfer.source.channel is ValueInputChannel.NEXT_REGIME_VALUE
    assert transfer.source.path == ("retirement",)
    assert isinstance(transfer.stored_sharding, jax.sharding.SingleDeviceSharding)
    assert isinstance(transfer.source_sharding, NamedSharding)
    assert transfer.source_sharding.spec == PartitionSpec()
    assert transfer.source_sharding.is_fully_replicated

    source_output_sharding = distributed[4]["working_life"].sharding
    assert isinstance(source_output_sharding, NamedSharding)
    assert transfer.source_sharding.mesh == source_output_sharding.mesh
    assert transfer.source_sharding.spec != source_output_sharding.spec

    single = _make_partially_distributed_model(distributed=False).solve(
        log_level="debug",
        params=params,
    )
    for period, regime_to_value in distributed.items():
        for regime_name, value in regime_to_value.items():
            np.testing.assert_array_equal(value, single[period][regime_name])


def test_distributed_action_grid_raises_at_regime_init():
    """Action grids cannot be distributed; regime finalization rejects one.

    Distribution is a property of state axes (which form the V-array shape).
    Marking an action grid as distributed has no consistent meaning under the
    current sharding model, so it is rejected when the model finalizes its
    regimes. (Continuous action grids never reach this check — they
    are rejected at grid init by `_fail_if_continuous_grid_distributed`.)
    """

    @categorical(ordered=False)
    class Choice:
        a: ScalarInt
        b: ScalarInt

    regime = UserRegime(
        functions={"utility": jnp.log},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        state_transitions={
            "wealth": lambda wealth, choice: wealth - choice,
        },
        actions={
            "choice": DiscreteGrid(category_class=Choice, distributed=True),
        },
        transition=lambda age: age,
    )
    with pytest.raises(RegimeInitializationError, match="distributed=True"):
        finalize_regimes(
            user_regimes={"regime": regime},
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
        )


@_skip_pytest_parallel
def test_v_array_has_nan_keeps_reduction_sharded_on_distributed_input():
    """`v_array_has_nan` returns a mesh-replicated scalar, not a single-device one.

    The reduction stays inside `@jax.jit` so XLA partitions it across the V-array's
    devices (per-device any → all-reduce → replicated scalar) instead of gathering
    the full V-array onto the default device first.
    """
    mesh = jax.make_mesh((4,), ("dev",))
    sharded = jax.device_put(
        jnp.zeros((8,), dtype=jnp.float32),
        NamedSharding(mesh, PartitionSpec("dev")),
    )

    result = v_array_has_nan(sharded)

    assert bool(result) is False
    assert result.sharding.num_devices == 4
    assert result.sharding.is_fully_replicated


@_skip_pytest_parallel
def test_v_array_has_inf_keeps_reduction_sharded_on_distributed_input():
    """`v_array_has_inf` returns a mesh-replicated scalar, not a single-device one."""
    mesh = jax.make_mesh((4,), ("dev",))
    sharded = jax.device_put(
        jnp.array([0.0, jnp.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        NamedSharding(mesh, PartitionSpec("dev")),
    )

    result = v_array_has_inf(sharded)

    assert bool(result) is True
    assert result.sharding.num_devices == 4
    assert result.sharding.is_fully_replicated
