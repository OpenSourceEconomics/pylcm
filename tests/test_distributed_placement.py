"""A solve places every regime on a submesh of the visible devices.

Runs on a four-CPU-device topology pinned at import; the file skips wholesale
in a process whose backend is already initialized, so it runs in its own
process. Placement is a partition of the solve, never a change to it: the
values two placements of one model publish name the same real numbers, and a
simulation reads them off the canonical layout either way.
"""

import dataclasses
import logging
import subprocess
import sys
from collections.abc import Hashable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from _lcm.execution import value_transfer as transfers_module
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    ResolvedCoreProgram,
    ValueRead,
)
from _lcm.execution.footprint import (
    ResidentInventory,
    concrete_device_bytes,
)
from _lcm.execution.output_layout import VALUE, resolve_output_layout
from _lcm.execution.scheduler import (
    BufferRegistry,
    PeriodTransferCache,
    ReleaseRecord,
    shares_a_buffer,
)
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
)
from _lcm.execution.workspace_planning import compiler_peak_bytes, plan_workspace
from _lcm.grids import categorical
from _lcm.grids.continuous import LinSpacedGrid
from _lcm.grids.discrete import DiscreteGrid
from _lcm.regime_building import processing
from _lcm.simulation.initial_conditions import build_initial_states
from _lcm.solution import backward_induction
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.solution.v_topology import _get_regime_V_shapes_and_shardings
from _lcm.typing import RegimeName
from lcm import fixed_transition
from lcm.ages import AgeGrid
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.model import Model
from lcm.regime import Regime as UserRegime
from lcm.solver_api import ContinuationReader
from lcm.solvers import GridSearch, Solver
from lcm.typing import ScalarInt
from tests.conftest import assert_agrees_to_ulp

# Run these tests on a four-CPU-device topology. The pin only applies in a
# process whose JAX backends are not yet initialized; otherwise the tests skip.
# The device-count update is attempted FIRST because it is the one that raises
# after initialization, which keeps the pin atomic.
try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _PYTEST_PARALLEL = False
except RuntimeError:
    _PYTEST_PARALLEL = True

_skip_pytest_parallel = pytest.mark.skipif(
    _PYTEST_PARALLEL, reason="Can't set num cpus in pytest paralellel"
)

_PARAMS = {"discount_factor": 0.95}


@categorical(ordered=False)
class _ThreeTypeRegimeId:
    """Regime vocabulary of the three-valued-type model."""

    working: ScalarInt
    retired: ScalarInt


@categorical(ordered=True)
class _Type:
    """A three-valued preference type; its extent is the mesh size."""

    low: ScalarInt
    mid: ScalarInt
    high: ScalarInt


def _make_three_type_model(
    *,
    distributed: bool,
    sharded: tuple[str, ...] = (),
    devices: tuple[int, ...] | None = None,
    solver: Solver | None = None,
) -> Model:
    """A working regime over a three-valued type beside a single-device terminal one.

    Both regimes are active before the final age and read nothing of each other
    within a period, so on four devices the working regime runs on three and the
    terminal one on the fourth. `sharded` names the same axis through
    `ExecutionConfig`; either spelling places the regime the same way.
    `devices` restricts the model to a subset of the four.
    """
    working = UserRegime(
        active=lambda age: age < 4,
        solver=GridSearch() if solver is None else solver,
        functions={
            "utility": lambda wealth, consumption, type1: (
                (jnp.log(consumption) + wealth * 0.001) * (type1 + 1)
            ),
        },
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=12)},
        state_transitions={"wealth": lambda wealth, consumption: wealth - consumption},
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(
            age >= 3, _ThreeTypeRegimeId.retired, _ThreeTypeRegimeId.working
        ),
    )
    retired = UserRegime(
        transition=None,
        functions={"utility": lambda wealth: wealth * 0.5},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=12)},
    )
    return Model(
        regimes={"working": working, "retired": retired},
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=_ThreeTypeRegimeId,
        states={"type1": DiscreteGrid(category_class=_Type)},
        state_transitions={"type1": fixed_transition("type1")},
        execution_config=ExecutionConfig(
            sharded_states=tuple(
                dict.fromkeys((*sharded, *(("type1",) if distributed else ())))
            ),
            devices=devices,
        ),
    )


@_skip_pytest_parallel
def test_sharded_state_from_execution_config_places_the_regime_on_the_submesh() -> None:
    """Declaring a state in `sharded_states` places its regime's nodes on a submesh."""
    model = _make_three_type_model(distributed=False, sharded=("type1",))

    assert model._regimes["working"].solution.submesh_device_ids == (0, 1, 2)


@_skip_pytest_parallel
def test_sharded_state_from_execution_config_shards_the_value() -> None:
    """A state named in `sharded_states` carries the same device axis as the field."""
    solution = _make_three_type_model(distributed=False, sharded=("type1",)).solve(
        params=_PARAMS, log_level="off"
    )
    mesh = solution.values[0]["working"].sharding.mesh  # ty: ignore[unresolved-attribute]

    assert tuple(device.id for device in mesh.devices.flat) == (0, 1, 2)


@_skip_pytest_parallel
def test_execution_devices_reports_the_configured_device_ids() -> None:
    """`Model.execution_devices` names exactly the ids the configuration gave."""
    model = _make_three_type_model(distributed=False, devices=(2, 3))

    assert model.execution_devices == (2, 3)


@_skip_pytest_parallel
def test_the_planner_partitions_the_models_own_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The submesh planner is given the model's device count, not JAX's."""
    recorded: list[int] = []
    original = processing.plan_submesh_placement

    def _recording(*, requests: Any, n_devices: int) -> Any:
        recorded.append(n_devices)
        return original(requests=requests, n_devices=n_devices)

    monkeypatch.setattr(processing, "plan_submesh_placement", _recording)
    _make_three_type_model(distributed=False, devices=(2, 3))

    assert recorded == [2]


@_skip_pytest_parallel
def test_a_sharded_regime_is_placed_on_the_configured_device_ids() -> None:
    """The planner's block positions are read back as the model's own device ids."""
    model = _make_three_type_model(
        distributed=False, sharded=("type1",), devices=(1, 2, 3)
    )

    assert model._regimes["working"].solution.submesh_device_ids == (1, 2, 3)


@_skip_pytest_parallel
def test_a_model_restricted_to_two_devices_publishes_every_value_on_them() -> None:
    """Every value a solve publishes lives on a device the configuration named."""
    solution = _make_three_type_model(distributed=False, devices=(2, 3)).solve(
        params=_PARAMS, log_level="off"
    )

    published_device_ids = {
        device.id
        for by_regime in solution.values.values()
        for value in by_regime.values()
        for device in value.sharding.device_set
    }

    assert published_device_ids <= {2, 3}


@_skip_pytest_parallel
def test_the_regime_beside_a_sharded_one_stays_on_the_configured_devices() -> None:
    """A single-device regime of a restricted model keeps off the excluded devices."""
    solution = _make_three_type_model(
        distributed=False, sharded=("type1",), devices=(1, 2, 3)
    ).solve(params=_PARAMS, log_level="off")
    value = solution.values[0]["retired"]

    assert {device.id for device in value.sharding.device_set} <= {1, 2, 3}


@_skip_pytest_parallel
def test_seeded_subject_states_of_a_restricted_model_stay_on_its_devices() -> None:
    """Per-subject simulate arrays are seeded on a device the configuration named."""
    model = _make_three_type_model(distributed=False, devices=(2, 3))
    states_per_regime = build_initial_states(
        initial_states={
            "wealth": jnp.full(4, 50.0),
            "type1": jnp.asarray([0, 1, 2, 0]),
        },
        regimes=model._regimes,
        device_ids=model.execution_devices,
    )

    seeded_device_ids = {
        device.id
        for regime_states in states_per_regime.values()
        for array in regime_states.values()
        for device in array.sharding.device_set
    }

    assert seeded_device_ids <= {2, 3}


@_skip_pytest_parallel
def test_a_three_valued_type_is_sharded_over_three_devices() -> None:
    """The working regime's value lives on a three-device mesh."""
    solution = _make_three_type_model(distributed=True).solve(
        params=_PARAMS, log_level="off"
    )
    value = solution.values[0]["working"]

    mesh = value.sharding.mesh  # ty: ignore[unresolved-attribute]

    assert tuple(device.id for device in mesh.devices.flat) == (0, 1, 2)


@_skip_pytest_parallel
def test_the_single_device_regime_takes_the_idle_device() -> None:
    """The terminal regime's value lives on the device the mesh leaves idle."""
    solution = _make_three_type_model(distributed=True).solve(
        params=_PARAMS, log_level="off"
    )
    value = solution.values[0]["retired"]

    assert value.sharding == jax.sharding.SingleDeviceSharding(jax.devices()[3])


@_skip_pytest_parallel
@pytest.mark.parametrize("regime", ["working", "retired"])
def test_two_placements_of_one_model_publish_the_same_values(
    *, regime: RegimeName
) -> None:
    """Placement partitions a solve without changing what it computes.

    Sharding the type axis over three devices and keeping the whole regime on
    one are the same arithmetic in a different partition, and XLA vectorizes
    each at its own width, so the two runs name the same real number rather
    than the same bit pattern.
    """
    placed = _make_three_type_model(distributed=True).solve(
        params=_PARAMS, log_level="off"
    )
    canonical = _make_three_type_model(distributed=False).solve(
        params=_PARAMS, log_level="off"
    )

    expected_roster = {
        period: {"working", "retired"} if period < 4 else {"retired"}
        for period in range(5)
    }
    assert {period: set(values) for period, values in placed.values.items()} == (
        expected_roster
    )
    assert {period: set(values) for period, values in canonical.values.items()} == (
        expected_roster
    )
    for period, active in expected_roster.items():
        if regime not in active:
            continue
        assert_agrees_to_ulp(
            got=np.asarray(placed.values[period][regime]),
            expected=np.asarray(canonical.values[period][regime]),
            n_ulp=8,
            err_msg=f"regime {regime!r}, period {period}",
        )


@_skip_pytest_parallel
def test_independent_regimes_of_one_period_share_one_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both regimes of a period are dispatched together in the period's first wave."""
    units_by_period: dict[int, int] = {}
    recorder = _WavePlanRecorder(
        planner=backward_induction.plan_period_waves, units_by_period=units_by_period
    )
    monkeypatch.setattr(backward_induction, "plan_period_waves", recorder)
    _make_three_type_model(distributed=True).solve(params=_PARAMS, log_level="off")

    assert units_by_period == {0: 2, 1: 2, 2: 2, 3: 2, 4: 1}


class _WavePlanRecorder:
    """Call the real wave planner and record each period's first wave width."""

    def __init__(self, *, planner: object, units_by_period: dict[int, int]) -> None:
        """Keep the planner to delegate to and the mapping to record into."""
        self._planner = planner
        self._units_by_period = units_by_period

    def __call__(self, **kwargs: object) -> object:
        """Plan the period's waves and record how many units the first one holds."""
        waves = self._planner(**kwargs)  # ty: ignore[call-non-callable]
        self._units_by_period[waves[0][0].period] = len(waves[0])
        return waves


@_skip_pytest_parallel
def test_a_model_with_one_regime_per_period_is_placed_on_device_zero() -> None:
    """Where nothing is co-active, a single-device regime keeps today's placement."""
    from tests.test_distributed import (  # noqa: PLC0415
        _make_correct_distributed_model,
    )

    solution = _make_correct_distributed_model(distributed=False).solve(
        params=_PARAMS, log_level="off"
    )
    value = solution.values[5]["retirement"]

    assert value.sharding.device_set == {jax.devices()[0]}


@_skip_pytest_parallel
def test_simulating_a_submesh_placed_solution_uses_the_subject_devices() -> None:
    """Subjects span four devices while the original three-device value survives."""
    model = _make_three_type_model(distributed=True)
    solution = model.solve(params=_PARAMS, log_level="off")
    view = solution._engine_view
    assert isinstance(view, OwnedSolutionView)
    original = view.values[0]["working"]
    original_values = np.asarray(original).copy()
    assert original.sharding.device_set == set(jax.devices()[:3])

    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.array([10.0, 20.0, 30.0, 40.0]),
            "type1": jnp.array([0, 1, 2, 1]),
            "age": jnp.zeros(4),
            "regime_id": jnp.array([0, 0, 0, 0]),
        },
        solution=solution,
        log_level="off",
        seed=42,
    )
    assert result.n_subjects == 4
    assert result.raw_results["working"][0].V_arr.sharding.device_set == (
        set(jax.devices())
    )
    np.testing.assert_array_equal(np.asarray(original), original_values)


@_skip_pytest_parallel
def test_simulating_co_active_single_device_regimes_matches_one_device(
    tmp_path: Path,
) -> None:
    """Values placed off device zero are brought back before simulation."""
    code = (
        "import jax; jax.config.update('jax_num_cpu_devices', 1); "
        "import sys; import numpy as np; "
        "from tests.test_distributed_placement import _simulate_two_single_regimes; "
        "np.save(sys.argv[1], _simulate_two_single_regimes())"
    )
    reference = tmp_path / "one-device.npy"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(reference)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, result.stderr

    np.testing.assert_array_equal(_simulate_two_single_regimes(), np.load(reference))


def _simulate_two_single_regimes() -> np.ndarray:
    """Solve and simulate the two-regime out-of-tree model; return its wealth.

    Its `alive` and `dead` regimes are co-active in every period but the last,
    so on four devices `alive` sits on device 0 and `dead` on device 1.
    """
    from tests.test_solver_api_out_of_tree import (  # noqa: PLC0415
        WealthSolver,
        _two_regime_model,
    )

    model = _two_regime_model(solver=WealthSolver())
    result = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": jnp.array([1.0, 2.0, 3.0, 4.0]),
            "age": jnp.zeros(4),
            "regime_id": jnp.array([0, 0, 0, 0]),
        },
        log_level="off",
    )
    return np.asarray(result.to_dataframe()["wealth"])


@categorical(ordered=False)
class _TwoMeshRegimeId:
    """Regime vocabulary of the two-sharded-regime model."""

    alpha: ScalarInt
    beta: ScalarInt
    retired: ScalarInt


def _make_two_mesh_model() -> Model:
    """Two co-active regimes over one three-valued type beside a terminal one.

    Both sharded regimes take the same three-device block, and the terminal
    regime — which does not read the type — takes the device that block leaves
    idle. Each sharded regime therefore reads the terminal value across
    disjoint devices, into the one replicated layout their shared mesh
    defines.
    """

    def _worker() -> UserRegime:
        return UserRegime(
            functions={
                "utility": lambda wealth, consumption, type1: (
                    jnp.log(consumption) + wealth * 0.001 * (type1 + 1)
                ),
            },
            states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=8)},
            state_transitions={
                "wealth": lambda wealth, consumption: wealth - consumption
            },
            actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=6)},
            transition=lambda age: jnp.where(
                age >= 0, _TwoMeshRegimeId.retired, _TwoMeshRegimeId.alpha
            ),
        )

    return Model(
        regimes={
            "alpha": _worker(),
            "beta": _worker(),
            "retired": UserRegime(
                transition=None,
                functions={"utility": lambda wealth: wealth * 0.5},
                states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=8)},
            ),
        },
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_TwoMeshRegimeId,
        states={"type1": DiscreteGrid(category_class=_Type)},
        execution_config=ExecutionConfig(sharded_states=("type1",)),
        state_transitions={"type1": fixed_transition("type1")},
    )


class _SharedCopyLog:
    """Registered shared-transfer copies and their liveness after every commit."""

    def __init__(self) -> None:
        """Start with no copy recorded and no commit observed."""
        self.copies: list[jax.Array] = []
        self.deleted_after_commit: list[bool] = []


class _RecordingTransferCache(PeriodTransferCache):
    """A period transfer cache that reports what it registers and releases."""

    __slots__ = ("log",)

    def put(
        self,
        *,
        transfer: ResolvedValueTransfer,
        array: jax.Array,
        stored: jax.Array,
    ) -> None:
        """Cache the copy, recording it when it occupies a buffer of its own."""
        super().put(transfer=transfer, array=array, stored=stored)
        if not shares_a_buffer(first=array, second=stored):
            self.log.copies.append(array)

    def commit_consumer(
        self, *, key: tuple[Hashable, Hashable]
    ) -> tuple[ReleaseRecord, ...]:
        """Commit, then record whether the period's copy is already deleted."""
        records = super().commit_consumer(key=key)
        if self.log.copies:
            self.log.deleted_after_commit.append(self.log.copies[-1].is_deleted())
        return records


class _RecordingTransferCacheFactory:
    """Build recording caches that all report into one log."""

    def __init__(self, *, log: _SharedCopyLog) -> None:
        """Keep the log every cache this factory builds reports into."""
        self._log = log

    def __call__(self, **kwargs: Any) -> _RecordingTransferCache:
        """Build one period's recording cache."""
        cache = _RecordingTransferCache(**kwargs)
        cache.log = self._log
        return cache


def _record_shared_copy_lifetime(*, monkeypatch: pytest.MonkeyPatch) -> _SharedCopyLog:
    """Solve the two-mesh model, recording each registered copy's liveness.

    Every commit of a shared-transfer key appends whether the copy the period
    registered is already deleted, so the recorded sequence says exactly when
    the copy was freed relative to its declared consumers.
    """
    log = _SharedCopyLog()
    monkeypatch.setattr(
        backward_induction,
        "PeriodTransferCache",
        _RecordingTransferCacheFactory(log=log),
    )
    _make_two_mesh_model().solve(params=_PARAMS, log_level="off")
    return log


@_skip_pytest_parallel
def test_a_cross_device_regime_value_copy_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading a value from off the reader's mesh makes a copy the engine owns."""
    log = _record_shared_copy_lifetime(monkeypatch=monkeypatch)

    assert log.copies


@_skip_pytest_parallel
def test_a_shared_copy_survives_its_first_consumers_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copy stays alive while a second declared consumer has not committed."""
    log = _record_shared_copy_lifetime(monkeypatch=monkeypatch)

    assert log.deleted_after_commit[0] is False


@_skip_pytest_parallel
def test_a_shared_copy_is_deleted_after_its_last_consumers_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copy is released once every declared consumer has committed."""
    log = _record_shared_copy_lifetime(monkeypatch=monkeypatch)

    assert log.deleted_after_commit[1] is True


@_skip_pytest_parallel
def test_two_co_active_sharded_regimes_take_the_same_block() -> None:
    """Blocks follow declaration order, so two equal meshes may overlap."""
    solution = _make_two_mesh_model().solve(params=_PARAMS, log_level="off")

    assert solution.values[0]["alpha"].sharding == solution.values[0]["beta"].sharding


@categorical(ordered=False)
class _TwoBlockRegimeId:
    """Regime vocabulary of the two-block model."""

    first: ScalarInt
    second: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _TwoValuedType:
    """A two-valued preference type; its extent is each block's mesh size."""

    low: ScalarInt
    high: ScalarInt


def _make_two_block_model(*, distributed: bool) -> Model:
    """Two sharded regimes over a two-valued type; the first enters the second.

    Both regimes carry the type, so each takes a two-device block, and on four
    devices the blocks are disjoint. The first regime's continuation therefore
    reads the second regime's value from devices its own mesh does not hold.
    """

    def _utility(*, wealth: Any, consumption: Any, type1: Any) -> Any:
        return (jnp.log(consumption) + wealth * 0.001) * (type1 + 1)

    def _next_wealth(*, wealth: Any, consumption: Any) -> Any:
        return wealth - consumption

    def _worker(*, transition: Any, active: Any) -> UserRegime:
        return UserRegime(
            functions={"utility": _utility},
            states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
            state_transitions={"wealth": _next_wealth},
            actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
            transition=transition,
            active=active,
        )

    first = _worker(
        transition=lambda age: jnp.where(
            age >= 1, _TwoBlockRegimeId.second, _TwoBlockRegimeId.first
        ),
        active=lambda age: age < 3,
    )
    second = _worker(
        transition=lambda age: jnp.where(
            age >= 3, _TwoBlockRegimeId.dead, _TwoBlockRegimeId.second
        ),
        active=lambda _age: True,
    )
    dead = UserRegime(
        transition=None,
        functions={"utility": lambda wealth, type1: 0.0 * wealth * type1},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        active=lambda age: age >= 4,
    )
    return Model(
        regimes={"first": first, "second": second, "dead": dead},
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=_TwoBlockRegimeId,
        states={"type1": DiscreteGrid(category_class=_TwoValuedType)},
        state_transitions={"type1": fixed_transition("type1")},
        execution_config=ExecutionConfig(
            sharded_states=("type1",) if distributed else ()
        ),
    )


@_skip_pytest_parallel
def test_a_value_read_across_disjoint_blocks_is_a_cross_mesh_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first block's regime reads the second block's value by cross-mesh copy."""
    captured: list[Any] = []
    original = backward_induction._attach_resolved_output_layout

    def capture(**kwargs: Any) -> Any:
        core = original(**kwargs)
        captured.append(core)
        return core

    monkeypatch.setattr(backward_induction, "_attach_resolved_output_layout", capture)
    _make_two_block_model(distributed=True).solve(params=_PARAMS, log_level="off")
    kinds = {
        transfer.kind
        for core in captured
        for transfer in core.input_transfer_plan
        if transfer.target.regime == "second"
    }

    assert ValueTransferKind.CROSS_MESH_COPY in kinds


@_skip_pytest_parallel
@pytest.mark.parametrize("regime", ["first", "second"])
def test_a_two_block_solve_publishes_the_single_device_values(
    *, regime: RegimeName
) -> None:
    """A cross-mesh copy delivers the stored values unchanged."""
    placed = _make_two_block_model(distributed=True).solve(
        params=_PARAMS, log_level="off"
    )
    reference = _make_two_block_model(distributed=False).solve(
        params=_PARAMS, log_level="off"
    )

    for period in placed.values:
        if regime in placed.values[period]:
            np.testing.assert_array_equal(
                np.asarray(placed.values[period][regime]),
                np.asarray(reference.values[period][regime]),
            )


def _nbegm_toy(*, distributed_kind: bool) -> Model:
    """The NB-EGM ride-along toy at its smallest grids.

    Its terminal `dead` regime does not read the ride-along type, so on four
    devices the sharded `alive` regime takes a two-device block and `dead` is
    placed on a device that block leaves free.
    """
    from tests.test_models import nbegm_ride_along_toy  # noqa: PLC0415

    return nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_periods=4,
        n_liquid=24,
        n_savings=32,
        distributed_kind=distributed_kind,
    )


def _nbegm_toy_params() -> dict[str, float]:
    """The toy's parameters."""
    from tests.test_models import nbegm_ride_along_toy  # noqa: PLC0415

    return nbegm_ride_along_toy.build_params()


def _carry_template_devices(
    *, model: Model, regime_name: RegimeName
) -> set[jax.Device]:
    """Return every device one regime's continuation template leaves sit on."""
    template = model._regimes[regime_name].solution.continuation_template
    assert isinstance(template, ContinuationReader)
    return {
        device
        for leaf in template.leaves().values()
        for device in leaf.sharding.device_set
    }


@_skip_pytest_parallel
def test_the_terminal_egm_regime_beside_a_sharded_one_is_placed_off_device_zero() -> (
    None
):
    """The block the sharded regime takes leaves the terminal regime elsewhere."""
    model = _nbegm_toy(distributed_kind=True)

    assert model._regimes["dead"].solution.submesh_device_ids != (0,)


@_skip_pytest_parallel
def test_a_placed_single_device_regime_keeps_its_carry_template_on_its_device() -> None:
    """An EGM carry template lives on the devices its regime was placed on."""
    model = _nbegm_toy(distributed_kind=True)
    placed = set(model._regimes["dead"].solution.placed_devices())

    assert _carry_template_devices(model=model, regime_name="dead") == placed


@_skip_pytest_parallel
@pytest.mark.parametrize("regime", ["alive", "dead"])
def test_the_nbegm_toy_publishes_the_same_values_under_both_placements(
    *, regime: RegimeName, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Co-mapped NB-EGM keeps ordinary ownership and the same solved values."""
    params = _nbegm_toy_params()
    run = backward_induction._run_period_kernel
    nominations: list[tuple[str, ...]] = []

    def observe(**kwargs: Any) -> Any:
        if kwargs["regime_name"] == "alive":
            nominations.extend(
                core.donated_arguments for core in kwargs["compiled_cores"].values()
            )
        return run(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(backward_induction, "_run_period_kernel", observe)
        placed = _nbegm_toy(distributed_kind=True).solve(params=params, log_level="off")
    assert nominations
    assert not any(nominations)
    canonical = _nbegm_toy(distributed_kind=False).solve(params=params, log_level="off")

    for period in placed.values:
        if regime not in placed.values[period]:
            continue
        assert_agrees_to_ulp(
            got=np.asarray(placed.values[period][regime]),
            expected=np.asarray(canonical.values[period][regime]),
            n_ulp=8,
            err_msg=f"regime {regime!r}, period {period}",
        )


@_skip_pytest_parallel
def test_simulation_topology_reads_a_proper_submesh_on_all_subject_devices() -> None:
    """The three-type value keeps its shape in a four-device replicated read."""
    model = _make_three_type_model(distributed=True)
    topologies = {
        phase: _get_regime_V_shapes_and_shardings(
            regimes=model._regimes,
            flat_params=model._process_params(_PARAMS),
            phase=phase,
        )
        for phase in ("solve", "simulate")
    }
    stored = topologies["solve"]["working"]
    read = topologies["simulate"]["working"]
    assert stored.shape == read.shape == (3, 12)
    assert {device.id for device in stored.sharding.device_set} == {0, 1, 2}
    assert {device.id for device in read.sharding.device_set} == {0, 1, 2, 3}
    assert read.sharding.is_fully_replicated


def _shape_only_transfer_inputs(*, values: Mapping[str, jax.Array]) -> jax.Array:
    """Both runtime copies are pruned; only the declared shape affects output."""
    return jnp.arange(values["first"].size, dtype=values["first"].dtype)


@_skip_pytest_parallel
@pytest.mark.parametrize("shared", [False, True])
def test_pruned_transfer_destinations_remain_budgeted_before_dispatch(
    *, monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    """Device3 originals and distinct device2 copies survive compiler pruning."""
    payload = 1024 * 1024
    dtype = jnp.zeros(()).dtype
    stored = jax.sharding.SingleDeviceSharding(jax.devices()[3])
    required = jax.sharding.SingleDeviceSharding(jax.devices()[2])
    source = jax.device_put(np.full(payload // dtype.itemsize, -3, dtype=dtype), stored)
    lowering_value = jax.device_put(source, required)
    address = ValueArtifactAddress(
        kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="done"
    )
    transfers = tuple(
        ResolvedValueTransfer(
            target=address,
            source=ValueConsumerAddress(
                source_period=0,
                source_regime="acting",
                core_key="main",
                channel=ValueInputChannel.NEXT_REGIME_VALUE,
                argument="values",
                path=(name,),
            ),
            kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
            stored_sharding=stored,
            source_sharding=required,
            expected_shape=source.shape,
            expected_dtype=source.dtype,
            reused_by_several_consumers=shared,
        )
        for name in ("first", "second")
    )
    program = ResolvedCoreProgram(
        name="main",
        function=_shape_only_transfer_inputs,
        arguments={
            "values": MappingProxyType(
                {"first": lowering_value, "second": lowering_value}
            )
        },
        static_kwargs={},
        requirements=CoreExecutionRequirements(
            value_reads=tuple(
                ValueRead(target=address, source=transfer.source)
                for transfer in transfers
            )
        ),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
        tile_widths={},
        specialization_key=(),
        input_transfer_plan=transfers,
    )
    metadata = backward_induction._ProgramExecutionMetadata(
        requirements=program.requirements,
        disposition=program.disposition,
        scope=program.scope,
        input_transfer_plan=transfers,
    )
    copies = backward_induction._period_copy_reservations(
        period=0,
        metadata={("acting", 0, "main"): metadata},
    )
    inventory = ResidentInventory(
        device_ids=(2,),
        live={},
        peer_bytes={2: 0},
        declared_inputs=(),
        shared_copies=copies,
    )
    layout = resolve_output_layout(
        core_key="main",
        value_template=lowering_value,
        state_order=("wealth",),
        output_roles=VALUE,
    )
    triple = ("acting", 0, "main")
    candidate = (triple, ())
    compiled: dict[Hashable, jax.stages.Compiled] = {}
    backward_induction._lower_and_compile_wave(
        new_lowerings={"transferred": candidate},
        resolved_programs={candidate: program},
        all_layouts={triple: layout},
        internal_templates={candidate: {}},
        donations={candidate: ()},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        n_triples_per_lowering={"transferred": 1},
        log_kernel_memory=False,
        n_workers=1,
        logger=logging.getLogger(__name__),
        compiled=compiled,
        labels={},
    )
    executable = compiled["transferred"]
    assert all(
        value is None for value in executable.input_shardings[1]["values"].values()
    )
    peak = compiler_peak_bytes(compiled=executable, widths={})
    copy_count = 1 if shared else 2

    def resident(comp: jax.stages.Compiled) -> int:
        return backward_induction._candidate_resident_bytes(
            compiled=comp,
            program=program,
            internal_arguments={},
            inventory=inventory,
        )

    assert resident(executable) == copy_count * payload
    copies_made: list[jax.Array] = []
    apply = transfers_module.apply_value_transfer

    def observe(*, value: object, transfer: ResolvedValueTransfer) -> jax.Array:
        result = apply(value=value, transfer=transfer)
        copies_made.append(result)
        return result

    monkeypatch.setattr(transfers_module, "apply_value_transfer", observe)
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        plan_workspace(
            axes=(),
            compile_candidate=lambda _widths: executable,
            budget_bytes=peak + copy_count * payload - 1,
            resident_bytes_for=resident,
        )
    assert copies_made == []
    generous = plan_workspace(
        axes=(),
        compile_candidate=lambda _widths: executable,
        budget_bytes=peak + copy_count * payload,
        resident_bytes_for=resident,
    )
    core = backward_induction._attach_resolved_output_layout(
        compiled=generous.compiled,
        layout=layout,
        tile_widths={},
        input_transfer_plan=transfers,
        name="main",
    )
    if shared:
        core = dataclasses.replace(
            core,
            transfer_cache=PeriodTransferCache(
                registry=BufferRegistry(),
                consumer_counts={(address, required): 1},
            ),
        )
    output = core(values={"first": source, "second": source})
    assert isinstance(output, jax.Array)
    jax.block_until_ready((output, copies_made))
    assert len(copies_made) == copy_count
    assert concrete_device_bytes(tree=copies_made)[2] == copy_count * payload
    assert (
        concrete_device_bytes(tree=(output, copies_made))[2]
        == (copy_count + 1) * payload
    )
    assert output.devices() == {jax.devices()[2]}
    assert source.devices() == {jax.devices()[3]}
    assert not source.is_deleted()
    np.testing.assert_array_equal(source, np.full(source.shape, -3))
    np.testing.assert_array_equal(output, np.arange(source.size))
