"""Reading block-major values on the devices is admitted against the device budget.

Single reads (`SolutionResult.value`, `solution.values[p][r]`) and the bulk
`solution.values.materialize()` are refused before anything is assembled when
what they place exceeds the budget on a device, counted per device on the value's
sharding. Admitted reads return exact, independently owned buffers, and a refused
read leaves the host-side save available.

The eight-device cases skip unless the process has eight CPU host devices
(`JAX_NUM_CPU_DEVICES=8`).
"""

from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution import block_major
from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    ExecutionConfig,
    InvariantBlockSchedule,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm._solver_api import entries
from lcm._solver_api.authority import _ArrayCopier
from lcm._solver_api.stores import ValueStore
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import SolutionResult
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)

pytestmark = pytest.mark.slow
_N_TYPES = 128
_N_WEALTH = 512


@categorical(ordered=False)
class _Types:
    t000: ScalarInt
    t001: ScalarInt
    t002: ScalarInt
    t003: ScalarInt
    t004: ScalarInt
    t005: ScalarInt
    t006: ScalarInt
    t007: ScalarInt
    t008: ScalarInt
    t009: ScalarInt
    t010: ScalarInt
    t011: ScalarInt
    t012: ScalarInt
    t013: ScalarInt
    t014: ScalarInt
    t015: ScalarInt
    t016: ScalarInt
    t017: ScalarInt
    t018: ScalarInt
    t019: ScalarInt
    t020: ScalarInt
    t021: ScalarInt
    t022: ScalarInt
    t023: ScalarInt
    t024: ScalarInt
    t025: ScalarInt
    t026: ScalarInt
    t027: ScalarInt
    t028: ScalarInt
    t029: ScalarInt
    t030: ScalarInt
    t031: ScalarInt
    t032: ScalarInt
    t033: ScalarInt
    t034: ScalarInt
    t035: ScalarInt
    t036: ScalarInt
    t037: ScalarInt
    t038: ScalarInt
    t039: ScalarInt
    t040: ScalarInt
    t041: ScalarInt
    t042: ScalarInt
    t043: ScalarInt
    t044: ScalarInt
    t045: ScalarInt
    t046: ScalarInt
    t047: ScalarInt
    t048: ScalarInt
    t049: ScalarInt
    t050: ScalarInt
    t051: ScalarInt
    t052: ScalarInt
    t053: ScalarInt
    t054: ScalarInt
    t055: ScalarInt
    t056: ScalarInt
    t057: ScalarInt
    t058: ScalarInt
    t059: ScalarInt
    t060: ScalarInt
    t061: ScalarInt
    t062: ScalarInt
    t063: ScalarInt
    t064: ScalarInt
    t065: ScalarInt
    t066: ScalarInt
    t067: ScalarInt
    t068: ScalarInt
    t069: ScalarInt
    t070: ScalarInt
    t071: ScalarInt
    t072: ScalarInt
    t073: ScalarInt
    t074: ScalarInt
    t075: ScalarInt
    t076: ScalarInt
    t077: ScalarInt
    t078: ScalarInt
    t079: ScalarInt
    t080: ScalarInt
    t081: ScalarInt
    t082: ScalarInt
    t083: ScalarInt
    t084: ScalarInt
    t085: ScalarInt
    t086: ScalarInt
    t087: ScalarInt
    t088: ScalarInt
    t089: ScalarInt
    t090: ScalarInt
    t091: ScalarInt
    t092: ScalarInt
    t093: ScalarInt
    t094: ScalarInt
    t095: ScalarInt
    t096: ScalarInt
    t097: ScalarInt
    t098: ScalarInt
    t099: ScalarInt
    t100: ScalarInt
    t101: ScalarInt
    t102: ScalarInt
    t103: ScalarInt
    t104: ScalarInt
    t105: ScalarInt
    t106: ScalarInt
    t107: ScalarInt
    t108: ScalarInt
    t109: ScalarInt
    t110: ScalarInt
    t111: ScalarInt
    t112: ScalarInt
    t113: ScalarInt
    t114: ScalarInt
    t115: ScalarInt
    t116: ScalarInt
    t117: ScalarInt
    t118: ScalarInt
    t119: ScalarInt
    t120: ScalarInt
    t121: ScalarInt
    t122: ScalarInt
    t123: ScalarInt
    t124: ScalarInt
    t125: ScalarInt
    t126: ScalarInt
    t127: ScalarInt


@categorical(ordered=False)
class _Regimes:
    working: ScalarInt
    terminal: ScalarInt


def _utility(
    *, wealth: ContinuousState, consumption: ContinuousAction, pref_type: DiscreteState
) -> FloatND:
    """Linear, finite utility; both state axes are genuinely used."""
    return wealth + 0.25 * consumption + 0.125 * pref_type


def _terminal(*, wealth: ContinuousState, pref_type: DiscreteState) -> FloatND:
    """Both axes remain in the terminal value."""
    return wealth + 0.5 * pref_type


def _go_terminal() -> ScalarInt:
    return _Regimes.terminal


def _leaf_bytes(*, n_wealth: int = _N_WEALTH) -> int:
    return _N_TYPES * n_wealth * np.dtype(jnp.asarray(0.0).dtype).itemsize


def _model(
    *,
    budget: int | None,
    n_wealth: int = _N_WEALTH,
    devices: tuple[int, ...] = (0,),
    action_partitions: int = 1,
) -> Model:
    """Two periods and many cheap independent types; no private fixture helpers."""
    ages = AgeGrid(start=0, inclusive_stop=1, step="Y")
    wealth = LinSpacedGrid(start=0, stop=1, n_points=n_wealth)
    states = {"wealth": wealth, "pref_type": DiscreteGrid(category_class=_Types)}
    working = Regime(
        regime_transitions=DeterministicTransition(func=_go_terminal),
        state_transitions={
            "wealth": fixed_transition("wealth"),
            "pref_type": fixed_transition("pref_type"),
        },
        actions={"consumption": LinSpacedGrid(start=0, stop=1, n_points=2)},
        functions={"utility": _utility},
    )
    terminal = Regime(
        regime_transitions=None,
        functions={"utility": _terminal},
    )
    return Model(
        regimes={"working": working, "terminal": terminal},
        states=states,
        ages=ages,
        regime_id_class=_Regimes,
        initial_nodes={0: "working"},
        edges={"working": {"terminal": 0}},
        execution_config=ExecutionConfig(
            devices=devices,
            sharded_states=("wealth",) if len(devices) > 1 else (),
            action_partitions={"working": action_partitions},
            device_memory_bytes=budget,
            device_memory_headroom_fraction=0.0,
            invariant_block_widths={"pref_type": 1},
            invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR,
            axis_widths={
                "cell": 64,
                "action_product": 1 if action_partitions > 1 else 2,
            },
        ),
    )


def _solve(*, budget: int | None, n_wealth: int = _N_WEALTH) -> SolutionResult:
    """An inability to admit the fixture is a setup failure, never a green test."""
    return _model(budget=budget, n_wealth=n_wealth).solve(
        params={"discount_factor": 0.9}, log_level="off"
    )


@pytest.mark.parametrize("n_wealth", [256, 512])
def test_single_value_larger_than_budget_is_refused(*, n_wealth: int) -> None:
    """A single value larger than the budget is refused before it is placed.

    The solve uses one type at a time. If a native backend cannot admit the
    fixture, this fails during setup; it does not count as fixing the read bug.
    """
    budget = _leaf_bytes(n_wealth=n_wealth) - 1
    solution = _solve(budget=budget, n_wealth=n_wealth)
    with pytest.raises(ExecutionPlanningError):
        solution.value(period=0, regime="working")


def test_bulk_materialization_accounts_for_source_and_copy(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Either reject before assembly, or avoid/charge the second full-value copy.

    The event count is a represented-buffer lower bound, not allocator telemetry.
    The copy observer performs exactly the original copy and holds no additional
    arrays after returning. A repair which hands out an already-independent
    upload needs no extra copy and can legitimately admit the final payload.
    """
    budget = 2 * _leaf_bytes()
    solution = _solve(budget=budget)
    original_assembly = block_major.RetainedComponentValues.assemble_host_value
    original_copy = entries._copy_artifact_array_leaf
    assembly_count = 0
    retained_output_bytes = 0
    observed_copy_peaks: list[int] = []

    # keyword-only-exempt: library-callback=pytest.MonkeyPatch.setattr
    def observe_assembly(
        self: block_major.RetainedComponentValues, *, period: int, regime: str
    ) -> np.ndarray:
        nonlocal assembly_count
        assembly_count += 1
        return original_assembly(self, period=period, regime=regime)

    def observe_copy(
        *, leaf: object, label: str, array_copier: _ArrayCopier | None = None
    ) -> jax.Array:
        nonlocal retained_output_bytes
        copied = original_copy(leaf=leaf, label=label, array_copier=array_copier)
        jax.block_until_ready(copied)
        # Single selected device: no replication/sharding conversion is needed.
        assert isinstance(leaf, jax.Array)
        assert copied is not leaf
        observed_copy_peaks.append(retained_output_bytes + leaf.nbytes + copied.nbytes)
        retained_output_bytes += copied.nbytes
        return copied

    monkeypatch.setattr(
        block_major.RetainedComponentValues, "assemble_host_value", observe_assembly
    )
    monkeypatch.setattr(entries, "_copy_artifact_array_leaf", observe_copy)
    try:
        values = cast("ValueStore", solution.values).materialize()
    except ExecutionPlanningError:
        assert assembly_count == 0, "Refusal occurred after a partial materialization."
        return
    jax.block_until_ready(values)
    final_bytes = sum(
        value.nbytes for period in values.values() for value in period.values()
    )
    assert final_bytes == budget
    assert all(peak <= budget for peak in observed_copy_peaks), (
        budget,
        observed_copy_peaks,
    )


def test_refused_value_still_saves_from_the_host(*, tmp_path: Path) -> None:
    """A device-read refusal must not make a complete host result unusable."""
    solution = _solve(budget=_leaf_bytes() - 1)
    with pytest.raises(ExecutionPlanningError):
        solution.value(period=0, regime="working")
    path = solution.save(path=tmp_path / "host-result")
    assert path.exists()


def test_admitted_reads_have_independent_buffers_and_exact_values() -> None:
    """Deleting one public read must not destroy the result or another read."""
    solution = _solve(budget=None)
    first = solution.value(period=0, regime="working")
    second = solution.value(period=0, regime="working")
    jax.block_until_ready((first, second))
    before = np.asarray(second).copy()
    first.delete()
    assert not second.is_deleted()
    np.testing.assert_array_equal(np.asarray(second), before)
    third = solution.value(period=0, regime="working")
    np.testing.assert_array_equal(np.asarray(third), before)
    assert np.asarray(third).dtype == before.dtype
    assert third.sharding == second.sharding
    # Byte check is deliberately stronger than an approximate numeric comparison.
    assert np.asarray(third).tobytes() == before.tobytes()


@pytest.mark.parametrize("partitions", [1, 2])
def test_single_value_guard_uses_per_device_replication(*, partitions: int) -> None:
    """Eight-device state sharding, with and without an action-replica axis.

    Run with the project's real eight-host-device profile. A one-device run
    skips these two cases explicitly and is not eight-device acceptance.
    """
    if jax.local_device_count() < 8:
        pytest.skip("Requires the declared eight-host-device native profile.")
    devices = tuple(range(8))
    per_device_bytes = _leaf_bytes() // (len(devices) // partitions)
    budget = per_device_bytes - 1
    solution = _model(
        budget=budget, devices=devices, action_partitions=partitions
    ).solve(params={"discount_factor": 0.9}, log_level="off")
    with pytest.raises(ExecutionPlanningError):
        solution.value(period=0, regime="working")


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_admitted_read_near_two_buffer_bound_preserves_bytes(*, offset: int) -> None:
    """Permit conservative admission or copy elimination; never alter the value.

    A conservative implementation may reject at 2*L-1; a one-upload ownership
    handoff can admit it. At and above 2*L the modeled source and copy fit.
    """
    leaf = _leaf_bytes(n_wealth=256)
    solution = _solve(budget=2 * leaf + offset, n_wealth=256)
    reference = _solve(budget=None, n_wealth=256)
    try:
        got = solution.value(period=0, regime="working")
    except ExecutionPlanningError:
        assert offset == -1
        return
    expected = reference.value(period=0, regime="working")
    jax.block_until_ready((got, expected))
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert got.sharding == expected.sharding
    assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()
