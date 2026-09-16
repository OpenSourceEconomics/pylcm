"""Native acceptance of trailing continuous solve-state shards.

Run alone in a fresh eight-CPU process; this module never sets device topology.
The Fraction oracle below independently evaluates the finite-grid equations in
Pro's verified ACCEPTANCE-WITNESS.md (continuous-state reply, 2026-09-14).
It does not use pylcm interpolation, reductions, or the native solved arrays.
CPU tests establish semantics and ownership only, never GPU performance.
"""

import dataclasses
from fractions import Fraction
from functools import cache
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dags.signature import rename_arguments

import _lcm.simulation.runtime as simulation_runtime
import _lcm.simulation.simulate as simulation
from _lcm.execution import output_layout, scheduler, value_transfer
from _lcm.execution.value_transfer import ValueInputChannel, ValueTransferKind
from _lcm.regime_building.max_Q_over_a import (
    get_max_Q_over_a,
    get_streaming_max_Q_over_a,
)
from _lcm.simulation import chunk_admission
from _lcm.variables import from_regime
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import ScalarInt
from tests.conftest import assert_agrees_to_ulp


@categorical(ordered=False)
class _Three:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    r0: ScalarInt
    r1: ScalarInt
    terminal: ScalarInt


def _landing(*, assets, decision, pref_type, spousal_income):
    return 15 - assets + (decision - 1) + (pref_type - spousal_income + 1) / 4


def _regime(*, source: int, identity: bool) -> Regime:
    def utility(*, assets, decision, pref_type, spousal_income):
        desired = ((pref_type + 2 * spousal_income + source) % 3) - 1
        return (
            -10
            - 20 * (decision - 1 - desired) ** 2
            - (assets + 2) ** 2 / 32
            + pref_type / 4
            - spousal_income / 8
            + source / 2
        )

    def probabilities(age):
        weight = 0.25 if source == 0 else 0.75
        return jnp.where(
            age < 1, jnp.array([weight, 1 - weight, 0]), jnp.array([0.0, 0.0, 1.0])
        )

    return Regime(
        active=lambda age: age < 2,
        transition=MarkovTransition(probabilities),
        actions={"decision": DiscreteGrid(_Three)},
        functions={"utility": utility, "landing": _landing},
        constraints={"feasible": lambda landing: (landing >= -4) & (landing <= 19)},
        state_transitions={
            "assets": fixed_transition("assets")
            if identity
            else lambda landing: landing,
            "pref_type": lambda pref_type: (pref_type + 1 + source) % 3,
            "spousal_income": lambda spousal_income: (spousal_income + 2) % 3,
        },
    )


def _model(
    *,
    devices: tuple[int, ...],
    widths: tuple[int, int] = (1, 1),
    sharded: bool = True,
    budget: int = 2**30,
    identity: bool = False,
    grid: Any = None,
    extra_shard: bool = False,
) -> Model:
    return Model(
        regimes={
            "r0": _regime(source=0, identity=identity),
            "r1": _regime(source=1, identity=identity),
            "terminal": Regime(
                active=lambda age: age == 2,
                transition=None,
                functions={
                    "utility": lambda assets, pref_type, spousal_income: (
                        -40 + pref_type + spousal_income / 4 - (assets - 5) ** 2 / 8
                    )
                },
            ),
        },
        # Assets deliberately precedes the discrete declarations: canonical
        # storage must still be (pref_type, spousal_income, assets).
        states={
            "assets": LinSpacedGrid(start=-4, stop=19, n_points=24)
            if grid is None
            else grid,
            "pref_type": DiscreteGrid(_Three),
            "spousal_income": DiscreteGrid(_Three),
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(
            devices=devices,
            sharded_states=(("assets", "pref_type") if extra_shard else ("assets",))
            if sharded
            else (),
            axis_widths={
                "action_product": widths[0],
                "cell": widths[1],
                "subject": 432,
            },
            device_memory_bytes=budget,
        ),
    )


@cache
def _reference() -> tuple[dict, dict]:
    """Exact independent finite-grid Bellman values and first maximizing actions."""
    coords = [(p, s, i) for p in range(3) for s in range(3) for i in range(24)]
    values = {
        (2, "terminal"): {
            (p, s, i): -40 + p + Fraction(s, 4) - Fraction((i - 9) ** 2, 8)
            for p, s, i in coords
        }
    }
    policies = {}

    def read(*, table: dict, p: int, s: int, landing: Fraction) -> Fraction:
        coordinate = landing + 4
        lower = min(coordinate.numerator // coordinate.denominator, 22)
        weight = coordinate - lower
        return (1 - weight) * table[p, s, lower] + weight * table[p, s, lower + 1]

    for period in (1, 0):
        for regime in range(2):
            table, policy = {}, {}
            for p, s, i in coords:
                assets = i - 4
                candidates = []
                for decision in range(3):
                    action = decision - 1
                    landing = 15 - assets + action + Fraction(p - s + 1, 4)
                    if not -4 <= landing <= 19:
                        continue
                    pp, ss = (p + 1 + regime) % 3, (s + 2) % 3
                    if period == 1:
                        continuation = read(
                            table=values[2, "terminal"], p=pp, s=ss, landing=landing
                        )
                    else:
                        weight = Fraction(1 if regime == 0 else 3, 4)
                        continuation = weight * read(
                            table=values[1, "r0"], p=pp, s=ss, landing=landing
                        ) + (1 - weight) * read(
                            table=values[1, "r1"], p=pp, s=ss, landing=landing
                        )
                    desired = ((p + 2 * s + regime) % 3) - 1
                    utility = (
                        -10
                        - 20 * (action - desired) ** 2
                        - Fraction((assets + 2) ** 2, 32)
                        + Fraction(p, 4)
                        - Fraction(s, 8)
                        + Fraction(regime, 2)
                    )
                    candidates.append((decision, utility + continuation / 2))
                decision, value = max(candidates, key=lambda item: item[1])
                table[p, s, i], policy[p, s, i] = value, decision
            values[period, f"r{regime}"], policies[period, f"r{regime}"] = table, policy
    return values, policies


def _require_eight() -> None:
    if jax.default_backend() != "cpu" or jax.device_count() != 8:
        pytest.skip("Requires a fresh eight-device CPU process")


def _assert_assets_shards(value: jax.Array) -> None:
    assert value.shape == (3, 3, 24)
    assert isinstance(value.sharding, jax.NamedSharding)
    assert value.sharding.spec == jax.P(None, None, "assets")
    shards = value.addressable_shards
    assert len(shards) == 8
    mesh_order = [device.id for device in value.sharding.mesh.devices.flat]
    covered = []
    global_ids = np.arange(216).reshape(3, 3, 24)
    for shard in shards:
        position = mesh_order.index(shard.device.id)
        assert shard.data.shape == (3, 3, 3)
        got = global_ids[shard.index]
        np.testing.assert_array_equal(
            got, global_ids[:, :, 3 * position : 3 * (position + 1)]
        )
        covered.extend(got.ravel().tolist())
    assert sorted(covered) == list(range(216))


@pytest.mark.parametrize("streamed", [False, True])
def test_trailing_untiled_axis_is_restored_inside_max_q(*, streamed: bool) -> None:
    def q(
        *,
        next_regime_to_V_arr: Any,
        pref_type: Any,
        spousal_income: Any,
        assets: Any,
        decision: Any,
    ) -> Any:
        del next_regime_to_V_arr
        return -2000.0 + 100 * pref_type + 10 * spousal_income + assets - (
            decision - 1
        ) ** 2, jnp.asarray(pref_type >= 0)

    kwargs: dict[str, Any] = {
        "Q_and_F": q,
        "batch_sizes": dict.fromkeys(("pref_type", "spousal_income", "assets"), 0),
        "action_names": ("decision",),
        "state_names": ("pref_type", "spousal_income", "assets"),
        "cell_width_keyword": "cell_width",
        "untiled_state_names": ("assets",),
    }
    builder = get_streaming_max_Q_over_a if streamed else get_max_Q_over_a
    function = builder(
        **kwargs, **({"action_width_keyword": "action_width"} if streamed else {})
    )
    result = jax.jit(
        function,
        static_argnames=("cell_width", "action_width") if streamed else ("cell_width",),
    )(
        next_regime_to_V_arr={},
        pref_type=jnp.arange(3),
        spousal_income=jnp.arange(3),
        assets=jnp.arange(-4, 20),
        decision=jnp.arange(3),
        cell_width=4,
        **({"action_width": 2} if streamed else {}),
    )
    expected = (
        -2000
        + 100 * np.arange(3)[:, None, None]
        + 10 * np.arange(3)[None, :, None]
        + np.arange(-4, 20)[None, None, :]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("widths", [(1, 1), (3, 9)])
def test_eight_assets_shards_use_full_reads_and_match_exact_bellman_reference(  # noqa: C901, PLR0915
    *, widths: tuple[int, int], monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    _require_eight()
    params = {"discount_factor": 0.5}
    reference_model = _model(devices=(0,), widths=widths, sharded=False)
    reference_solution = reference_model.solve(params=params, log_level="off")
    model = _model(devices=tuple(range(8)), widths=widths)
    materialized, born = [], []
    apply = value_transfer.apply_value_transfer
    execute = output_layout.execute_with_pending_work

    def observe_transfer(**call: Any) -> Any:
        transfer = call["transfer"]
        copied = apply(**call)
        if transfer.source.channel is ValueInputChannel.NEXT_REGIME_VALUE:
            assert transfer.kind is ValueTransferKind.ALL_GATHER
            assert copied.is_fully_replicated
            assert len(copied.sharding.device_set) == 8
            _assert_assets_shards(call["value"])
            np.testing.assert_array_equal(copied, call["value"])
            materialized.append((transfer.target, transfer.source_sharding))
        return copied

    def observe_birth(**call: Any) -> Any:
        output = execute(**call)
        for leaf in jax.tree.leaves(output):
            if isinstance(leaf, jax.Array) and leaf.shape == (3, 3, 24):
                _assert_assets_shards(leaf)
                born.append(leaf.shape)
        return output

    with monkeypatch.context() as probe:
        probe.setattr(value_transfer, "apply_value_transfer", observe_transfer)
        probe.setattr(output_layout, "execute_with_pending_work", observe_birth)
        solution = model.solve(params=params, log_level="off")
    assert len(born) == 5
    assert len(materialized) == 3  # shared terminal replica, then r0 and r1
    assert len(set(materialized)) == 3
    exact, policies = _reference()
    originals, max_ulp = [], 0.0
    assert {(t, r) for t, arrays in solution.values.items() for r in arrays} == set(
        exact
    )
    for period, arrays in solution.values.items():
        for regime, value in arrays.items():
            _assert_assets_shards(value)
            snapshot = np.asarray(value).copy()
            originals.append((value, snapshot))
            expected = np.array(
                [float(x) for x in exact[period, regime].values()], dtype=snapshot.dtype
            ).reshape(3, 3, 24)
            assert_agrees_to_ulp(got=snapshot, expected=expected, n_ulp=8)
            assert_agrees_to_ulp(
                got=snapshot,
                expected=reference_solution.values[period][regime],
                n_ulp=8,
            )
            max_ulp = max(
                max_ulp,
                float(
                    np.max(np.abs(snapshot - expected) / np.abs(np.spacing(expected)))
                ),
            )
    record_property("max_reference_ulp", max_ulp)
    # Keep both independent solutions alive during a repeated solve and replay.
    second = model.solve(params=params, log_level="off")
    assert second is not solution
    coordinates = np.array(
        [(p, s, i - 4) for p in range(3) for s in range(3) for i in range(24)]
    )
    for period in (0, 1):
        initial = {
            "pref_type": np.tile(coordinates[:, 0], 2).astype(np.int32),
            "spousal_income": np.tile(coordinates[:, 1], 2).astype(np.int32),
            "assets": np.tile(coordinates[:, 2], 2).astype(float),
            "regime_id": np.repeat(np.arange(2, dtype=np.int32), 216),
            "age": np.full(432, period, dtype=float),
        }
        result = model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            seed=42,
            log_level="off",
        )
        control = reference_model.simulate(
            params=params,
            solution=reference_solution,
            initial_conditions=initial,
            seed=42,
            log_level="off",
        )
        for regime in ("r0", "r1"):
            data = result.raw_results[regime][period]
            mask = np.asarray(data.in_regime)
            assert mask.sum() == 216
            expected_actions = np.array(list(policies[period, regime].values()))
            np.testing.assert_array_equal(
                np.asarray(data.actions["decision"])[mask], expected_actions
            )
        for got, want in zip(
            jax.tree.leaves(result.raw_results),
            jax.tree.leaves(control.raw_results),
            strict=True,
        ):
            if np.issubdtype(np.asarray(want).dtype, np.inexact):
                assert_agrees_to_ulp(got=got, expected=want, n_ulp=8)
            else:
                np.testing.assert_array_equal(got, want)
    if widths == (1, 1):
        _assert_seventeen_rows(
            model=model,
            reference_model=reference_model,
            solution=solution,
            reference_solution=reference_solution,
            monkeypatch=monkeypatch,
        )
    record_property("represented_policy_disagreements", 0)
    record_property("represented_policy_max_regret", 0)
    for original, snapshot in originals:
        assert not original.is_deleted()
        _assert_assets_shards(original)
        np.testing.assert_array_equal(original, snapshot)
    for arrays in second.values.values():
        for value in arrays.values():
            _assert_assets_shards(value)
            assert not value.is_deleted()


def _record_profiled_executables(
    *, monkeypatch: pytest.MonkeyPatch, profiles: dict
) -> None:
    """Record the declared output layout of every chunk-profiled executable.

    A budgeted call that reuses an identical abstract chunk profile does not
    recompile its declared bodies, so `prepare_abstract` fires only on the call
    that builds a given profile. Read the declared layouts off the selected
    chunk profile itself, which every call resolves before it admits and
    dispatches: the guarded contract is that dispatch executes exactly the
    stages that profile declared, never that the profile happened to be
    compiled during this call.
    """
    runtime_class = simulation_runtime.SimulationRuntime
    prepare = runtime_class.prepare_abstract
    profile_chunk = chunk_admission._ChunkProfiler.profile_widths
    selected_profiles: list[Any] = []

    def record_executable(executable: Any) -> None:
        assert isinstance(executable, jax.stages.Compiled)
        profiles[id(executable)] = tuple(
            (leaf.shape, leaf.dtype, leaf.sharding)
            for leaf in jax.tree.leaves(executable.out_info)
        )

    def observe_prepare(self: Any, **call: Any) -> Any:
        prepared = prepare(self, **call)
        record_executable(prepared.executable)
        return prepared

    def observe_chunk_profile(self: Any, **call: Any) -> Any:
        profile = profile_chunk(self, **call)
        # Keep the profile alive, so no stage executable it declares can be
        # collected and have its identity recycled while `profiles` is read.
        selected_profiles.append(profile)
        for stage in (*profile.stages, *profile.host_stages):
            record_executable(stage.executable)
        return profile

    monkeypatch.setattr(runtime_class, "prepare_abstract", observe_prepare)
    monkeypatch.setattr(
        chunk_admission._ChunkProfiler, "profile_widths", observe_chunk_profile
    )


def _observe_subject_programs(*, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Observe required input shards and the exact inferred output layouts."""
    runtime_class = simulation_runtime.SimulationRuntime
    place = simulation_runtime.place_simulation_arguments
    dispatch = runtime_class.dispatch
    compiled_call = simulation_runtime.CompiledSimulationProgram.__call__
    profiles = {}
    placed_counts = []
    profiled_runs: list[int] = []
    _record_profiled_executables(monkeypatch=monkeypatch, profiles=profiles)

    def observe_place(**call: Any) -> Any:
        placed = place(**call)
        for name in call["subject_arg_names"]:
            for value in jax.tree.leaves(placed.get(name, ())):
                if not isinstance(value, jax.Array) or not value.ndim:
                    continue
                assert value.shape[0] == 24
                assert len(value.addressable_shards) == 8
                rows = []
                for shard in value.addressable_shards:
                    indices = np.arange(24)[shard.index[0]]
                    assert len(indices) == 3
                    rows.extend(indices.tolist())
                assert sorted(rows) == list(range(24))
                placed_counts.append(24)
        return placed

    def observe_dispatch(self: Any, **call: Any) -> Any:
        # Every dispatch must run a body the selected chunk profile declared.
        # Observe that through the executable it actually runs, rather than
        # through a compile that a warm profile legitimately skips.
        before = len(profiled_runs)
        result = dispatch(self, **call)
        assert len(profiled_runs) > before, (
            "Dispatch ran no chunk-profiled compiled program"
        )
        return result

    def observe_compiled(
        self: simulation_runtime.CompiledSimulationProgram, **call: Any
    ) -> Any:
        assert id(self.executable) in profiles, (
            "Dispatch selected an unprofiled executable"
        )
        profiled_runs.append(id(self.executable))
        expected = profiles[id(self.executable)]
        result = compiled_call(self, **call)
        for value, (shape, dtype, sharding) in zip(
            jax.tree.leaves(result), expected, strict=True
        ):
            assert value.shape == shape
            assert value.dtype == dtype
            assert sharding is not None
            assert value.sharding.is_equivalent_to(sharding, value.ndim)
        return result

    monkeypatch.setattr(simulation_runtime, "place_simulation_arguments", observe_place)
    monkeypatch.setattr(runtime_class, "dispatch", observe_dispatch)
    monkeypatch.setattr(
        simulation_runtime.CompiledSimulationProgram, "__call__", observe_compiled
    )
    return placed_counts


def _assert_seventeen_rows(
    *,
    model: Model,
    reference_model: Model,
    solution: Any,
    reference_solution: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect actual 24-row subject shards, then compare the 17 returned rows."""

    run_chunk = simulation._simulate_subject_chunk
    padded_calls = []

    def observe_chunk(**call: Any) -> Any:
        result = run_chunk(**call)
        assert call["original_n_subjects"] == 17
        assert call["n_subjects"] == 24
        assert result["r0"][0].V_arr.shape[0] == 24
        padded_calls.append(24)
        return result

    initial = {
        "pref_type": np.arange(17, dtype=np.int32) % 3,
        "spousal_income": (np.arange(17, dtype=np.int32) // 3) % 3,
        "assets": np.array([-4.0, 17.0, *np.linspace(-3, 18, 15)]),
        "regime_id": np.arange(17, dtype=np.int32) % 2,
        "age": np.zeros(17),
    }
    params = {"discount_factor": 0.5}
    for seed in (0, 42):
        with monkeypatch.context() as probe:
            probe.setattr(simulation, "_simulate_subject_chunk", observe_chunk)
            placed_counts = _observe_subject_programs(monkeypatch=probe)
            result = model.simulate(
                params=params,
                solution=solution,
                initial_conditions=initial,
                seed=seed,
                log_level="off",
            )
        assert placed_counts
        control = reference_model.simulate(
            params=params,
            solution=reference_solution,
            initial_conditions=initial,
            seed=seed,
            log_level="off",
        )
        assert result.n_subjects == control.n_subjects == 17
        assert jax.tree.structure(result.raw_results) == jax.tree.structure(
            control.raw_results
        )
        for got, want in zip(
            jax.tree.leaves(result.raw_results),
            jax.tree.leaves(control.raw_results),
            strict=True,
        ):
            if np.issubdtype(np.asarray(want).dtype, np.inexact):
                assert_agrees_to_ulp(got=got, expected=want, n_ulp=8)
            else:
                np.testing.assert_array_equal(got, want)
    assert padded_calls == [24, 24]


def test_continuous_identity_still_materializes_full_continuation(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_eight()
    transfers = []
    apply = value_transfer.apply_value_transfer

    def observe(**call: Any) -> Any:
        copied = apply(**call)
        if call["transfer"].source.channel is ValueInputChannel.NEXT_REGIME_VALUE:
            assert call["transfer"].kind is ValueTransferKind.ALL_GATHER
            assert copied.is_fully_replicated
            transfers.append(call["transfer"])
        return copied

    monkeypatch.setattr(value_transfer, "apply_value_transfer", observe)
    solution = _model(devices=tuple(range(8)), identity=True).solve(
        params={"discount_factor": 0.5}, log_level="off"
    )
    assert len(transfers) == 3
    _assert_assets_shards(solution.values[0]["r0"])


@pytest.mark.parametrize("unsupported", ["irregular", "mixed_shards"])
def test_unsupported_continuous_sharding_refuses_at_construction(
    *, unsupported: str
) -> None:
    _require_eight()
    with pytest.raises(
        ExecutionPlanningError, match=r"[Ss]hard|LinSpacedGrid|continuous"
    ):
        _model(
            devices=tuple(range(8)),
            extra_shard=unsupported == "mixed_shards",
            grid=IrregSpacedGrid(points=tuple(range(-4, 20)))
            if unsupported == "irregular"
            else None,
        )


def test_budget_below_full_replica_refuses_without_transfer_and_preserves_owner(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_eight()
    params = {"discount_factor": 0.5}
    sufficient = _model(devices=tuple(range(8)))
    retained = sufficient.solve(params=params, log_level="off")
    originals = [
        (value, np.asarray(value).copy())
        for arrays in retained.values.values()
        for value in arrays.values()
    ]
    itemsize = originals[0][1].dtype.itemsize
    constrained = _model(devices=tuple(range(8)), budget=216 * itemsize // 2)

    def forbidden(**call: Any) -> Any:
        del call
        pytest.fail(
            "A budget below one full continuation replica must refuse before copying"
        )

    with monkeypatch.context() as probe:
        probe.setattr(value_transfer, "apply_value_transfer", forbidden)
        with pytest.raises(ExecutionPlanningError):
            constrained.solve(params=params, log_level="off")
    # Recovery is an actual sufficient-budget solve with the old owner still live.
    recovered = sufficient.solve(params=params, log_level="off")
    _assert_assets_shards(recovered.values[0]["r0"])
    for value, snapshot in originals:
        assert not value.is_deleted()
        np.testing.assert_array_equal(value, snapshot)


def test_renamed_trailing_axis_maps_blocks_in_mesh_order() -> None:
    """Cheap canonical/layout check; the public producer is tested separately."""

    _require_eight()
    regime = Regime(
        transition=None,
        states={
            "liquid": LinSpacedGrid(start=-4, stop=19, n_points=24),
            "pref_type": DiscreteGrid(_Three),
            "spousal_income": DiscreteGrid(_Three),
        },
        functions={
            "utility": lambda liquid, pref_type, spousal_income: (
                liquid + pref_type + spousal_income
            )
        },
    )
    variables = from_regime(
        user_regime=regime, sharded_state_names=frozenset({"liquid"})
    )
    assert tuple(variables) == ("pref_type", "spousal_income", "liquid")
    mesh_devices = tuple(reversed(jax.devices()))
    sharding = jax.NamedSharding(
        jax.sharding.Mesh(np.array(mesh_devices), ("liquid",)),
        jax.P(None, None, "liquid"),
    )
    index_map = sharding.devices_indices_map((3, 3, 24))
    global_ids = np.arange(216).reshape(3, 3, 24)
    for position, device in enumerate(mesh_devices):
        np.testing.assert_array_equal(
            global_ids[index_map[device]],
            global_ids[:, :, 3 * position : 3 * (position + 1)],
        )


def _renamed_functions(*, regime: Regime) -> dict[str, Any]:
    """Rename arguments of the fixture's ordinary callable economic nodes."""
    result = {}
    for key, function in regime.functions.items():
        assert callable(function)
        result[key] = rename_arguments(function, mapper={"assets": "liquid"})
    return result


def _renamed_public_model(*, original: Model) -> Model:
    """Apply only a state-name bijection to the existing fixture declarations."""
    regimes = {}
    for source, name in enumerate(("r0", "r1")):
        regime = _regime(source=source, identity=False)
        regimes[name] = dataclasses.replace(
            regime,
            functions=_renamed_functions(regime=regime),
            state_transitions={
                "liquid" if key == "assets" else key: law
                for key, law in regime.state_transitions.items()
            },
        )
    terminal = original.user_regimes["terminal"]
    regimes["terminal"] = dataclasses.replace(
        terminal,
        states={},
        functions=_renamed_functions(regime=terminal),
    )
    return Model(
        regimes=regimes,
        states={
            "liquid" if key == "assets" else key: grid
            for key, grid in terminal.states.items()
        },
        ages=original.ages,
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(
            devices=tuple(reversed(range(8))),
            sharded_states=("liquid",),
            axis_widths={"action_product": 3, "cell": 9, "subject": 432},
            device_memory_bytes=2**30,
        ),
    )


def _assert_renamed_values(*, original: Any, renamed: Any) -> None:
    exact, _ = _reference()
    assert set(renamed.values) == set(original.values)
    for period, values in original.values.items():
        assert set(renamed.values[period]) == set(values)
        for regime, value in values.items():
            _assert_assets_shards(value)
            actual = renamed.values[period][regime]
            assert actual.shape == (3, 3, 24)
            assert isinstance(actual.sharding, jax.NamedSharding)
            assert actual.sharding.spec == jax.P(None, None, "liquid")
            assert tuple(
                device.id for device in actual.sharding.mesh.devices.flat
            ) == tuple(range(8))
            assert len(actual.addressable_shards) == 8
            cells = np.arange(216).reshape(3, 3, 24)
            for shard in actual.addressable_shards:
                assert shard.data.shape == (3, 3, 3)
                position = shard.device.id
                np.testing.assert_array_equal(
                    cells[shard.index], cells[:, :, 3 * position : 3 * (position + 1)]
                )
            assert_agrees_to_ulp(got=actual, expected=value, n_ulp=8)
            expected = np.array(
                [float(x) for x in exact[period, regime].values()], dtype=actual.dtype
            ).reshape(3, 3, 24)
            assert_agrees_to_ulp(got=actual, expected=expected, n_ulp=8)


def test_public_state_rename_and_reversed_device_selection_preserve_solve_and_rng() -> (
    None
):
    """Identifier changes preserve the model; caller device order normalizes."""
    _require_eight()
    original = _model(devices=tuple(range(8)), widths=(3, 9))
    renamed = _renamed_public_model(original=original)
    assert original.execution_devices == renamed.execution_devices == tuple(range(8))
    params = {"discount_factor": 0.5}
    solution = original.solve(params=params, log_level="off")
    renamed_solution = renamed.solve(params=params, log_level="off")
    _assert_renamed_values(original=solution, renamed=renamed_solution)
    initial = {
        "assets": np.array([-4.0, 17.0, *np.linspace(-3, 18, 15)]),
        "pref_type": np.arange(17, dtype=np.int32) % 3,
        "spousal_income": (np.arange(17, dtype=np.int32) // 3) % 3,
        "regime_id": np.arange(17, dtype=np.int32) % 2,
        "age": np.zeros(17),
    }
    renamed_initial = {
        "liquid" if key == "assets" else key: value for key, value in initial.items()
    }
    for seed in (0, 42):
        result = original.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            seed=seed,
            log_level="off",
        )
        renamed_result = renamed.simulate(
            params=params,
            solution=renamed_solution,
            initial_conditions=renamed_initial,
            seed=seed,
            log_level="off",
        )
        assert result.n_subjects == renamed_result.n_subjects == 17
        assert isinstance(result.raw_results, MappingProxyType)
        assert isinstance(renamed_result.raw_results, MappingProxyType)
        assert tuple(renamed_result.raw_results) == tuple(result.raw_results)
        for regime, periods in renamed_result.raw_results.items():
            assert isinstance(periods, MappingProxyType)
            assert isinstance(result.raw_results[regime], MappingProxyType)
            assert tuple(periods) == tuple(result.raw_results[regime])
        normalized = MappingProxyType(
            {
                regime: MappingProxyType(
                    {
                        period: dataclasses.replace(
                            data,
                            states=MappingProxyType(
                                {
                                    "assets" if name == "liquid" else name: array
                                    for name, array in data.states.items()
                                }
                            ),
                        )
                        for period, data in periods.items()
                    }
                )
                for regime, periods in renamed_result.raw_results.items()
            }
        )
        assert jax.tree.structure(normalized) == jax.tree.structure(result.raw_results)
        for got, expected in zip(
            jax.tree.leaves(normalized),
            jax.tree.leaves(result.raw_results),
            strict=True,
        ):
            if np.issubdtype(np.asarray(expected).dtype, np.inexact):
                assert_agrees_to_ulp(got=got, expected=expected, n_ulp=8)
            else:
                np.testing.assert_array_equal(got, expected)
    _assert_renamed_values(original=solution, renamed=renamed_solution)


def test_shared_native_all_gather_releases_after_both_consumers_are_ready(  # noqa: C901, PLR0915
    *, monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    """Observe the real solve cache; wider-replica aliases have separate tests."""
    _require_eight()
    model = _model(devices=tuple(range(8)), widths=(3, 9))
    tracked: dict[str, Any] = {}
    materializations = []
    commits = []
    hits = []
    barriers = []
    apply = value_transfer.apply_value_transfer
    put = scheduler.PeriodTransferCache.put
    get = scheduler.PeriodTransferCache.get
    commit = scheduler.PeriodTransferCache.commit_consumer
    release = scheduler.release_closed_artifacts
    wait = jax.block_until_ready

    def observe_apply(**kwargs: Any) -> Any:
        result = apply(**kwargs)
        transfer = kwargs["transfer"]
        materializations.append((transfer.target, transfer.source_sharding))
        return result

    def observe_put(self: Any, **kwargs: Any) -> None:
        transfer, copied, stored = (
            kwargs["transfer"],
            kwargs["array"],
            kwargs["stored"],
        )
        if not tracked and transfer.kind is ValueTransferKind.ALL_GATHER:
            _assert_assets_shards(stored)
            assert copied.is_fully_replicated
            assert len(copied.addressable_shards) == 8
            assert all(
                shard.data.shape == (3, 3, 24) for shard in copied.addressable_shards
            )
            aliased = scheduler.shares_a_buffer(first=copied, second=stored)
            record_property("native_all_gather_aliases_source", aliased)
            assert not aliased
            tracked.update(
                cache=self,
                key=(transfer.target, transfer.source_sharding),
                copy=copied,
                source=stored,
                expected=np.asarray(stored).copy(),
            )
            np.testing.assert_array_equal(copied, stored)
        put(self, **kwargs)

    def observe_get(self: Any, **kwargs: Any) -> Any:
        result = get(self, **kwargs)
        if tracked and self is tracked["cache"] and result is not None:
            transfer = kwargs["transfer"]
            if (transfer.target, transfer.source_sharding) == tracked["key"]:
                assert result is tracked["copy"]
                assert not result.is_deleted()
                hits.append(result)
        return result

    def observe_commit(self: Any, **kwargs: Any) -> Any:
        selected = (
            tracked and self is tracked["cache"] and kwargs["key"] == tracked["key"]
        )
        if selected:
            assert not tracked["copy"].is_deleted()
        result = commit(self, **kwargs)
        if selected:
            commits.append(tracked["copy"].is_deleted())
            assert commits in ([False], [False, True])
            np.testing.assert_array_equal(tracked["source"], tracked["expected"])
        return result

    def observe_wait(tree: Any) -> Any:
        result = wait(tree)
        if tracked.get("releasing", False):
            ids = tuple(id(leaf) for leaf in jax.tree.leaves(tree))
            if ids == tracked["pending_ids"]:
                assert not tracked["copy"].is_deleted()
                barriers.append(ids)
        return result

    def observe_release(**kwargs: Any) -> Any:
        selected = (
            tracked
            and any(
                array is tracked["copy"]
                for array in kwargs["arrays_by_artifact"].values()
            )
            and tracked["key"] in kwargs["artifacts"]
        )
        if not selected:
            return release(**kwargs)
        assert commits == [False]
        pending = kwargs["pending_outputs"]
        assert len(pending) == 2
        tracked["pending_ids"] = tuple(
            id(leaf) for leaf in jax.tree.leaves(tuple(pending))
        )
        tracked["releasing"] = True
        before = kwargs["before_delete"]

        def observe_before_delete(*, arrays: Any) -> None:
            assert len(barriers) == 1
            assert any(array is tracked["copy"] for array in arrays)
            assert not tracked["copy"].is_deleted()
            np.testing.assert_array_equal(tracked["source"], tracked["expected"])
            if before is not None:
                before(arrays=arrays)

        kwargs["before_delete"] = observe_before_delete
        try:
            return release(**kwargs)
        finally:
            tracked["releasing"] = False

    with monkeypatch.context() as probe:
        probe.setattr(value_transfer, "apply_value_transfer", observe_apply)
        probe.setattr(scheduler.PeriodTransferCache, "put", observe_put)
        probe.setattr(scheduler.PeriodTransferCache, "get", observe_get)
        probe.setattr(scheduler.PeriodTransferCache, "commit_consumer", observe_commit)
        probe.setattr(scheduler, "release_closed_artifacts", observe_release)
        probe.setattr(jax, "block_until_ready", observe_wait)
        solution = model.solve(params={"discount_factor": 0.5}, log_level="off")
    assert tracked
    assert materializations.count(tracked["key"]) == 1
    assert hits
    assert commits == [False, True]
    assert len(barriers) == 1
    assert tracked["copy"].is_deleted()
    _assert_assets_shards(tracked["source"])
    np.testing.assert_array_equal(tracked["source"], tracked["expected"])
    np.testing.assert_array_equal(solution.values[2]["terminal"], tracked["expected"])
