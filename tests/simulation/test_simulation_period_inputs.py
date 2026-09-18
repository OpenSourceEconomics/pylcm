"""Concrete raw operands and selected replay occurrences in one regime unit."""

from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.gated_edges import D_KEY_SUFFIX
from _lcm.simulation import gated_routing
from _lcm.simulation.period_inputs import (
    GATE_FOLD,
    GATE_ROUTE,
    POLICY_SCORE,
    acquire_gate_inputs,
    gate_reads,
    unit_value_reads,
)
from _lcm.simulation.value_reads import PeriodSimulationReads
from lcm.solver_api import SIMULATION_POLICY, ActionOutput
from tests.regime_building.test_collective_regime_simulate import _solve_dissolution


@pytest.fixture(scope="module")
def dissolution():
    """A real solved consent gate, with D true only at the middle wage."""
    return _solve_dissolution()


def test_gate_occurrences_resolve_to_actual_raw_operands(dissolution) -> None:
    _, regimes, _, _, values, flags = dissolution
    reads = gate_reads(
        regime=regimes["married"],
        name="married",
        period=0,
        values=values,
        flags=flags,
    )
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[0],),
        reads_by_unit={"married": reads},
        release_enabled=True,
    )
    operands = {}
    for family in (GATE_FOLD, GATE_ROUTE):
        selected = tuple(read for read in reads if read.source.core_key == family)
        edge_values, edge_flags = acquire_gate_inputs(
            reads=selected,
            owner=owner,
            name="married",
            values=values,
            flags=flags,
        )
        assert tuple(edge_values) == ("married_ir",)
        assert set(edge_values["married_ir"]) == {"married_ir", "single_f", "single_m"}
        assert edge_flags["married_ir"].dtype == jnp.dtype(bool)
        np.testing.assert_array_equal(edge_flags["married_ir"], [False, True, False])
        arguments = {"edge_values": edge_values, "edge_flags": edge_flags}
        for read in selected:
            assert read.source.argument is not None
            actual: object = arguments[read.source.argument]
            for component in read.source.path:
                assert isinstance(actual, Mapping)
                assert isinstance(component, str)
                actual = actual[component]
            original = (flags if read.source.argument == "edge_flags" else values)[
                read.target.period
            ][read.target.regime]
            assert actual is original
        operands[family] = arguments
    for reference in ("married_ir", "single_f", "single_m"):
        assert (
            operands[GATE_FOLD]["edge_values"]["married_ir"][reference]
            is (operands[GATE_ROUTE]["edge_values"]["married_ir"][reference])
        )
    owner.commit(unit="married", outputs=operands)
    owner.finish()


@pytest.mark.parametrize("observe_derived", [False, True])
def test_raw_boolean_gate_fold_and_route_preserve_the_dissolution_decision(
    *,
    dissolution,
    observe_derived: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, regimes, ids, params, values, flags = dissolution
    married = regimes["married"]
    reads = gate_reads(
        regime=married,
        name="married",
        period=0,
        values=values,
        flags=flags,
    )
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[0],),
        reads_by_unit={"married": reads},
        release_enabled=True,
    )
    snapshots = []
    barriers = []
    block = jax.block_until_ready

    def observed_block(tree):
        barriers.append(tree)
        return block(tree)

    monkeypatch.setattr(jax, "block_until_ready", observed_block)
    callback = snapshots.append if observe_derived else None
    fold_values, fold_flags = acquire_gate_inputs(
        reads=tuple(read for read in reads if read.source.core_key == GATE_FOLD),
        owner=owner,
        name="married",
        values=values,
        flags=flags,
    )
    folded = gated_routing.simulation_gate_fold(
        regime=married,
        regime_name="married",
        regimes=regimes,
        period=0,
        next_regime_to_V_arr=values[1],
        base_state_action_spaces={
            name: regime.solution.state_action_space(regime_params=params[name])
            for name, regime in regimes.items()
        },
        edge_values=fold_values,
        edge_flags=fold_flags,
        flat_params=params,
        on_derived=callback,
    )
    # A pointwise edge carries target stakeholder operands followed by raw D.
    # Its decision evaluates the fallback after interpolation, not in this stack.
    np.testing.assert_array_equal(
        folded["married_ir"], [[2, 1, 0], [-np.inf, -np.inf, 1], [6, 3, 0]]
    )
    fold_snapshots = list(snapshots)
    route_values, route_flags = acquire_gate_inputs(
        reads=tuple(read for read in reads if read.source.core_key == GATE_ROUTE),
        owner=owner,
        name="married",
        values=values,
        flags=flags,
    )
    roles = jnp.full(3, married.stakeholder_names_to_ids["f"], dtype=jnp.int32)
    states, routed_ids, routed_roles = gated_routing.simulation_gate_route(
        regime=married,
        fold_period=1,
        edge_values=route_values,
        edge_flags=route_flags,
        next_states=MappingProxyType(
            {
                "married_ir": MappingProxyType({"wage": jnp.array([1.0, 2.0, 3.0])}),
                "single_f": MappingProxyType({"wage": jnp.full(3, -999.0)}),
                "single_m": MappingProxyType({"wage": jnp.full(3, -999.0)}),
            }
        ),
        regime_names_to_ids=ids,
        new_subject_regime_ids=jnp.full(3, ids["married_ir"], dtype=jnp.int32),
        subjects_in_regime=jnp.ones(3, dtype=bool),
        flat_params=params,
        own_stakeholder=roles,
        new_own_stakeholder=roles,
        on_derived=callback,
    )
    np.testing.assert_array_equal(
        routed_ids, [ids["married_ir"], ids["single_f"], ids["married_ir"]]
    )
    np.testing.assert_array_equal(states["single_f"]["wage"], [-999.0, 2.0, -999.0])
    np.testing.assert_array_equal(states["single_m"]["wage"], [-999.0, 2.0, -999.0])
    assert (
        route_flags["married_ir"] is fold_flags["married_ir"] is flags[1]["married_ir"]
    )
    assert flags[1]["married_ir"].dtype == jnp.dtype(bool)
    assert len(barriers) == (2 if observe_derived else 0)
    if observe_derived:
        assert set(fold_snapshots[-1]) == {"next_values"}
        assert snapshots[-1] == {}
        fold_mapping = next(
            snapshot["same_period_mappings"]["married_ir"]
            for snapshot in fold_snapshots
            if "same_period_mappings" in snapshot
        )
        route_mapping = next(
            snapshot["same_period_mappings"]["married_ir"]
            for snapshot in snapshots[len(fold_snapshots) :]
            if "same_period_mappings" in snapshot
        )
        d_key = f"married_ir{D_KEY_SUFFIX}"
        assert fold_mapping[d_key].dtype == jnp.asarray(0.0).dtype
        assert route_mapping[d_key].dtype == jnp.asarray(0.0).dtype
        assert fold_mapping[d_key] is not route_mapping[d_key]
        assert (
            fold_mapping["single_f"]
            is route_mapping["single_f"]
            is values[1]["single_f"]
        )
    owner.commit(unit="married", outputs=(folded, states, routed_ids, routed_roles))
    owner.finish()


@pytest.mark.parametrize("external", [False, True])
def test_selected_external_reader_omits_unused_legacy_policy_occurrences(
    *,
    dissolution,
    external: bool,
) -> None:
    _, regimes, _, _, values, flags = dissolution

    def reader(*, states, fallback_actions):
        del states
        return ActionOutput(actions=fallback_actions)

    reads = unit_value_reads(
        regime=regimes["married"],
        name="married",
        period=0,
        values=values,
        flags=flags,
        policy={"legacy": jnp.array([1.0, 2.0])},
        reader=reader if external else None,
    )
    policy_reads = [
        read for read in reads if read.target.artifact_key == SIMULATION_POLICY
    ]
    score_reads = [read for read in reads if read.source.core_key == POLICY_SCORE]
    if external:
        assert not policy_reads
        assert not score_reads
        assert any(
            read.source.core_key == "simulation_external_replay_score" for read in reads
        )
    else:
        assert len(policy_reads) == 1
        assert policy_reads[0].source.argument == "payload"
        assert policy_reads[0].source.path == ("legacy",)
        assert score_reads
