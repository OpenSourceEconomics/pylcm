"""Model-building helpers hand back immutable containers.

Each helper returns a `MappingProxyType`, a `tuple` or a `frozenset`, never the
mutable `dict`, `list` or `set` it assembles internally.
"""

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from _lcm.model_graph import (
    _declared_gated_edges,
    _derived_target_ages,
    _single_destination_laws,
    _targets_by_period,
    _transition_laws,
)
from _lcm.params.processing import (
    _edge_arg_names,
    _edge_slot_hints,
    _edges_branch,
    _leaf_paths,
    _validated_arg_names,
    find_param_candidates,
    get_flat_param_names,
)
from _lcm.regime_building.broadcast import (
    _composed_resources_edge,
    _draw_law_roots,
    _incoming_edge_roots,
    _joint_phase_closure,
    _law_roots,
    _merge_one_slot,
    _model_slot_value_errors,
    _needed_names,
    _phase_fixed_point,
    _transition_roots,
    _valuation_roots,
    _value_aware_roots,
)
from _lcm.regime_building.diagnostics import _wrap_with_reduction
from _lcm.regime_building.fixed_components import _collect_groups, _create_splits
from _lcm.regime_building.invariant_blocking import (
    _block_major_failures,
    _regime_failures,
    _request_failures,
)
from _lcm.regime_building.invariant_components import (
    _channel_refusals,
    _identity_law_states,
)
from _lcm.regime_building.phases import normalize_regime_phases
from lcm import AgeGrid, Model, Transition
from lcm_examples.tiny import get_model

_IMMUTABLE = (MappingProxyType, tuple, frozenset)


@pytest.fixture(scope="module")
def model() -> Model:
    return get_model()


def _regime_cases(model: Model) -> dict[str, Callable[[], object]]:
    name = "working_life"
    regime = model.user_regimes[name]
    laws = model.graph.laws
    spec = normalize_regime_phases(regime, law=laws[name])
    specs = {
        regime_name: normalize_regime_phases(user_regime, law=laws[regime_name])
        for regime_name, user_regime in model.user_regimes.items()
    }
    unkept = dict.fromkeys(model.user_regimes, frozenset())
    shared: dict[str, Any] = {
        "specs": specs,
        "user_regimes": model.user_regimes,
        "laws": laws,
        "broadcast_variables": unkept,
        "koopmans_aggregator": lambda: 0.0,
        "kept": unkept,
        "all_regime_names": frozenset(model.user_regimes),
        "ages": None,
        "active_periods_by_regime": None,
    }
    return {
        "valuation_roots": lambda: _valuation_roots(
            regime=regime, law=laws[name], phase="solve", koopmans_aggregator=None
        ),
        "transition_roots": lambda: _transition_roots(
            regime=regime, law=laws[name], phase="solve"
        ),
        "value_aware_roots": lambda: _value_aware_roots(regime=regime),
        "incoming_edge_roots": lambda: _incoming_edge_roots(
            regime_name=name, laws=laws
        ),
        "law_roots": lambda: _law_roots(
            phase_slice=spec.solution,
            candidate_targets=frozenset(model.user_regimes),
            kept=unkept,
        ),
        "draw_law_roots": lambda: _draw_law_roots(
            phase_slice=spec.solution, regime=regime, reads=()
        ),
        "composed_resources_edge": lambda: _composed_resources_edge(
            user_regime=regime, pool={}
        ),
        "needed_names": lambda: _needed_names(
            phase_slice=spec.solution,
            regime_name=name,
            user_regime=regime,
            laws=laws,
            phase_name="solution",
            koopmans_aggregator=lambda: 0.0,
            candidate_targets=frozenset(model.user_regimes),
            kept=unkept,
            ages=None,
            active_periods=None,
        ),
        "phase_fixed_point": lambda: _phase_fixed_point(
            **shared, phase_name="solution"
        ),
        "joint_phase_closure": lambda: _joint_phase_closure(**shared),
        "identity_law_states": lambda: _identity_law_states(specs),
        "regime_failures": lambda: _regime_failures(
            regime_name=name, regime=regime, law=laws[name], carried=()
        ),
        "edge_arg_names": lambda: _edge_arg_names(model._regimes),
        "edges_branch": lambda: _edges_branch(model._regimes),
        "collect_groups": lambda: _collect_groups(
            regimes=model.user_regimes, state_transitions={}
        ),
    }


_REGIME_CASES = (
    "valuation_roots",
    "transition_roots",
    "value_aware_roots",
    "incoming_edge_roots",
    "law_roots",
    "draw_law_roots",
    "composed_resources_edge",
    "needed_names",
    "phase_fixed_point",
    "joint_phase_closure",
    "identity_law_states",
    "regime_failures",
    "edge_arg_names",
    "edges_branch",
    "collect_groups",
)


@pytest.mark.parametrize("case", _REGIME_CASES)
def test_model_building_helper_returns_an_immutable_container(
    *, model: Model, case: str
) -> None:
    """Helpers reading a built model's regimes return immutable containers."""
    assert isinstance(_regime_cases(model)[case](), _IMMUTABLE)


_AGES = AgeGrid(start=0, inclusive_stop=2, step="Y")
_RESOLVED = {"b": frozenset({0, 1})}
_TEMPLATE = MappingProxyType({"utility": MappingProxyType({"beta": "float"})})

_PLAIN_CASES: dict[str, Callable[[], object]] = {
    "merge_one_slot": lambda: _merge_one_slot(
        slot_name="functions",
        regime_name="a",
        regime_slot={"f": None},
        model_slot={},
    ),
    "model_slot_value_errors": lambda: _model_slot_value_errors(
        model_slots={"functions": {"f": None}}
    ),
    "channel_refusals": lambda: _channel_refusals(
        phase="solve", state_name="x", regimes={}, carriers=frozenset()
    ),
    "request_failures": lambda: _request_failures(
        user_regimes={}, block_widths={"x": 2}, sharded_states=frozenset()
    ),
    "block_major_failures": lambda: _block_major_failures(
        user_regimes={}, laws={}, block_widths={}
    ),
    "edge_slot_hints": lambda: _edge_slot_hints(
        unknown={"a__beta"}, template_flat={"edges__a__f__beta": "float"}
    ),
    "find_param_candidates": lambda: find_param_candidates(
        qname="a__utility__beta", params_flat={"beta": 0.9}
    ),
    "leaf_paths": lambda: _leaf_paths(_TEMPLATE),
    "validated_arg_names": lambda: _validated_arg_names(
        func_name="utility", params={"beta": "float"}, regime_name="a"
    ),
    "flat_param_names": lambda: get_flat_param_names(_TEMPLATE),
    "targets_by_period": lambda: _targets_by_period(resolved=_RESOLVED, ages=_AGES),
    "single_destination_laws": lambda: _single_destination_laws(
        source="a", resolved=_RESOLVED, ages=_AGES, side="solve"
    ),
    "transition_laws": lambda: _transition_laws(
        source="a",
        transition=Transition(law="b"),
        resolved=_RESOLVED,
        ages=_AGES,
        side="solve",
    ),
    "derived_target_ages": lambda: _derived_target_ages(
        transition=Transition(law="b"), ages=_AGES, fallback_phases=("solve",)
    ),
    "declared_gated_edges": lambda: _declared_gated_edges(
        source="a",
        declared={"solve": {}, "simulate": {}},
        resolved={"solve": MappingProxyType({}), "simulate": MappingProxyType({})},
    ),
}


@pytest.mark.parametrize("case", list(_PLAIN_CASES))
def test_declaration_helper_returns_an_immutable_container(case: str) -> None:
    """Helpers over plain declarations return immutable containers."""
    assert isinstance(_PLAIN_CASES[case](), _IMMUTABLE)


def test_create_splits_returns_immutable_splits_and_parts() -> None:
    """Both halves of the fixed-component inventory are immutable mappings."""
    splits, parts = _create_splits(regimes={}, groups={}, occupied=set())
    assert (type(splits), type(parts)) == (MappingProxyType, MappingProxyType)


def _intermediates() -> tuple:
    values = jnp.array([[1.0, jnp.nan], [2.0, 3.0]])
    feasible = jnp.array([[True, True], [False, True]])
    return (
        values,
        feasible,
        values,
        values,
        MappingProxyType({"b": jnp.full((2, 2), 0.5)}),
    )


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_diagnostic_reductions_are_immutable_at_both_levels(*, jit: bool) -> None:
    """The reductions and their per-target probabilities are immutable mappings."""
    reduce = _wrap_with_reduction(func=_intermediates, variable_names=("x", "a"))
    out = (jax.jit(reduce) if jit else reduce)()
    assert (type(out), type(out["regime_probs"])) == (
        MappingProxyType,
        MappingProxyType,
    )


def test_diagnostic_reductions_round_trip_through_jax_tree_utilities() -> None:
    """Flattening and unflattening the reductions restores the same mappings."""
    reduce = _wrap_with_reduction(func=_intermediates, variable_names=("x", "a"))
    out = reduce()
    leaves, treedef = jax.tree_util.tree_flatten(out)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert (type(rebuilt), dict(rebuilt["regime_probs"]).keys()) == (
        MappingProxyType,
        {"b"},
    )
