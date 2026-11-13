"""Original primitive metadata controls fibre reads and rectangle exclusions."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import textwrap
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.fibre_source import FibreSource, TargetSupport
from _lcm_regime_finder.fibre_facts import FibreLiveness, calculate_fibre_liveness
from _lcm_regime_finder.records import JSONValue
from tests.regime_finder.test_model_adapter import function, grid, snapshot
from tests.regime_finder.test_query_owners import policy_snapshot
from tests.regime_finder.test_round4_semantic import captured_gate_graph
from tests.regime_finder.test_graph_builtin_processes import _target_process_capture

if TYPE_CHECKING:
    from lcm.typing import RegimeName


def _object(value: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(value, dict)
    return value


def _name(value: JSONValue) -> str:
    assert isinstance(value, str)
    return value


def _rows(value: JSONValue) -> list[JSONValue]:
    assert isinstance(value, list)
    return value


def _original() -> dict[str, JSONValue]:
    source = snapshot()
    _fields(source)["constraints"] = {}
    return source


def _fields(
    source: dict[str, JSONValue], regime: RegimeName = "original"
) -> dict[str, JSONValue]:
    return _object(_object(_object(source["user_regimes"])[regime])["fields"])


def _normalized(source: dict[str, JSONValue]) -> dict[str, JSONValue]:
    return normalize_declarations(source)


def _source(original: dict[str, JSONValue]) -> FibreSource:
    assert _rows(_normalized(original)["cells"])
    return FibreSource(original)


def _cell_id(
    source: dict[str, JSONValue], period: int = 0, regime: RegimeName = "original"
) -> str:
    cells = [
        _object(row)
        for row in _rows(_normalized(source)["cells"])
        if _object(row)["period"] == period and _object(row)["regime"] == regime
    ]
    assert len(cells) == 1
    identifier = cells[0]["id"]
    assert isinstance(identifier, str)
    return identifier


def _relation_id(
    source: dict[str, JSONValue], *, kind: str = "transition", phase: str = "solve"
) -> str:
    cid = _cell_id(source)
    relations = [
        _object(row)
        for row in _rows(_normalized(source)["relations"])
        if _object(row)["source"] == cid
        and _object(row)["kind"] == kind
        and _object(row)["phase"] == phase
    ]
    assert len(relations) == 1
    identifier = relations[0]["id"]
    assert isinstance(identifier, str)
    return identifier


def test_source_ids_are_original_declaration_order_and_input_is_unchanged() -> None:
    original = _original()
    before = deepcopy(original)
    declarations = _normalized(original)
    source = _source(original)
    assert source.get_cell_ids() == tuple(
        _object(c)["id"] for c in _rows(declarations["cells"])
    )
    assert source.get_relation_ids() == tuple(
        _object(r)["id"] for r in _rows(declarations["relations"])
    )
    assert original == before


def test_solve_only_and_simulation_only_consumers_both_remain_live() -> None:
    original = _original()
    _object(_fields(original)["functions"])["utility"] = {
        "kind": "dataclass",
        "type": "lcm.Phased",
        "fields": {
            "solve": function("def utility(x):\n    return x\n", ["x"]),
            "simulate": function(
                "def utility(history):\n    return history\n", ["history"]
            ),
        },
    }
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset({"x", "history"})


def test_reported_output_is_a_direct_consumer() -> None:
    original = _original()
    fields = _fields(original)
    _object(fields["functions"])["reported_history"] = function(
        "def reported_history(history):\n    return history\n", ["history"]
    )
    fields["output_only_functions"] = ["reported_history"]
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset({"x", "history"})


def test_transitive_function_and_category_code_alias_keep_the_state() -> None:
    original = _original()
    functions = _object(_fields(original)["functions"])
    functions["utility"] = function(
        "def utility(intermediate):\n    return intermediate\n", ["intermediate"]
    )
    functions["intermediate"] = function(
        "def intermediate(history_code):\n    return history_code\n", ["history_code"]
    )
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset({"history"})


@pytest.mark.parametrize("registered", [True, False])
def test_only_registered_conditional_shock_laws_are_direct_consumers(
    registered: bool,
) -> None:
    original = _original()
    _fields(original)["shocks"] = ["offers"] if registered else []
    original["shocks"] = {
        "offers": {
            "kind": "dataclass",
            "type": "lcm.shocks.ShockGroup",
            "fields": {
                "shocks": {
                    "offer": {
                        "kind": "dataclass",
                        "type": "lcm.shocks.Discrete",
                        "fields": {
                            "probs": function(
                                "def probs(history):\n    return history\n", ["history"]
                            )
                        },
                    }
                }
            },
        }
    }
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset(
        {"x", "history"} if registered else {"x"}
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [(0, frozenset({"history"})), (1, frozenset({"history", "x"}))],
)
def test_status_branch_only_removes_ordinary_state_with_joint_proof(
    status: int, expected: frozenset[str]
) -> None:
    original = _original()
    _object(_fields(original)["functions"])["utility"] = function(
        "def utility(history, x):\n    return 0 if history == 0 else x\n",
        ["history", "x"],
    )
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original),
        statuses={"history": status},
        removal_order=("history",),
    )
    assert result.live_coordinates == expected
    fibre = {"history": status, "x": 0}
    for history in (0, 1):
        for wealth in (0, 1):
            candidate = {"history": history, "x": wealth}
            if all(candidate[k] == fibre[k] for k in result.live_coordinates):
                assert (0 if history == 0 else wealth) == (0 if status == 0 else 0)


def test_date_specialization_uses_original_period_and_keeps_both_branches_where_needed() -> (
    None
):
    original = _original()
    _object(_fields(original)["functions"])["utility"] = function(
        "def utility(history, x, period):\n    return x if period == 0 else history\n",
        ["history", "x", "period"],
    )
    source = _source(original)
    assert source.get_direct_reads(
        cell_id=_cell_id(original, 0), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"x"})
    assert source.get_direct_reads(
        cell_id=_cell_id(original, 1), statuses={}, removal_order=()
    ).live_coordinates == frozenset({"history"})


def test_opaque_source_retains_all_states_and_smaller_producer_reads_are_not_authority() -> (
    None
):
    original = _original()
    callback = function("def utility(x):\n    return unknown_operation(x)\n", ["x"])
    callback.update(reads=[], expression="0", status="proved")
    _object(_fields(original)["functions"])["utility"] = callback
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset({"x", "history"})


@pytest.mark.parametrize("missing", [False, True])
def test_update_reads_follow_original_law_and_missing_law_is_conservative(
    missing: bool,
) -> None:
    original = _original()
    transitions = _object(_fields(original)["state_transitions"])
    if missing:
        transitions.pop("x")
    else:
        transitions["x"] = function(
            "def next_x(history):\n    return history\n", ["history"]
        )
    result = _source(original).get_update_reads(
        relation_id=_relation_id(original),
        target_coordinate="x",
        statuses={},
        removal_order=(),
    )
    assert result.live_coordinates == frozenset(
        {"x", "history"} if missing else {"history"}
    )


def test_same_period_value_read_incidence_preserves_projector_inputs() -> None:
    original = _original()
    _fields(original)["same_period_refs"] = {
        "current": {
            "kind": "dataclass",
            "type": "lcm.ProjectedRegimeValue",
            "fields": {
                "regime": "original",
                "kind": "Raw",
                "field": "value",
                "observes": [],
                "projection": {
                    "x": function(
                        "def project(history):\n    return history\n", ["history"]
                    ),
                    "history": function(
                        "def keep(history):\n    return history\n", ["history"]
                    ),
                },
            },
        }
    }
    source = _source(original)
    relation = _relation_id(original, kind="value_read")
    assert relation in source.get_relation_ids()
    assert source.get_update_reads(
        relation_id=relation, target_coordinate="x", statuses={}, removal_order=()
    ).live_coordinates == frozenset({"history"})


@pytest.mark.parametrize(
    ("stochastic", "expected"),
    [
        (False, TargetSupport(frozenset({0}), True)),
        (True, TargetSupport(frozenset({0, 1}), False)),
    ],
)
def test_stochastic_scalar_zero_is_not_a_deterministic_target_code(
    stochastic: bool, expected: TargetSupport
) -> None:
    original = _original()
    callback = function("def next_x():\n    return 0\n", [])
    _object(_fields(original)["state_transitions"])["x"] = (
        {
            "kind": "dataclass",
            "type": "lcm.StochasticTransition",
            "fields": {"func": callback},
        }
        if stochastic
        else callback
    )
    result = _source(original).get_target_support(
        relation_id=_relation_id(original), target_coordinate="x", statuses={}
    )
    assert result == expected


def test_unknown_target_support_keeps_entire_original_grid() -> None:
    original = _original()
    _object(_fields(original)["state_transitions"])["x"] = function(
        "def next_x(x):\n    return unknown_operation(x)\n", ["x"]
    )
    result = _source(original).get_target_support(
        relation_id=_relation_id(original), target_coordinate="x", statuses={}
    )
    assert result == TargetSupport(frozenset({0, 1}), False)


@pytest.mark.parametrize(
    ("rectangle", "excluded"),
    [
        ({"x": (0,), "history": (0,)}, frozenset({0})),
        ({"x": (0, 1), "history": (0, 1)}, frozenset()),
    ],
)
def test_action_exclusion_checks_full_rectangle_including_nonmember_points(
    rectangle: Mapping[str, tuple[int, ...]], excluded: frozenset[int]
) -> None:
    original = _original()
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(choice, x, history):\n    return (choice != 0) | (x != history)\n",
            ["choice", "x", "history"],
        )
    }
    result = _source(original).get_excluded_action_values(
        cell_id=_cell_id(original), action="choice", status_values=rectangle
    )
    assert result == excluded
    for action in result:
        assert all(
            not ((action != 0) or (x != history))
            for x in rectangle["x"]
            for history in rectangle["history"]
        )
    member_fibres = ((0, 0), (1, 1))
    assert all(not ((0 != 0) or (x != history)) for x, history in member_fibres)
    assert (0 != 0) or (0 != 1)


def test_other_actions_ordinary_states_and_runtime_inputs_stay_free_for_exclusion() -> (
    None
):
    original = _original()
    _object(_fields(original)["actions"])["other"] = grid([0, 1])
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(choice, other, x, beta):\n    return (choice != 0) | (other == 1) | (x == 1) | (beta > 0)\n",
            ["choice", "other", "x", "beta"],
        )
    }
    result = _source(original).get_excluded_action_values(
        cell_id=_cell_id(original), action="choice", status_values={"history": (0,)}
    )
    assert result == frozenset()


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "cell",
        "relation",
        "target_coordinate",
        "status_coordinate",
        "status_code",
        "action",
        "rectangle_code",
        "empty_rectangle",
    ],
)
def test_malformed_original_or_lookup_fails_clearly(defect: str) -> None:
    original = _original()
    if defect == "schema":
        original["schema"] = "producer-declarations"
        with pytest.raises(ValueError):
            FibreSource(original)
        return
    source = _source(original)
    with pytest.raises(ValueError):
        if defect == "cell":
            source.get_direct_reads(cell_id="missing", statuses={}, removal_order=())
        elif defect == "relation":
            source.get_update_reads(
                relation_id="missing",
                target_coordinate="x",
                statuses={},
                removal_order=(),
            )
        elif defect == "target_coordinate":
            source.get_update_reads(
                relation_id=_relation_id(original),
                target_coordinate="missing",
                statuses={},
                removal_order=(),
            )
        elif defect in {"status_coordinate", "status_code"}:
            statuses = (
                {"missing": 0} if defect == "status_coordinate" else {"history": 7}
            )
            source.get_direct_reads(
                cell_id=_cell_id(original),
                statuses=statuses,
                removal_order=tuple(statuses),
            )
        else:
            source.get_excluded_action_values(
                cell_id=_cell_id(original),
                action="missing" if defect == "action" else "choice",
                status_values={"history": () if defect == "empty_rectangle" else (7,)},
            )


@pytest.mark.parametrize("composed", [False, True])
def test_update_draws_use_each_target_and_phase_law_context(composed: bool) -> None:
    conditions = {
        "target_a": ("a_solve", "a_simulate"),
        "target_b": ("b_solve", "b_simulate"),
    }
    original = _target_process_capture(
        source_condition="source_history",
        target_conditions=conditions,
        composed=composed,
    )
    declarations = _normalized(original)
    cells = {
        _name(_object(cell)["id"]): _object(cell)
        for cell in _rows(declarations["cells"])
    }
    source = _source(original)
    matched = 0
    for raw in _rows(declarations["relations"]):
        relation = _object(raw)
        if relation["kind"] != "transition":
            continue
        targets = _rows(relation["targets"])
        assert len(targets) == 1
        target_id = targets[0]
        assert isinstance(target_id, str)
        target = cells[target_id]["regime"]
        assert isinstance(target, str)
        condition = conditions[target][0 if relation["phase"] == "solve" else 1]
        relation_id = relation["id"]
        assert isinstance(relation_id, str)
        result = source.get_update_reads(
            relation_id=relation_id,
            target_coordinate="x" if composed else "z",
            statuses={},
            removal_order=(),
        )
        assert result.live_coordinates == frozenset({condition})
        assert "source_history" not in result.live_coordinates
        matched += 1
    assert matched == 4


def test_raising_original_callback_is_inspected_without_execution() -> None:
    original = _original()
    _object(_fields(original)["functions"])["utility"] = function(
        'def utility(x, history):\n    raise AssertionError("Economic callback executed")\n',
        ["x", "history"],
    )
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses={}, removal_order=()
    )
    assert result.live_coordinates == frozenset({"x", "history"})


@pytest.mark.parametrize(
    ("fixed", "expected"),
    [
        ({}, TargetSupport(frozenset({0, 1}), False)),
        ({"history": 1}, TargetSupport(frozenset({1}), True)),
    ],
)
def test_typed_identity_support_uses_only_the_fixed_original_status(
    fixed: Mapping[str, int], expected: TargetSupport
) -> None:
    original = _original()
    _object(_fields(original)["state_transitions"])["history"] = {
        "kind": "object",
        "type": "_lcm.identity_transition._IdentityTransition",
        "fields": {"state_name": "history"},
    }
    result = _source(original).get_target_support(
        relation_id=_relation_id(original), target_coordinate="history", statuses=fixed
    )
    assert result == expected


def test_missing_update_does_not_invent_identity_target_support() -> None:
    original = _original()
    _object(_fields(original)["state_transitions"]).pop("history")
    result = _source(original).get_target_support(
        relation_id=_relation_id(original),
        target_coordinate="history",
        statuses={"history": 1},
    )
    assert result == TargetSupport(frozenset({0, 1}), False)


def test_numeric_discrete_action_values_are_distinct_from_excluded_codes() -> None:
    original = _original()
    _object(_fields(original)["actions"])["choice"] = grid([10.0, 20.0])
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(choice_code):\n    return choice_code != 0\n", ["choice_code"]
        )
    }
    result = _source(original).get_excluded_action_values(
        cell_id=_cell_id(original), action="choice", status_values={}
    )
    assert result == frozenset({0})


def test_native_capture_identity_and_raising_callbacks_are_metadata_only() -> None:
    script = textwrap.dedent(
        """
        from lcm import DiscreteGrid, InitialNodes, Model, Periods, Regime, categorical, fixed_transition
        from lcm.typing import DiscreteState, FloatND, ScalarInt
        from _lcm_regime_finder.declarations import normalize_declarations
        from _lcm_regime_finder.fibre_source import FibreSource, TargetSupport
        from _lcm_regime_finder.model_snapshot import read_model

        @categorical(ordered=False)
        class Binary:
            low: ScalarInt
            high: ScalarInt

        @categorical(ordered=False)
        class Regimes:
            alive: ScalarInt
            final: ScalarInt

        def utility(x: DiscreteState, history: DiscreteState) -> FloatND:
            raise AssertionError("Economic callback executed")

        model = Model(
            n_periods=2,
            regime_id_class=Regimes,
            regimes={"alive": Regime(
                states={"x": DiscreteGrid(Binary), "history": DiscreteGrid(Binary)},
                functions={"utility": utility},
                state_transitions={"x": fixed_transition("x"), "history": fixed_transition("history")},
            ), "final": Regime(
                states={"x": DiscreteGrid(Binary), "history": DiscreteGrid(Binary)},
                functions={"utility": utility},
            )},
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={"alive": {"final": Periods(values=(0,))}},
            enable_jit=False,
        )
        original = read_model(model)
        declarations = normalize_declarations(original)
        source = FibreSource(original)
        cell = next(row for row in declarations["cells"] if row["period"] == 0)
        relation = next(row for row in declarations["relations"] if row["source"] == cell["id"] and row["phase"] == "solve")
        assert source.get_direct_reads(cell_id=cell["id"], statuses={}, removal_order=()).live_coordinates == frozenset({"x", "history"})
        assert source.get_target_support(relation_id=relation["id"], target_coordinate="history", statuses={"history": 1}) == TargetSupport(frozenset({1}), True)
        """
    )
    command = [
        "pixi",
        "run",
        "--as-is",
        "--manifest-path",
        str(Path(__file__).resolve().parents[2] / "pyproject.toml"),
        "-e",
        os.environ.get("PIXI_ENVIRONMENT_NAME", "type-checking"),
        "python",
        "-c",
        script,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("slot", ["utility", "update"])
def test_unknown_process_namespace_does_not_grant_helper_input_authority(
    slot: str,
) -> None:
    original = _original()
    unknown: dict[str, JSONValue] = {
        "kind": "dataclass",
        "type": "_lcm.processes.CustomUnspecifiedProcess",
        "fields": {"helper": function("def helper(x):\n    return x\n", ["x"])},
    }
    if slot == "utility":
        _object(_fields(original)["functions"])["utility"] = unknown
    else:
        _object(_fields(original)["state_transitions"])["x"] = unknown
    source = _source(original)
    result = (
        source.get_direct_reads(
            cell_id=_cell_id(original), statuses={}, removal_order=()
        )
        if slot == "utility"
        else source.get_update_reads(
            relation_id=_relation_id(original),
            target_coordinate="x",
            statuses={},
            removal_order=(),
        )
    )
    assert result.live_coordinates == frozenset({"x", "history"})


@pytest.mark.parametrize(
    ("simulation_values", "expected"),
    [([0], frozenset({"history"})), ([0, 1], frozenset({"history", "x"}))],
)
@pytest.mark.parametrize("statuses", [{}, {"history": 1, "x": 0}])
def test_solve_imputation_reads_history_without_replacing_realised_carried_state(
    simulation_values: list[int], expected: frozenset[str], statuses: Mapping[str, int]
) -> None:
    original = _original()
    fields = _fields(original)
    _object(fields["states"])["x"] = {
        "kind": "dataclass",
        "type": "lcm.Phased",
        "fields": {
            "solve": function(
                "def imputed(history):\n    return history\n", ["history"]
            ),
            "simulate": grid(simulation_values),
        },
    }
    _object(fields["functions"])["utility"] = function(
        "def utility(x):\n    return x\n", ["x"]
    )
    result = _source(original).get_direct_reads(
        cell_id=_cell_id(original), statuses=statuses, removal_order=tuple(statuses)
    )
    assert result.live_coordinates == expected


@pytest.mark.parametrize("missing_update", [False, True])
def test_cell_facts_preserve_original_order_and_literal_multistep_dependencies(
    missing_update: bool,
) -> None:
    original = _original()
    transitions = _object(_fields(original)["state_transitions"])
    if missing_update:
        transitions.pop("x")
    else:
        transitions["x"] = function(
            "def next_x(history):\n    return history\n", ["history"]
        )
    declarations = _normalized(original)
    source = _source(original)
    facts = source.get_cell_facts()
    original_ids = tuple(
        _name(_object(cell)["id"]) for cell in _rows(declarations["cells"])
    )
    assert tuple(fact.identifier for fact in facts) == original_ids
    assert all(fact.coordinates == frozenset({"x", "history"}) for fact in facts)
    assert all(fact.direct.live_coordinates == frozenset({"x"}) for fact in facts)
    assert all(fact.direct.removal_order == () for fact in facts)
    expected = tuple(
        FibreLiveness(
            identifier,
            frozenset({"x", "history"} if index < len(original_ids) - 1 else {"x"}),
        )
        for index, identifier in enumerate(original_ids)
    )
    assert calculate_fibre_liveness(facts) == expected
    first = facts[0]
    assert len(first.links) == 2
    assert {link.phase for link in first.links} == {"solve", "simulation"}
    for link in first.links:
        updates = {row.coordinate: row.reads.live_coordinates for row in link.updates}
        assert updates == {
            "x": frozenset({"x", "history"} if missing_update else {"history"}),
            "history": frozenset({"history"}),
        }


def test_joint_and_ordinary_producers_form_one_complete_link_per_phase() -> None:
    original = _original()
    fields = _fields(original)
    _object(fields["state_transitions"]).pop("x")
    fields["joint_transitions"] = {
        "original": {
            "lottery": {
                "kind": "dataclass",
                "type": "lcm.JointTransition",
                "fields": {
                    "support_size": 2,
                    "support": [0, 1],
                    "probabilities": function(
                        "def probabilities():\n    return [0.5, 0.5]\n", []
                    ),
                    "outputs": {
                        "x": function(
                            "def draw(history, lottery):\n    return history\n",
                            ["history", "lottery"],
                        )
                    },
                },
            }
        }
    }
    declarations = _normalized(original)
    source = _source(original)
    first = source.get_cell_facts()[0]
    assert len(first.links) == 2
    assert {link.phase for link in first.links} == {"solve", "simulation"}
    for link in first.links:
        expected_relations = tuple(
            _name(_object(row)["id"])
            for row in _rows(declarations["relations"])
            if _object(row)["source"] == first.identifier
            and _object(row)["targets"] == [link.target]
            and _object(row)["phase"] == link.phase
        )
        assert link.source_relations == expected_relations
        assert len(link.source_relations) == 2
        assert {update.coordinate for update in link.updates} == {"x", "history"}
        assert {
            update.coordinate: update.reads.live_coordinates for update in link.updates
        } == {"x": frozenset({"history"}), "history": frozenset({"history"})}
    assert calculate_fibre_liveness(source.get_cell_facts())[
        0
    ].live_coordinates == frozenset({"x", "history"})


def test_cell_links_preserve_target_phase_specific_process_certificates() -> None:
    conditions = {
        "target_a": ("a_solve", "a_simulate"),
        "target_b": ("b_solve", "b_simulate"),
    }
    original = _target_process_capture(
        source_condition="source_history", target_conditions=conditions
    )
    declarations = _normalized(original)
    cells = {
        _name(_object(cell)["id"]): _object(cell)
        for cell in _rows(declarations["cells"])
    }
    facts = _source(original).get_cell_facts()
    source = next(
        fact for fact in facts if cells[fact.identifier]["regime"] == "source"
    )
    assert len(source.links) == 4
    for link in source.links:
        target = cells[link.target]
        regime = _name(target["regime"])
        assert link.phase in {"solve", "simulation"}
        expected_condition = conditions[regime][0 if link.phase == "solve" else 1]
        updates = {
            update.coordinate: update.reads.live_coordinates for update in link.updates
        }
        assert set(updates) == set(_object(target["states"]))
        assert updates["z"] == frozenset({expected_condition})
        assert all(
            not reads for coordinate, reads in updates.items() if coordinate != "z"
        )
    expected_source_live = frozenset(
        {"z", "a_solve", "a_simulate", "b_solve", "b_simulate"}
    )
    assert (
        next(
            row
            for row in calculate_fibre_liveness(facts)
            if row.identifier == source.identifier
        ).live_coordinates
        == expected_source_live
    )


@pytest.mark.parametrize("kind", ["value_read", "exit"])
def test_current_value_and_exit_targets_keep_explicit_projector_links(
    kind: str,
) -> None:
    original = _original()
    target: dict[str, JSONValue] = {
        "kind": "dataclass",
        "type": "lcm.ExitTarget" if kind == "exit" else "lcm.ProjectedRegimeValue",
        "fields": {
            "regime": "original",
            "kind": "Raw",
            "field": "value",
            "observes": [],
            "projection": {
                "x": function(
                    "def project(history):\n    return history\n", ["history"]
                ),
                "history": function(
                    "def keep(history):\n    return history\n", ["history"]
                ),
            },
        },
    }
    if kind == "exit":
        _fields(original)["exits"] = {
            "leave": {
                "kind": "dataclass",
                "type": "lcm.Exit",
                "fields": {
                    "targets": {"self": target},
                    "rule": "unilateral",
                    "observes": [],
                },
            }
        }
    else:
        _fields(original)["same_period_refs"] = {"current": target}
    declarations = _normalized(original)
    facts = _source(original).get_cell_facts()
    first = facts[0]
    desired = {
        _name(_object(row)["id"])
        for row in _rows(declarations["relations"])
        if _object(row)["source"] == first.identifier
        and _object(row)["kind"] == kind
        and _object(row)["targets"]
    }
    assert desired
    selected = tuple(
        link for link in first.links if set(link.source_relations) & desired
    )
    assert {rid for link in selected for rid in link.source_relations} == desired
    assert all(link.target == first.identifier for link in selected)
    assert all(
        {update.coordinate for update in link.updates} == {"x", "history"}
        for link in selected
    )
    assert all(
        next(
            update.reads.live_coordinates
            for update in link.updates
            if update.coordinate == "x"
        )
        == frozenset({"history"})
        for link in selected
    )
    assert calculate_fibre_liveness(facts)[0].live_coordinates == frozenset(
        {"x", "history"}
    )


def test_fallback_is_a_value_link_but_gate_computation_is_not_a_self_link() -> None:
    original = captured_gate_graph()
    declarations = _normalized(original)
    source = _source(original)
    facts = source.get_cell_facts()
    by_relation = {
        _name(_object(row)["id"]): _object(row)
        for row in _rows(declarations["relations"])
    }
    linked_relations = {
        rid for fact in facts for link in fact.links for rid in link.source_relations
    }
    fallback_ids = {
        rid for rid, row in by_relation.items() if row["kind"] == "fallback"
    }
    assert fallback_ids and fallback_ids <= linked_relations
    assert all(
        by_relation[rid]["kind"] not in {"gate", "arrival"} for rid in linked_relations
    )
    assert all(link.target != fact.identifier for fact in facts for link in fact.links)
    landing = next(
        fact
        for fact in facts
        if any(
            by_relation[rid]["kind"] == "fallback"
            for link in fact.links
            for rid in link.source_relations
        )
    )
    assert landing.direct.live_coordinates == frozenset({"history"})


def test_imputation_stays_a_provider_instead_of_an_artificial_self_value_link() -> None:
    original = _original()
    _object(_fields(original)["states"])["x"] = {
        "kind": "dataclass",
        "type": "lcm.Phased",
        "fields": {
            "solve": function(
                "def imputed(history):\n    return history\n", ["history"]
            ),
            "simulate": grid([0, 1]),
        },
    }
    declarations = _normalized(original)
    facts = _source(original).get_cell_facts()
    imputation_ids = {
        _name(_object(row)["id"])
        for row in _rows(declarations["relations"])
        if _object(row)["kind"] == "imputation"
    }
    assert imputation_ids
    assert all(
        not (set(link.source_relations) & imputation_ids)
        for fact in facts
        for link in fact.links
    )
    assert all(link.target != fact.identifier for fact in facts for link in fact.links)
    assert all(
        fact.direct.live_coordinates == frozenset({"history", "x"}) for fact in facts
    )


def test_query_only_target_qualification_retains_every_original_source_coordinate() -> (
    None
):
    original = policy_snapshot()
    declarations = _normalized(original)
    source = _source(original)
    facts = source.get_cell_facts()
    assert any(
        "policy query" in qualification for qualification in source.source_obligations
    )
    assert tuple(fact.identifier for fact in facts) == source.get_cell_ids()
    assert all(
        fact.direct.live_coordinates == frozenset(_object(_object(cell)["states"]))
        for fact, cell in zip(facts, _rows(declarations["cells"]), strict=True)
    )
    query_ids = {
        _name(_object(row)["id"])
        for row in _rows(declarations["relations"])
        if _object(row)["kind"] == "policy_query"
    }
    assert query_ids
    assert all(
        not (set(link.source_relations) & query_ids)
        for fact in facts
        for link in fact.links
    )


def test_new_target_coordinate_has_an_explicit_constant_update() -> None:
    original = _original()
    original["n_periods"] = 2
    fields = _fields(original)
    fields["functions"] = {"utility": function("def utility():\n    return 0\n", [])}
    fields["state_transitions"] = {
        "fresh": function("def fresh():\n    return 0\n", [])
    }
    _object(original["user_regimes"])["target"] = {
        "kind": "dataclass",
        "type": "lcm.Regime",
        "fields": {
            "states": {"fresh": grid([0, 1])},
            "actions": {},
            "constraints": {},
            "functions": {
                "utility": function(
                    "def utility(fresh):\n    return fresh\n", ["fresh"]
                )
            },
            "state_transitions": {},
        },
    }
    reach = _object(_object(original["reachability"])["fields"])
    reach["nodes"] = [[0, "original"], [1, "target"]]
    for phase in ("solution", "simulation"):
        _object(_object(reach[phase])["fields"])["targets_by_period"] = [
            {"original": ["target"]}
        ]
    before = deepcopy(original)
    facts = _source(original).get_cell_facts()
    assert isinstance(facts, tuple)
    assert len(facts) == 2
    assert facts[0].direct.live_coordinates == frozenset()
    assert all(
        link.updates[0].coordinate == "fresh"
        and link.updates[0].reads.live_coordinates == frozenset()
        for link in facts[0].links
    )
    assert calculate_fibre_liveness(facts) == (
        FibreLiveness(facts[0].identifier, frozenset()),
        FibreLiveness(facts[1].identifier, frozenset({"fresh"})),
    )
    assert original == before


@pytest.mark.parametrize("period", [0, 1])
@pytest.mark.parametrize(
    "clock", ["half_year_ages", "integer_ages", "periods", "periods_free_age"]
)
def test_date_specialization_uses_numeric_original_age_or_actual_period(
    clock: str, period: int
) -> None:
    original = _original()
    original["n_periods"] = 2
    dates: list[JSONValue]
    if clock == "half_year_ages":
        dates = [{"fraction": "1/2"}, {"fraction": "3/2"}]
    else:
        dates = [0, 1]
    uses_age_grid = clock in {"half_year_ages", "integer_ages"}
    original["ages"] = (
        {"kind": "object", "type": "lcm.AgeGrid", "fields": {"exact_values": dates}}
        if uses_age_grid
        else None
    )
    graph = _object(_object(original["graph"])["fields"])
    graph["coordinate_kind"] = "age" if uses_age_grid else "period"
    graph["initial_nodes"] = [[dates[0], "original"]]
    reach = _object(_object(original["reachability"])["fields"])
    reach["nodes"] = [[date, "original"] for date in dates]
    for phase in ("solution", "simulation"):
        _object(_object(reach[phase])["fields"])["targets_by_period"] = [
            {"original": ["original"]}
        ]
    argument = "period" if clock == "periods" else "age"
    _object(_fields(original)["functions"])["utility"] = function(
        f"def utility(x, history, {argument}):\n    return x if {argument} < 1 else history\n",
        ["x", "history", argument],
    )
    declarations = _normalized(original)
    cell = next(
        _object(row)
        for row in _rows(declarations["cells"])
        if _object(row)["period"] == period
    )
    if clock == "half_year_ages":
        assert cell["age"] == ("1/2", "3/2")[period]
        assert isinstance(cell["age"], str)
    elif clock == "integer_ages":
        assert cell["age"] == period and type(cell["age"]) is int
    else:
        assert cell["age"] is None
    result = _source(original).get_direct_reads(
        cell_id=_name(cell["id"]), statuses={}, removal_order=()
    )
    expected = frozenset(
        {"x", "history"}
        if clock == "periods_free_age"
        else {"x"}
        if period == 0
        else {"history"}
    )
    assert result.live_coordinates == expected


def _report_original(*, numeric_values: bool = False) -> dict[str, JSONValue]:
    original: dict[str, JSONValue] = snapshot()
    _fields(original)["constraints"] = {
        "allowed": function(
            "def allowed(choice_code, period):\n    return (choice_code == 0) | (period < 2)\n",
            ["choice_code", "period"],
        )
    }
    if numeric_values:
        _object(_fields(original)["states"])["x"] = grid([10.0, 20.0])
        _object(_fields(original)["actions"])["choice"] = grid([10.0, 20.0])
        _object(_fields(original)["state_transitions"])["x"] = function(
            "def next_x():\n    return 20.0\n", []
        )
    return original


def _assert_source_report(
    report: dict[str, JSONValue],
    original: dict[str, JSONValue],
    *,
    conservative_update: bool = False,
) -> None:
    assert report["schema"] == "regime-source-facts/v1"
    assert report["status"] == "SOURCE_FACTS_BUILT"
    assert report["transformation_authorized"] is False
    assert isinstance(json.dumps(report, allow_nan=False), str)
    declarations = _normalized(original)
    cells = tuple(_object(row) for row in _rows(report["cells"]))
    assert tuple(row["id"] for row in cells) == tuple(
        _object(row)["id"] for row in _rows(declarations["cells"])
    )
    assert tuple(_object(row)["id"] for row in _rows(report["relations"])) == tuple(
        _object(row)["id"] for row in _rows(declarations["relations"])
    )
    exclusions = _object(report["excluded_action_values"])
    for cell, declared in zip(cells, _rows(declarations["cells"]), strict=True):
        assert cell["direct_reads"] == ["x"]
        has_successor = any(
            _object(row)["source"] == cell["id"] and bool(_object(row)["targets"])
            for row in _rows(declarations["relations"])
        )
        assert cell["live_coordinates"] == (
            ["history", "x"] if conservative_update and has_successor else ["x"]
        )
        period = _object(declared)["period"]
        assert isinstance(period, int)
        expected = [1] if period >= 2 else []
        assert _object(exclusions[_name(cell["id"])])["choice"] == expected
    assert all(isinstance(item, str) for item in _rows(report["source_obligations"]))
    assert "recommendation" not in report
    assert "optimality_certificate" not in report


def test_source_report_is_json_safe_original_facts_without_layout_search() -> None:
    from _lcm_regime_finder import frontier, production

    original = _report_original()
    before = deepcopy(original)
    with (
        patch.object(
            frontier,
            "search_frontier",
            side_effect=AssertionError("Source facts must not run a layout search"),
        ),
        patch.object(
            production.MetadataAnalysis,
            "run",
            side_effect=AssertionError("Source facts must not run an optimizer"),
        ),
    ):
        report = _source(original).get_report()
    _assert_source_report(report, original)
    assert original == before


@pytest.mark.parametrize("stochastic", [False, True])
def test_source_report_keeps_nominal_codes_and_unknown_categorical_support(
    stochastic: bool,
) -> None:
    original = _report_original(numeric_values=True)
    if stochastic:
        _object(_fields(original)["state_transitions"])["x"] = {
            "kind": "dataclass",
            "type": "lcm.StochasticTransition",
            "fields": {"func": function("def probabilities():\n    return 0\n", [])},
        }
    report = _source(original).get_report()
    _assert_source_report(report, original, conservative_update=stochastic)
    first = next(
        _object(row)
        for row in _rows(report["relations"])
        if _object(row)["id"] == _relation_id(original)
    )
    support = _object(_object(first["categorical_support"])["x"])
    assert support["possible_codes"] == ([0, 1] if stochastic else [1])
    assert support["proved"] is (not stochastic)


def test_source_report_preserves_query_qualification_and_conservative_live_states() -> (
    None
):
    original = policy_snapshot()
    report = _source(original).get_report()
    assert report["status"] == "SOURCE_FACTS_BUILT"
    assert report["transformation_authorized"] is False
    assert any(
        "policy query" in _name(item) for item in _rows(report["source_obligations"])
    )
    assert all(
        set(_rows(_object(cell)["live_coordinates"])) == {"x", "history"}
        for cell in _rows(report["cells"])
    )


def test_source_report_uses_each_targets_original_categorical_domain() -> None:
    original = _report_original()
    original["n_periods"] = 2
    regimes = _object(original["user_regimes"])
    for target, values in (("small", [0, 1]), ("large", [10, 20, 30])):
        regimes[target] = deepcopy(regimes["original"])
        _object(_fields(original, target)["states"])["x"] = grid(values)
    _fields(original)["state_transitions"] = {
        "x": {
            "small": function("def next_x():\n    return 0\n", []),
            "large": function("def next_x():\n    return 20\n", []),
        },
        "history": function(
            "def next_history(history):\n    return history\n", ["history"]
        ),
    }
    reach = _object(_object(original["reachability"])["fields"])
    reach["nodes"] = [[0, "original"], [1, "small"], [1, "large"]]
    for phase in ("solution", "simulation"):
        _object(_object(reach[phase])["fields"])["targets_by_period"] = [
            {"original": ["small", "large"]}
        ]
    report = _source(original).get_report()
    cells = {_name(_object(row)["id"]): _object(row) for row in _rows(report["cells"])}
    checked = set()
    for row in _rows(report["relations"]):
        relation = _object(row)
        if "x" not in _object(relation["categorical_support"]):
            continue
        targets = _rows(relation["targets"])
        assert len(targets) == 1
        target = _name(cells[_name(targets[0])]["regime"])
        support = _object(_object(relation["categorical_support"])["x"])
        assert support["proved"] is True
        assert support["possible_codes"] == ([0] if target == "small" else [1])
        checked.add((target, _name(relation["phase"])))
    assert checked == {
        (target, phase)
        for target in ("small", "large")
        for phase in ("solve", "simulation")
    }
    assert report["transformation_authorized"] is False


def test_source_report_qualifies_uninterpretable_categorical_payload() -> None:
    original = _report_original()
    unavailable: JSONValue = {"kind": "opaque", "type": "UnavailableCategoricalPayload"}
    for slot, coordinate in (("states", "x"), ("actions", "choice")):
        domain = _object(_object(_fields(original)[slot])[coordinate])
        _object(domain["fields"])["values"] = unavailable
    report = _source(original).get_report()
    assert report["status"] == "SOURCE_FACTS_BUILT"
    assert report["transformation_authorized"] is False
    assert any(
        "payload needs interpretation" in _name(item).lower()
        for item in _rows(report["source_obligations"])
    )
    for cell in _rows(report["cells"]):
        assert (
            _object(
                _object(report["excluded_action_values"])[_name(_object(cell)["id"])]
            )["choice"]
            == []
        )
    supports = tuple(
        _object(_object(_object(row)["categorical_support"])["x"])
        for row in _rows(report["relations"])
    )
    assert supports
    assert all(
        item["proved"] is False and "target_domains" in item for item in supports
    )
    assert all("possible_codes" not in item for item in supports)
