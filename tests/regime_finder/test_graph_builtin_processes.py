"""Intrinsic process dependencies and query identity survive graph-only scope."""

from copy import deepcopy
from collections.abc import Mapping

import pytest
from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder.records import JSONValue
from _lcm_regime_finder import verify_production_result
from tests.regime_finder.test_model_adapter import snapshot, function
from tests.regime_finder.test_query_owners import policy_snapshot


def process(name):
    module = "ar1" if "AR1" in name else "iid"
    return {
        "kind": "dataclass",
        "type": f"_lcm.processes.{module}.{name}",
        "fields": {
            "n_points": 3,
            "state_conditioned": None,
            "sigma": 1.0,
            "mu": 0.0,
            "rho": 0.8,
        },
    }


def _check_analysis(source):
    result = analyze_production(source, limits={"seconds": 15}, search_mode="exact")
    assert (result["status"]) == ("COMPLETE"), str(result.get("semantic_check"))[
        :2000
    ] + str(result["analysis"]["obligations"])
    assert (verify_production_result(source, result)["status"]) == ("PASS")
    return result


def test_persistent_process_keeps_its_current_state_dependency() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"]["x"] = process("RouwenhorstAR1Process")
    del fields["state_transitions"]["x"]
    result = _check_analysis(source)
    bad = deepcopy(result)
    cb = next(
        c
        for c in bad["analysis"]["declarations"]["callbacks"].values()
        if c.get("original", {}).get("type", "").endswith("RouwenhorstAR1Process")
    )
    cb.update(arguments=[], reads=[])
    assert (verify_production_result(source, bad)["status"]) == ("FAIL")


def test_new_iid_is_drawn_in_arrival_regime_without_a_current_value() -> None:
    source = snapshot()
    arrival = deepcopy(source["user_regimes"]["original"])
    arrival["fields"]["states"]["fresh"] = process("NormalIIDProcess")
    arrival["fields"]["functions"]["utility"] = function(
        "def utility(x,fresh):\n    return x+fresh\n", ["x", "fresh"]
    )
    source["user_regimes"]["arrival"] = arrival
    source["reachability"]["fields"]["nodes"] = [[0, "original"]] + [
        [t, "arrival"] for t in range(1, 6)
    ]
    rows = source["reachability"]["fields"]["solution"]["fields"]["targets_by_period"]
    for t in range(5):
        rows[t] = {"original" if t == 0 else "arrival": ["arrival"]}
    _check_analysis(source)


def test_query_identity_is_preserved_without_numerical_factorization_claim() -> None:
    result = _check_analysis(policy_snapshot())
    assert result["analysis"]["declarations"]["query_owners"]


def test_unknown_process_class_is_not_dispatched_as_a_library_primitive() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"]["x"] = process("RouwenhorstAR1Process")
    fields["states"]["x"]["type"] = "foreign.RouwenhorstAR1Process"
    del fields["state_transitions"]["x"]
    result = analyze_production(source, limits={"seconds": 15}, search_mode="exact")
    assert (result["status"]) != ("COMPLETE")


def test_lowered_joint_normal_coordinate_is_an_iid_latent_axis() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    coordinate = process("NormalIIDProcess")
    coordinate["type"] = "_lcm.processes.joint_normal_lowering._JointNormalCoordinate"
    coordinate["fields"]["group"] = "income"
    coordinate["fields"]["joint"] = {
        "kind": "dataclass",
        "type": "_lcm.processes.joint_normal.JointNormalIIDProcess",
        "fields": {
            "states": ["income", "return"],
            "cov": [[1, 0.5], [0.5, 1]],
            "mean": [0, 0],
        },
    }
    fields["states"]["x"] = coordinate
    del fields["state_transitions"]["x"]
    result = _check_analysis(source)
    callbacks = [
        cb
        for cb in result["analysis"]["declarations"]["callbacks"].values()
        if cb.get("original", {}).get("type") == coordinate["type"]
    ]
    assert callbacks
    assert all(cb["arguments"] == [] for cb in callbacks)


def test_empty_target_map_does_not_create_a_next_state_provider() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["state_transitions"]["absent"] = {}
    result = _check_analysis(source)
    assert all(
        "next_absent" not in c["functions"]
        for c in result["analysis"]["declarations"]["cells"]
    )


def test_canonical_identity_preserves_a_categorical_characteristic() -> None:
    from _lcm_regime_finder.semantic_check import _identity_callback

    raw = {
        "arguments": ["history"],
        "bound_self": {
            "type": "_lcm.identity_transition._IdentityTransition",
            "fields": {"_state_name": "history"},
        },
    }
    assert _identity_callback({"original": raw}, "history")
    assert not (_identity_callback({"original": raw}, "other"))
    raw["arguments"] = []
    assert not (_identity_callback({"original": raw}, "history"))


def test_equal_persistent_process_specs_keep_distinct_state_owners() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"] = {
        name: process("RouwenhorstAR1Process") for name in ("x", "history")
    }
    fields["state_transitions"] = {}
    fields["functions"]["utility"] = function(
        "def utility(history,period):\n    return history if period>=3 else 0\n",
        ["history", "period"],
    )
    result = _check_analysis(source)
    assert all(
        "history" in axes and "x" not in axes
        for axes in result["analysis"]["dependency_closure"].values()
    )
    from _lcm_regime_finder.production import MetadataAnalysis
    from _lcm_regime_finder.semantic_check import check_semantics

    declaration = deepcopy(result["analysis"]["declarations"])
    relation = next(r for r in declaration["relations"] if r["kind"] == "transition")
    relation["state_updates"]["history"] = relation["state_updates"]["x"]
    changed = MetadataAnalysis(declaration).run()
    checked = check_semantics(source, declaration, changed)
    assert any(
        e["kind"] == "original process dependency owner changed"
        for e in checked["errors"]
    ), checked


def test_phased_process_owner_is_the_axis_not_the_phase_slot() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    process_grid = {
        "kind": "dataclass",
        "type": "lcm.phased.Phased",
        "fields": {
            "solve": process("RouwenhorstAR1Process"),
            "simulate": process("TauchenAR1Process"),
        },
    }
    fields["states"]["x"] = process_grid
    del fields["state_transitions"]["x"]
    result = _check_analysis(source)
    callbacks = [
        cb
        for cb in result["analysis"]["declarations"]["callbacks"].values()
        if cb.get("original", {}).get("type", "").startswith("_lcm.processes.ar1.")
    ]
    assert callbacks
    assert all(cb["arguments"] == ["x"] for cb in callbacks)


def test_independent_check_rejects_miswired_intrinsic_draw_providers() -> None:
    from _lcm_regime_finder.semantic_check import check_semantics

    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"]["x"] = process("RouwenhorstAR1Process")
    del fields["state_transitions"]["x"]
    result = _check_analysis(source)
    analysis = deepcopy(result["analysis"])
    declaration = analysis["declarations"]
    for cell in declaration["cells"]:
        assert "next_x" in cell["functions"]
        cell["functions"]["next_x"] = cell["functions"]["utility"]
        for providers in cell["phase_functions"].values():
            assert "next_x" in providers
            providers["next_x"] = providers["utility"]
    checked = check_semantics(source, declaration, analysis, check_coverage=False)
    assert checked["status"] == "FAIL", checked
    assert any(
        error["kind"] == "original intrinsic draw provider changed"
        for error in checked["errors"]
    ), checked


def test_independent_check_rejects_erased_conditioned_process_inputs() -> None:
    from _lcm_regime_finder.semantic_check import check_semantics

    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"]["x"] = process("NormalIIDProcess")
    fields["states"]["x"]["fields"]["sigma"] = {
        "kind": "dataclass",
        "type": "_lcm.processes.base.StateConditioned",
        "fields": {"on": "history", "by": {"0": 0.5, "1": 1.0}},
    }
    del fields["state_transitions"]["x"]
    result = _check_analysis(source)
    analysis = deepcopy(result["analysis"])
    declaration = analysis["declarations"]
    callbacks = [
        cb
        for cb in declaration["callbacks"].values()
        if cb.get("original", {}).get("type", "").endswith("NormalIIDProcess")
    ]
    assert callbacks
    for callback in callbacks:
        assert "history" in callback["arguments"]
        callback.update(arguments=[], reads=[])
    checked = check_semantics(source, declaration, analysis, check_coverage=False)
    assert checked["status"] == "FAIL", checked


def test_missing_conditioning_input_is_rejected_before_discovery() -> None:
    from _lcm_regime_finder.declarations import DeclarationError, normalize_declarations

    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"]["x"] = process("NormalIIDProcess")
    fields["states"]["x"]["fields"]["sigma"] = {
        "kind": "dataclass",
        "type": "_lcm.processes.base.StateConditioned",
        "fields": {"by": {"0": 0.5, "1": 1.0}},
    }
    del fields["state_transitions"]["x"]
    with pytest.raises(DeclarationError, match="no captured conditioning state"):
        normalize_declarations(source)


@pytest.mark.parametrize("process_name", ["RouwenhorstAR1Process", "NormalIIDProcess"])
def test_equal_specs_do_not_authorize_source_provider_owner_swaps(
    process_name: str,
) -> None:
    from _lcm_regime_finder.production import MetadataAnalysis
    from _lcm_regime_finder.semantic_check import check_semantics

    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"] = {axis: process(process_name) for axis in ("x", "history")}
    fields["state_transitions"] = {}
    fields["functions"]["utility"] = function(
        "def utility(next_x):\n    return next_x\n", ["next_x"]
    )
    result = _check_analysis(source)
    declaration = deepcopy(result["analysis"]["declarations"])
    for cell in declaration["cells"]:
        for providers in cell["phase_functions"].values():
            assert providers["next_x"] != providers["next_history"]
            providers["next_x"] = providers["next_history"]
        cell["functions"]["next_x"] = cell["functions"]["next_history"]
    analysis = MetadataAnalysis(declaration).run()
    checked = check_semantics(source, declaration, analysis)
    assert checked["status"] == "FAIL", checked
    assert any(
        error["kind"] == "original intrinsic draw provider owner changed"
        for error in checked["errors"]
    ), checked


def test_equal_iid_specs_keep_distinct_source_draw_providers() -> None:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"] = {axis: process("NormalIIDProcess") for axis in ("x", "history")}
    fields["state_transitions"] = {}
    fields["functions"]["utility"] = function(
        "def utility(next_x, next_history):\n    return next_x + next_history\n",
        ["next_x", "next_history"],
    )
    result = _check_analysis(source)
    for cell in result["analysis"]["declarations"]["cells"]:
        for providers in cell["phase_functions"].values():
            assert providers["next_x"] != providers["next_history"]


def _json_object(value: JSONValue) -> dict[str, JSONValue]:
    assert isinstance(value, dict)
    return value


def _json_rows(value: JSONValue) -> list[dict[str, JSONValue]]:
    assert isinstance(value, list)
    return [_json_object(item) for item in value]


def _input_names(value: JSONValue) -> frozenset[str]:
    assert isinstance(value, list)
    names = []
    for name in value:
        assert isinstance(name, str)
        names.append(name)
    return frozenset(names)


def _conditioned_ar1_declaration(
    *, conditioning: str | None, low_sigma: float = 0.5
) -> dict[str, JSONValue]:
    return {
        "kind": "dataclass",
        "type": "_lcm.processes.ar1.TauchenAR1Process",
        "fields": {
            "n_points": 3,
            "gauss_hermite": False,
            "sigma": 1.0,
            "mu": 0.0,
            "rho": 0.5,
            "n_std": 2.0,
            "state_conditioned": {
                "kind": "dataclass",
                "type": "_lcm.processes.base.StateConditioned",
                "fields": {"on": conditioning, "by": {"low": low_sigma, "high": 1.0}},
            }
            if conditioning is not None
            else None,
        },
    }


def _conditioned_ar1_capture(
    *,
    axis: str = "z",
    conditioning: str = "history",
    conditioned: bool = True,
    reverse_order: bool = False,
) -> dict[str, JSONValue]:
    original: dict[str, JSONValue] = snapshot()
    declaration = _json_object(
        _json_object(_json_object(original["user_regimes"])["original"])["fields"]
    )
    states: dict[str, JSONValue] = {
        axis: _conditioned_ar1_declaration(
            conditioning=conditioning if conditioned else None
        ),
        conditioning: {
            "kind": "object",
            "type": "lcm.DiscreteGrid",
            "fields": {
                "values": [0, 1],
                "codes": [0, 1],
                "categories": ["low", "high"],
            },
        },
    }
    declaration.update(
        states=dict(reversed(tuple(states.items()))) if reverse_order else states,
        actions={},
        constraints={},
        functions={
            "utility": function(
                f"def utility({axis}, {conditioning}):\n    return {axis} + {conditioning}\n",
                [axis, conditioning],
            )
        },
        state_transitions={
            conditioning: function(
                f"def next_state({conditioning}):\n    return {conditioning}\n",
                [conditioning],
            )
        },
    )
    return original


@pytest.mark.parametrize(
    ("axis", "conditioning"),
    [("z", "history"), ("income", "status"), ("state_b", "state_a")],
)
@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_persistent_process_inputs_do_not_depend_on_names_or_declaration_order(
    *, axis: str, conditioning: str, conditioned: bool, reverse_order: bool
) -> None:
    source = _conditioned_ar1_capture(
        axis=axis,
        conditioning=conditioning,
        conditioned=conditioned,
        reverse_order=reverse_order,
    )
    result = _check_analysis(source)
    expected = {axis, conditioning} if conditioned else {axis}
    declarations = _json_object(result["analysis"]["declarations"])
    callbacks = _json_object(declarations["callbacks"])
    transitions = [
        row
        for row in _json_rows(declarations["relations"])
        if row["kind"] == "transition"
    ]
    assert transitions
    for relation in transitions:
        callback = _json_object(
            callbacks[str(_json_object(relation["state_updates"])[axis])]
        )
        assert _input_names(callback["arguments"]) == expected
        assert _input_names(callback["reads"]) == expected


@pytest.mark.parametrize("mutation", ["drop_condition", "drop_axis", "extra_input"])
@pytest.mark.parametrize("field", ["arguments", "reads", "both"])
def test_conditioned_process_rejects_missing_or_extra_declared_inputs(
    *, mutation: str, field: str
) -> None:
    from _lcm_regime_finder.production import MetadataAnalysis
    from _lcm_regime_finder.semantic_check import check_semantics

    source = _conditioned_ar1_capture()
    result = _check_analysis(source)
    declarations = deepcopy(result["analysis"]["declarations"])
    names = {
        "drop_condition": ["z"],
        "drop_axis": ["history"],
        "extra_input": ["history", "z", "invented"],
    }[mutation]
    mutated = []
    for callback in _json_rows(list(declarations["callbacks"].values())):
        original = callback.get("original")
        if (
            not isinstance(original, dict)
            or original.get("type") != "_lcm.processes.ar1.TauchenAR1Process"
        ):
            continue
        for selected_field in ("arguments", "reads") if field == "both" else (field,):
            callback[selected_field] = names
        mutated.append(callback)
    assert mutated
    analysis = MetadataAnalysis(declarations, search_mode="heuristic").run()
    checked = check_semantics(source, declarations, analysis, check_coverage=False)
    assert checked["status"] == "FAIL", checked


def test_conditioned_process_rejects_foreign_original_law_with_the_same_axis() -> None:
    from _lcm_regime_finder.production import MetadataAnalysis
    from _lcm_regime_finder.semantic_check import check_semantics

    source = _conditioned_ar1_capture()
    regimes = _json_object(source["user_regimes"])
    foreign = deepcopy(_json_object(regimes["original"]))
    _json_object(_json_object(foreign["fields"])["states"])["z"] = (
        _conditioned_ar1_declaration(conditioning="history", low_sigma=0.25)
    )
    regimes["foreign"] = foreign
    reachability = _json_object(_json_object(source["reachability"])["fields"])
    nodes = reachability["nodes"]
    assert isinstance(nodes, list)
    nodes.extend([[period, "foreign"] for period in range(6)])
    for phase in ("solution", "simulation"):
        rows = _json_object(_json_object(reachability[phase])["fields"])[
            "targets_by_period"
        ]
        for targets in _json_rows(rows):
            targets["foreign"] = ["foreign"]
    starts = _json_object(_json_object(source["graph"])["fields"])["initial_nodes"]
    assert isinstance(starts, list)
    starts.append([0, "foreign"])
    result = _check_analysis(source)
    declarations = deepcopy(result["analysis"]["declarations"])
    cells = {cell["id"]: cell for cell in declarations["cells"]}
    original_relation = next(
        row
        for row in declarations["relations"]
        if row["kind"] == "transition"
        and row["phase"] == "solve"
        and cells[row["source"]]["regime"] == "original"
    )
    foreign_relation = next(
        row
        for row in declarations["relations"]
        if row["kind"] == "transition"
        and row["phase"] == "solve"
        and cells[row["source"]]["regime"] == "foreign"
    )
    original_relation["state_updates"]["z"] = foreign_relation["state_updates"]["z"]
    analysis = MetadataAnalysis(declarations, search_mode="heuristic").run()
    checked = check_semantics(source, declarations, analysis, check_coverage=False)
    assert checked["status"] == "FAIL", checked
    assert any(
        row["kind"] == "original state update consumer changed"
        for row in checked["errors"]
    ), checked


@pytest.mark.parametrize("invalid_condition", [None, 0, ["history"]])
def test_original_process_without_a_conditioning_name_has_no_source_authority(
    invalid_condition: int | list[str] | None,
) -> None:
    source = _conditioned_ar1_capture()
    regimes = _json_object(source["user_regimes"])
    states = _json_object(
        _json_object(_json_object(regimes["original"])["fields"])["states"]
    )
    process_fields = _json_object(_json_object(states["z"])["fields"])
    _json_object(_json_object(process_fields["state_conditioned"])["fields"])["on"] = (
        invalid_condition
    )
    result = analyze_production(source)
    assert result["status"] in {"UNRESOLVED", "INVALID"}, result
    assert "conditioning" in result.get("error", "").lower(), result
    assert any(
        row.get("kind") == "metadata_admission"
        and "conditioning" in row.get("detail", "").lower()
        for row in result.get("analysis", {}).get("obligations", [])
    )
    assert all(
        row["authority"] != "source_checked"
        for row in result.get("recommendations", [])
    )


def _conditioned_iid_declaration(conditioning: str | None) -> dict[str, JSONValue]:
    declaration = _conditioned_ar1_declaration(conditioning=conditioning)
    declaration["type"] = "_lcm.processes.iid.NormalIIDProcess"
    _json_object(declaration["fields"]).pop("rho")
    return declaration


def _target_process_capture(
    *,
    source_condition: str | None,
    target_conditions: Mapping[str, tuple[str | None, str | None]],
    composed: bool = False,
    target_carries_process: bool = True,
) -> dict[str, JSONValue]:
    """Original-only graph fixture; it does not claim native constructor admission."""
    original: dict[str, JSONValue] = snapshot()
    base = _json_object(
        _json_object(_json_object(original["user_regimes"])["original"])["fields"]
    )
    names = {source_condition} | {
        condition for phases in target_conditions.values() for condition in phases
    }
    names.discard(None)
    histories: dict[str, JSONValue] = {
        name: {
            "kind": "object",
            "type": "lcm.DiscreteGrid",
            "fields": {
                "values": [0, 1],
                "codes": [0, 1],
                "categories": ["low", "high"],
            },
        }
        for name in names
        if name is not None
    }
    continuous: dict[str, JSONValue] = {
        "kind": "dataclass",
        "type": "_lcm.grids.continuous.LinSpacedGrid",
        "fields": {"start": -2.0, "stop": 2.0, "n_points": 3},
    }
    source_states = dict(histories, z=_conditioned_iid_declaration(source_condition))
    if composed:
        source_states["x"] = deepcopy(continuous)
    payoff = "x" if composed else "z"
    transitions: dict[str, JSONValue] = {
        name: function("def reset():\n    return 0\n", []) for name in histories
    }
    if composed:
        transitions["x"] = function(
            "def next_x(next_z):\n    return next_z\n", ["next_z"]
        )
    base.update(
        states=source_states,
        actions={},
        constraints={},
        functions={
            "utility": function(
                f"def utility({payoff}):\n    return {payoff}\n", [payoff]
            )
        },
        state_transitions=transitions,
    )
    regimes: dict[str, JSONValue] = {
        "source": {"kind": "dataclass", "type": "lcm.Regime", "fields": base}
    }
    for target, (solve, simulation) in target_conditions.items():
        states = deepcopy(histories)
        if target_carries_process:
            states["z"] = {
                "kind": "dataclass",
                "type": "lcm.phased.Phased",
                "fields": {
                    "solve": _conditioned_iid_declaration(solve),
                    "simulate": _conditioned_iid_declaration(simulation),
                },
            }
        if composed:
            states["x"] = deepcopy(continuous)
        regimes[target] = {
            "kind": "dataclass",
            "type": "lcm.Regime",
            "fields": {
                "states": states,
                "actions": {},
                "functions": {
                    "utility": function(
                        f"def utility({payoff}):\n    return {payoff}\n", [payoff]
                    )
                },
                "state_transitions": {},
            },
        }
    targets = list(target_conditions)
    reachability: dict[str, JSONValue] = {
        "kind": "dataclass",
        "fields": {
            "nodes": [[0, "source"]] + [[1, target] for target in targets],
            "solution": {
                "kind": "dataclass",
                "fields": {"targets_by_period": [{"source": targets}]},
            },
            "simulation": {
                "kind": "dataclass",
                "fields": {"targets_by_period": [{"source": targets}]},
            },
        },
    }
    original.update(
        n_periods=2,
        user_regimes=regimes,
        reachability=reachability,
        graph={
            "kind": "dataclass",
            "fields": {
                "reachability": reachability,
                "initial_nodes": [[0, "source"]],
                "laws": {},
            },
        },
    )
    return original


@pytest.mark.parametrize("composed", [False, True])
def test_each_transition_uses_its_target_and_phase_conditioning_law(
    composed: bool,
) -> None:
    conditions = {
        "target_a": ("a_solve", "a_simulate"),
        "target_b": ("b_solve", "b_simulate"),
    }
    source = _target_process_capture(
        source_condition="source_history",
        target_conditions=conditions,
        composed=composed,
    )
    result = _check_analysis(source)
    declaration = result["analysis"]["declarations"]
    cells = {cell["id"]: cell for cell in declaration["cells"]}
    callbacks = [
        (
            relation,
            cells[relation["targets"][0]],
            declaration["callbacks"][relation["state_updates"]["z"]],
        )
        for relation in declaration["relations"]
        if relation["kind"] == "transition" and "z" in relation["state_updates"]
    ]
    assert len(callbacks) == 4
    for relation, target, callback in callbacks:
        phase = relation["phase"]
        condition = conditions[target["regime"]][0 if phase == "solve" else 1]
        assert _input_names(callback["arguments"]) == {condition}
        assert _input_names(callback["reads"]) == {condition}
        side = "solve" if phase == "solve" else "simulate"
        assert callback["source_path"].startswith(
            f"/user_regimes/{target['regime']}/fields/states/z/fields/{side}"
        )
        assert callback["original"] == _conditioned_iid_declaration(condition)
    source_cell = next(cell for cell in cells.values() if cell["regime"] == "source")
    live = result["analysis"]["dependency_closure"][source_cell["id"]]
    expected = {condition for phases in conditions.values() for condition in phases}
    assert expected <= set(live)
    assert "source_history" not in live


def test_an_unused_source_process_law_does_not_retain_its_conditioner() -> None:
    source = _target_process_capture(
        source_condition="source_history", target_conditions={"target": (None, None)}
    )
    result = _check_analysis(source)
    cell = next(
        cell
        for cell in result["analysis"]["declarations"]["cells"]
        if cell["regime"] == "source"
    )
    assert "source_history" not in result["analysis"]["dependency_closure"][cell["id"]]


@pytest.mark.parametrize("target_carries_process", [False, True])
def test_composed_next_state_uses_the_draw_law_for_that_transition_context(
    target_carries_process: bool,
) -> None:
    source = _target_process_capture(
        source_condition="source_history",
        target_conditions={"target": ("target_history", "target_history")},
        composed=True,
        target_carries_process=target_carries_process,
    )
    result = _check_analysis(source)
    cell = next(
        cell
        for cell in result["analysis"]["declarations"]["cells"]
        if cell["regime"] == "source"
    )
    live = result["analysis"]["dependency_closure"][cell["id"]]
    expected = "target_history" if target_carries_process else "source_history"
    unused = "source_history" if target_carries_process else "target_history"
    assert expected in live
    assert unused not in live


@pytest.mark.parametrize("same_payload", [False, True])
@pytest.mark.parametrize("replacement", ["source", "foreign_target"])
def test_independent_replay_rejects_a_process_provider_from_another_owner(
    *, same_payload: bool, replacement: str
) -> None:
    from _lcm_regime_finder.production import MetadataAnalysis
    from _lcm_regime_finder.semantic_check import check_semantics

    source = _target_process_capture(
        source_condition="target_history" if same_payload else "source_history",
        target_conditions={
            "target_a": ("target_history", "target_history"),
            "target_b": ("target_history", "target_history")
            if same_payload
            else ("other_history", "other_history"),
        },
    )
    result = _check_analysis(source)
    declarations = deepcopy(result["analysis"]["declarations"])
    cells = {cell["id"]: cell for cell in declarations["cells"]}
    transitions = [
        row
        for row in declarations["relations"]
        if row["kind"] == "transition" and row["phase"] == "solve"
    ]
    selected = next(
        row for row in transitions if cells[row["targets"][0]]["regime"] == "target_a"
    )
    current = declarations["callbacks"][selected["state_updates"]["z"]]
    assert current["source_path"].startswith("/user_regimes/target_a/fields/states/z")
    if replacement == "source":
        source_cell = cells[selected["source"]]
        provider = source_cell["phase_functions"]["solve"]["next_z"]
    else:
        foreign = next(
            row
            for row in transitions
            if cells[row["targets"][0]]["regime"] == "target_b"
        )
        provider = foreign["state_updates"]["z"]
    assert provider != selected["state_updates"]["z"]
    selected["state_updates"]["z"] = provider
    analysis = MetadataAnalysis(declarations, search_mode="heuristic").run()
    checked = check_semantics(source, declarations, analysis, check_coverage=False)
    assert checked["status"] == "FAIL", checked
