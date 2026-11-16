"""A cell splits into status lanes that need different states.

A lane fixes some status states and leaves the others free, and the lanes
partition the joint status rectangle. In the one-status model, the unemployed
neither read experience nor carry it forward. In the two-status model, only the
employed and healthy need experience, and health matters only to the employed:
the joint split needs 5 rows, the best split by one status 7, and the unsplit
cell 12. Exact backends offer the splits; the metadata backend,
which has no per-status facts, does not.
"""

from copy import deepcopy

import jax.numpy as jnp
import pytest

from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder.semantic_check import check_semantics
from lcm.graphs import StatusLane, analyze_regime_structure
from lcm import DiscreteGrid, InitialNodes, Model, Periods, Regime, categorical
from lcm.typing import DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class Employment:
    unemployed: ScalarInt
    employed: ScalarInt


@categorical(ordered=False)
class Health:
    unhealthy: ScalarInt
    healthy: ScalarInt


@categorical(ordered=False)
class Experience:
    none: ScalarInt
    some: ScalarInt
    much: ScalarInt


@categorical(ordered=False)
class Regimes:
    working_age: ScalarInt
    final: ScalarInt


def wage(experience: DiscreteState, employment: DiscreteState) -> FloatND:
    """Only the employed earn from experience."""
    return jnp.where(employment == Employment.employed, 1.0 + experience, 0.0)


def next_experience(
    experience: DiscreteState, employment: DiscreteState
) -> DiscreteState:
    """Unemployment resets experience."""
    return jnp.where(employment == Employment.employed, experience, Experience.none)


def next_employment(employment: DiscreteState) -> DiscreteState:
    return employment


def joint_wage(
    experience: DiscreteState, employment: DiscreteState, health: DiscreteState
) -> FloatND:
    """Experience pays only the employed and healthy; health matters only at work."""
    at_work = jnp.where(health == Health.healthy, 1.0 + experience, 0.0)
    return jnp.where(employment == Employment.employed, at_work, 0.0)


def joint_next_experience(
    experience: DiscreteState, employment: DiscreteState, health: DiscreteState
) -> DiscreteState:
    """Experience is kept only by the employed and healthy."""
    at_work = jnp.where(health == Health.healthy, experience, Experience.none)
    return jnp.where(employment == Employment.employed, at_work, Experience.none)


def next_health(health: DiscreteState) -> DiscreteState:
    return health


def _model(states, utility, transitions) -> Model:
    return Model(
        n_periods=2,
        regime_id_class=Regimes,
        regimes={
            "working_age": Regime(
                states=states,
                functions={"utility": utility},
                state_transitions=transitions,
            ),
            "final": Regime(states=states, functions={"utility": utility}),
        },
        initial_nodes=InitialNodes(by_period={0: "working_age"}),
        edges={"working_age": {"final": Periods(values=(0,))}},
        fixed_params={"discount_factor": 1.0},
        enable_jit=False,
    )


def model(*, status_matters: bool = True) -> Model:
    return _model(
        {
            "employment": DiscreteGrid(Employment),
            "experience": DiscreteGrid(Experience),
        },
        wage if status_matters else _always_experience,
        {
            "experience": next_experience if status_matters else _carry,
            "employment": next_employment,
        },
    )


def joint_model() -> Model:
    return _model(
        {
            "employment": DiscreteGrid(Employment),
            "health": DiscreteGrid(Health),
            "experience": DiscreteGrid(Experience),
        },
        joint_wage,
        {
            "experience": joint_next_experience,
            "employment": next_employment,
            "health": next_health,
        },
    )


def _always_experience(experience: DiscreteState, employment: DiscreteState) -> FloatND:
    return 1.0 + experience + 0 * employment


def _carry(experience: DiscreteState, employment: DiscreteState) -> DiscreteState:
    return experience + 0 * employment


def _analyze(facts_backend: str, fixture=None):
    return analyze_production(
        fixture if fixture is not None else model(),
        facts_backend=facts_backend,
        search_mode="heuristic",
        limits={"seconds": 120},
    )


def _lanes(split):
    """Map each lane's fixed statuses to its kept states."""
    return {
        tuple(sorted(lane["statuses"].items())): set(lane["states"])
        for lane in split["lanes"]
    }


@pytest.mark.parametrize("facts_backend", ["dense", "implicit"])
def test_exact_backends_split_by_the_status_that_drops_a_state(facts_backend) -> None:
    result = _analyze(facts_backend)
    splits = result["analysis"]["status_splits"]
    assert splits["status"] == "OFFERED", splits
    assert len(splits["cells"]) == 2, "both regimes split"
    for split in splits["cells"]:
        assert split["axes"] == ["employment"]
        assert _lanes(split) == {
            (("employment", 0),): set(),
            (("employment", 1),): {"experience"},
        }
        assert (split["state_rows"], split["unsplit_state_rows"]) == (1 + 3, 2 * 3)
    assert result["semantic_check"]["status"] != "FAIL", result["semantic_check"]


@pytest.mark.parametrize("facts_backend", ["dense", "implicit"])
def test_exact_backends_split_the_joint_status_rectangle(facts_backend) -> None:
    result = _analyze(facts_backend, joint_model())
    splits = result["analysis"]["status_splits"]
    assert splits["status"] == "OFFERED", splits
    assert len(splits["cells"]) == 2, "both regimes split"
    for split in splits["cells"]:
        assert split["axes"] == ["employment", "health"]
        assert _lanes(split) == {
            (("employment", 0),): set(),
            (("employment", 1), ("health", 0)): set(),
            (("employment", 1), ("health", 1)): {"experience"},
        }
        assert (split["state_rows"], split["unsplit_state_rows"]) == (
            1 + 1 + 3,
            2 * 2 * 3,
        )
    assert result["semantic_check"]["status"] != "FAIL", result["semantic_check"]


def test_metadata_backend_states_why_it_offers_no_split() -> None:
    splits = _analyze("metadata")["analysis"]["status_splits"]
    assert splits["status"] == "NOT_OFFERED"
    assert "per-status facts" in splits["reason"]


def test_a_status_that_changes_nothing_is_not_split() -> None:
    splits = _analyze("dense", model(status_matters=False))["analysis"]["status_splits"]
    assert splits["status"] == "NONE_USEFUL"
    assert not splits["cells"]


@pytest.mark.parametrize(
    "forgery", ["drop_needed_state", "understate_rows", "overlap", "gap"]
)
def test_checker_refutes_a_forged_split(forgery) -> None:
    result = _analyze("dense", joint_model())
    analysis = deepcopy(result["analysis"])
    cell = result["analysis"]["status_splits"]["cells"][0]["cell"]
    split = next(o for o in analysis["interfaces"][cell]["options"] if "lanes" in o)
    domain = analysis["derived_domains"][next(iter(split["states"].values()))]
    copies = (split["lanes"], domain["lanes"])
    if forgery == "drop_needed_state":
        for lanes in copies:
            for lane in lanes:
                lane["states"].pop("experience", None)
        domain["size"] = 3
        expected = "status lane drops a needed state"
    elif forgery == "understate_rows":
        domain["size"] = 1
        expected = "status split rows differ from its lanes"
    elif forgery == "overlap":
        for lanes in copies:
            lanes.append(deepcopy(lanes[0]))
        domain["size"] += 1
        expected = "status lanes do not partition the status points"
    else:
        for lanes in copies:
            lanes.pop()
        domain["size"] -= 3
        expected = "status lanes do not partition the status points"
    check = check_semantics(
        analysis["declarations"]["original"]
        if "original" in analysis["declarations"]
        else result["analysis"]["declarations"],
        analysis["declarations"],
        analysis,
    )
    assert any(error["kind"] == expected for error in check["errors"]), check["errors"]


def test_the_recommended_graph_shares_one_split_block() -> None:
    result = _analyze("dense")
    recommendation = result["recommendations"][0]
    scenario = result["scenarios"][recommendation.get("scenario_index", 0)]
    graph = next(
        c for c in scenario["candidates"] if c["id"] == recommendation["graph_id"]
    )
    (block,) = graph["blocks"]
    assert list(block["state_axes"]) == ["lanes[employment]"]
    assert graph["cost"]["work"] == 8, (
        "4 lane rows in each of 2 cells, against 12 unsplit"
    )
    assert scenario["independent_check"]["status"] == "PASS"


def test_the_typed_result_lists_the_lanes() -> None:
    result = analyze_regime_structure(
        joint_model(), facts_backend="dense", search_mode="heuristic"
    )
    (block,) = result.recommendations[0].blocks
    assert block.states == ("employment", "health", "experience")
    assert set(block.lanes) == {
        StatusLane(statuses=(("employment", 0),), states=()),
        StatusLane(statuses=(("employment", 1), ("health", 0)), states=()),
        StatusLane(statuses=(("employment", 1), ("health", 1)), states=("experience",)),
    }


def test_the_report_names_the_lanes(tmp_path) -> None:
    result = analyze_regime_structure(
        joint_model(), facts_backend="dense", search_mode="heuristic"
    )
    result.write_report(tmp_path / "report")
    text = "\n".join(
        p.read_text()
        for p in (tmp_path / "report").rglob("*")
        if p.suffix in {".md", ".html"}
    )
    assert (
        "split by employment, health [employment 0: no other states; "
        "employment 1, health 0: no other states; employment 1, health 1: experience]"
        in text
    )


def test_the_split_search_reads_each_cell_and_status_point_once(monkeypatch) -> None:
    """Successors' needs are shared across the tree and the cells, not recomputed.

    Only the producer's reads count; the checker re-derives the lanes on its own.
    """
    import _lcm_regime_finder.facts as facts_module
    import _lcm_regime_finder.production as production

    calls = []
    inside = []
    original_reads, original_needs = facts_module.current_reads, production.lane_needs

    def counting_reads(facts, cell, stops=frozenset(), statuses=None):
        if inside:
            calls.append((cell["id"], tuple(sorted((statuses or {}).items()))))
        return original_reads(facts, cell, stops, statuses)

    def producer_needs(*args, **kwargs):
        inside.append(True)
        try:
            return original_needs(*args, **kwargs)
        finally:
            inside.pop()

    monkeypatch.setattr(facts_module, "current_reads", counting_reads)
    monkeypatch.setattr(production, "lane_needs", producer_needs)
    _analyze("dense", joint_model())
    assert calls, "positive control: the split search reads at fixed statuses"
    assert len(calls) == len(set(calls))


def _reads_all(
    experience: DiscreteState, employment: DiscreteState, health: DiscreteState
) -> FloatND:
    return 1.0 + experience + employment + health


def _keep_experience(experience: DiscreteState) -> DiscreteState:
    return experience


def test_a_cell_where_no_state_can_drop_builds_no_tree(monkeypatch) -> None:
    """Fully fixed points and one-status fibres decide that no split saves rows.

    With three statuses the whole tree has 3 * 3 * 4 = 36 nodes; the check reads the
    unsplit cell, 12 points and 6 + 6 + 4 fibres.
    """
    import _lcm_regime_finder.production as production

    original = production.lane_needs
    nodes = {}

    def counting(facts, cell, live, statuses, *args):
        nodes.setdefault(cell["id"], set()).add(tuple(sorted(statuses.items())))
        return original(facts, cell, live, statuses, *args)

    monkeypatch.setattr(production, "lane_needs", counting)
    fixture = _model(
        {
            "employment": DiscreteGrid(Employment),
            "health": DiscreteGrid(Health),
            "experience": DiscreteGrid(Experience),
        },
        _reads_all,
        {
            "experience": _keep_experience,
            "employment": next_employment,
            "health": next_health,
        },
    )
    splits = _analyze("dense", fixture)["analysis"]["status_splits"]
    assert splits["status"] == "NONE_USEFUL"
    assert nodes, "positive control: the producer reads lane needs"
    assert {len(visited) for visited in nodes.values()} == {1 + 12 + 6 + 6 + 4}
