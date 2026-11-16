"""A cell splits by a status whose values need different states.

The unemployed neither read experience nor carry it forward, so a split by
employment drops experience from the unemployed lane. Exact backends offer
that split; the metadata backend, which has no per-status facts, does not.
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


def model(*, status_matters: bool = True) -> Model:
    utility = wage if status_matters else _always_experience
    return Model(
        n_periods=2,
        regime_id_class=Regimes,
        regimes={
            "working_age": Regime(
                states={
                    "employment": DiscreteGrid(Employment),
                    "experience": DiscreteGrid(Experience),
                },
                functions={"utility": utility},
                state_transitions={
                    "experience": next_experience if status_matters else _carry,
                    "employment": next_employment,
                },
            ),
            "final": Regime(
                states={
                    "employment": DiscreteGrid(Employment),
                    "experience": DiscreteGrid(Experience),
                },
                functions={"utility": utility},
            ),
        },
        initial_nodes=InitialNodes(by_period={0: "working_age"}),
        edges={"working_age": {"final": Periods(values=(0,))}},
        fixed_params={"discount_factor": 1.0},
        enable_jit=False,
    )


def _always_experience(experience: DiscreteState, employment: DiscreteState) -> FloatND:
    return 1.0 + experience + 0 * employment


def _carry(experience: DiscreteState, employment: DiscreteState) -> DiscreteState:
    return experience + 0 * employment


def _analyze(facts_backend: str, *, status_matters: bool = True):
    return analyze_production(
        model(status_matters=status_matters),
        facts_backend=facts_backend,
        search_mode="heuristic",
        limits={"seconds": 120},
    )


@pytest.mark.parametrize("facts_backend", ["dense", "implicit"])
def test_exact_backends_split_by_the_status_that_drops_a_state(facts_backend) -> None:
    result = _analyze(facts_backend)
    splits = result["analysis"]["status_splits"]
    assert splits["status"] == "OFFERED", splits
    assert len(splits["cells"]) == 2, "both regimes split"
    for split in splits["cells"]:
        assert split["axis"] == "employment"
        lanes = {
            tuple(lane["codes"]["employment"]): set(lane["states"])
            for lane in split["lanes"]
        }
        assert lanes == {(0,): set(), (1,): {"experience"}}
        assert (split["state_rows"], split["unsplit_state_rows"]) == (1 + 3, 2 * 3)
    assert result["semantic_check"]["status"] != "FAIL", result["semantic_check"]


def test_metadata_backend_states_why_it_offers_no_split() -> None:
    splits = _analyze("metadata")["analysis"]["status_splits"]
    assert splits["status"] == "NOT_OFFERED"
    assert "per-status facts" in splits["reason"]


def test_a_status_that_changes_nothing_is_not_split() -> None:
    splits = _analyze("dense", status_matters=False)["analysis"]["status_splits"]
    assert splits["status"] == "NONE_USEFUL"
    assert not splits["cells"]


@pytest.mark.parametrize("forgery", ["drop_needed_state", "understate_rows", "overlap"])
def test_checker_refutes_a_forged_split(forgery) -> None:
    result = _analyze("dense")
    analysis = deepcopy(result["analysis"])
    cell = result["analysis"]["status_splits"]["cells"][0]["cell"]
    split = next(o for o in analysis["interfaces"][cell]["options"] if "lanes" in o)
    domain = analysis["derived_domains"][next(iter(split["states"].values()))]
    if forgery == "drop_needed_state":
        for lanes in (split["lanes"], domain["lanes"]):
            for lane in lanes:
                lane["states"].pop("experience", None)
        domain["size"] = 2
        expected = "status lane drops a needed state"
    elif forgery == "understate_rows":
        domain["size"] = 1
        expected = "status split rows differ from its lanes"
    else:
        for lanes in (split["lanes"], domain["lanes"]):
            lanes[0]["codes"]["employment"] = [0, 1]
        expected = "status lanes do not partition the axis codes"
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
        model(), facts_backend="dense", search_mode="heuristic"
    )
    (block,) = result.recommendations[0].blocks
    assert block.states == ("employment", "experience")
    assert block.lanes == (
        StatusLane(status="employment", codes=(0,), states=()),
        StatusLane(status="employment", codes=(1,), states=("experience",)),
    )


def test_the_report_names_the_lanes(tmp_path) -> None:
    result = analyze_regime_structure(
        model(), facts_backend="dense", search_mode="heuristic"
    )
    result.write_report(tmp_path / "report")
    text = "\n".join(
        p.read_text()
        for p in (tmp_path / "report").rglob("*")
        if p.suffix in {".md", ".html"}
    )
    assert (
        "split by employment [employment 0: no other states; employment 1: experience]"
        in text
    )
