"""The layout search of the Supplement (lcm-paper, supplement-choice.tex).

Dense regimes pool nonterminal regimes of one kind by the taste-shock steps
(Def. "Dense model", :126-146); every terminal regime stays a regime of its own.
"""

import jax.numpy as jnp
import pytest

from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.graph_scope import apply_scope
from _lcm_regime_finder.facts import Record
from _lcm_regime_finder.layout_search import LayoutProblem, dense_regimes, regime_kinds
from _lcm_regime_finder.model_snapshot import read_model
from _lcm_regime_finder.records import JSONValue
from lcm import DiscreteGrid, InitialNodes, Model, Periods, Regime, categorical
from lcm.typing import DiscreteState, FloatND, ScalarInt


def _kind(*, terminal=False, key="k", taste=None, discrete=True):
    return {
        "terminal": terminal,
        "key": key,
        "taste_shocks": taste,
        "discrete_choice": discrete,
    }


def test_terminal_regimes_never_pool() -> None:
    kinds = {
        "a": _kind(),
        "b": _kind(),
        "dead": _kind(terminal=True),
        "gone": _kind(terminal=True),
    }
    assert dense_regimes(kinds) == [("a", "b"), ("dead",), ("gone",)]


def test_regimes_of_different_kinds_stay_apart() -> None:
    kinds = {"a": _kind(key="grid"), "b": _kind(key="dcegm")}
    assert dense_regimes(kinds) == [("a",), ("b",)]


def test_a_shocked_regime_stays_apart_from_degenerate_regimes_with_choice() -> None:
    """Step (i) pools equal laws only; two shocked regimes are not shown equal."""
    kinds = {"a": _kind(), "b": _kind(), "c": _kind(taste="ev"), "d": _kind(taste="ev")}
    assert dense_regimes(kinds) == [("a", "b"), ("c",), ("d",)]


def test_step_iii_pools_regimes_without_choice_with_the_one_shocked_regime() -> None:
    kinds = {
        "a": _kind(discrete=False),
        "b": _kind(discrete=False),
        "c": _kind(taste="ev"),
    }
    assert dense_regimes(kinds) == [("a", "b", "c")]


def test_step_iii_needs_exactly_one_shocked_regime() -> None:
    kinds = {
        "a": _kind(discrete=False),
        "c": _kind(taste="ev"),
        "d": _kind(taste="ev"),
    }
    assert dense_regimes(kinds) == [("a",), ("c",), ("d",)]


@categorical(ordered=False)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class Stages:
    young: ScalarInt
    old: ScalarInt
    dead: ScalarInt


def _utility(health: DiscreteState) -> FloatND:
    return jnp.where(health == Health.good, 1.0, 0.0)


def _stay(health: DiscreteState) -> DiscreteState:
    return health


def life_model() -> Model:
    states = {"health": DiscreteGrid(Health)}
    return Model(
        n_periods=3,
        regime_id_class=Stages,
        regimes={
            "young": Regime(
                states=states,
                functions={"utility": _utility},
                state_transitions={"health": _stay},
            ),
            "old": Regime(
                states=states,
                functions={"utility": _utility},
                state_transitions={"health": _stay},
            ),
            "dead": Regime(states=states, functions={"utility": _utility}),
        },
        initial_nodes=InitialNodes(by_period={0: "young"}),
        edges={
            "young": {"old": Periods(values=(0,))},
            "old": {"dead": Periods(values=(1,))},
        },
        fixed_params={"discount_factor": 1.0},
        enable_jit=False,
    )


def _declarations(model: Model) -> dict[str, JSONValue]:
    original = read_model(model)
    return apply_scope(original, normalize_declarations(original))


def test_a_model_pools_its_living_regimes_and_keeps_the_terminal_one_apart() -> None:
    kinds = regime_kinds(_declarations(life_model()))
    assert kinds["dead"]["terminal"] and not kinds["young"]["terminal"]
    assert kinds["young"]["key"] == kinds["old"]["key"]
    assert dense_regimes(kinds) == [("dead",), ("old", "young")]


def test_a_fibre_keeps_what_its_status_point_needs_statuses_included() -> None:
    """The least profile (Def. "Profiles from the facts", :403-428) at full points.

    Experience pays only the employed and healthy, and health matters only at
    work. Statuses are released in name order while the release adds no read of
    the released status, the others still fixed (Remark "Reads must be joint").
    Unhealthy, the payoff is 0 whoever works, so employment is released first and
    health stays live; healthy and unemployed, employment stays live instead.
    """
    from _lcm_regime_finder.facts import fibre_live, make_facts

    from .test_status_splitting import joint_model

    declarations = _declarations(joint_model())
    facts = make_facts("dense", declarations, {"seconds": 120}, lambda: None)
    cell = next(c for c in facts.cells.values() if c["regime"] == "working_age")
    memo: dict[tuple[str, tuple[tuple[str, int], ...]], set[str]] = {}
    expected = {
        (0, 0): {"health"},
        (0, 1): {"employment"},
        (1, 0): {"health"},
        (1, 1): {"employment", "health", "experience"},
    }
    for (employment, health), live in expected.items():
        for experience in (0, 1, 2):
            point = {
                "employment": employment,
                "health": health,
                "experience": experience,
            }
            assert fibre_live(facts, cell, point, memo) == live, point


def _search(model: Model, **limits: int) -> tuple[Record, LayoutProblem]:
    from _lcm_regime_finder.facts import make_facts
    from _lcm_regime_finder.layout_search import search_layouts

    declarations = _declarations(model)
    facts = make_facts("dense", declarations, {"seconds": 120}, lambda: None)
    budget = {"max_blocks": 64, **limits}
    return (
        search_layouts(declarations, facts, budget, lambda: None),
        LayoutProblem(declarations, facts, budget, lambda: None),
    )


@pytest.mark.parametrize("fixture", ["joint", "life"])
def test_the_recurrence_matches_literal_enumeration(fixture) -> None:
    """w(n) and ev*(k) equal the literal minimum over every partition and cut."""
    from .layout_oracle import literal_tables
    from .test_status_splitting import joint_model

    result, problem = _search(joint_model() if fixture == "joint" else life_model())
    exact, at_most = literal_tables(problem)
    assert {e["blocks"]: (e["evaluations"], e["storage"]) for e in result["exact"]} == {
        n: v for n, v in exact.items() if n >= len(problem.dense)
    }
    curve = {
        p["max_blocks"]: (p["evaluations"], p["blocks"], p["storage"])
        for p in result["curve"]
    }
    assert {k: curve[k] for k in at_most} == at_most
    assert all(curve[k] == at_most[max(at_most)] for k in curve if k > max(at_most))


def test_a_given_layout_is_priced_from_its_fibres() -> None:
    """Pooling young and old pays both regimes' labels in both periods."""
    from _lcm_regime_finder.layout_search import analyze_layouts

    result = analyze_layouts(
        life_model(),
        limits={"max_blocks": 8},
        price=[
            {"regimes": ["young", "old"], "periods": [0, 1]},
            {"regimes": ["dead"], "periods": [2]},
        ],
    )
    pooled, dead = result["priced"]
    assert (pooled["evaluations"], pooled["storage"]) == (2 * 2 * 2, 2 * 2 * 2)
    assert pooled["states"] == {"regime": ["old", "young"], "health": [0, 1]}
    assert dead["evaluations"] == 2
    assert result["original_count"] == 3 * 2
    assert result["dense_count"] == 2 * 2 * 2 + 2, (
        "ev^0 charges both labels in both periods"
    )
    assert set(result["phase_timings"]) == {"facts", "search"}


@pytest.mark.parametrize("fixture", ["joint", "life"])
def test_the_independent_checker_passes_the_search(fixture) -> None:
    from _lcm_regime_finder.facts import make_facts
    from _lcm_regime_finder.layout_check import check_layouts

    from .test_status_splitting import joint_model

    declarations = _declarations(joint_model() if fixture == "joint" else life_model())
    facts = make_facts("dense", declarations, {"seconds": 120}, lambda: None)
    result, _ = _search(joint_model() if fixture == "joint" else life_model())
    assert check_layouts(declarations, facts, result) == {
        "status": "PASS",
        "errors": [],
    }


@pytest.mark.parametrize(
    "forgery", ["underprice", "drop_lane", "bad_cut", "inert", "envelope"]
)
def test_the_independent_checker_refutes_forgeries(forgery) -> None:
    import copy

    from _lcm_regime_finder.facts import make_facts
    from _lcm_regime_finder.layout_check import check_layouts

    from .test_status_splitting import joint_model

    declarations = _declarations(joint_model())
    facts = make_facts("dense", declarations, {"seconds": 120}, lambda: None)
    result, _ = _search(joint_model())
    forged = copy.deepcopy(result)
    point = forged["curve"][1]
    if forgery == "underprice":
        point["layout"][0]["blocks"][0]["evaluations"] -= 1
    elif forgery == "drop_lane":
        point["layout"][0]["lanes"].pop()
    elif forgery == "bad_cut":
        point["layout"][0]["cuts"] = [0]
    elif forgery == "inert":
        forged["inert_statuses"] = ["employment", "experience", "health"]
    else:
        point["evaluations"] -= 1
    assert check_layouts(declarations, facts, forged)["status"] == "FAIL"


@categorical(ordered=False)
class Work:
    rest: ScalarInt
    work: ScalarInt


def _work_utility(health: DiscreteState, work: DiscreteState) -> FloatND:
    return jnp.where(health == Health.good, 1.0 + work, 0.0)


@categorical(ordered=False)
class Ends:
    young: ScalarInt
    dead: ScalarInt


def test_layouts_run_with_default_limits_on_a_model_with_a_discrete_action() -> None:
    from _lcm_regime_finder.layout_search import analyze_layouts

    states = {"health": DiscreteGrid(Health)}
    model = Model(
        n_periods=2,
        regime_id_class=Ends,
        regimes={
            "young": Regime(
                states=states,
                actions={"work": DiscreteGrid(Work)},
                functions={"utility": _work_utility},
                state_transitions={"health": _stay},
            ),
            "dead": Regime(states=states, functions={"utility": _utility}),
        },
        initial_nodes=InitialNodes(by_period={0: "young"}),
        edges={"young": {"dead": Periods(values=(0,))}},
        fixed_params={"discount_factor": 1.0},
        enable_jit=False,
    )
    result = analyze_layouts(model)
    assert result["status"] == "COMPLETE"
    assert result["curve"][0]["evaluations"] == 2 * 2 + 2


def _impute_pension(aime: float) -> float:
    return aime * 0.1


def _keep_aime(aime: float) -> float:
    return aime


def _next_cash(*, cash: float, consumption: float, pension: float) -> float:
    return cash - consumption + pension


def _log_utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _affordable(*, consumption: float, cash: float) -> bool:
    return consumption <= cash


def _carried_model() -> Model:
    """Pension wealth is imputed from AIME within the cell; `noise` is read by nothing."""
    from lcm import LinSpacedGrid, Phased

    working = Regime(
        states={
            "cash": LinSpacedGrid(start=1.0, stop=10.0, n_points=4),
            "aime": LinSpacedGrid(start=1.0, stop=5.0, n_points=3),
            "noise": LinSpacedGrid(start=0.0, stop=1.0, n_points=5),
            "pension": Phased(
                solve=_impute_pension,
                simulate=LinSpacedGrid(start=0.0, stop=2.0, n_points=4),
            ),
        },
        state_transitions={
            "cash": _next_cash,
            "aime": _keep_aime,
            "noise": _keep_noise,
            "pension": _grow_pension,
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=3)},
        constraints={"affordable": _affordable},
        functions={"utility": _log_utility},
    )
    return Model(
        n_periods=3,
        regime_id_class=Ends,
        regimes={"young": working, "dead": Regime(functions={"utility": _zero})},
        initial_nodes=InitialNodes(by_period={0: "young"}),
        edges={"young": {"young": Periods(values=(0,)), "dead": Periods(values=(1,))}},
        fixed_params={"discount_factor": 1.0},
        enable_jit=False,
    )


def _keep_noise(noise: float) -> float:
    return noise


def _grow_pension(pension: float) -> float:
    return pension * 1.03


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def test_an_in_cell_imputation_is_neither_a_link_nor_a_reason_to_keep_everything() -> (
    None
):
    """The imputation adds what it reads for a live imputed state, at the same fibre."""
    from _lcm_regime_finder.layout_search import analyze_layouts

    result = analyze_layouts(_carried_model(), limits={"max_blocks": 6})
    first = result["curve"][0]["layout"]
    states = {
        name for group in first for block in group["blocks"] for name in block["states"]
    }
    assert {"cash", "aime"} <= states
    assert "noise" not in states


def test_a_block_records_the_values_it_excludes_for_an_action_its_members_lack() -> (
    None
):
    """(E3) with the fill rule: a member without `work` can only take its first value."""
    from _lcm_regime_finder.layout_search import analyze_layouts

    states = {"health": DiscreteGrid(Health)}
    model = Model(
        n_periods=3,
        regime_id_class=Stages,
        regimes={
            "young": Regime(
                states=states,
                actions={"work": DiscreteGrid(Work)},
                functions={"utility": _work_utility},
                state_transitions={"health": _stay},
            ),
            "old": Regime(
                states=states,
                functions={"utility": _utility},
                state_transitions={"health": _stay},
            ),
            "dead": Regime(states=states, functions={"utility": _utility}),
        },
        initial_nodes=InitialNodes(by_period={0: "young"}),
        edges={
            "young": {"old": Periods(values=(0,))},
            "old": {"dead": Periods(values=(1,))},
        },
        fixed_params={"discount_factor": 1.0},
        enable_jit=False,
    )
    result = analyze_layouts(
        model,
        price=[
            {"regimes": ["old"], "periods": [1]},
            {"regimes": ["young", "old"], "periods": [0, 1]},
        ],
    )
    alone, pooled = result["priced"]
    assert alone["actions"] == {} and alone["excluded_actions"] == {"work": [1]}
    assert pooled["actions"] == {"work": 2} and pooled["excluded_actions"] == {}


@categorical(ordered=False)
class WideCover:
    none: ScalarInt
    some: ScalarInt
    full: ScalarInt


@categorical(ordered=False)
class NarrowCover:
    none: ScalarInt
    some: ScalarInt


def _no_cover() -> DiscreteState:
    return jnp.asarray(0)


def _covered_utility(health: DiscreteState, cover: DiscreteState) -> FloatND:
    return jnp.where(health == Health.good, 1.0, 0.0) + cover


def test_a_status_whose_grid_differs_across_members_is_not_certified_inert() -> None:
    """(I1) takes parts over the dense regime's grid (:1743-1747).

    Old has no `full` cover, so the part (old, full) of a kept fibre is not kept.
    Contracting cover is then a restriction, and the curve an upper bound.
    """
    from _lcm_regime_finder.layout_search import analyze_layouts

    model = Model(
        n_periods=3,
        regime_id_class=Stages,
        regimes={
            "young": Regime(
                states={
                    "health": DiscreteGrid(Health),
                    "cover": DiscreteGrid(WideCover),
                },
                functions={"utility": _covered_utility},
                state_transitions={"health": _stay, "cover": {"old": _no_cover}},
            ),
            "old": Regime(
                states={
                    "health": DiscreteGrid(Health),
                    "cover": DiscreteGrid(NarrowCover),
                },
                functions={"utility": _covered_utility},
                state_transitions={"health": _stay, "cover": _no_cover},
            ),
            "dead": Regime(
                states={"health": DiscreteGrid(Health)}, functions={"utility": _utility}
            ),
        },
        initial_nodes=InitialNodes(by_period={0: "young"}),
        edges={
            "young": {"old": Periods(values=(0,))},
            "old": {"dead": Periods(values=(1,))},
        },
        fixed_params={"discount_factor": 1.0},
        enable_jit=False,
    )
    result = analyze_layouts(model, limits={"max_blocks": 6})
    assert "cover" not in result["inert_statuses"]
    assert "cover" in result["restricted_statuses"]
    assert result["status"] == "RESTRICTED"
    assert result["floor"] is None, "a floor of contracted lanes bounds nothing"
    assert "differs" in result["restriction_reasons"]["cover"]
    first, *rest = result["curve"]
    assert first["exact"], "one block per dense regime is exact in any lane family"
    assert not any(point["exact"] for point in rest)
    full = analyze_layouts(model, limits={"max_blocks": 6, "contract_unproved": False})
    assert full["status"] == "COMPLETE"
    assert {
        lane["statuses"].get("cover") for lanes in full["lanes"] for lane in lanes
    } >= {0, 1, 2}
    assert all(
        f["evaluations"] <= r["evaluations"]
        for f, r in zip(full["curve"], result["curve"], strict=True)
    )
