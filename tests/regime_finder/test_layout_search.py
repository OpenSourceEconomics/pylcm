"""The layout search of the Supplement (lcm-paper, supplement-choice.tex).

Dense regimes pool nonterminal regimes of one kind by the taste-shock steps
(Def. "Dense model", :126-146); every terminal regime stays a regime of its own.
"""

import jax.numpy as jnp
import pytest

from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.graph_scope import apply_scope
from _lcm_regime_finder.layout_search import dense_regimes, regime_kinds
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
