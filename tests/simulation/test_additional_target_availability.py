"""Availability classification for deferred simulation targets."""

import inspect
from typing import cast

from dags import dag as dags_dag

from _lcm.engine import Regime
from _lcm.simulation import additional_targets as target_module


def test_all_candidates_are_classified_from_one_dependency_graph(monkeypatch):
    """`available_targets` classifies one regime from one dependency graph."""

    def source(x):
        return x

    def split(source):
        return source

    def decision(split):
        return split

    def downstream(decision):
        return decision

    def safe(source):
        return source

    pool = {
        "source": source,
        "split": split,
        "decision": decision,
        "downstream": downstream,
        "safe": safe,
    }
    regime = cast("Regime", object())
    monkeypatch.setattr(target_module, "_build_functions_pool", lambda _regime: pool)
    monkeypatch.setattr(
        target_module,
        "_unresolvable_transition_names",
        lambda _regime: set(),
    )
    monkeypatch.setattr(
        target_module,
        "_phase_split_transition_names",
        lambda _regime: {"split"},
    )

    original_create_dag = dags_dag.create_dag
    calls = 0

    def counted_create_dag(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_create_dag(*args, **kwargs)

    monkeypatch.setattr(dags_dag, "create_dag", counted_create_dag)

    classify = inspect.unwrap(target_module._decision_only_target_names)
    decision_only = classify(
        regime=regime,
        candidates={"decision", "downstream", "safe"},
    )

    assert decision_only == {"decision", "downstream"}
    assert calls == 1
