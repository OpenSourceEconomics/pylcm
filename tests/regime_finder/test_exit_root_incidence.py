"""The independent source check binds the incidence of an exit's root relation.

An exit's availability and settlement default reach the dependency graph through the
exit's root relation, the exit relation without targets. Its callbacks, its phase and
its support are what make the states the exit reads live. A report whose root lost its
callback, claims zero support, holds in one phase only or lists its callback twice
fails the source check, even when every callback record, the exit's payload and its
observed shocks are authentic and every price is recomputed from the altered
declarations.
"""

from copy import deepcopy
from typing import Literal

import pytest

from _lcm_regime_finder import find_regimes, verify_certificate
from _lcm_regime_finder.facts import FactsBackendName, Record
from _lcm_regime_finder.layout_check import check_layouts
from _lcm_regime_finder.layout_search import search_layouts
from _lcm_regime_finder.model_snapshot import read_model
from _lcm_regime_finder.production import MetadataAnalysis, original_cost, recommend
from _lcm_regime_finder.semantic_check import check_semantics
from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    Regime,
    Unilateral,
    categorical,
)
from lcm.typing import BoolND, DiscreteState, FloatND, ScalarInt

BACKENDS: tuple[FactsBackendName, ...] = ("dense", "metadata")

type Mutation = Literal[
    "erase_callbacks", "zero_support", "simulation_phase", "duplicate_callback"
]

# The two alterations that drop the exit's live state `z` from the dependency closure.
DROPPING: tuple[Mutation, ...] = ("erase_callbacks", "zero_support")


@categorical(ordered=True)
class _Binary:
    zero: ScalarInt
    one: ScalarInt


@categorical(ordered=True)
class _Ternary:
    zero: ScalarInt
    one: ScalarInt
    two: ScalarInt


@categorical(ordered=False)
class _Regimes:
    couple: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt


def _stay_value() -> float:
    return 1.0


def _outside_value(x: DiscreteState) -> FloatND:
    return 2.0 + x


def _zero_x() -> int:
    return 0


def _exit_available(z: DiscreteState) -> BoolND:
    return z == 1


def _model(*, three_codes: bool) -> Model:
    """A couple whose exit is available only at `z == 1`; no other calculation reads `z`."""
    return Model(
        regimes={
            "couple": Regime(
                states={"z": DiscreteGrid(_Ternary if three_codes else _Binary)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _stay_value, "m": _stay_value}
                    )
                },
                exit=Exit(
                    targets={
                        "f": ExitTarget(regime="single_f", projection={"x": _zero_x}),
                        "m": ExitTarget(regime="single_m", projection={"x": _zero_x}),
                    },
                    rule=Unilateral(),
                    available=_exit_available,
                ),
            ),
            "single_f": Regime(
                states={"x": DiscreteGrid(_Binary)},
                functions={"utility": _outside_value},
            ),
            "single_m": Regime(
                states={"x": DiscreteGrid(_Binary)},
                functions={"utility": _outside_value},
            ),
        },
        n_periods=1,
        regime_id_class=_Regimes,
        initial_nodes=InitialNodes(by_period={0: ("couple", "single_f", "single_m")}),
        edges={},
        enable_jit=False,
    )


def _couple_id(report: Record) -> str:
    return next(
        c["id"]
        for c in report["analysis"]["declarations"]["cells"]
        if c["regime"] == "couple"
    )


def _altered_report(
    *, backend: FactsBackendName, mutation: Mutation, three_codes: bool
) -> tuple[Record, Record]:
    """Return the capture and a finder report whose exit root alone is altered.

    Only the root relation changes. The altered report's dependency closure, layouts,
    costs, recommendation and its own check records are rebuilt from the altered
    declarations, the way a finder that wrote such a root would produce them.
    """
    captured = read_model(_model(three_codes=three_codes))
    genuine = find_regimes(captured, facts_backend=backend)
    declarations = deepcopy(genuine["analysis"]["declarations"])
    couple = _couple_id(genuine)
    (root,) = [
        r
        for r in declarations["relations"]
        if r["kind"] == "exit" and r["source"] == couple and r["targets"] == []
    ]
    match mutation:
        case "erase_callbacks":
            root["callbacks"] = []
        case "zero_support":
            root["support"] = "zero"
        case "simulation_phase":
            root["phase"] = "simulation"
        case "duplicate_callback":
            root["callbacks"] = root["callbacks"] * 2
    metadata = MetadataAnalysis(
        declarations,
        {"seconds": 60, "metadata_categories": 100_000},
        facts_backend=backend,
    )
    if mutation == "erase_callbacks":
        analysis = metadata.run()
    else:
        # The source check visits the authentic callback even under a false zero
        # support, so the altered report keeps the genuine interfaces and closure.
        analysis = deepcopy(genuine["analysis"])
        analysis["declarations"] = declarations
    layouts = search_layouts(
        declarations, metadata.facts, metadata.limits, metadata.guard
    )
    cost = original_cost(declarations, metadata.domains)
    altered = deepcopy(genuine)
    altered.update(
        analysis=analysis,
        layouts=layouts,
        original_cost=cost,
        recommendation=recommend(
            layouts, cost, genuine.get("recommendation_preferences")
        ),
        semantic_check=check_semantics(captured, declarations, analysis),
        layout_check=check_layouts(declarations, metadata.facts, layouts),
    )
    return captured, altered


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("three_codes", [False, True])
def test_genuine_exit_root_passes(
    *, backend: FactsBackendName, three_codes: bool
) -> None:
    """The finder's own report keeps `z` live through the exit and passes both checks."""
    captured = read_model(_model(three_codes=three_codes))
    report = find_regimes(captured, facts_backend=backend)
    verified = verify_certificate(captured, report)
    assert (
        report["status"],
        "z" in report["analysis"]["dependency_closure"][_couple_id(report)],
        verified["status"],
        verified["source_semantics"]["status"],
        verified["layout_check"]["status"],
    ) == ("COMPLETE", True, "PASS", "PASS", "PASS")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mutation", DROPPING)
@pytest.mark.parametrize("three_codes", [False, True])
def test_exit_root_that_drops_the_exit_state_fails_the_source_check(
    *, backend: FactsBackendName, mutation: Mutation, three_codes: bool
) -> None:
    """Erased callbacks or a false zero support drop `z`, and the source check says so."""
    captured, altered = _altered_report(
        backend=backend, mutation=mutation, three_codes=three_codes
    )
    closure = MetadataAnalysis(
        altered["analysis"]["declarations"],
        {"seconds": 60, "metadata_categories": 100_000},
        facts_backend=backend,
    ).run()["dependency_closure"]
    verified = verify_certificate(captured, altered)
    kinds = {error["kind"] for error in verified["source_semantics"]["errors"]}
    assert (
        "z" in closure[_couple_id(altered)],
        verified["status"],
        verified["source_semantics"]["status"],
        "original exit root incidence changed" in kinds,
    ) == (False, "FAIL", "FAIL", True)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mutation", ["simulation_phase", "duplicate_callback"])
def test_exit_root_with_changed_phase_or_callback_list_fails_the_source_check(
    *, backend: FactsBackendName, mutation: Mutation
) -> None:
    """A root that holds in one phase only, or lists its callback twice, is rejected."""
    captured, altered = _altered_report(
        backend=backend, mutation=mutation, three_codes=False
    )
    verified = verify_certificate(captured, altered)
    kinds = {error["kind"] for error in verified["source_semantics"]["errors"]}
    assert (
        verified["status"],
        verified["source_semantics"]["status"],
        "original exit root incidence changed" in kinds,
    ) == ("FAIL", "FAIL", True)
