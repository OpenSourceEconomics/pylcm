"""The regime finder on collective models: pooling, counts, renegotiation and exits.

Every model is a constructed `Model` from the collective, exit, renegotiation, gate
and shared-problem tests, or a few-line model of the same kind. The certificate
tests alter a genuine report the way an unsound finder would write it and check that
the independent replay rejects it.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from functools import cache
from math import prod

import numpy as np
import pytest

import _lcm.regime_building.max_Q_over_a as max_Q_over_a
from _lcm_regime_finder import find_regimes, verify_certificate
from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.facts import FactsBackendName, Record, make_facts
from _lcm_regime_finder.layout_search import LayoutProblem, search_layouts
from _lcm_regime_finder.model_snapshot import read_model
from _lcm_regime_finder.records import JSONValue
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
from lcm.typing import BoolND, DiscreteState, FloatND, RegimeName, ScalarInt
from tests import test_transition_roles as transition_roles
from tests.collective_fixtures import make_two_stakeholder_model
from tests.regime_building import test_gate_shared_fallback as gate_fallback
from tests.regime_finder import native_conditional_current_shocks as observed_offer
from tests.simulation import test_query_only_participation_actors as value_readers
from tests.solution import test_mutual_consent_settlement as settlement
from tests.solution import test_renegotiation as renegotiation
from tests.solution import test_shared_problem as shared_problem

BACKENDS: tuple[FactsBackendName, ...] = ("dense", "metadata")

type Report = Record


@categorical(ordered=True)
class _Binary:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class _RoleRegimes:
    single: ScalarInt
    couple: ScalarInt


def _wife(x: DiscreteState) -> FloatND:
    return 1.0 + x


def _husband() -> float:
    return 1.0


def _stay() -> float:
    return 0.5


def _carry(w: DiscreteState) -> DiscreteState:
    return w


def _role_model() -> Model:
    """A single who may join a couple as its wife, whose value alone reads `x`."""
    return Model(
        regimes={
            "single": Regime(
                states={"w": DiscreteGrid(_Binary)},
                functions={"utility": _stay},
                exit=Exit(
                    targets={
                        "self": ExitTarget(
                            regime="couple", role="f", projection={"x": _carry}
                        )
                    },
                    rule=Unilateral(),
                ),
            ),
            "couple": Regime(
                states={"x": DiscreteGrid(_Binary)},
                functions={
                    "utility": CollectiveUtility(utilities={"f": _wife, "m": _husband})
                },
            ),
        },
        n_periods=1,
        regime_id_class=_RoleRegimes,
        initial_nodes=InitialNodes(by_period={0: ("single", "couple")}),
        edges={},
        enable_jit=False,
    )


def _equal_exit_couples() -> Model:
    """The hand-declared twin couples, both dissolving under one rule."""
    regimes = shared_problem._twin_regimes()
    regimes["couple_m"] = regimes["couple_u"]
    return shared_problem._model(regimes=regimes, skill=True)


_MODELS: dict[str, Callable[[], Model]] = {
    "roles": lambda: transition_roles.build_model(edge_form="roles"),
    "plain": lambda: transition_roles.build_model(edge_form="plain"),
    "exact_gate": lambda: transition_roles.build_model(edge_form="exact_gate"),
    "hand_gate": lambda: transition_roles.build_model(edge_form="hand_gate"),
    "twin": lambda: shared_problem._model(
        regimes=shared_problem._twin_regimes(), skill=True
    ),
    "equal_exits": _equal_exit_couples,
    "shared": lambda: shared_problem._model(
        regimes=shared_problem._shared_regimes(skill=True), skill=True
    ),
    "renegotiate_1p": lambda: renegotiation._model(values=renegotiation._VALUES),
    "renegotiate_2p": lambda: renegotiation._model(
        values=renegotiation._VALUES, n_periods=2
    ),
    "two_stakeholder": lambda: make_two_stakeholder_model()[0],
    "role": _role_model,
    "observed_offer": observed_offer.model,
    "settle": lambda: settlement._settlement_model(division="transfer"),
    "survivor_gate": lambda: gate_fallback.build_model(route_form="survivor"),
    "shared_fallback": lambda: gate_fallback.build_model(route_form="both_phases"),
    "value_reader": lambda: value_readers.household_model(target_payoff=5, budget=None),
}


@cache
def _model(label: str) -> Model:
    return _MODELS[label]()


@cache
def _captured(label: str) -> Report:
    """The model's captured declarations, which a report and its replay share."""
    return read_model(_model(label))


@cache
def _report(label: str, backend: FactsBackendName) -> Report:
    return find_regimes(_captured(label), facts_backend=backend)


def _named(report: Report, cell_id: str) -> str:
    cells = {c["id"]: c for c in report["analysis"]["declarations"]["cells"]}
    return f"{cells[cell_id]['regime']}@{cells[cell_id]['period']}"


def _pooled(report: Report, *regimes: str) -> bool:
    return any(
        set(regimes) <= set(group) for group in report["layouts"]["dense_regimes"]
    )


def _layouts_without(report: Report, *, key: str, contract: bool = True) -> Report:
    """Search the report's declarations again with `key` removed from every cell."""
    declarations = deepcopy(report["analysis"]["declarations"])
    for cell in declarations["cells"]:
        cell.pop(key, None)
    limits = {"seconds": 60, "metadata_categories": 100_000}
    facts = make_facts(report["facts_backend"], declarations, limits, lambda: None)
    if contract:
        return search_layouts(declarations, facts, limits, lambda: None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LayoutProblem, "_inert_statuses", lambda _: frozenset())
        return search_layouts(declarations, facts, limits, lambda: None)


def _verify_altered(
    label: str, backend: FactsBackendName, *, layouts: dict[str, JSONValue]
) -> Report:
    altered = deepcopy(_report(label, backend))
    altered["layouts"] = layouts
    return verify_certificate(_captured(label), altered)


def _verify_with_cells(
    label: str,
    backend: FactsBackendName,
    *,
    regime: RegimeName,
    key: str,
    value: JSONValue,
) -> Report:
    altered = deepcopy(_report(label, backend))
    for cell in altered["analysis"]["declarations"]["cells"]:
        if cell["regime"] == regime:
            cell[key] = value
    return verify_certificate(_captured(label), altered)


@pytest.mark.parametrize("backend", BACKENDS)
def test_collective_regime_does_not_pool_with_a_single_agent_regime(
    backend: FactsBackendName,
) -> None:
    """The couple of `Transition(roles=...)` stays apart from the single."""
    assert not _pooled(_report("roles", backend), "couple", "single")


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_agent_regimes_of_one_kind_still_pool(
    backend: FactsBackendName,
) -> None:
    """Without stakeholders, the same couple and single pool as one dense regime."""
    assert _pooled(_report("plain", backend), "couple", "single")


@pytest.mark.parametrize("backend", BACKENDS)
def test_regimes_with_different_exits_do_not_pool(backend: FactsBackendName) -> None:
    """A unilateral and a mutual-consent couple stay apart."""
    assert not _pooled(_report("twin", backend), "couple_m", "couple_u")


@pytest.mark.parametrize("backend", BACKENDS)
def test_regimes_with_equal_exits_still_pool(backend: FactsBackendName) -> None:
    """Two couples with one stakeholder tuple and one exit pool."""
    assert _pooled(_report("equal_exits", backend), "couple_m", "couple_u")


def test_normalized_cells_carry_their_stakeholders() -> None:
    """Each cell names its regime's stakeholders, `None` for a single agent."""
    cells = _report("roles", "dense")["analysis"]["declarations"]["cells"]
    assert {c["regime"]: c.get("stakeholders", "absent") for c in cells} == {
        "single": None,
        "couple": ["f", "m"],
        "single_end": None,
        "couple_end": ["f", "m"],
    }


@pytest.mark.parametrize(
    ("label", "key", "regimes"),
    [
        ("roles", "stakeholders", ("couple", "single")),
        ("twin", "exit_identity", ("couple_m", "couple_u")),
    ],
)
def test_certificate_rejects_a_layout_that_pools_unequal_regimes(
    label: str, key: str, regimes: tuple[str, ...]
) -> None:
    """A layout curve searched without the regimes' kind fails the layout check."""
    genuine = verify_certificate(_captured(label), _report(label, "dense"))
    layouts = _layouts_without(_report(label, "dense"), key=key)
    altered = _verify_altered(label, "dense", layouts=layouts)
    assert (
        genuine["layout_check"]["status"],
        any(set(regimes) <= set(group) for group in layouts["dense_regimes"]),
        altered["layout_check"]["status"],
    ) == ("PASS", True, "FAIL")


@pytest.mark.parametrize(
    ("label", "regime", "key", "value"),
    [
        ("roles", "couple", "stakeholders", None),
        ("twin", "couple_m", "exit_identity", None),
        ("renegotiate_2p", "couple", "renegotiated_states", []),
    ],
)
def test_certificate_rejects_cells_that_lose_their_kind(
    label: str, regime: RegimeName, key: str, value: JSONValue
) -> None:
    """The source check rejects cells whose stakeholders, exit or renegotiation changed."""
    genuine = verify_certificate(_captured(label), _report(label, "dense"))
    verified = _verify_with_cells(label, "dense", regime=regime, key=key, value=value)
    assert (
        genuine["source_semantics"]["status"] == "FAIL",
        verified["source_semantics"]["status"],
    ) == (False, "FAIL")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("label", ["renegotiate_1p", "renegotiate_2p"])
def test_renegotiation_state_is_never_a_status(
    label: str, backend: FactsBackendName
) -> None:
    """The renegotiated weight is neither contracted, nor split, nor carried as a status."""
    layouts = _report(label, backend)["layouts"]
    lane_statuses = {
        name
        for lanes in layouts["lanes"]
        for lane in lanes
        for name in lane["statuses"]
    }
    assert ("mu" in layouts["inert_statuses"], "mu" in lane_statuses) == (False, False)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("label", ["renegotiate_1p", "renegotiate_2p"])
def test_renegotiation_state_is_live_and_kept_whole(
    label: str, backend: FactsBackendName
) -> None:
    """Every couple cell keeps `mu`, and every block holding the couple stores all its codes."""
    report = _report(label, backend)
    live = {
        _named(report, c): "mu" in v
        for c, v in report["analysis"]["dependency_closure"].items()
        if _named(report, c).startswith("couple")
    }
    grids = {
        str(block["states"].get("mu"))
        for point in report["layouts"]["curve"]
        for group in point["layout"]
        if "couple" in group["dense_regime"]
        for block in group["blocks"]
    }
    assert (set(live.values()), grids) == ({True}, {"4"})


def test_certificate_rejects_a_layout_that_splits_the_renegotiation_state() -> None:
    """Lanes split by the renegotiated weight fail the layout check."""
    report = _report("renegotiate_2p", "dense")
    genuine = verify_certificate(_captured("renegotiate_2p"), report)
    layouts = _layouts_without(report, key="renegotiated_states", contract=False)
    altered = _verify_altered("renegotiate_2p", "dense", layouts=layouts)
    split = any(
        "mu" in lane["statuses"] for lanes in layouts["lanes"] for lane in lanes
    )
    assert (
        genuine["layout_check"]["status"],
        split,
        altered["layout_check"]["status"],
    ) == (
        "PASS",
        True,
        "FAIL",
    )


def _engine_counts() -> tuple[int, int]:
    """Solve the two-stakeholder model; return its stored values and action values.

    The collective readout runs once per state node of each regime-period and
    receives every stakeholder's action values there; the stored value carries
    the state axes and a trailing stakeholder axis.
    """
    per_node: list[int] = []
    readout = max_Q_over_a.collective_readout

    def recording(
        *,
        stakeholder_Q: Mapping[str, FloatND],
        feasibility: BoolND,
        weights: Mapping[str, FloatND | float],
        action_axes: tuple[int, ...],
    ) -> tuple[dict[str, FloatND], BoolND]:
        per_node.append(sum(prod(q.shape) for q in stakeholder_Q.values()))
        return readout(
            stakeholder_Q=stakeholder_Q,
            feasibility=feasibility,
            weights=weights,
            action_axes=action_axes,
        )

    model, params = make_two_stakeholder_model()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(max_Q_over_a, "collective_readout", recording)
        values = model.solve(params=params, log_level="off").values
    shapes = [
        np.asarray(v).shape for regimes in values.values() for v in regimes.values()
    ]
    assert len(per_node) == len(shapes) and len(set(per_node)) == 1
    stored = sum(prod(shape) for shape in shapes)
    return stored, sum(prod(shape[:-1]) for shape in shapes) * per_node[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_storage_counts_the_stakeholder_axis(backend: FactsBackendName) -> None:
    """The finder's stored values of the model as written equal the solve's."""
    stored, _ = _engine_counts()
    report = _report("two_stakeholder", backend)
    assert (
        report["original_cost"]["storage"],
        report["layouts"]["curve"][0]["storage"],
    ) == (stored, stored)


@pytest.mark.parametrize("backend", BACKENDS)
def test_evaluations_count_the_stakeholder_axis(backend: FactsBackendName) -> None:
    """The finder's evaluations of the model as written equal the solve's action values."""
    _, evaluated = _engine_counts()
    report = _report("two_stakeholder", backend)
    assert (report["original_cost"]["work"], report["layouts"]["original_count"]) == (
        evaluated,
        evaluated,
    )


def test_exit_rule_kind_tie_and_compares_enter_the_exit_identity() -> None:
    """Eight exit rules of one couple give eight exit identities."""
    identities = set()
    for rule in ("unilateral", "mutual"):
        for compares in ("each_action", "optimum"):
            for tie in ("stay", "leave"):
                model = settlement._comparison_model(
                    rule=settlement._rule(rule=rule, compares=compares, tie=tie)
                )
                cells = normalize_declarations(read_model(model))["cells"]
                identities |= {
                    c.get("exit_identity") for c in cells if c["regime"] == "couple"
                }
    assert len(identities - {None}) == 8


@pytest.mark.parametrize("backend", BACKENDS)
def test_exit_role_keeps_the_projected_state_live_and_is_reported(
    backend: FactsBackendName,
) -> None:
    """A single joining a couple as `f` keeps `w`, and its exit names the role."""
    report = _report("role", backend)
    closure = {
        _named(report, c): v
        for c, v in report["analysis"]["dependency_closure"].items()
    }
    roles = {
        r.get("stakeholder")
        for r in report["analysis"]["declarations"]["relations"]
        if r["kind"] == "exit" and r["targets"]
    }
    assert (closure["single@0"], roles) == (["w"], {"f"})


@pytest.mark.parametrize("backend", BACKENDS)
def test_exit_observes_keeps_the_law_state_live_and_is_reported(
    backend: FactsBackendName,
) -> None:
    """An observed offer whose law reads the history keeps it, and the exit names it."""
    report = _report("observed_offer", backend)
    closure = {
        _named(report, c): v
        for c, v in report["analysis"]["dependency_closure"].items()
    }
    observed = [
        r.get("observes")
        for r in report["analysis"]["declarations"]["relations"]
        if r["kind"] == "exit" and not r["targets"]
    ]
    assert (closure["source@0"], observed) == (["lagged_school"], [["offer"]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_settlement_reads_stay_live(backend: FactsBackendName) -> None:
    """The settlement's default and projections keep the case they read."""
    report = _report("settle", backend)
    closure = {
        _named(report, c): v
        for c, v in report["analysis"]["dependency_closure"].items()
    }
    assert closure["couple@0"] == ["case"]


@pytest.mark.parametrize("label", ["survivor_gate", "shared_fallback"])
def test_gates_with_legs_and_shared_fallbacks_stay_complete(label: str) -> None:
    """A gate with stakeholder legs and fallbacks, one of them shared, is checked."""
    report = _report(label, "dense")
    fallbacks = [
        r
        for r in report["analysis"]["declarations"]["relations"]
        if r["kind"] == "fallback"
    ]
    assert (
        report["status"],
        bool(fallbacks),
        all(r["ownership"].get("leg") for r in fallbacks),
    ) == (
        "COMPLETE",
        True,
        True,
    )


def test_roles_equal_their_lowered_gate() -> None:
    """`Transition(roles=...)` and its hand-lowered exact gate give one layout curve."""
    curves = [
        [
            (p["blocks"], p["evaluations"], p["storage"])
            for p in _report(label, "dense")["layouts"]["curve"]
        ]
        for label in ("roles", "exact_gate")
    ]
    assert curves[0] == curves[1]


@pytest.mark.parametrize("label", ["shared", "value_reader"])
def test_problem_holders_and_value_readers_stay_unresolved(label: str) -> None:
    """A `Problem` holder and a reader of a collective value keep an open obligation."""
    assert _report(label, "dense")["status"] == "UNRESOLVED"


def test_gate_whose_fallback_is_its_source_shows_no_internal_error() -> None:
    """A hand-written gate falling back to the source regime reports no engine exception."""
    report = _report("hand_gate", "dense")
    assert "NotImplementedError" not in str(report.get("error", ""))
