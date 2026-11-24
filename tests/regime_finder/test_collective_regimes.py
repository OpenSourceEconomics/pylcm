"""The regime finder on collective models: pooling, counts, renegotiation and exits.

Every model is a constructed `Model` from the collective, exit, renegotiation and
gate tests, or a few-line model of the same kind. The certificate
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
    Consent,
    Counterparty,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    MutualConsent,
    Periods,
    Regime,
    Unilateral,
    categorical,
    fixed_transition,
)
from lcm.exits import ExitRule
from lcm.shocks import GaussHermite, Normal, Product, ShockGroup
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteState,
    FloatND,
    RegimeName,
    ScalarInt,
)
from tests import test_transition_roles as transition_roles
from tests.collective_fixtures import make_two_member_model
from tests.regime_building import test_gate_shared_fallback as gate_fallback
from tests.regime_finder import native_conditional_current_shocks as observed_offer
from tests.solution import test_mutual_consent_settlement as settlement
from tests.solution import test_renegotiation as renegotiation

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


@categorical(ordered=True)
class _Skill:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _Household:
    couple_u: ScalarInt
    couple_m: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt
    couple_end: ScalarInt
    single_end: ScalarInt


def _twin_wife(*, wealth: ContinuousState, match: FloatND) -> FloatND:
    return 3.0 * wealth + match


def _twin_husband(*, wealth: ContinuousState, skill: DiscreteState) -> FloatND:
    return wealth + skill


def _twin_outside(wealth: ContinuousState) -> FloatND:
    return 2.0 * wealth


def _twin_regimes() -> dict[RegimeName, Regime]:
    """Two couples that differ only in the rule of their exit, and their singles.

    One couple dissolves unless both members are willing to stay, the other only
    if neither is. The couples read a current shock and the model-level state
    `skill`; every regime moves on to a terminal regime of its kind.
    """

    def couple(*, last: bool, rule: ExitRule | None) -> Regime:
        return Regime(
            states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
            state_transitions={} if last else {"wealth": fixed_transition("wealth")},
            functions={
                "utility": CollectiveUtility(
                    utilities={"f": _twin_wife, "m": _twin_husband}
                )
            },
            shocks=("match",),
            exit=None
            if rule is None
            else Exit(
                rule=rule,
                targets={
                    member: ExitTarget(
                        regime=f"single_{member}",
                        projection={
                            "wealth": fixed_transition("wealth"),
                            "skill": fixed_transition("skill"),
                        },
                    )
                    for member in ("f", "m")
                },
            ),
        )

    def single(*, last: bool) -> Regime:
        return Regime(
            states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
            state_transitions={} if last else {"wealth": fixed_transition("wealth")},
            functions={"utility": _twin_outside},
        )

    return {
        "couple_u": couple(last=False, rule=Unilateral()),
        "couple_m": couple(last=False, rule=MutualConsent()),
        "single_f": single(last=False),
        "single_m": single(last=False),
        "couple_end": couple(last=True, rule=None),
        "single_end": single(last=True),
    }


def _twin_model(regimes: Mapping[RegimeName, Regime]) -> Model:
    """Build the two-period couples model with the model-level state `skill`."""
    return Model(
        regimes=regimes,
        states={"skill": DiscreteGrid(category_class=_Skill)},
        state_transitions={"skill": fixed_transition("skill")},
        shocks={
            "match": ShockGroup(
                shocks={"match": Normal(mu=0.0, sigma=0.5)},
                nodes=Product(match=GaussHermite(n=2)),
            )
        },
        n_periods=2,
        regime_id_class=_Household,
        initial_nodes=InitialNodes(by_period={0: ("couple_u", "couple_m")}),
        edges={
            source: {target: Periods(values=(0,))}
            for source, target in (
                ("couple_u", "couple_end"),
                ("couple_m", "couple_end"),
                ("single_f", "single_end"),
                ("single_m", "single_end"),
            )
        },
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _equal_exit_couples() -> Model:
    """The twin couples, both dissolving under one rule."""
    regimes = _twin_regimes()
    regimes["couple_m"] = regimes["couple_u"]
    return _twin_model(regimes)


@categorical(ordered=False)
class _ValueReaderRegimes:
    source: ScalarInt
    formation: ScalarInt
    counterparty: ScalarInt
    done: ScalarInt
    couple_done: ScalarInt
    target: ScalarInt


def _one() -> float:
    return 1.0


def _three() -> float:
    return 3.0


def _five() -> float:
    return 5.0


def _seven() -> float:
    return 7.0


def _eight() -> float:
    return 8.0


def _ten() -> float:
    return 10.0


def _twenty() -> float:
    return 20.0


def _value_reader_model() -> Model:
    """A single forms a couple under `Consent`, reading the couple's member values."""
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": _one},
                exit=Exit(
                    targets={
                        "self": ExitTarget(regime="formation", role="f", projection={})
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="counterparty", role="m", projection={}
                        )
                    ),
                ),
            ),
            "formation": Regime(
                functions={
                    "utility": CollectiveUtility(utilities={"f": _eight, "m": _seven})
                }
            ),
            "counterparty": Regime(functions={"utility": _three}),
            "done": Regime(functions={"utility": _five}),
            "couple_done": Regime(
                functions={
                    "utility": CollectiveUtility(utilities={"f": _ten, "m": _twenty})
                }
            ),
            "target": Regime(functions={"utility": _five}),
        },
        n_periods=2,
        regime_id_class=_ValueReaderRegimes,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={
            "source": {"done": Periods(values=(0,))},
            "formation": {"couple_done": Periods(values=(0,))},
            "counterparty": {"target": Periods(values=(0,))},
        },
    )


_MODELS: dict[str, Callable[[], Model]] = {
    "roles": lambda: transition_roles.build_model(edge_form="roles"),
    "plain": lambda: transition_roles.build_model(edge_form="plain"),
    "exact_gate": lambda: transition_roles.build_model(edge_form="exact_gate"),
    "hand_gate": lambda: transition_roles.build_model(edge_form="hand_gate"),
    "twin": lambda: _twin_model(_twin_regimes()),
    "equal_exits": _equal_exit_couples,
    "renegotiate_1p": lambda: renegotiation._model(values=renegotiation._VALUES),
    "renegotiate_2p": lambda: renegotiation._model(
        values=renegotiation._VALUES, n_periods=2
    ),
    "two_member": lambda: make_two_member_model()[0],
    "role": _role_model,
    "observed_offer": observed_offer.model,
    "settle": lambda: settlement._settlement_model(division="transfer"),
    "survivor_gate": lambda: gate_fallback.build_model(route_form="survivor"),
    "shared_fallback": lambda: gate_fallback.build_model(route_form="both_phases"),
    "value_reader": lambda: _value_reader_model(),
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
    """Without members, the same couple and single pool as one dense regime."""
    assert _pooled(_report("plain", backend), "couple", "single")


@pytest.mark.parametrize("backend", BACKENDS)
def test_regimes_with_different_exits_do_not_pool(backend: FactsBackendName) -> None:
    """A unilateral and a mutual-consent couple stay apart."""
    assert not _pooled(_report("twin", backend), "couple_m", "couple_u")


@pytest.mark.parametrize("backend", BACKENDS)
def test_regimes_with_equal_exits_still_pool(backend: FactsBackendName) -> None:
    """Two couples with one member tuple and one exit pool."""
    assert _pooled(_report("equal_exits", backend), "couple_m", "couple_u")


def test_normalized_cells_carry_their_members() -> None:
    """Each cell names its regime's members, `None` for a single agent."""
    cells = _report("roles", "dense")["analysis"]["declarations"]["cells"]
    assert {c["regime"]: c.get("member_axis", "absent") for c in cells} == {
        "single": None,
        "couple": ["f", "m"],
        "single_end": None,
        "couple_end": ["f", "m"],
    }


@pytest.mark.parametrize(
    ("label", "key", "regimes"),
    [
        ("roles", "member_axis", ("couple", "single")),
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
        ("roles", "couple", "member_axis", None),
        ("twin", "couple_m", "exit_identity", None),
        ("renegotiate_2p", "couple", "renegotiated_states", []),
    ],
)
def test_certificate_rejects_cells_that_lose_their_kind(
    label: str, regime: RegimeName, key: str, value: JSONValue
) -> None:
    """The source check rejects cells whose members, exit or renegotiation changed."""
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
    """Solve the two-member model; return its stored values and action values.

    The collective readout runs once per state node of each regime-period and
    receives every member's action values there; the stored value carries
    the state axes and a trailing member axis.
    """
    per_node: list[int] = []
    readout = max_Q_over_a.collective_readout

    def recording(
        *,
        member_Q: Mapping[str, FloatND],
        feasibility: BoolND,
        weights: Mapping[str, FloatND | float],
        action_axes: tuple[int, ...],
    ) -> tuple[dict[str, FloatND], BoolND]:
        per_node.append(sum(prod(q.shape) for q in member_Q.values()))
        return readout(
            member_Q=member_Q,
            feasibility=feasibility,
            weights=weights,
            action_axes=action_axes,
        )

    model, params = make_two_member_model()
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
def test_storage_counts_the_member_axis(backend: FactsBackendName) -> None:
    """The finder's stored values of the model as written equal the solve's."""
    stored, _ = _engine_counts()
    report = _report("two_member", backend)
    assert (
        report["original_cost"]["storage"],
        report["layouts"]["curve"][0]["storage"],
    ) == (stored, stored)


@pytest.mark.parametrize("backend", BACKENDS)
def test_evaluations_count_the_member_axis(backend: FactsBackendName) -> None:
    """The finder's evaluations of the model as written equal the solve's action values."""
    _, evaluated = _engine_counts()
    report = _report("two_member", backend)
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
        r.get("member")
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
    """A gate with member legs and fallbacks, one of them shared, is checked."""
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


def test_readers_of_a_collective_value_stay_unresolved() -> None:
    """A reader of a collective value keeps an open obligation."""
    assert _report("value_reader", "dense")["status"] == "UNRESOLVED"


def test_gate_whose_fallback_is_its_source_shows_no_internal_error() -> None:
    """A hand-written gate falling back to the source regime reports no engine exception."""
    report = _report("hand_gate", "dense")
    assert "NotImplementedError" not in str(report.get("error", ""))
