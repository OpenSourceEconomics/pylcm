"""A query below a formation exit runs once per class of its callers' rows it reads.

Each single may marry into one couple by consent. Its rows cross its own wage and
taste shocks, an offer group (whether an offer arrives, and the offered partner's
experience) and the couple shocks it observes. The couple map never reads the
single's taste, and the stay benchmarks read neither the offered partner nor most
couple shocks. Each query therefore runs once per distinct combination of the
caller columns it, its guards and its own descendants read, and its consumers
gather that answer back to every caller row.
"""

import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from types import MappingProxyType
from typing import ClassVar, TypedDict, Unpack

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dags import with_signature

from _lcm import exit_queries, policy_queries, query_information
from _lcm.policy_queries import QueryKind, QueryNode
from _lcm.regime_building import Q_and_F
from _lcm.solution.contract import SolverBuildContext
from _lcm.typing import PytreeValue, QAndFKwargs
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
    Periods,
    Regime,
    RuleResult,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.shocks import (
    Discrete,
    GaussHermite,
    Normal,
    Product,
    QuasiMonteCarlo,
    ShockGroup,
)
from lcm.typing import BoolND, FloatND, IntND, RegimeName, ScalarInt
from tests.conftest import assert_agrees_to_ulp

_N_EXP = 3
_N_OFFERED = 3


@categorical(ordered=False)
class _Regimes:
    single_f: ScalarInt
    single_m: ScalarInt
    couple: ScalarInt
    alone_f: ScalarInt
    alone_m: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt
    couple_terminal: ScalarInt
    alone_f_terminal: ScalarInt
    alone_m_terminal: ScalarInt


@categorical(ordered=False)
class _FormationRegimes:
    single_f: ScalarInt
    single_m: ScalarInt
    couple: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt
    couple_terminal: ScalarInt


@categorical(ordered=False)
class _Offer:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class _Offered:
    low: ScalarInt
    mid: ScalarInt
    high: ScalarInt


def _single_f(
    *, experience: IntND, work: IntND, wage_f: FloatND, taste_f: FloatND
) -> FloatND:
    return (
        1 + work * (2 + 0.2 * experience + wage_f) + (2.45 + 0.1 * taste_f) * (1 - work)
    )


def _single_m(
    *, experience: IntND, work: IntND, wage_m: FloatND, taste_m: FloatND
) -> FloatND:
    return (
        1 + work * (3 + 0.2 * experience + wage_m) + (2.45 + 0.1 * taste_m) * (1 - work)
    )


def _alone_f(*, experience: IntND, work: IntND, wage_f: FloatND) -> FloatND:
    return 1 + work * (2 + 0.2 * experience + wage_f) + 2.4 * (1 - work)


def _alone_m(*, experience: IntND, work: IntND, wage_m: FloatND) -> FloatND:
    return 1 + work * (3 + 0.2 * experience + wage_m) + 2.4 * (1 - work)


def _consumption(
    *,
    experience_f: IntND,
    experience_m: IntND,
    work_f: IntND,
    work_m: IntND,
    wage_f: FloatND,
    wage_m: FloatND,
    match: FloatND,
) -> FloatND:
    return (
        1
        + work_f * (2 + 0.2 * experience_f + wage_f)
        + work_m * (3 + 0.2 * experience_m + wage_m)
        + match
    )


def _lasting_consumption(
    *,
    experience_f: IntND,
    experience_m: IntND,
    work_f: IntND,
    work_m: IntND,
    wage_f: FloatND,
    wage_m: FloatND,
    match: FloatND,
    last_match: FloatND,
) -> FloatND:
    return last_match + _consumption(
        experience_f=experience_f,
        experience_m=experience_m,
        work_f=work_f,
        work_m=work_m,
        wage_f=wage_f,
        wage_m=wage_m,
        match=match,
    )


def _gift(*, experience: IntND) -> FloatND:
    return 0.05 * experience


def _take_gift(*, gift: FloatND) -> FloatND:
    return gift


def _carry_match(*, match: FloatND) -> FloatND:
    return match


def _couple_f(*, consumption: FloatND, work_f: IntND) -> FloatND:
    return consumption / 2 + 2.45 * (1 - work_f) + 1


def _couple_m(*, consumption: FloatND, work_m: IntND) -> FloatND:
    return consumption / 2 + 2.5 * (1 - work_m) + 1


def _experience(experience: IntND) -> IntND:
    return experience


def _experience_f(experience_f: IntND) -> IntND:
    return experience_f


def _experience_m(experience_m: IntND) -> IntND:
    return experience_m


def _offered_m(offered_m: IntND) -> IntND:
    return offered_m


def _offered_f(offered_f: IntND) -> IntND:
    return offered_f


def _offer_f(offer_f: IntND) -> BoolND:
    return offer_f == 1


def _offer_m(offer_m: IntND) -> BoolND:
    return offer_m == 1


def _next_experience(*, experience: IntND, work: IntND) -> IntND:
    return jnp.minimum(experience + work, _N_EXP - 1)


def _next_f(*, experience_f: IntND, work_f: IntND) -> IntND:
    return jnp.minimum(experience_f + work_f, _N_EXP - 1)


def _next_m(*, experience_m: IntND, work_m: IntND) -> IntND:
    return jnp.minimum(experience_m + work_m, _N_EXP - 1)


class _StrictParticipation:
    """Keep the couple's best action that both spouses prefer to dissolving."""

    reads_outside_values: ClassVar[bool] = True

    def __call__(
        self,
        *,
        Q_f: FloatND,
        Q_m: FloatND,
        exit_payoffs: Mapping[str, Mapping[str, FloatND]],
        alternative_ids: IntND,
        pareto_weights: Mapping[str, FloatND],
    ) -> RuleResult:
        q_f, q_m = Q_f.reshape(-1), Q_m.reshape(-1)
        outside = exit_payoffs["dissolve"]
        kept = (q_f > outside["f"]) & (q_m > outside["m"])
        objective = pareto_weights["f"] * q_f + pareto_weights["m"] * q_m
        stay = jnp.argmax(jnp.where(kept, objective, -jnp.inf))
        return RuleResult(
            action=alternative_ids[jnp.where(jnp.any(kept), stay, q_f.size)]
        )


def _formation(
    *, member: str, nodes: QuasiMonteCarlo | None = None, gift: bool = False
) -> Exit:
    spouse = "m" if member == "f" else "f"
    offered = _offered_m if member == "f" else _offered_f
    return Exit(
        targets={
            "self": ExitTarget(
                regime="couple",
                role=member,
                projection={
                    f"experience_{member}": _experience,
                    f"experience_{spouse}": offered,
                },
                utility_adjustment=_take_gift if gift else None,
            )
        },
        rule=Consent(
            counterparty=Counterparty(
                regime=f"single_{spouse}",
                role=spouse,
                projection={"experience": offered},
            )
        ),
        available=_offer_f if member == "f" else _offer_m,
        observes=("wage_f", "wage_m", "match", f"offer_{member}", f"offered_{spouse}"),
        nodes=nodes,
    )


def _dissolve() -> Exit:
    return Exit(
        targets={
            member: ExitTarget(
                regime=f"alone_{member}",
                projection={
                    "experience": _experience_f if member == "f" else _experience_m
                },
            )
            for member in ("f", "m")
        },
        rule=_StrictParticipation(),
    )


def _gauss(*, name: str, n: int, sigma: float) -> ShockGroup:
    return ShockGroup(
        shocks={name: Normal(mu=0.0, sigma=sigma)},
        nodes=Product(**{name: GaussHermite(n=n)}),
    )


def _model(
    *,
    dissolution: bool = False,
    n_match: int = 2,
    lasting_match: bool = False,
    nodes: QuasiMonteCarlo | None = None,
    offer_probs: tuple[float, float] = (0.4, 0.6),
    offered_probs: tuple[float, float, float] = (0.2, 0.5, 0.3),
    query_cell: int | None = 4,
    n_periods: int = 2,
    gift: bool = False,
) -> Model:
    """Two singles marry by consent; with `dissolution` the couple may split.

    With `lasting_match` the couple carries its match shock into its terminal
    period, so the couple's continuation reads that shock. With `nodes` each
    single draws the rows of its formation choice that its stay problem ignores.
    With `query_cell=None` the chunk width is left to the engine. Every regime
    lives through the periods before the last one and ends in its terminal twin.
    With `gift` a single who marries is paid a gift its regime computes.
    """
    experience = DiscreteGrid(values=jnp.arange(_N_EXP))
    work = DiscreteGrid(values=jnp.arange(2))
    regimes = {
        f"single_{member}": Regime(
            states={"experience": experience},
            actions={"work": work},
            functions={"utility": _single_f if member == "f" else _single_m}
            | ({"gift": _gift} if gift else {}),
            shocks=(f"wage_{member}", f"taste_{member}", f"marriage_from_{member}"),
            state_transitions={"experience": _next_experience},
            exits={"marry": _formation(member=member, nodes=nodes, gift=gift)},
        )
        for member in ("f", "m")
    }
    regimes["couple"] = Regime(
        states={"experience_f": experience, "experience_m": experience},
        actions={"work_f": work, "work_m": work},
        functions={
            "utility": CollectiveUtility(utilities={"f": _couple_f, "m": _couple_m}),
            "consumption": _consumption,
        },
        state_transitions={
            "experience_f": _next_f,
            "experience_m": _next_m,
            **(
                {"last_match": {"couple_terminal": _carry_match}}
                if lasting_match
                else {}
            ),
        },
        shocks=("wage_f", "wage_m", "match"),
        exits={"dissolve": _dissolve()} if dissolution else {},
    )
    if dissolution:
        for member in ("f", "m"):
            regimes[f"alone_{member}"] = Regime(
                states={"experience": experience},
                actions={"work": work},
                functions={"utility": _alone_f if member == "f" else _alone_m},
                shocks=(f"wage_{member}",),
                state_transitions={"experience": _next_experience},
            )
    for name, regime in tuple(regimes.items()):
        regimes[f"{name}_terminal"] = replace(
            regime,
            functions={
                name: function
                for name, function in regime.functions.items()
                if name != "gift"
            },
            state_transitions={},
            shocks=tuple(
                group for group in regime.shocks if not group.startswith("marriage")
            ),
            exits={},
        )
    if lasting_match:
        regimes["couple_terminal"] = replace(
            regimes["couple_terminal"],
            states={
                **regimes["couple"].states,
                "last_match": LinSpacedGrid(start=-1.0, stop=1.0, n_points=2),
            },
            functions={
                **regimes["couple"].functions,
                "consumption": _lasting_consumption,
            },
        )
    marriage = {
        f"marriage_from_{member}": ShockGroup(
            shocks={
                f"offer_{member}": Discrete(categories=_Offer, probs=offer_probs),
                f"offered_{'m' if member == 'f' else 'f'}": Discrete(
                    categories=_Offered, probs=offered_probs
                ),
            }
        )
        for member in ("f", "m")
    }
    sources = ("couple", "single_f", "single_m") + (
        ("alone_f", "alone_m") if dissolution else ()
    )
    return Model(
        regimes={
            name: regime
            for name, regime in regimes.items()
            if dissolution or not name.startswith("alone")
        },
        shocks={
            "wage_f": _gauss(name="wage_f", n=2, sigma=0.5),
            "wage_m": _gauss(name="wage_m", n=2, sigma=0.4),
            "match": _gauss(name="match", n=n_match, sigma=0.3),
            "taste_f": _gauss(name="taste_f", n=3, sigma=1.0),
            "taste_m": _gauss(name="taste_m", n=3, sigma=1.0),
            **marriage,
        },
        n_periods=n_periods,
        regime_id_class=_Regimes if dissolution else _FormationRegimes,
        initial_nodes=InitialNodes(by_period={0: ("couple", "single_f", "single_m")}),
        edges={
            name: {
                **(
                    {name: Periods(values=tuple(range(n_periods - 2)))}
                    if n_periods > 2
                    else {}
                ),
                f"{name}_terminal": Periods(values=(n_periods - 2,)),
            }
            for name in sources
        },
        execution_config=ExecutionConfig(
            device_memory_bytes=None,
            axis_widths={} if query_cell is None else {"query_cell": query_cell},
        ),
    )


def _solve(model: Model) -> dict[str, np.ndarray]:
    solution = model.solve(params={"discount_factor": 0.9}, log_level="off")
    return {
        f"{period}/{regime}": np.asarray(value)
        for period, values in solution.values.items()
        for regime, value in values.items()
    }


def _evaluated_rows(
    *, model: Model, monkeypatch: pytest.MonkeyPatch
) -> dict[tuple[str, str, str], int]:
    """Action problems each (root, kind, target) query evaluates per source cell.

    A query's answer has one leading axis per ancestor class it runs over, and
    within each entry it evaluates its own rows or their pinned representatives.
    Every lane runs: a guard selects under vmap.
    """
    rows = {}
    original = query_information.InformationQueryCell.__call__

    # keyword-only-exempt: primary-argument=self
    def record(
        self: query_information.InformationQueryCell,
        *,
        live: BoolND,
        source: query_information.SourceStates,
        dependencies: query_information.Dependencies,
    ) -> policy_queries.PolicyAnswer:
        answer = original(self, live=live, source=source, dependencies=dependencies)
        root = next(
            node.read.target for node in self.definitions.values() if not node.ancestors
        )
        local = query_information._has_local_information(
            node=self.node, period=self.node.read.period
        )
        own = self._axis(name=self.name, node=self.node) if local else 1
        rows[(root, self.node.read.kind.name, self.node.read.target)] = (
            math.prod(answer[2].shape) * own
        )
        return answer

    monkeypatch.setattr(query_information.InformationQueryCell, "__call__", record)
    _solve(model)
    return rows


def test_each_caller_prices_a_couple_once_per_read_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per single state, a caller has 144 rows: wage 2 x taste 3 x offer 2 x
    offered 3 x observed wage 2 x match 2.

    The couple reads all but taste, so it runs 48 times. Its offer flag is read by
    the exit's availability guard, so offer stays a class. The own stay benchmark
    reads wage, taste and the guard: 12. The counterparty reads the offered
    partner, the observed wage and the guard (12 classes), times its own 3 taste
    rows.
    """
    rows = _evaluated_rows(model=_model(), monkeypatch=monkeypatch)
    for member, spouse in (("f", "m"), ("m", "f")):
        root = f"single_{member}"
        assert rows[(root, "FULL", root)] == 144
        assert rows[(root, "PARTICIPATING", "couple")] == 48
        assert rows[(root, "RAW", root)] == 12
        assert rows[(root, "RAW", f"single_{spouse}")] == 36
    # The couple prices each (couple state, couple row, offer flag) once per
    # period: 3 single states x 48, not 3 x 144.
    assert 3 * rows[("single_f", "PARTICIPATING", "couple")] == _N_EXP * 48


def _definitions(*, model: Model, root: RegimeName) -> Mapping[str, QueryNode]:
    kernel = model._regimes[root].solution.period_kernels[0]
    assert isinstance(kernel, policy_queries._PolicyQueryKernel)
    return kernel.definitions


def _key(
    *, definitions: Mapping[str, QueryNode], kind: QueryKind, target: RegimeName
) -> str:
    (key,) = (
        name
        for name, node in definitions.items()
        if node.read.kind is kind and node.read.target == target
    )
    return key


def test_read_columns_follow_projections_guards_copies_and_descendants() -> None:
    """Each benchmark reads exactly the caller columns its value can depend on.

    The couple's dissolution benchmarks reach the single's columns only through
    the couple frame's pinned copies and the guards above them.
    """
    definitions = _definitions(model=_model(dissolution=True), root="single_f")
    root = _key(definitions=definitions, kind=QueryKind.FULL, target="single_f")

    def reads(*, target: RegimeName, kind: QueryKind = QueryKind.RAW) -> frozenset[str]:
        return query_information.read_columns(
            name=_key(definitions=definitions, kind=kind, target=target),
            ancestor=root,
            definitions=definitions,
        )

    assert reads(target="single_f") == {"wage_f", "taste_f", "offer_f"}
    assert reads(target="single_m") == {"offer_f", "offered_m", "wage_m"}
    assert reads(target="alone_f", kind=QueryKind.PARTICIPATING) == {
        "wage_f",
        "offer_f",
    }
    # The husband leaves with the offered experience the projection gave him.
    assert reads(target="alone_m", kind=QueryKind.PARTICIPATING) == {
        "wage_m",
        "offer_f",
        "offered_m",
    }
    # The couple's own reads, joined with its dissolution benchmarks'.
    assert reads(target="couple", kind=QueryKind.PARTICIPATING) == {
        "wage_f",
        "offer_f",
        "offered_m",
        "wage_m",
        "match",
    }


def test_an_unattributable_physical_read_keeps_every_caller_row() -> None:
    """A projection that takes every physical frame may read any caller column."""
    definitions = dict(_definitions(model=_model(), root="single_f"))
    root = _key(definitions=definitions, kind=QueryKind.FULL, target="single_f")
    name = _key(definitions=definitions, kind=QueryKind.RAW, target="single_f")
    node = definitions[name]

    def opaque(**operands: PytreeValue) -> PytreeValue:
        return operands["experience"]

    projection = {
        state: with_signature(
            opaque,
            args=["experience", policy_queries._PHYSICAL_CONTEXTS],
            enforce=False,
        )
        for state in node.read.projection
    }
    definitions[name] = replace(node, read=replace(node.read, projection=projection))
    every = frozenset(
        query_information._parent_sources(
            parent=definitions[root], definitions=definitions
        )
    )
    assert (
        query_information.read_columns(
            name=name, ancestor=root, definitions=definitions
        )
        == every
    )
    assert (
        query_information.ancestor_classes(
            name=name, ancestor=root, definitions=definitions
        )
        is None
    )


def _scalar_lanes(
    *,
    function: Callable[[PytreeValue], PytreeValue],
    operands: PytreeValue,
    batch_size: int | None = None,
) -> PytreeValue:
    """Every lane alone, so each guard branches instead of selecting."""
    del batch_size
    return jax.lax.map(function, operands)


def _every_row_values(
    *, model: Callable[[], Model], monkeypatch: pytest.MonkeyPatch
) -> dict[str, np.ndarray]:
    """Every query at every caller row, in scalar branches: the reference."""
    with monkeypatch.context() as patch:
        patch.setattr(policy_queries, "map_lanes", _scalar_lanes)
        patch.setattr(query_information, "pinned_support", lambda **_: None)
        patch.setattr(query_information, "ancestor_classes", lambda **_: None)
        patch.setattr(query_information, "own_stay_classes", lambda **_: None)
        patch.setattr(query_information, "continuation_members", lambda **_: None)
        return _solve(model())


@pytest.mark.parametrize("variant", ["formation", "dissolution", "lasting_match"])
def test_read_classes_reproduce_every_caller_row(
    *, variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gathering class answers back to caller rows equals evaluating every row.

    With dissolution the gathers compose over three generations: the formation
    root, the couple's classes of it, and the dissolution benchmarks' coarser
    classes of both. With a lasting match the couple's continuation reads a
    shock each single observes.
    """

    def model() -> Model:
        return _model(
            dissolution=variant == "dissolution",
            lasting_match=variant == "lasting_match",
        )

    got = _solve(model())
    expected = _every_row_values(model=model, monkeypatch=monkeypatch)
    assert got.keys() == expected.keys()
    for name, value in expected.items():
        assert np.all(np.isfinite(got[name])), name
        assert_agrees_to_ulp(got=got[name], expected=value, n_ulp=16, err_msg=name)


def test_dissolution_benchmarks_run_once_per_class_of_both_ancestors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below the couple, a dissolution benchmark runs once per class of the
    formation's rows and once per couple row.

    The wife's reads her own wage and the offer guard (4 classes); the husband's
    also reads the offered experience he leaves with (12).
    """
    rows = _evaluated_rows(model=_model(dissolution=True), monkeypatch=monkeypatch)
    assert rows[("single_f", "PARTICIPATING", "alone_f")] == 4
    assert rows[("single_f", "PARTICIPATING", "alone_m")] == 12
    assert rows[("single_f", "PARTICIPATING", "couple")] == 48


def test_no_query_reads_more_rows_than_its_consumer() -> None:
    """A producer's columns are a subset of its consumer's, so gathers compose."""
    definitions = _definitions(model=_model(dissolution=True), root="single_f")
    for name, node in definitions.items():
        for consumer, producer in ((name, child) for child in node.inputs.values()):
            for ancestor in definitions[producer].ancestors:
                if ancestor not in definitions[consumer].ancestors:
                    continue
                assert query_information.read_columns(
                    name=producer, ancestor=ancestor, definitions=definitions
                ) <= query_information.read_columns(
                    name=consumer, ancestor=ancestor, definitions=definitions
                ), (consumer, producer, ancestor)


def _root_stay_maps(
    *, model: Model, monkeypatch: pytest.MonkeyPatch
) -> dict[RegimeName, tuple[int, ...]]:
    """Lane maps each root's stay problem runs in, inside one source cell.

    The root's action problem runs inside nested lane maps: the chunk's source
    cells, then one map per independent group of shocks it reads. The product
    of the maps inside the cell map is the count of stay problems per source
    cell.
    """
    sizes: list[int] = []
    counts: dict[RegimeName, tuple[int, ...]] = {}
    map_lanes = policy_queries.map_lanes
    query_action = policy_queries._query_action

    def counted_map(
        *,
        function: Callable[[PytreeValue], PytreeValue],
        operands: PytreeValue,
        batch_size: int | None = None,
    ) -> PytreeValue:
        size = jax.tree.leaves(operands)[0].shape[0]

        def inner(lane: PytreeValue) -> PytreeValue:
            sizes.append(size)
            try:
                return function(lane)
            finally:
                sizes.pop()

        return map_lanes(function=inner, operands=operands, batch_size=batch_size)

    def counted_action(
        *,
        node: QueryNode,
        period: int,
        arguments: Mapping[str, PytreeValue],
        dependencies: Mapping[str, policy_queries.PolicyAnswer],
        participate: bool = True,
        **actions: PytreeValue,
    ) -> policy_queries.StayTable:
        if not node.ancestors and node.read.kind is QueryKind.FULL:
            counts[node.read.target] = tuple(sizes[1:])
        return query_action(
            node=node,
            period=period,
            arguments=arguments,
            dependencies=dependencies,
            participate=participate,
            **actions,
        )

    monkeypatch.setattr(policy_queries, "map_lanes", counted_map)
    monkeypatch.setattr(policy_queries, "_query_action", counted_action)
    _solve(model)
    return counts


def test_a_caller_evaluates_its_own_stay_once_per_class_it_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single's 144 rows differ in shocks its own stay problem never reads: the
    offer group and the observed couple shocks matter only to its exit. Its stay
    problem reads its wage and taste, so it runs 6 times; the exit decision still
    runs at every row."""
    maps = _root_stay_maps(model=_model(), monkeypatch=monkeypatch)
    # The couple's stay problem reads every couple shock.
    assert {
        name: math.prod(maps[name]) for name in ("single_f", "single_m", "couple")
    } == {
        "single_f": 6,
        "single_m": 6,
        "couple": 8,
    }


def test_a_stay_problem_maps_each_independent_shock_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The couple's 8 rows are the product of three shocks of 2 nodes each.

    Its stay problem runs in one lane map per shock, so a function of one shock
    is computed at that shock's 2 nodes and not at the 8 joint rows.
    """
    maps = _root_stay_maps(model=_model(), monkeypatch=monkeypatch)
    assert maps["couple"] == (2, 2, 2)


def test_stay_reads_are_the_stay_problem_s_shocks() -> None:
    """The own-stay classes follow the stay problem's shocks; a guarded query
    keeps every row, its guard deciding per row whether it runs at all."""
    definitions = _definitions(model=_model(dissolution=True), root="single_f")
    root = _key(definitions=definitions, kind=QueryKind.FULL, target="single_f")
    assert query_information.stay_read_shocks(name=root, definitions=definitions) == {
        "wage_f",
        "taste_f",
    }
    couple = _key(
        definitions=definitions, kind=QueryKind.PARTICIPATING, target="couple"
    )
    assert (
        query_information.own_stay_classes(name=couple, definitions=definitions) is None
    )


class _ContinuationKwargs(TypedDict):
    """Arguments of one continuation evaluation."""

    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND]
    zero: FloatND
    states_actions_params: QAndFKwargs


def _continuation_values(
    *, model: Model, monkeypatch: pytest.MonkeyPatch
) -> dict[RegimeName, int]:
    """Continuation values computed over one solve, by the regime they belong to.

    A run-time callback on the continuation's result sees one element per value
    the solve computes: a lane map that leaves the continuation's inputs alone
    leaves the result unbatched, so its lanes share one evaluation.
    """
    seen: Counter[RegimeName] = Counter()
    targets: list[RegimeName] = []
    call_cell = query_information.InformationQueryCell.__call__
    call_ce = Q_and_F._ComputeCE.__call__

    # keyword-only-exempt: primary-argument=self
    def cell(
        self: query_information.InformationQueryCell,
        *,
        live: BoolND,
        source: query_information.SourceStates,
        dependencies: query_information.Dependencies,
    ) -> policy_queries.PolicyAnswer:
        targets.append(self.node.read.target)
        try:
            return call_cell(self, live=live, source=source, dependencies=dependencies)
        finally:
            targets.pop()

    def continuation(
        self: Q_and_F._ComputeCE, **kwargs: Unpack[_ContinuationKwargs]
    ) -> Q_and_F.ContinuationAggregate:
        target = targets[-1]
        leaves, tree = jax.tree.flatten(call_ce(self, **kwargs))

        def record(value: np.ndarray) -> np.ndarray:
            seen[target] += value.size
            return value

        leaves[0] = jax.pure_callback(
            record,
            jax.ShapeDtypeStruct(leaves[0].shape, leaves[0].dtype),
            leaves[0],
            vmap_method="expand_dims",
        )
        return jax.tree.unflatten(tree, leaves)

    monkeypatch.setattr(query_information.InformationQueryCell, "__call__", cell)
    monkeypatch.setattr(Q_and_F._ComputeCE, "__call__", continuation)
    _solve(model)
    return dict(seen)


@pytest.mark.parametrize("n_match", [2, 3])
def test_a_query_computes_its_continuation_once_per_class_that_fixes_it(
    *, n_match: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A continuation runs once per combination of the caller shocks that fix it.

    The couple's continuation follows the offered partner, and the offer guard
    decides whether it runs: 6 classes per single state, whatever the wage and
    match supports. Each of its lanes holds 4 actions for 2 spouses, and it runs
    in 3 x 6 lanes below each single and in 12 lanes of its own.

    A single's continuation holds 2 actions. It runs once per own state (3),
    per offer flag as the own stay benchmark (3 x 2), and per offer flag and
    offered partner as the counterparty (3 x 6).
    """
    values = _continuation_values(
        model=_model(n_match=n_match), monkeypatch=monkeypatch
    )
    assert values == {
        "couple": (18 + 18 + 12) * 8,
        "single_f": (3 + 6 + 18) * 2,
        "single_m": (3 + 6 + 18) * 2,
    }


@pytest.mark.parametrize("n_match", [2, 3])
def test_a_continuation_reading_an_observed_shock_runs_once_per_class(
    *, n_match: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A continuation that reads a shock its caller observes runs once per class.

    The couple carries its match shock into its terminal period, and each single
    observes that shock from the couple it may form. The offer guard and the
    offered partner realise the observation, so the continuation has
    6 x `n_match` classes per single state. Each of the couple's own 12 lanes maps
    its shocks one group at a time, so the continuation runs once per node of the
    match shock it reads (`n_match`), not at every joint row (4 x `n_match`).
    """
    values = _continuation_values(
        model=_model(n_match=n_match, lasting_match=True), monkeypatch=monkeypatch
    )
    assert values == {
        "couple": (18 * n_match + 18 * n_match + 12 * n_match) * 8,
        "single_f": (3 + 6 + 18) * 2,
        "single_m": (3 + 6 + 18) * 2,
    }


def _weighted_rows(
    *, model: Model, monkeypatch: pytest.MonkeyPatch
) -> dict[tuple[str, str, str], int]:
    """Rows each (root, kind, target) query weights by its own shock factors.

    A query forms one weighted term per row it passes to the weighting, in every
    lane it runs in.
    """
    rows: dict[tuple[str, str, str], int] = {}
    keys: list[tuple[str, str, str]] = []
    call_cell = query_information.InformationQueryCell.__call__
    weighted_term = query_information.factored_weighted_term

    # keyword-only-exempt: primary-argument=self
    def cell(
        self: query_information.InformationQueryCell,
        *,
        live: BoolND,
        source: query_information.SourceStates,
        dependencies: query_information.Dependencies,
    ) -> policy_queries.PolicyAnswer:
        root = next(
            node.read.target for node in self.definitions.values() if not node.ancestors
        )
        keys.append((root, self.node.read.kind.name, self.node.read.target))
        try:
            return call_cell(self, live=live, source=source, dependencies=dependencies)
        finally:
            keys.pop()

    def counted_term(*, factors: FloatND, values: FloatND) -> FloatND:
        rows[keys[-1]] = values.shape[0]
        return weighted_term(factors=factors, values=values)

    monkeypatch.setattr(query_information.InformationQueryCell, "__call__", cell)
    monkeypatch.setattr(query_information, "factored_weighted_term", counted_term)
    _solve(model)
    return rows


@pytest.mark.parametrize("n_match", [2, 3])
def test_a_query_weights_one_term_per_representative_of_its_rows(
    *, n_match: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A query's rows that differ only in shocks its caller observes hold one answer.

    Each single observes all three couple shocks, so the couple below it has one
    representative row whatever the wage and match supports; it weights that one
    answer by the mass of the rows it stands for. The counterparty's wage is
    observed and its taste is not, so it keeps its 3 taste rows. The single's own
    stay benchmark repeats the caller's row.
    """
    rows = _weighted_rows(model=_model(n_match=n_match), monkeypatch=monkeypatch)
    assert {key[1:]: count for key, count in rows.items() if key[0] == "single_f"} == {
        ("FULL", "single_f"): 144 * n_match // 2,
        ("PARTICIPATING", "couple"): 1,
        ("RAW", "single_f"): 1,
        ("RAW", "single_m"): 3,
    }


def test_a_drawn_formation_crosses_its_stay_rows_with_the_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With 4 draws on the exit a single has 30 rows, not 144.

    Its stay problem reads its wage and taste, whose 6 node combinations stay
    exact. Every stay row meets the same 4 draws among offers, with the observed
    couple shocks, and one row that carries the mass without an offer.
    """
    rows = _evaluated_rows(
        model=_model(nodes=QuasiMonteCarlo(n=4, seed=0)), monkeypatch=monkeypatch
    )
    assert {root: rows[(root, "FULL", root)] for root in ("single_f", "single_m")} == {
        "single_f": 30,
        "single_m": 30,
    }


def test_a_drawn_formation_keeps_one_stay_problem_per_stay_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draws never multiply the stay problem: it runs once per wage and taste."""
    maps = _root_stay_maps(
        model=_model(nodes=QuasiMonteCarlo(n=4, seed=0)), monkeypatch=monkeypatch
    )
    assert {root: math.prod(maps[root]) for root in ("single_f", "single_m")} == {
        "single_f": 6,
        "single_m": 6,
    }


def test_a_drawn_formation_equals_the_product_when_the_draws_are_exact() -> None:
    """A drawn formation value equals the product value under dyadic laws.

    Among offers the offered partner has masses 1/4, 1/2 and 1/4, and each observed
    couple shock has masses 1/2. 64 scrambled Sobol points in those three
    dimensions put exactly 64 times its mass on every node combination, so the
    equal-weight average over the draws is the expectation given an offer.
    """
    dyadic = partial(_model, offer_probs=(0.5, 0.5), offered_probs=(0.25, 0.5, 0.25))
    got = _solve(dyadic(nodes=QuasiMonteCarlo(n=64, seed=0)))
    expected = _solve(dyadic())
    np.testing.assert_allclose(
        np.concatenate([got[name].ravel() for name in sorted(expected)]),
        np.concatenate([expected[name].ravel() for name in sorted(expected)]),
        rtol=1e-12,
    )


@pytest.mark.parametrize("offer", [1 / 64, 0.0])
def test_a_drawn_formation_draws_only_offers(offer: float) -> None:
    """A rare or impossible offer still integrates exactly.

    Draws fall only where the exit is available, so all 64 integrate the offered
    partner and the couple shocks exactly as in the dyadic case, while the mass
    without an offer enters once at the stay value. Without an offer at all the
    value is the stay value.
    """
    rare = partial(
        _model, offer_probs=(1 - offer, offer), offered_probs=(0.25, 0.5, 0.25)
    )
    got = _solve(rare(nodes=QuasiMonteCarlo(n=64, seed=0)))
    expected = _solve(rare())
    np.testing.assert_allclose(
        np.concatenate([got[name].ravel() for name in sorted(expected)]),
        np.concatenate([expected[name].ravel() for name in sorted(expected)]),
        rtol=1e-12,
    )


# Two factors of a 4-node group whose node masses are 1/4, 0, 1/2, 1/4.
_MASS = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5]])


def test_draw_picks_each_node_in_proportion_to_its_mass() -> None:
    """Evenly spread uniforms pick each node as often as its mass says."""
    row, _ = query_information._draw(
        weights=jnp.asarray(_MASS), uniform=jnp.asarray((np.arange(8) + 0.5) / 8)
    )
    assert np.bincount(np.asarray(row), minlength=4).tolist() == [2, 0, 4, 2]


def test_draw_scales_a_node_s_own_mass_to_the_group_total() -> None:
    """A drawn node's mass times its scale is the group's total mass."""
    row, scale = query_information._draw(
        weights=jnp.asarray(2 * _MASS), uniform=jnp.asarray((np.arange(8) + 0.5) / 8)
    )
    np.testing.assert_allclose(
        np.asarray(scale) * (4 * _MASS.prod(axis=0))[np.asarray(row)],
        np.full(8, 4.0),
        rtol=1e-15,
    )


def test_a_regime_rejects_exits_that_name_different_draw_rules() -> None:
    """A regime's exits share one draw rule."""
    with pytest.raises(RegimeInitializationError, match="share one draw rule"):
        Regime(
            states={"experience": DiscreteGrid(values=jnp.arange(_N_EXP))},
            actions={"work": DiscreteGrid(values=jnp.arange(2))},
            functions={"utility": _single_f},
            shocks=("wage_f", "taste_f", "marriage_from_f"),
            state_transitions={"experience": _next_experience},
            exits={
                f"marry_{seed}": _formation(
                    member="f", nodes=QuasiMonteCarlo(n=4, seed=seed)
                )
                for seed in (0, 1)
            },
        )


@pytest.mark.parametrize(("query_cell", "expected"), [(None, 7281), (4, 4)])
def test_an_unset_chunk_width_fills_one_step_with_the_widest_query(
    *, monkeypatch: pytest.MonkeyPatch, query_cell: int | None, expected: int
) -> None:
    """A single's 144 rows get 7281 cells per chunk, and a set width is kept.

    7281 cells of 144 rows are the most that stay within the 1048576 lanes of one
    step, so a query with few rows does not pay one launch per 64 cells.
    """
    requested = {}
    build = exit_queries.build_policy_query_kernel

    def record(
        *,
        root_context: SolverBuildContext,
        period: int,
        definitions: Mapping[str, QueryNode],
        root: str,
        chunk_size: int,
    ) -> policy_queries._PolicyQueryKernel:
        requested[definitions[root].read.target] = chunk_size
        return build(
            root_context=root_context,
            period=period,
            definitions=definitions,
            root=root,
            chunk_size=chunk_size,
        )

    monkeypatch.setattr(exit_queries, "build_policy_query_kernel", record)
    _model(query_cell=query_cell)
    assert requested["single_f"] == expected
