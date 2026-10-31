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
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dags import with_signature

from _lcm import policy_queries, query_information
from _lcm.policy_queries import QueryKind, QueryNode
from _lcm.typing import PytreeValue
from lcm import (
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    Periods,
    Regime,
    RuleResult,
    categorical,
)
from lcm.shocks import Discrete, GaussHermite, Normal, Product, ShockGroup
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


def _formation(member: str) -> Exit:
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


def _model(*, dissolution: bool = False) -> Model:
    """Two singles marry by consent; with `dissolution` the couple may split."""
    experience = DiscreteGrid(values=jnp.arange(_N_EXP))
    work = DiscreteGrid(values=jnp.arange(2))
    regimes = {
        f"single_{member}": Regime(
            states={"experience": experience},
            actions={"work": work},
            functions={"utility": _single_f if member == "f" else _single_m},
            shocks=(f"wage_{member}", f"taste_{member}", f"marriage_from_{member}"),
            state_transitions={"experience": _next_experience},
            exits={"marry": _formation(member)},
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
        state_transitions={"experience_f": _next_f, "experience_m": _next_m},
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
            state_transitions={},
            shocks=tuple(
                group for group in regime.shocks if not group.startswith("marriage")
            ),
            exits={},
        )
    marriage = {
        f"marriage_from_{member}": ShockGroup(
            shocks={
                f"offer_{member}": Discrete(categories=_Offer, probs=(0.4, 0.6)),
                f"offered_{'m' if member == 'f' else 'f'}": Discrete(
                    categories=_Offered, probs=(0.2, 0.5, 0.3)
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
            "match": _gauss(name="match", n=2, sigma=0.3),
            "taste_f": _gauss(name="taste_f", n=3, sigma=1.0),
            "taste_m": _gauss(name="taste_m", n=3, sigma=1.0),
            **marriage,
        },
        n_periods=2,
        regime_id_class=_Regimes if dissolution else _FormationRegimes,
        initial_nodes=InitialNodes(by_period={0: ("couple", "single_f", "single_m")}),
        edges={name: {f"{name}_terminal": Periods(values=(0,))} for name in sources},
        execution_config=ExecutionConfig(
            device_memory_bytes=None, axis_widths={"query_cell": 4}
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
        return _solve(model())


@pytest.mark.parametrize("dissolution", [False, True], ids=["formation", "dissolution"])
def test_read_classes_reproduce_every_caller_row(
    *, dissolution: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gathering class answers back to caller rows equals evaluating every row.

    With dissolution the gathers compose over three generations: the formation
    root, the couple's classes of it, and the dissolution benchmarks' coarser
    classes of both.
    """

    def model() -> Model:
        return _model(dissolution=dissolution)

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


def _root_stay_problems(
    *, model: Model, monkeypatch: pytest.MonkeyPatch
) -> dict[RegimeName, int]:
    """Stay problems each root evaluates per source cell.

    The root's action problem runs inside nested lane maps: the chunk's source
    cells, then its own rows (or the classes of them it evaluates). The product
    of the maps inside the cell map is the count per source cell.
    """
    sizes: list[int] = []
    counts: dict[RegimeName, int] = {}
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
            counts[node.read.target] = math.prod(sizes[1:])
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
    counts = _root_stay_problems(model=_model(), monkeypatch=monkeypatch)
    assert counts["single_f"] == 6
    assert counts["single_m"] == 6
    # The couple's stay problem reads every couple shock.
    assert counts["couple"] == 8


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
