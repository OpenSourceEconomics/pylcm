"""An exact composition action enumerates only the nodes where it is available.

A single's composition action is available only where an offer arrives. The offer and
the offered partner form one shock group, and the single's stay problem reads none of
it. Where no offer arrives the single stays, whoever the partner would have been, so the
solve enumerates the group's nodes with an offer and one more node without, which
carries the whole mass without an offer at the value of staying. Every other shock group
keeps its nodes. The rewrite is exact: the solved values are those of the full
enumeration to rounding, and simulation, which realises its shocks, is untouched.
"""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm import composition_queries, query_information
from _lcm.policy_queries import PolicyAnswer, _PolicyQueryKernel
from lcm import (
    CollectiveUtility,
    CompositionAction,
    Destination,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    LinSpacedGrid,
    Model,
    PCOnActions,
    Periods,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import OffGridStateError
from lcm.shocks import Discrete, GaussHermite, Normal, Product, QuasiMonteCarlo
from lcm.shocks import ShockGroup as Group
from lcm.typing import BoolND, FloatND, IntND, RegimeName, ScalarInt
from tests.conftest import assert_agrees_to_ulp
from tests.solution.test_formation_read_classes import _model as _formation

_N_EXP = 3


def _solve(model: Model) -> dict[str, np.ndarray]:
    """Every solved value array, keyed by period and regime.

    A one-period model has no continuation and so no discount factor.
    """
    params = {"discount_factor": 0.9} if model.n_periods > 1 else {}
    solution = model.solve(params=params, log_level="off")
    return {
        f"{period}/{regime}": np.asarray(value)
        for period, values in solution.values.items()
        for regime, value in values.items()
    }


@categorical(ordered=False)
class _Regimes:
    single: ScalarInt
    partnered: ScalarInt
    single_terminal: ScalarInt
    partnered_terminal: ScalarInt


@categorical(ordered=False)
class _OneRegime:
    single: ScalarInt
    partnered: ScalarInt


@categorical(ordered=False)
class _Offer:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class _Partner:
    low: ScalarInt
    mid: ScalarInt
    high: ScalarInt


def _single(*, experience: IntND, work: IntND, wage: FloatND) -> FloatND:
    return 1 + work * (2 + 0.2 * experience + wage) + 2.45 * (1 - work)


def _single_reading_partner(
    *, experience: IntND, work: IntND, wage: FloatND, offered: IntND
) -> FloatND:
    return _single(experience=experience, work=work, wage=wage) + 0.01 * offered


def _partnered(*, experience: IntND, partner: IntND, work: IntND) -> FloatND:
    return 1.5 + 0.4 * partner + work * (2 + 0.2 * experience) + 2.0 * (1 - work)


def _flat(*, experience: IntND, work: IntND) -> FloatND:
    return 2.0 + 0 * (experience + work)


def _flat_partnered(*, experience: IntND, partner: IntND, work: IntND) -> FloatND:
    """Below, equal to and above the single's 2 for a low, mid and high partner."""
    return 1.5 + 0.5 * partner + 0 * (experience + work)


def _experience(experience: IntND) -> IntND:
    return experience


def _partner(offered: IntND) -> IntND:
    return offered


def _offer(offer: IntND) -> BoolND:
    return offer == _Offer.yes


def _always(offer: IntND) -> BoolND:
    return offer >= _Offer.no


def _never(offer: IntND) -> BoolND:
    return offer > _Offer.yes


def _offer_with_experience(*, offer: IntND, experience: IntND) -> BoolND:
    return (offer == _Offer.yes) & (experience >= 0)


def _offer_unless_new(experience: IntND) -> FloatND:
    """No offer arrives without experience; half the time otherwise."""
    rate = jnp.where(experience == 0, 0.0, 0.5)
    return jnp.stack([1 - rate, rate], axis=-1)


def _partner_not_low(partner: IntND) -> BoolND:
    return partner > _Partner.low


def _next_experience(*, experience: IntND, work: IntND) -> IntND:
    return jnp.minimum(experience + work, _N_EXP - 1)


def _unilateral(
    *,
    available: Callable[..., BoolND] | None = _offer,
    tie: Literal["stay", "leave"] = "stay",
    offer_probs: tuple[float, float] | Callable[..., FloatND] = (0.4, 0.6),
    partner_probs: tuple[float, float, float] = (0.2, 0.5, 0.3),
    stay_reads_partner: bool = False,
    infeasible_low_partner: bool = False,
    composition_action_declared: bool = True,
    one_period: bool = False,
) -> Model:
    """A single may leave unilaterally for a partnered regime when an offer arrives.

    The single's rows cross its wage (2 nodes) with the marriage group: an offer
    flag (2) and the offered partner (3), who becomes the partnered regime's
    partner state. The partnered regime observes no shock, so the single's rows
    are a product of its own two groups. With `one_period` both regimes are
    terminal and their utilities flat, so leaving ties staying for a mid partner.
    """
    experience = DiscreteGrid(values=jnp.arange(_N_EXP))
    work = DiscreteGrid(values=jnp.arange(2))
    composition_action_ = CompositionAction(
        destinations={
            "self": Destination(
                regime="partnered",
                projection={"experience": _experience, "partner": _partner},
            )
        },
        rule=PCOnActions(tie=tie),
        available=available,
    )
    regimes = {
        "single": Regime(
            states={"experience": experience},
            actions={"work": work},
            functions={
                "utility": _flat
                if one_period
                else _single_reading_partner
                if stay_reads_partner
                else _single
            },
            shocks=("wage", "marriage"),
            state_transitions={} if one_period else {"experience": _next_experience},
            composition_action=composition_action_
            if composition_action_declared
            else None,
        ),
        "partnered": Regime(
            states={
                "experience": experience,
                "partner": DiscreteGrid(category_class=_Partner),
            },
            actions={"work": work},
            functions={"utility": _flat_partnered if one_period else _partnered},
            constraints={"partner_not_low": _partner_not_low}
            if infeasible_low_partner
            else {},
            state_transitions={}
            if one_period
            else {
                "experience": _next_experience,
                "partner": fixed_transition("partner"),
            },
        ),
    }
    if not one_period:
        for name, regime in tuple(regimes.items()):
            regimes[f"{name}_terminal"] = regime.replace(
                composition_action=None,
                state_transitions={},
                shocks=tuple(group for group in regime.shocks if group != "marriage"),
                functions={"utility": _single}
                if name == "single"
                else regime.functions,
            )
    return Model(
        regimes=regimes,
        shocks={
            "wage": Group(
                shocks={"wage": Normal(mu=0.0, sigma=0.5)},
                nodes=Product(wage=GaussHermite(n=2)),
            ),
            "marriage": Group(
                shocks={
                    "offer": Discrete(categories=_Offer, probs=offer_probs),
                    "offered": Discrete(categories=_Partner, probs=partner_probs),
                }
            ),
        },
        n_periods=1 if one_period else 2,
        regime_id_class=_OneRegime if one_period else _Regimes,
        initial_nodes=InitialNodes(by_period={0: ("single", "partnered")}),
        edges={}
        if one_period
        else {
            name: {f"{name}_terminal": Periods(values=(0,))}
            for name in ("single", "partnered")
        },
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _root_rows(*, model: Model, root: RegimeName) -> int:
    """Rows the planned root query of `root` evaluates per source cell in period 0."""
    kernel = model._regimes[root].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    node = kernel.definitions[kernel.root]
    return query_information.support_size(node=node, period=node.read.period)


def _build(
    *, model: Callable[[], Model], collapse: bool, monkeypatch: pytest.MonkeyPatch
) -> Model:
    """Build `model` with the solve enumerating available nodes only, or all."""
    with monkeypatch.context() as patch:
        patch.setattr(composition_queries, "_COLLAPSE_UNAVAILABLE_NODES", collapse)
        return model()


def _assert_values_agree(
    *, got: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]
) -> None:
    """Every solution array agrees to 16 units in the last place at its scale.

    Non-finite entries must match exactly. The collapsed sum adds the unavailable
    masses before weighting the stay value, the full one weights each of them, so
    the two round differently at the scale of the array's values.
    """
    assert got.keys() == expected.keys()
    for name, value in expected.items():
        finite = np.isfinite(value)
        assert_agrees_to_ulp(
            got=got[name],
            expected=value,
            n_ulp=16,
            operand_magnitude=float(np.max(np.abs(value[finite]), initial=0.0)),
            err_msg=name,
        )


# Each case: a model, its root, and its root rows per source cell with the switch on
# and off. Formation: wage 2 x taste 3 x observed wage 2 x match 2 = 24 other rows,
# times (3 offers + 1) or the marriage group's 6 nodes. PCOnActions: wage 2 = other
# rows, times (3 offers + 1) or 6.
_ROW_CASES = {
    "formation": (partial(_formation), "single_f", (3 + 1) * 24, 6 * 24),
    "unilateral": (_unilateral, "single", (3 + 1) * 2, 6 * 2),
    "never_available": (partial(_unilateral, available=_never), "single", 2, 12),
    "always_available": (partial(_unilateral, available=_always), "single", 12, 12),
    "reads_a_state": (
        partial(_unilateral, available=_offer_with_experience),
        "single",
        12,
        12,
    ),
    "stay_reads_the_group": (
        partial(_unilateral, stay_reads_partner=True),
        "single",
        12,
        12,
    ),
    # 6 stay rows, each with 4 draws among offers and one row without an offer.
    "drawn": (
        partial(_formation, nodes=QuasiMonteCarlo(n=4, seed=0)),
        "single_f",
        30,
        30,
    ),
}


@pytest.mark.parametrize("case", list(_ROW_CASES))
def test_an_exact_composition_action_choice_enumerates_its_available_nodes_and_one_more(
    case: str,
) -> None:
    """The root evaluates (available nodes + 1) x (other rows) where the rewrite
    applies, and every row where it does not.

    It does not apply when every node is available, when the predicate reads a
    state, when the stay problem reads the group, or when the choice is drawn.
    """
    model, root, expected, _ = _ROW_CASES[case]
    assert _root_rows(model=model(), root=root) == expected


@pytest.mark.parametrize("case", list(_ROW_CASES))
def test_the_switch_restores_the_full_enumeration(
    *, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the rewrite switched off, the root evaluates every node of every group."""
    model, root, _, expected = _ROW_CASES[case]
    built = _build(model=model, collapse=False, monkeypatch=monkeypatch)
    assert _root_rows(model=built, root=root) == expected


_EQUIVALENCE_MODELS: Mapping[str, Callable[[], Model]] = {
    "consent": partial(_formation),
    "consent_dissolution": partial(_formation, dissolution=True),
    "unilateral": _unilateral,
    "unilateral_state_dependent_offer": partial(
        _unilateral, offer_probs=_offer_unless_new
    ),
    "unilateral_no_offer": partial(_unilateral, offer_probs=(1.0, 0.0)),
    "unilateral_infeasible_low_partner": partial(
        _unilateral, infeasible_low_partner=True
    ),
    "unilateral_never_available": partial(_unilateral, available=_never),
    "tie_stay": partial(_unilateral, one_period=True, tie="stay"),
    "tie_leave": partial(_unilateral, one_period=True, tie="leave"),
}


@pytest.mark.parametrize("stored", [False, True], ids=["point", "stored"])
@pytest.mark.parametrize("case", list(_EQUIVALENCE_MODELS))
def test_enumerating_available_nodes_gives_the_full_enumeration_s_values(
    *, case: str, stored: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every solved value array agrees with the full enumeration to rounding.

    The cases cover a consent formation and a unilateral composition action, an offer
    whose probability is zero at some cells or everywhere, a destination infeasible at
    some available nodes, a composition action available nowhere, and ties between
    staying and leaving under either tie rule; stored outside values on and off.
    """
    monkeypatch.setattr(composition_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
    model = _EQUIVALENCE_MODELS[case]
    _assert_values_agree(
        got=_solve(_build(model=model, collapse=True, monkeypatch=monkeypatch)),
        expected=_solve(_build(model=model, collapse=False, monkeypatch=monkeypatch)),
    )


def test_an_infeasible_destination_is_infeasible_at_some_available_nodes() -> None:
    """The infeasible-destination case is real: a low partner has no action."""
    values = _solve(_unilateral(infeasible_low_partner=True))
    assert np.isneginf(values["0/partnered"][:, _Partner.low]).all()


@pytest.mark.parametrize("stored", [False, True], ids=["point", "stored"])
def test_a_composition_action_available_nowhere_gives_the_value_of_staying(
    *, stored: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A composition action available at no node leaves the single's value unchanged.

    The value is that of a single without the composition action, under dyadic
    marriage masses that sum to exactly one.

    The two are different programs, so they may round apart by the code each
    compiles to; at single precision they do by one unit in the last place.
    """
    monkeypatch.setattr(composition_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
    dyadic = partial(
        _unilateral, offer_probs=(0.5, 0.5), partner_probs=(0.25, 0.5, 0.25)
    )
    got = _solve(dyadic(available=_never))
    expected = _solve(dyadic(composition_action_declared=False))
    _assert_values_agree(
        got={"0/single": got["0/single"]}, expected={"0/single": expected["0/single"]}
    )


def test_with_every_node_available_the_values_are_the_full_enumeration_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing collapses if leaving is available everywhere: bit for bit."""
    model = partial(_unilateral, available=_always)
    got = _solve(_build(model=model, collapse=True, monkeypatch=monkeypatch))
    expected = _solve(_build(model=model, collapse=False, monkeypatch=monkeypatch))
    np.testing.assert_equal(got, expected)


def _simulate(model: Model) -> pd.DataFrame:
    params = {"discount_factor": 0.9}
    solution = model.solve(params=params, log_level="off")
    n_agents = 12
    return model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(n_agents, dtype=jnp.int32),
            "regime_id": jnp.full(n_agents, _Regimes.single, dtype=jnp.int32),
            "experience": jnp.arange(n_agents, dtype=jnp.int32) % _N_EXP,
        },
        seed=11,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)


def test_simulation_is_the_same_with_and_without_the_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulation realises each agent's shocks and enumerates no node: the panel
    with a fixed seed is identical whichever way the solve enumerated.
    """
    pd.testing.assert_frame_equal(
        _simulate(_build(model=_unilateral, collapse=True, monkeypatch=monkeypatch)),
        _simulate(_build(model=_unilateral, collapse=False, monkeypatch=monkeypatch)),
        check_exact=True,
    )


@categorical(ordered=False)
class _PairRegimes:
    source: ScalarInt
    destination: ScalarInt


@categorical(ordered=False)
class _LandingRegimes:
    source: ScalarInt
    destination: ScalarInt
    source_terminal: ScalarInt
    destination_terminal: ScalarInt


@categorical(ordered=False)
class _Binary:
    common: ScalarInt
    rare: ScalarInt


def _unless_both_rare(*, a: IntND, b: IntND) -> BoolND:
    return (a == _Binary.common) | (b == _Binary.common)


def _only_both_rare(*, a: IntND, b: IntND) -> BoolND:
    return (a == _Binary.rare) & (b == _Binary.rare)


def _nowhere(*, a: IntND, b: IntND) -> BoolND:
    return (a + b) < 0


def _everywhere(*, a: IntND, b: IntND) -> BoolND:
    return (a + b) >= 0


def _zero(*, wealth: FloatND, work: IntND) -> FloatND:
    return 0.0 * (wealth + work)


def _never_feasible(*, wealth: FloatND, work: IntND) -> BoolND:
    return wealth + work < 0


def _same_wealth(wealth: FloatND) -> FloatND:
    return wealth


def _past_the_grid(wealth: FloatND) -> FloatND:
    return wealth + 2.0


def _rare_pair(
    *,
    p: float,
    stay: float,
    infeasible: bool = False,
    several: bool = False,
    available: Callable[..., BoolND] = _unless_both_rare,
    landing: bool = False,
    leave: float = 0.0,
    drawn: bool = False,
) -> Model:
    """A source may leave for a destination worth zero unless two independent draws
    `a` and `b` of one group are both rare, each with probability `p`; there it
    stays at `stay <= 0`, or nowhere when the stay is infeasible.

    The source's value is the unavailable nodes' mass times the stay value, `p**2 *
    stay` under the default predicate. With `several`, a fair draw in the same group
    makes two nodes unavailable, and a fair draw in a group of its own, which the
    stay reads, gives the node standing for them two rows.

    With `landing` the source is a household of two, and staying moves its wealth past a
    raise-rule grid in a second period. The destination is worth `leave` instead of
    zero. With `drawn` the composition action choice draws the group's nodes where the
    composition action is available and adds one sample for the others.
    """
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    work = DiscreteGrid(values=jnp.arange(2))

    def staying(*, wealth: FloatND, work: IntND) -> FloatND:
        return stay + 0.0 * (wealth + work)

    def staying_reading_d(*, wealth: FloatND, work: IntND, d: IntND) -> FloatND:
        return stay + 0.0 * (wealth + work + d)

    def leaving(*, wealth: FloatND, work: IntND) -> FloatND:
        return leave + 0.0 * (wealth + work)

    pair = {name: Discrete(categories=_Binary, probs=(1.0 - p, p)) for name in "ab"}
    utility = staying
    if several:
        pair["c"] = Discrete(categories=_Binary, probs=(0.5, 0.5))
        utility = staying_reading_d
    shocks = {"pair": Group(shocks=pair)}
    if several:
        shocks["other"] = Group(
            shocks={"d": Discrete(categories=_Binary, probs=(0.5, 0.5))}
        )
    members = ("f", "m") if landing else ("self",)
    regimes = {
        "source": Regime(
            states={"wealth": wealth},
            actions={"work": work},
            functions={
                "utility": CollectiveUtility(utilities=dict.fromkeys(members, utility))
                if landing
                else utility
            },
            constraints={"never_feasible": _never_feasible} if infeasible else {},
            shocks=tuple(shocks),
            state_transitions={"wealth": _past_the_grid} if landing else {},
            composition_action=CompositionAction(
                destinations=dict.fromkeys(
                    members,
                    Destination(
                        regime="destination", projection={"wealth": _same_wealth}
                    ),
                ),
                rule=PCOnActions(tie="stay"),
                available=available,
                nodes=QuasiMonteCarlo(n=4, seed=0) if drawn else None,
            ),
        ),
        "destination": Regime(
            states={"wealth": wealth},
            actions={"work": work},
            functions={"utility": leaving},
            state_transitions={"wealth": _same_wealth} if landing else {},
        ),
    }
    if not landing:
        return Model(
            regimes=regimes,
            shocks=shocks,
            n_periods=1,
            regime_id_class=_PairRegimes,
            initial_nodes=InitialNodes(by_period={0: ("source", "destination")}),
            edges={},
            execution_config=ExecutionConfig(device_memory_bytes=None),
        )
    terminal = Regime(
        states={"wealth": LinSpacedGrid(start=0, stop=1, n_points=2, off_grid="raise")},
        actions={"work": work},
        functions={
            "utility": CollectiveUtility(utilities=dict.fromkeys(members, _zero))
        },
    )
    return Model(
        regimes={
            **regimes,
            "source_terminal": terminal,
            "destination_terminal": regimes["destination"].replace(
                state_transitions={}
            ),
        },
        shocks=shocks,
        n_periods=2,
        regime_id_class=_LandingRegimes,
        initial_nodes=InitialNodes(by_period={0: ("source", "destination")}),
        edges={
            name: {f"{name}_terminal": Periods(values=(0,))}
            for name in ("source", "destination")
        },
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _rare_probability_and_stay(kind: str) -> tuple[float, float]:
    """A rare probability and a stay value for one class of unavailable mass.

    Every unavailable node's mass is `p**2` (halved with `several`), and the
    source's value `p**2 * stay` is a normal number at the active precision:
    - `subnormal`: the node mass is a subnormal number;
    - `below_format`: it is below the smallest subnormal by hundreds of binades;
    - `just_below_format`: it is just below the smallest subnormal.
    """
    x64 = jnp.asarray(0.0).dtype == jnp.float64
    exponents = {
        "subnormal": (-520, 64) if x64 else (-70, 32),
        "below_format": (-600, 900) if x64 else (-80, 120),
        "just_below_format": (-538, 64) if x64 else (-76, 32),
    }[kind]
    return 2.0 ** exponents[0], -(2.0 ** exponents[1])


# Each case: rare probability and stay value, infeasible stay and composition action
# predicate. A string names a class of `_rare_probability_and_stay`, which sets the
# first two.
_MASS_CASES: Mapping[
    str, tuple[tuple[float, float] | str, bool, Callable[..., BoolND]]
] = {
    "ordinary": ((0.25, -4.0), False, _unless_both_rare),
    "ordinary_large_stay": ((0.1, -3.7e6), False, _unless_both_rare),
    "ordinary_infeasible": ((0.25, -4.0), True, _unless_both_rare),
    "zero_probability": ((0.0, -4.0), False, _unless_both_rare),
    "zero_probability_infeasible": ((0.0, -4.0), True, _unless_both_rare),
    "all_unavailable": ((0.25, -4.0), False, _nowhere),
    "all_unavailable_infeasible": ((0.25, -4.0), True, _nowhere),
    "none_unavailable": ((0.25, -4.0), False, _everywhere),
    "subnormal": ("subnormal", False, _unless_both_rare),
    "subnormal_infeasible": ("subnormal", True, _unless_both_rare),
    "below_format": ("below_format", False, _unless_both_rare),
    "below_format_infeasible": ("below_format", True, _unless_both_rare),
    "just_below_format": ("just_below_format", False, _unless_both_rare),
}


# The cases at which the full enumeration and the collapsed one already agreed bit for
# bit before the unavailable nodes kept their own factors.
_BIT_FOR_BIT_CASES = frozenset(
    {
        "ordinary",
        "ordinary_large_stay",
        "ordinary_infeasible",
        "zero_probability",
        "zero_probability_infeasible",
        "all_unavailable",
        "all_unavailable_infeasible",
        "none_unavailable",
    }
)


def _mass_case(case: str) -> tuple[float, float, bool, Callable[..., BoolND]]:
    """The case's rare probability, stay value, infeasible stay and predicate."""
    sizes, infeasible, available = _MASS_CASES[case]
    p, stay = _rare_probability_and_stay(sizes) if isinstance(sizes, str) else sizes
    return p, stay, infeasible, available


def _solve_recording_support(
    *, model: Model, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, np.ndarray], list[bool]]:
    """Solve, and record the support of the source's choice to leave at every cell."""
    kernel = model._regimes["source"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    call = query_information.InformationQueryCell.__call__
    support: list[bool] = []

    # keyword-only-exempt: library-callback=InformationQueryCell.__call__
    def recording(
        self: query_information.InformationQueryCell,
        *,
        live: BoolND,
        source: query_information.SourceStates,
        dependencies: query_information.Dependencies,
    ) -> PolicyAnswer:
        answer = call(self, live=live, source=source, dependencies=dependencies)
        if self.name == kernel.root and self.outside is None:
            jax.debug.callback(lambda flag: support.append(bool(flag)), answer[2])
        return answer

    with monkeypatch.context() as patch:
        patch.setattr(query_information.InformationQueryCell, "__call__", recording)
        values = _solve(model)
    return values, sorted(support)


@pytest.mark.parametrize("stored", [False, True], ids=["point", "stored"])
@pytest.mark.parametrize("several", [False, True], ids=["one_row", "several_rows"])
@pytest.mark.parametrize("case", list(_MASS_CASES))
def test_the_node_standing_for_unavailable_nodes_carries_their_whole_mass(
    *, case: str, several: bool, stored: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enumerating the available nodes and one more gives the full enumeration's
    values, non-finite entries and support, whatever the size of the unavailable
    nodes' mass: ordinary, zero, subnormal or below the smallest subnormal, against
    a finite or an infeasible stay, and bit for bit where the mass is ordinary or
    zero. It evaluates the rows of the available nodes and of one more. The full
    enumeration's source value is the closed form wherever the probability is a
    power of two.
    """
    p, stay, infeasible, available = _mass_case(case)
    monkeypatch.setattr(composition_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
    model = partial(
        _rare_pair,
        p=p,
        stay=stay,
        infeasible=infeasible,
        several=several,
        available=available,
    )
    on = _build(model=model, collapse=True, monkeypatch=monkeypatch)
    off = _build(model=model, collapse=False, monkeypatch=monkeypatch)

    nodes, other_rows = (8, 2) if several else (4, 1)
    if available is _everywhere:
        unavailable = 0
    else:
        unavailable = nodes if available is _nowhere else nodes // 4
    kept = nodes - unavailable + (unavailable > 0)
    assert _root_rows(model=on, root="source") == kept * other_rows
    assert _root_rows(model=off, root="source") == nodes * other_rows

    got, got_support = _solve_recording_support(model=on, monkeypatch=monkeypatch)
    expected, expected_support = _solve_recording_support(
        model=off, monkeypatch=monkeypatch
    )
    if available is _everywhere or p == 0.0:
        closed_form = 0.0
    elif infeasible:
        closed_form = -np.inf
    else:
        # `p * p` alone can leave the double format; the source value does not.
        closed_form = stay if available is _nowhere else p * (p * stay)
    if p == 0.0 or np.frexp(p)[0] == 0.5:
        np.testing.assert_array_equal(
            expected["0/source"], np.full_like(expected["0/source"], closed_form)
        )
    assert got_support == expected_support
    assert len(got_support) > 0
    if case in _BIT_FOR_BIT_CASES:
        np.testing.assert_equal(got, expected)
    else:
        _assert_values_agree(got=got, expected=expected)


def _refusal(model: Model) -> str:
    """The message of the solve's `OffGridStateError`."""
    with pytest.raises(OffGridStateError) as raised:
        _solve(model)
    return str(raised.value)


@pytest.mark.parametrize("stored", [False, True], ids=["point", "stored"])
@pytest.mark.parametrize("several", [False, True], ids=["one_row", "several_rows"])
@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "zero_probability",
        "all_unavailable",
        "subnormal",
        "below_format",
        "just_below_format",
    ],
)
def test_the_node_standing_for_unavailable_nodes_counts_their_landings(
    *, case: str, several: bool, stored: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stay that moves wealth past a raise-rule grid is refused with the full
    enumeration's count of landings, whatever the size of the unavailable nodes'
    mass: every action at every wealth lands once per node of positive mass.
    """
    p, stay, _, available = _mass_case(case)
    monkeypatch.setattr(composition_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
    model = partial(
        _rare_pair, p=p, stay=stay, several=several, available=available, landing=True
    )
    got = _refusal(_build(model=model, collapse=True, monkeypatch=monkeypatch))
    expected = _refusal(_build(model=model, collapse=False, monkeypatch=monkeypatch))

    nodes, other_rows = (8, 2) if several else (4, 1)
    present = (nodes // 4 if p == 0.0 else nodes) * other_rows
    assert got == expected
    assert "regime='source', target='source_terminal'" in expected
    assert expected.endswith(f"count={2 * 2 * present}")


# Each drawn case: a class of unavailable mass, and whether the composition action is
# available only where both draws are rare, so that the available nodes carry the rare
# mass.
_DRAWN_CASES: Mapping[str, tuple[str, bool]] = {
    "ordinary": ("ordinary", False),
    "zero_probability": ("zero_probability", False),
    "all_unavailable": ("all_unavailable", False),
    "subnormal": ("subnormal", False),
    "subnormal_infeasible": ("subnormal_infeasible", False),
    "below_format": ("below_format", False),
    "below_format_infeasible": ("below_format_infeasible", False),
    "just_below_format": ("just_below_format", False),
    "ordinary_available": ("ordinary", True),
    "subnormal_available": ("subnormal", True),
    "below_format_available": ("below_format", True),
}


@pytest.mark.parametrize("several", [False, True], ids=["one_row", "several_rows"])
@pytest.mark.parametrize("case", list(_DRAWN_CASES))
def test_a_drawn_composition_action_choice_carries_the_whole_mass_of_either_side(
    *, case: str, several: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Available nodes plus one sample for the rest match the full enumeration.

    The draw keeps every node where the composition action is available and takes one
    sample for the others. It gives the full enumeration's values, non-finite
    entries and support, whatever the size of either side's mass.

    Every draw among the available nodes leaves for a destination worth the same, so the
    draws integrate that side exactly. Where the composition action is available only if
    both draws are rare, the destination is worth `-stay` and staying zero, so the rare
    mass weights the leaving value: `p**2 * -stay`.
    """
    mass_case, rare_available = _DRAWN_CASES[case]
    p, stay, infeasible, available = _mass_case(mass_case)
    model = partial(
        _rare_pair,
        p=p,
        stay=0.0 if rare_available else stay,
        leave=-stay if rare_available else 0.0,
        infeasible=infeasible,
        several=several,
        available=_only_both_rare if rare_available else available,
    )
    got, got_support = _solve_recording_support(
        model=partial(model, drawn=True)(), monkeypatch=monkeypatch
    )
    expected, expected_support = _solve_recording_support(
        model=_build(model=model, collapse=False, monkeypatch=monkeypatch),
        monkeypatch=monkeypatch,
    )
    if rare_available and np.frexp(p)[0] == 0.5:
        np.testing.assert_array_equal(
            expected["0/source"],
            np.full_like(expected["0/source"], p * (p * -stay)),
        )
    assert got_support == expected_support
    assert len(got_support) > 0
    _assert_values_agree(got=got, expected=expected)


def test_a_query_with_a_collapsed_group_refuses_to_keep_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collapsed plan is not its plan's product, so no group of it can be kept,
    and its standing rows are weighted only on the integrating route.
    """
    model = _build(
        model=partial(_rare_pair, p=0.25, stay=-4.0, several=True),
        collapse=True,
        monkeypatch=monkeypatch,
    )
    kernel = model._regimes["source"].solution.period_kernels[0]
    assert isinstance(kernel, _PolicyQueryKernel)
    node = kernel.definitions[kernel.root]
    assert node.available_nodes is not None
    with pytest.raises(ValueError, match="keeps plan groups as axes"):
        query_information.InformationQueryCell(
            name=kernel.root,
            node=node,
            definitions=kernel.definitions,
            period=node.read.period,
            fixed={},
            projection_operands={},
            information_arguments={},
            kept=(0,),
        )
