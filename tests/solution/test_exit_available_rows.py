"""An exact exit choice enumerates only the nodes where its exit is available.

A single's exit is available only where an offer arrives. The offer and the offered
partner form one shock group, and the single's stay problem reads none of it. Where
no offer arrives the single stays, whoever the partner would have been, so the solve
enumerates the group's nodes with an offer and one more node without, which carries
the whole mass without an offer at the value of staying. Every other shock group
keeps its nodes. The rewrite is exact: the solved values are those of the full
enumeration to rounding, and simulation, which realises its shocks, is untouched.
"""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm import exit_queries, query_information
from _lcm.policy_queries import _PolicyQueryKernel
from lcm import (
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    Periods,
    Regime,
    Unilateral,
    categorical,
    fixed_transition,
)
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
    exit_declared: bool = True,
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
    exit_ = Exit(
        targets={
            "self": ExitTarget(
                regime="partnered",
                projection={"experience": _experience, "partner": _partner},
            )
        },
        rule=Unilateral(tie=tie),
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
            exit=exit_ if exit_declared else None,
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
                exit=None,
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
        patch.setattr(exit_queries, "_COLLAPSE_UNAVAILABLE_NODES", collapse)
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
# times (3 offers + 1) or the marriage group's 6 nodes. Unilateral: wage 2 = other
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
def test_an_exact_exit_choice_enumerates_its_available_nodes_and_one_more(
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

    The cases cover a consent formation and a unilateral exit, an offer whose
    probability is zero at some cells or everywhere, a destination infeasible at
    some available nodes, an exit available nowhere, and ties between staying and
    leaving under either tie rule; stored outside values on and off.
    """
    monkeypatch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
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
def test_an_exit_available_nowhere_gives_the_value_of_staying(
    *, stored: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an exit available at no node, the single's value is that of a single
    without the exit, under dyadic marriage masses that sum to exactly one.

    The two are different programs, so they may round apart by the code each
    compiles to; at single precision they do by one unit in the last place.
    """
    monkeypatch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
    dyadic = partial(
        _unilateral, offer_probs=(0.5, 0.5), partner_probs=(0.25, 0.5, 0.25)
    )
    got = _solve(dyadic(available=_never))
    expected = _solve(dyadic(exit_declared=False))
    _assert_values_agree(
        got={"0/single": got["0/single"]}, expected={"0/single": expected["0/single"]}
    )


def test_with_every_node_available_the_values_are_the_full_enumeration_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing collapses when the exit is available everywhere: bit for bit."""
    model = partial(_unilateral, available=_always)
    got = _solve(_build(model=model, collapse=True, monkeypatch=monkeypatch))
    expected = _solve(_build(model=model, collapse=False, monkeypatch=monkeypatch))
    np.testing.assert_equal(got, expected)


def _simulate(model: Model) -> pd.DataFrame:
    params = {"discount_factor": 0.9}
    solution = model.solve(params=params, log_level="off")
    n_subjects = 12
    return model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(n_subjects, dtype=jnp.int32),
            "regime_id": jnp.full(n_subjects, _Regimes.single, dtype=jnp.int32),
            "experience": jnp.arange(n_subjects, dtype=jnp.int32) % _N_EXP,
        },
        seed=11,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)


def test_simulation_is_the_same_with_and_without_the_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulation realises each subject's shocks and enumerates no node: the panel
    with a fixed seed is identical whichever way the solve enumerated.
    """
    pd.testing.assert_frame_equal(
        _simulate(_build(model=_unilateral, collapse=True, monkeypatch=monkeypatch)),
        _simulate(_build(model=_unilateral, collapse=False, monkeypatch=monkeypatch)),
        check_exact=True,
    )
