"""An outside read is aligned when the model build proves it lands on stored nodes.

A read of a destination's problem is aligned when every state its projection writes
is provably a point of the destination's grid, and every shock group of the
destination is either fixed by the caller's frame, with the caller's own nodes, or
integrated without being observed. The proof runs when the model is built and is
conservative: a read the build cannot prove aligned is evaluated at its exact point.
"""

import re
from collections.abc import Callable, Mapping

import jax.numpy as jnp
import pytest

from _lcm import exit_queries, policy_queries
from _lcm.policy_queries import QueryKind, QueryNode
from lcm import (
    AgeGrid,
    CollectiveUtility,
    Consent,
    Counterparty,
    DeterministicTransition,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    Regime,
    Transition,
    Unilateral,
    categorical,
    fixed_transition,
)
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import Discrete, GaussHermite, Normal, Product, ShockGroup
from lcm.typing import BoolND, FloatND, IntND, RegimeName, ScalarInt
from lcm_examples.iskhakov_et_al_2017 import WEALTH_GRID, dead
from tests.test_models.deterministic.dcegm_variants import dcegm_retirement
from tests.test_models.deterministic.retirement_only import retirement_edges

_KNOTS = (0.0, 1.5, 3.5, 7.5, 15.0)


@categorical(ordered=False)
class _Pair:
    source: ScalarInt
    destination: ScalarInt


@categorical(ordered=False)
class _Marriage:
    single_f: ScalarInt
    single_m: ScalarInt
    couple: ScalarInt


@categorical(ordered=False)
class _Retirement:
    retirement: ScalarInt
    dead: ScalarInt
    source: ScalarInt


@categorical(ordered=False)
class _Offer:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class _Offered:
    low: ScalarInt
    high: ScalarInt


def _three_codes() -> DiscreteGrid:
    return DiscreteGrid(values=jnp.arange(3))


def _thirds() -> LinSpacedGrid:
    return LinSpacedGrid(start=0, stop=1, n_points=3)


def _knots() -> IrregSpacedGrid:
    return IrregSpacedGrid(points=jnp.asarray(_KNOTS))


def _kind_utility(kind: IntND) -> FloatND:
    return 1.0 + 0.5 * kind


def _kind_utility_of_experience(experience: IntND) -> FloatND:
    return 1.0 + 0.5 * experience


def _wealth_utility(wealth: FloatND) -> FloatND:
    return 1.0 + wealth


def _code_utility(code: IntND) -> FloatND:
    return 1.0 + 0.1 * code


def _experience_utility(experience: FloatND) -> FloatND:
    return jnp.sqrt(experience)


def _experience_f_utility(experience_f: IntND) -> FloatND:
    return 1.0 + 0.1 * experience_f


def _offer_utility(*, wealth: FloatND, offer: IntND) -> FloatND:
    return 1.0 + wealth + 0.1 * offer


def _eps_utility(*, wealth: FloatND, eps: FloatND) -> FloatND:
    return 1.0 + wealth + 0.1 * eps


def _pair_utility(*, wealth: FloatND, first: FloatND) -> FloatND:
    return 1.0 + wealth + 0.1 * first


def _both_utility(*, wealth: FloatND, first: FloatND, second: FloatND) -> FloatND:
    return 1.0 + wealth + 0.1 * first + 0.2 * second


def _kind(kind: IntND) -> IntND:
    return kind


def _wealth(wealth: FloatND) -> FloatND:
    return wealth


def _experience_f(experience_f: IntND) -> IntND:
    return experience_f


def _offer(offer: IntND) -> IntND:
    return offer


def _half(wealth: FloatND) -> FloatND:
    return 0.5 + 0 * wealth


def _quarter(wealth: FloatND) -> FloatND:
    return 0.25 + 0 * wealth


def _on_knots(code: IntND) -> FloatND:
    return jnp.asarray([0.0, 1.5, 3.5, 7.5])[code]


def _off_knots(code: IntND) -> FloatND:
    return jnp.asarray([1.0, 4.0, 8.0, 12.0])[code]


def _shifted(*, wealth: FloatND, shift: FloatND) -> FloatND:
    return wealth + shift


def _sign(eps: FloatND) -> IntND:
    return jnp.where(eps > 0, 1, 0)


def _offer_group() -> ShockGroup:
    return ShockGroup(shocks={"offer": Discrete(categories=_Offer, probs=(0.4, 0.6))})


def _eps_group() -> ShockGroup:
    return ShockGroup(
        shocks={"eps": Normal(mu=0.0, sigma=1.0)},
        nodes=Product(eps=GaussHermite(n=2)),
    )


def _pair_group() -> ShockGroup:
    return ShockGroup(
        shocks={
            "first": Normal(mu=0.0, sigma=1.0),
            "second": Normal(mu=0.0, sigma=0.5),
        },
        nodes=Product(first=GaussHermite(n=2), second=GaussHermite(n=2)),
    )


def _pair_model(
    *,
    source: tuple[str, DiscreteGrid | LinSpacedGrid],
    source_utility: Callable[..., FloatND],
    destination: tuple[str, DiscreteGrid | LinSpacedGrid | IrregSpacedGrid],
    destination_utility: Callable[..., FloatND],
    projection: Callable[..., FloatND | IntND],
    shocks: Mapping[str, ShockGroup] | None = None,
    observes: tuple[str, ...] = (),
    destination_shocks: tuple[str, ...] = (),
) -> Model:
    """A source leaves unilaterally for a destination; both are solved in period 0."""
    groups = {} if shocks is None else dict(shocks)
    return Model(
        regimes={
            "source": Regime(
                states={source[0]: source[1]},
                functions={"utility": source_utility},
                shocks=tuple(groups),
                exit=Exit(
                    targets={
                        "self": ExitTarget(
                            regime="destination",
                            projection={destination[0]: projection},
                        )
                    },
                    rule=Unilateral(),
                    observes=observes,
                ),
            ),
            "destination": Regime(
                states={destination[0]: destination[1]},
                functions={"utility": destination_utility},
                shocks=destination_shocks,
            ),
        },
        shocks=groups,
        regime_id_class=_Pair,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("source", "destination")}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _definitions(*, model: Model, root: RegimeName) -> Mapping[str, QueryNode]:
    kernel = model._regimes[root].solution.period_kernels[0]
    assert isinstance(kernel, policy_queries._PolicyQueryKernel)
    return kernel.definitions


def _read_key(
    *, definitions: Mapping[str, QueryNode], target: RegimeName, depth: int
) -> str:
    (key,) = (
        name
        for name, node in definitions.items()
        if node.read.kind is QueryKind.RAW
        and node.read.target == target
        and len(node.ancestors) == depth
    )
    return key


def _reason(*, model: Model, root: RegimeName, key: str) -> str:
    """Why the build leaves the read `key` unaligned; empty when it is aligned."""
    from _lcm.outside_values import UnalignedOutsideRead, align_outside_read  # noqa: PLC0415

    result = align_outside_read(
        key=key,
        definitions=_definitions(model=model, root=root),
        canonical_regimes=model._regimes,
        value_periods_by_regime={
            name: regime.active_periods for name, regime in model._regimes.items()
        },
    )
    return result.reason if isinstance(result, UnalignedOutsideRead) else ""


_CASES = {
    "identity onto the same discrete grid": (
        lambda: _pair_model(
            source=("kind", _three_codes()),
            source_utility=_kind_utility,
            destination=("kind", _three_codes()),
            destination_utility=_kind_utility,
            projection=_kind,
        ),
        None,
    ),
    "identity onto the same linspaced grid": (
        lambda: _pair_model(
            source=("wealth", _thirds()),
            source_utility=_wealth_utility,
            destination=("wealth", _thirds()),
            destination_utility=_wealth_utility,
            projection=_wealth,
        ),
        None,
    ),
    "identity onto a differently named state on the same grid": (
        lambda: _pair_model(
            source=("experience_f", _three_codes()),
            source_utility=_experience_f_utility,
            destination=("experience", _three_codes()),
            destination_utility=_kind_utility_of_experience,
            projection=_experience_f,
        ),
        None,
    ),
    "a discrete shock code onto a discrete grid": (
        lambda: _pair_model(
            source=("wealth", _thirds()),
            source_utility=_offer_utility,
            destination=("kind", DiscreteGrid(values=jnp.arange(2))),
            destination_utility=_kind_utility,
            projection=_offer,
            shocks={"offer": _offer_group()},
            observes=("offer",),
        ),
        None,
    ),
    "a constant grid point": (
        lambda: _pair_model(
            source=("wealth", _thirds()),
            source_utility=_wealth_utility,
            destination=("wealth", _thirds()),
            destination_utility=_wealth_utility,
            projection=_half,
        ),
        None,
    ),
    "codes onto irregular knots": (
        lambda: _pair_model(
            source=("code", DiscreteGrid(values=jnp.arange(4))),
            source_utility=_code_utility,
            destination=("experience", _knots()),
            destination_utility=_experience_utility,
            projection=_on_knots,
        ),
        None,
    ),
    "identity onto a coarser linspaced grid": (
        lambda: _pair_model(
            source=("wealth", _thirds()),
            source_utility=_wealth_utility,
            destination=("wealth", LinSpacedGrid(start=0, stop=1, n_points=2)),
            destination_utility=_wealth_utility,
            projection=_wealth,
        ),
        r"'wealth'.*0\.5",
    ),
    "a constant off the grid": (
        lambda: _pair_model(
            source=("wealth", _thirds()),
            source_utility=_wealth_utility,
            destination=("wealth", _thirds()),
            destination_utility=_wealth_utility,
            projection=_quarter,
        ),
        r"'wealth'.*0\.25",
    ),
    "codes off irregular knots": (
        lambda: _pair_model(
            source=("code", DiscreteGrid(values=jnp.arange(4))),
            source_utility=_code_utility,
            destination=("experience", _knots()),
            destination_utility=_experience_utility,
            projection=_off_knots,
        ),
        r"'experience'.*1\.0\b",
    ),
    "a runtime parameter": (
        lambda: _pair_model(
            source=("wealth", _thirds()),
            source_utility=_wealth_utility,
            destination=("wealth", _thirds()),
            destination_utility=_wealth_utility,
            projection=_shifted,
        ),
        r"'wealth'.*parameter",
    ),
    "a continuous shock": (
        lambda: _pair_model(
            source=("wealth", _thirds()),
            source_utility=_eps_utility,
            destination=("kind", DiscreteGrid(values=jnp.arange(2))),
            destination_utility=_kind_utility,
            projection=_sign,
            shocks={"eps": _eps_group()},
            observes=("eps",),
        ),
        r"'kind'.*continuous shock 'eps'",
    ),
    "a runtime-points irregular grid": (
        lambda: _pair_model(
            source=("wealth", _thirds()),
            source_utility=_wealth_utility,
            destination=("wealth", IrregSpacedGrid(n_points=3)),
            destination_utility=_wealth_utility,
            projection=_wealth,
        ),
        r"'wealth'.*run time",
    ),
}


@pytest.mark.parametrize(("model", "reason"), list(_CASES.values()), ids=list(_CASES))
def test_a_projection_is_aligned_exactly_when_it_lands_on_the_grid(
    *, model: Callable[[], Model], reason: str | None
) -> None:
    """A read is aligned exactly when every projected state is a destination node.

    An unaligned read names the state and the first value off the grid, or what
    makes the value unprovable when the model is built.
    """
    built = model()
    definitions = _definitions(model=built, root="source")
    key = _read_key(definitions=definitions, target="destination", depth=1)
    aligned = definitions[key].outside_value is not None
    if reason is None:
        assert aligned
    else:
        found = re.search(reason, _reason(model=built, root="source", key=key))
        assert (aligned, found is not None) == (False, True)


def test_a_partly_observed_group_makes_a_read_unaligned() -> None:
    """The source observes one shock of a destination group and not the other.

    A stored array would have to keep one shock of the group as an axis and
    integrate the other, so the read is evaluated at its point.
    """
    model = _pair_model(
        source=("wealth", _thirds()),
        source_utility=_pair_utility,
        destination=("wealth", _thirds()),
        destination_utility=_both_utility,
        projection=_wealth,
        shocks={"pair": _pair_group()},
        destination_shocks=("pair",),
    )
    definitions = _definitions(model=model, root="source")
    key = _read_key(definitions=definitions, target="destination", depth=1)
    found = re.search(
        r"partly observed.*'pair'", _reason(model=model, root="source", key=key)
    )
    assert (definitions[key].outside_value is None, found is not None) == (True, True)


def test_an_observed_destination_group_is_kept_from_the_caller_that_draws_it() -> None:
    """The source draws `eps` and observes it; the destination's `eps` is a copy.

    The aligned read keeps the destination's `eps` group as an axis, whose node the
    source's own frame enumerates.
    """
    model = _pair_model(
        source=("wealth", _thirds()),
        source_utility=_eps_utility,
        destination=("wealth", _thirds()),
        destination_utility=_eps_utility,
        projection=_wealth,
        shocks={"eps": _eps_group()},
        observes=("eps",),
        destination_shocks=("eps",),
    )
    definitions = _definitions(model=model, root="source")
    key = _read_key(definitions=definitions, target="destination", depth=1)
    (root,) = (name for name, node in definitions.items() if not node.ancestors)
    assert definitions[key].outside_value == policy_queries.OutsideValueRead(
        label="destination:eps", groups=("eps",), origins=(root,)
    )


def _experience(experience: FloatND) -> FloatND:
    return experience


def _experience_f_state(experience_f: FloatND) -> FloatND:
    return experience_f


def _experience_m_state(experience_m: FloatND) -> FloatND:
    return experience_m


def _offered_experience(offered_m: IntND) -> FloatND:
    return jnp.asarray([1.5, 4.0])[offered_m]


def _has_offer(offer: IntND) -> BoolND:
    return offer == 1


def _wife(*, experience_f: FloatND, work: IntND) -> FloatND:
    return jnp.sqrt(experience_f) + work


def _husband(*, experience_m: FloatND, work: IntND) -> FloatND:
    return jnp.sqrt(experience_m) + 1 - work


def _marriage_model() -> Model:
    """A single forms a couple by consent; the couple dissolves into the singles.

    Experience lives on irregular knots. The offered husband's experience is 1.5 or
    4.0, so the husband's dissolution read can land between two knots.
    """
    knots = _knots()
    return Model(
        regimes={
            "single_f": Regime(
                states={"experience": knots},
                functions={"utility": _experience_utility},
                shocks=("marriage",),
                exit=Exit(
                    targets={
                        "self": ExitTarget(
                            regime="couple",
                            role="f",
                            projection={
                                "experience_f": _experience,
                                "experience_m": _offered_experience,
                            },
                        )
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="single_m",
                            role="m",
                            projection={"experience": _offered_experience},
                        )
                    ),
                    available=_has_offer,
                    observes=("offer", "offered_m"),
                ),
            ),
            "single_m": Regime(
                states={"experience": knots},
                functions={"utility": _experience_utility},
            ),
            "couple": Regime(
                states={"experience_f": knots, "experience_m": knots},
                actions={"work": DiscreteGrid(values=jnp.arange(2))},
                functions={
                    "utility": CollectiveUtility(utilities={"f": _wife, "m": _husband})
                },
                exit=Exit(
                    targets={
                        "f": ExitTarget(
                            regime="single_f",
                            projection={"experience": _experience_f_state},
                        ),
                        "m": ExitTarget(
                            regime="single_m",
                            projection={"experience": _experience_m_state},
                        ),
                    },
                    rule=Unilateral(),
                ),
            ),
        },
        shocks={
            "marriage": ShockGroup(
                shocks={
                    "offer": Discrete(categories=_Offer, probs=(0.4, 0.6)),
                    "offered_m": Discrete(categories=_Offered, probs=(0.5, 0.5)),
                }
            )
        },
        regime_id_class=_Marriage,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f", "single_m", "couple")}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def test_a_formation_s_nested_reads_align_the_own_state_not_an_off_grid_partner() -> (
    None
):
    """The wife leaves with her own experience, a knot; the husband may leave at 4.0.

    The wife's dissolution read is an identity of her experience through the
    couple's `experience_f`, so it is aligned. The husband's lands at the offered
    experience 4.0, between the knots 3.5 and 7.5, so it is not.
    """
    model = _marriage_model()
    definitions = _definitions(model=model, root="single_f")
    wife = _read_key(definitions=definitions, target="single_f", depth=2)
    husband = _read_key(definitions=definitions, target="single_m", depth=2)
    observed = (
        definitions[wife].outside_value is not None,
        definitions[husband].outside_value is not None,
        bool(
            re.search(
                r"'experience'.*4\.0",
                _reason(model=model, root="single_f", key=husband),
            )
        ),
    )
    assert observed == (True, False, True)


def test_point_reads_switch_leaves_every_read_unaligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With stored reads switched off no read is aligned; with them on, one is."""

    def aligned() -> bool:
        definitions = _definitions(model=_marriage_model(), root="single_f")
        return any(node.outside_value is not None for node in definitions.values())

    stored = aligned()
    monkeypatch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", False)
    assert (stored, aligned()) == (True, False)


def _to_dead() -> ScalarInt:
    return _Retirement.dead


def _log_wealth(wealth: FloatND) -> FloatND:
    return jnp.log(wealth)


def _richer(wealth: FloatND) -> FloatND:
    return wealth + 1.0


def _retirement_model(*, projection: Callable[..., FloatND]) -> Model:
    """A grid-search source leaves unilaterally for a DC-EGM retirement problem."""
    ages = AgeGrid(start=40, inclusive_stop=60, step="10Y")
    return Model(
        regimes={
            "source": Regime(
                states={"wealth": WEALTH_GRID},
                functions={"utility": _log_wealth},
                state_transitions={"wealth": fixed_transition("wealth")},
                exit=Exit(
                    targets={
                        "self": ExitTarget(
                            regime="retirement", projection={"wealth": projection}
                        )
                    },
                    rule=Unilateral(),
                ),
            ),
            "retirement": dcegm_retirement,
            "dead": dead,
        },
        ages=ages,
        regime_id_class=_Retirement,
        initial_nodes=InitialNodes(by_age={40: ("source", "retirement")}),
        edges={
            **retirement_edges(ages),
            "source": Transition(
                targets={"dead": (40,)}, law=DeterministicTransition(func=_to_dead)
            ),
        },
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def test_an_unaligned_read_of_an_endogenous_grid_problem_is_refused_at_build() -> None:
    """An endogenous-grid problem cannot be evaluated at an off-grid point."""
    with pytest.raises(
        UnsupportedOperationError,
        match=r"reads 'retirement', whose DCEGM solver cannot be evaluated at a "
        r"point.*'wealth'",
    ):
        _retirement_model(projection=_richer)


def test_an_aligned_read_of_an_endogenous_grid_problem_is_refused_until_produced() -> (
    None
):
    """An aligned read would take a stored array, which no endogenous-grid solve
    produces.
    """
    with pytest.raises(UnsupportedOperationError, match=r"not produced yet"):
        _retirement_model(projection=_wealth)
