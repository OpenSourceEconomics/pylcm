"""An observed shock has one law at the source and at the landing state."""

import importlib
import importlib.util
from collections.abc import Callable
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.grids import Grid
from _lcm.shock_registry import ShockBinding, create_shock_registry
from lcm import (
    CollectiveUtility,
    CompositionAction,
    Consent,
    Counterparty,
    Destination,
    DiscreteGrid,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    PCOnActions,
    PCOnWeight,
    Regime,
    categorical,
)
from lcm.exceptions import ShockDeclarationError
from lcm.shocks import (
    Discrete,
    Law,
    Normal,
    Product,
    ShockGroup,
    UserNodes,
)
from lcm.typing import BoolND, ContinuousState, DiscreteState, FloatND, ScalarInt


@categorical(ordered=True)
class _Outcome:
    low: ScalarInt
    high: ScalarInt


def _probabilities(state: FloatND) -> FloatND:
    high = 0.25 + state / 4
    return jnp.stack((1 - high, high))


def _payoff(eps: ScalarInt) -> FloatND:
    return 1.0 * eps


def _identity(state: FloatND) -> FloatND:
    return state


def _changed(state: FloatND) -> FloatND:
    return state / 2


def _binding(
    *, grid: Grid | None = None, probabilities: Callable[..., FloatND] = _probabilities
) -> ShockBinding:
    regime = Regime(
        states={"state": grid or LinSpacedGrid(start=0, stop=1, n_points=2)},
        functions={"utility": _payoff},
        shocks=("innovation",),
    )
    return create_shock_registry(
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Discrete(categories=_Outcome, probs=probabilities)},
                nodes=Product(),
            )
        },
        regimes={"alive": regime},
        legacy_regimes={"alive": regime},
        model_states={},
    ).bindings["alive", "solve", "innovation"]


def _validator() -> Callable[..., None]:
    assert importlib.util.find_spec("_lcm.shock_projection") is not None, (
        "Observed law admission must be implemented."
    )
    return cast(
        "Callable[..., None]",
        importlib.import_module("_lcm.shock_projection").validate_observed_laws,
    )


def test_formation_only_declaration_is_not_a_raw_read() -> None:
    """A raw stay problem ignores a partner law declared solely for formation."""
    binding = _binding()
    _validator()(
        source="home",
        target="alive",
        physical_inputs=frozenset(),
        source_physical_inputs=frozenset({"eps"}),
        projection={"state": _changed},
        source_bindings={"innovation": binding},
        target_bindings={"innovation": binding},
        source_domains=binding.state_domains,
        target_domains=binding.state_domains,
    )


def _off_grid_change(state: FloatND) -> FloatND:
    return jnp.where((state == 0) | (state == 1), state, state + 1)


def _discarded_computation(state: FloatND) -> FloatND:
    unused = state * 2  # noqa: F841 - discarded equation must not certify continuous identity
    return state


def _identity_through_helper(mapped: FloatND) -> FloatND:
    return mapped


def _finite_identity(state: ScalarInt) -> ScalarInt:
    return jnp.where(state == 0, jnp.int32(0), jnp.int32(1))


def _finite_swap(state: ScalarInt) -> ScalarInt:
    return jnp.int32(1) - state


def _parameter_probabilities(*, state: FloatND, p: FloatND) -> FloatND:
    high = p + state / 4
    return jnp.stack((1 - high, high))


@pytest.mark.parametrize("project", [_off_grid_change, _discarded_computation])
def test_continuous_identity_is_not_a_grid_node_comparison(
    project: Callable[..., FloatND],
) -> None:
    """Off-grid changes and discarded equations cannot prove continuous identity."""
    binding = _binding()
    with pytest.raises(ShockDeclarationError, match=r"'eps'.*'alive'.*state 'state'"):
        _validator()(
            source="home",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": project},
            source_bindings={"innovation": binding},
            target_bindings={"innovation": binding},
            source_domains=binding.state_domains,
            target_domains=binding.state_domains,
        )


def test_effective_helper_composition_preserves_direct_return() -> None:
    """The complete effective projection, including helpers, is the proof agent."""
    binding = _binding()
    _validator()(
        source="home",
        target="alive",
        physical_inputs=frozenset({"eps"}),
        source_physical_inputs=frozenset({"eps"}),
        projection={"state": _identity_through_helper},
        projection_functions={"mapped": _identity},
        source_bindings={"innovation": binding},
        target_bindings={"innovation": binding},
        source_domains=binding.state_domains,
        target_domains=binding.state_domains,
    )


@pytest.mark.parametrize("preserves", [True, False])
def test_discrete_identity_checks_the_complete_code_domain(*, preserves: bool) -> None:
    """Every valid code must map to itself under the same physical-value codec."""
    binding = _binding(grid=DiscreteGrid(values=jnp.array([10.0, 20.0])))
    kwargs = {
        "source": "home",
        "target": "alive",
        "physical_inputs": frozenset({"eps"}),
        "source_physical_inputs": frozenset({"eps"}),
        "projection": {"state": _finite_identity if preserves else _finite_swap},
        "source_bindings": {"innovation": binding},
        "target_bindings": {"innovation": binding},
        "source_domains": binding.state_domains,
        "target_domains": binding.state_domains,
    }
    if preserves:
        _validator()(**kwargs)
    else:
        with pytest.raises(
            ShockDeclarationError, match=r"'eps'.*'alive'.*state 'state'"
        ):
            _validator()(**kwargs)


def test_matching_codes_with_different_value_domains_are_refused() -> None:
    """Equal integer positions cannot hide different physical state domains."""
    source = _binding(grid=DiscreteGrid(values=jnp.array([10.0, 20.0])))
    target = _binding(grid=DiscreteGrid(values=jnp.array([10.0, 30.0])))
    with pytest.raises(
        ShockDeclarationError,
        match=r"'eps'.*'alive'.*(functions or parameters|state 'state')",
    ):
        _validator()(
            source="home",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": _identity},
            source_bindings={"innovation": source},
            target_bindings={"innovation": target},
            source_domains=source.state_domains,
            target_domains=target.state_domains,
        )


def test_matching_parameter_values_cannot_replace_binding_identity() -> None:
    """A shared partition needs the same runtime operand path, not a sampled value."""
    binding = _binding(probabilities=_parameter_probabilities)
    with pytest.raises(
        ShockDeclarationError, match=r"'eps'.*'alive'.*functions or parameters"
    ):
        _validator()(
            source="home",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": _identity},
            source_bindings={"innovation": binding},
            target_bindings={"innovation": binding},
            source_domains=binding.state_domains,
            target_domains=binding.state_domains,
            source_parameter_paths={"innovation__p": ("source", "p")},
            target_parameter_paths={"innovation__p": ("target", "p")},
        )


def test_target_only_hidden_category_has_no_source_partition() -> None:
    """A permission to reference a group does not create a source category read."""
    binding = _binding()
    _validator()(
        source="home",
        target="alive",
        physical_inputs=frozenset({"eps"}),
        source_physical_inputs=frozenset(),
        projection={"state": _changed},
        source_bindings={"innovation": binding},
        target_bindings={"innovation": binding},
        source_domains=binding.state_domains,
        target_domains=binding.state_domains,
    )


def _helper_probabilities(share: FloatND) -> FloatND:
    return jnp.stack((1 - share, share))


def _share(state: FloatND) -> FloatND:
    return 0.25 + state / 4


def _different_share(state: FloatND) -> FloatND:
    return 0.25 + state / 8


def test_phase_specific_law_helper_bindings_must_match() -> None:
    """A shared group declaration does not equate different phase helper bodies."""
    from dataclasses import replace  # noqa: PLC0415

    base = _binding()
    group = ShockGroup(
        shocks={"eps": Discrete(categories=_Outcome, probs=_helper_probabilities)},
        nodes=Product(),
    )
    source = replace(base, group=group, functions={"share": _share})
    target = replace(base, group=group, functions={"share": _different_share})
    with pytest.raises(
        ShockDeclarationError, match=r"'eps'.*'alive'.*functions or parameters"
    ):
        _validator()(
            source="home",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": _identity},
            source_bindings={"innovation": source},
            target_bindings={"innovation": target},
            source_domains=source.state_domains,
            target_domains=target.state_domains,
        )


def _effectful_identity(state: FloatND) -> FloatND:
    import jax  # noqa: PLC0415

    jax.debug.print("state={state}", state=state)
    return state


def test_effectful_projection_is_not_an_identity_proof() -> None:
    """An effectful staged projection cannot certify shared-partition identity."""
    binding = _binding()
    with pytest.raises(ShockDeclarationError, match=r"'eps'.*'alive'.*state 'state'"):
        _validator()(
            source="home",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": _effectful_identity},
            source_bindings={"innovation": binding},
            target_bindings={"innovation": binding},
            source_domains=binding.state_domains,
            target_domains=binding.state_domains,
        )


def _wide_probabilities(state: ScalarInt) -> FloatND:
    high = jnp.where(state == 0, 0.25, 0.5)
    return jnp.stack((1 - high, high))


def _bitwise_identity(state: ScalarInt) -> ScalarInt:
    return state | 0


@pytest.mark.parametrize("direct", [False, True])
def test_large_finite_domains_need_a_direct_identity_proof(*, direct: bool) -> None:
    """Bound exhaustive build-time work while admitting a direct return at any size."""
    import numpy as np  # noqa: PLC0415

    binding = _binding(
        grid=DiscreteGrid(values=np.arange(5000, dtype=np.int32)),
        probabilities=_wide_probabilities,
    )
    kwargs = {
        "source": "home",
        "target": "alive",
        "physical_inputs": frozenset({"eps"}),
        "source_physical_inputs": frozenset({"eps"}),
        "projection": {"state": _identity if direct else _bitwise_identity},
        "source_bindings": {"innovation": binding},
        "target_bindings": {"innovation": binding},
        "source_domains": binding.state_domains,
        "target_domains": binding.state_domains,
    }
    if direct:
        _validator()(**kwargs)
    else:
        with pytest.raises(
            ShockDeclarationError, match=r"'eps'.*'alive'.*state 'state'"
        ):
            _validator()(**kwargs)


# A household `pair` (assets, skill) dissolves: each member lands alone with half the
# assets and keeps skill. The husband's destination draws group `bonuses` and pays
# assets + bonus. The pair draws `bonuses` too and his stay payoff reads it.
_ASSETS = np.array([2.0, 4.0, 6.0])
_SKILL = np.array([1.0, 2.0])
_Z = np.array([-1.5, 0.0, 1.0])
_W = np.array([0.25, 0.5, 0.25])


@categorical(ordered=False)
class _PairIds:
    pair: ScalarInt
    alone_f: ScalarInt
    alone_m: ScalarInt


@categorical(ordered=False)
class _FormationIds:
    single_f: ScalarInt
    single_m: ScalarInt
    couple: ScalarInt


@categorical(ordered=True)
class _Theta:
    low: ScalarInt
    high: ScalarInt


def _half(assets: ContinuousState) -> ContinuousState:
    return 0.5 * assets


def _keep_assets(assets: ContinuousState) -> ContinuousState:
    return assets


def _keep_skill(skill: ContinuousState) -> ContinuousState:
    return skill


def _wife_stay(assets: ContinuousState) -> FloatND:
    return 10.0 + 0.0 * assets


def _husband_stay(*, assets: ContinuousState, bonus: FloatND) -> FloatND:
    return 5.0 + 0.5 * bonus + 0.0 * assets


def _wife_stay_bargained(*, assets: ContinuousState, theta: DiscreteState) -> FloatND:
    return 10.0 + 0.0 * assets + 0.0 * theta


def _husband_stay_bargained(
    *, assets: ContinuousState, bonus: FloatND, theta: DiscreteState
) -> FloatND:
    return 5.0 + 0.5 * bonus + 0.0 * assets + 0.0 * theta


def _alone_f(*, assets: ContinuousState, skill: ContinuousState) -> FloatND:
    return 0.0 * assets + 0.0 * skill


def _alone_m(
    *, assets: ContinuousState, skill: ContinuousState, bonus: FloatND
) -> FloatND:
    return assets + bonus + 0.0 * skill


def _bonus_seen(bonus: FloatND) -> BoolND:
    return bonus > -100.0


def _mean_from_assets(assets: ContinuousState) -> FloatND:
    return assets


def _mean_from_skill(skill: ContinuousState) -> FloatND:
    return 2.0 * skill


def _spread_from_skill(skill: ContinuousState) -> FloatND:
    return skill


def _bonuses(law: Law) -> ShockGroup:
    return ShockGroup(
        shocks={"bonus": law}, nodes=UserNodes(nodes=_Z[:, None], weights=_W)
    )


def _dissolving_pair(*, law: Law, rule: str = "unilateral") -> Model:
    """The pair; `rule` adds a renegotiation or an availability reading the shock."""
    assets = IrregSpacedGrid(points=_ASSETS)
    skill = IrregSpacedGrid(points=_SKILL)
    landing = IrregSpacedGrid(points=_ASSETS / 2)
    projection = {"assets": _half, "skill": _keep_skill}
    bargained = rule == "renegotiate"
    return Model(
        regimes={
            "pair": Regime(
                states={
                    "assets": assets,
                    "skill": skill,
                    **({"theta": DiscreteGrid(_Theta)} if bargained else {}),
                },
                functions={
                    "utility": CollectiveUtility(
                        utilities={
                            "f": _wife_stay_bargained if bargained else _wife_stay,
                            "m": _husband_stay_bargained
                            if bargained
                            else _husband_stay,
                        }
                    )
                },
                composition_action=CompositionAction(
                    destinations={
                        "f": Destination(regime="alone_f", projection=projection),
                        "m": Destination(regime="alone_m", projection=projection),
                    },
                    rule=PCOnWeight(state="theta", tie="leave")
                    if bargained
                    else PCOnActions(tie="leave"),
                    available=_bonus_seen if rule == "available" else None,
                ),
                shocks=("bonuses",),
            ),
            "alone_f": Regime(
                states={"assets": landing, "skill": skill},
                functions={"utility": _alone_f},
            ),
            "alone_m": Regime(
                states={"assets": landing, "skill": skill},
                functions={"utility": _alone_m},
                shocks=("bonuses",),
            ),
        },
        edges={},
        shocks={"bonuses": _bonuses(law)},
        n_periods=1,
        regime_id_class=_PairIds,
        initial_nodes=InitialNodes(by_period={0: ("pair",)}),
    )


def _single(*, assets: ContinuousState, skill: ContinuousState) -> FloatND:
    return 1.0 + 0.0 * assets + 0.0 * skill


def _single_seeing_bonus(
    *, assets: ContinuousState, skill: ContinuousState, bonus: FloatND
) -> FloatND:
    return 1.0 + 0.5 * bonus + 0.0 * assets + 0.0 * skill


def _couple_f(
    *, assets: ContinuousState, skill: ContinuousState, bonus: FloatND
) -> FloatND:
    return 3.0 + 0.0 * assets + 0.0 * skill + 0.0 * bonus


def _formation(*, law: Law) -> Model:
    """A single man marries by consent into a couple that draws `bonuses`."""
    assets = IrregSpacedGrid(points=_ASSETS)
    skill = IrregSpacedGrid(points=_SKILL)
    return Model(
        regimes={
            "single_f": Regime(
                states={"assets": assets, "skill": skill},
                functions={"utility": _single},
            ),
            "single_m": Regime(
                states={"assets": assets, "skill": skill},
                functions={"utility": _single_seeing_bonus},
                shocks=("bonuses",),
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="couple",
                            role="m",
                            projection={"assets": _half, "skill": _keep_skill},
                        )
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="single_f",
                            role="f",
                            projection={"assets": _keep_assets, "skill": _keep_skill},
                        )
                    ),
                ),
            ),
            "couple": Regime(
                states={"assets": IrregSpacedGrid(points=_ASSETS / 2), "skill": skill},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _couple_f, "m": _alone_m}
                    )
                },
                shocks=("bonuses",),
            ),
        },
        edges={},
        shocks={"bonuses": _bonuses(law)},
        n_periods=1,
        regime_id_class=_FormationIds,
        initial_nodes=InitialNodes(by_period={0: ("single_m", "single_f", "couple")}),
    )


def _model(*, law: Law, rule: str) -> Model:
    if rule == "consent":
        return _formation(law=law)
    return _dissolving_pair(law=law, rule=rule)


_MEAN_FROM_ASSETS = Normal(mu=_mean_from_assets, sigma=1.0)


@pytest.mark.parametrize(
    ("law", "rule"),
    [(_MEAN_FROM_ASSETS, "unilateral")],
    ids=["shared-normal"],
)
def test_observed_law_reading_a_state_the_composition_action_changes_is_refused(
    *, law: Law, rule: str
) -> None:
    """The source's decision and the landing would read the draw under two laws."""
    source, target = (
        ("single_m", "couple") if rule == "consent" else ("pair", "alone_m")
    )
    with pytest.raises(
        ShockDeclarationError,
        match=rf"'{source}'.*'bonus'.*'bonuses'.*'{target}'.*'assets'.*"
        rf"Draw 'bonus' in '{source}'",
    ):
        _model(law=law, rule=rule)


def test_observed_law_reading_only_kept_states_has_one_law() -> None:
    """Mean and spread read skill, which separation keeps, so the draw has one law.

    The husband's value is sum_k w_k max(5 + b_k / 2, a / 2 + b_k) with
    b_k = 2 s + s z_k.
    """
    model = _dissolving_pair(law=Normal(mu=_mean_from_skill, sigma=_spread_from_skill))
    values = np.asarray(model.solve(params={}, log_level="off").values[0]["pair"])
    bonus = 2 * _SKILL[:, None] + _SKILL[:, None] * _Z
    expected = np.maximum(5 + bonus / 2, _ASSETS[:, None, None] / 2 + bonus) @ _W
    np.testing.assert_allclose(values[..., 1], expected, rtol=1e-6)


# A single woman (edu_f) draws a marriage offer (offer_edu, probabilities 0.3 / 0.7)
# and forms a couple by consent, landing with edu_m = offer_edu. Only the couple draws
# the husband's shock; the formation observes it, and its law reads edu_m: a job with
# probability 0.2 + 0.6 edu_m (literal nodes), or a wage N(-0.5 + 2 edu_m, 1) on nodes
# that move with edu_m. The law exists at the landing state alone.
_OFFER_PROBS = np.array([0.3, 0.7])


def _offer_probs() -> FloatND:
    return jnp.asarray(_OFFER_PROBS)


def _job_probs(edu_m: ScalarInt) -> FloatND:
    high = 0.2 + 0.6 * edu_m
    return jnp.stack((1 - high, high))


def _wage_mean(edu_m: ScalarInt) -> FloatND:
    return -0.5 + 2.0 * edu_m


def _keep_edu_f(edu_f: ScalarInt) -> ScalarInt:
    return edu_f


def _husband_from_offer(offer_edu: ScalarInt) -> ScalarInt:
    return offer_edu


def _stay_job(edu_f: ScalarInt) -> FloatND:
    return 1.8 + 0.0 * edu_f


def _stay_wage(edu_f: ScalarInt) -> FloatND:
    return 1.0 + 0.0 * edu_f


def _single_man(edu: ScalarInt) -> FloatND:
    return 0.5 + 0.0 * edu


def _wife_job(*, edu_f: ScalarInt, edu_m: ScalarInt, job_m: ScalarInt) -> FloatND:
    return 1.0 + 1.5 * job_m + 0.0 * edu_f + 0.0 * edu_m


def _husband_job(*, edu_m: ScalarInt, job_m: ScalarInt) -> FloatND:
    return 1.0 + 2.0 * job_m + 0.0 * edu_m


def _wife_wage(*, edu_f: ScalarInt, edu_m: ScalarInt, wage_m: FloatND) -> FloatND:
    return wage_m + 0.0 * edu_f + 0.0 * edu_m


def _husband_wage(*, edu_m: ScalarInt, wage_m: FloatND) -> FloatND:
    return 10.0 + 0.0 * wage_m + 0.0 * edu_m


def _destination_law_formation(*, shock: str) -> Model:
    binary = DiscreteGrid(values=jnp.arange(2))
    if shock == "job_m":
        group = ShockGroup(
            shocks={"job_m": Discrete(categories=_Outcome, probs=_job_probs)}
        )
        stay, wife, husband = _stay_job, _wife_job, _husband_job
    else:
        group = ShockGroup(
            shocks={"wage_m": Normal(mu=_wage_mean, sigma=1.0)},
            nodes=UserNodes(nodes=_Z[:, None], weights=_W),
        )
        stay, wife, husband = _stay_wage, _wife_wage, _husband_wage
    return Model(
        regimes={
            "single_f": Regime(
                states={"edu_f": binary},
                functions={"utility": stay},
                shocks=("marriage",),
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="couple",
                            role="f",
                            projection={
                                "edu_f": _keep_edu_f,
                                "edu_m": _husband_from_offer,
                            },
                        )
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="single_m",
                            role="m",
                            projection={"edu": _husband_from_offer},
                        )
                    ),
                    observes=("offer_edu", shock),
                ),
            ),
            "single_m": Regime(
                states={"edu": binary}, functions={"utility": _single_man}
            ),
            "couple": Regime(
                states={"edu_f": binary, "edu_m": binary},
                functions={
                    "utility": CollectiveUtility(utilities={"f": wife, "m": husband})
                },
                shocks=("jobs_m",),
            ),
        },
        edges={},
        shocks={
            "marriage": ShockGroup(
                shocks={"offer_edu": Discrete(categories=_Outcome, probs=_offer_probs)}
            ),
            "jobs_m": group,
        },
        n_periods=1,
        regime_id_class=_FormationIds,
        initial_nodes=InitialNodes(by_period={0: ("single_f", "single_m", "couple")}),
    )


@pytest.mark.parametrize(("shock", "expected"), [("job_m", 2.234), ("wage_m", 1.4375)])
def test_observed_law_of_a_group_only_the_destination_draws_is_the_landing_law(
    *, shock: str, expected: float
) -> None:
    """The wife's value is sum_o P(o) E[max(stay, couple) | edu_m = o]: for the job,
    0.3 (0.2 * 2.5 + 0.8 * 1.8) + 0.7 (0.8 * 2.5 + 0.2 * 1.8) = 2.234; for the wage,
    0.3 * 1 + 0.7 sum_k w_k max(1, 1.5 + z_k) = 1.4375. A mover's observed draw is
    the couple's draw."""
    model = _destination_law_formation(shock=shock)
    values = np.asarray(model.solve(params={}, log_level="off").values[0]["single_f"])
    np.testing.assert_allclose(values, expected, rtol=1e-6)

    n = 200
    frame = model.simulate(
        params={},
        initial_conditions={
            "period": jnp.zeros(n, dtype=jnp.int32),
            "regime_id": jnp.zeros(n, dtype=jnp.int32),
            "edu_f": jnp.arange(n, dtype=jnp.int32) % 2,
            "edu": jnp.zeros(n, dtype=jnp.int32),
            "edu_m": jnp.zeros(n, dtype=jnp.int32),
        },
        seed=11,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    movers = frame[frame["regime_name"] == "couple"]
    assert len(movers) > 0
    np.testing.assert_array_equal(movers[shock], movers[f"source__{shock}"])
