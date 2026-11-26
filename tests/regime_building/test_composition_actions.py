"""A composition action declares a current choice and immutable mapped targets."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import pytest

import lcm
from _lcm.composition_actions import composition_action_without_pruned_states
from _lcm.grids import Grid
from lcm import CompositionAction, Consent, Counterparty, Destination, Model
from lcm.composition_actions import CompositionRule, PCOnActions
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.typing import BoolND, FloatND, IntND, StateName, UserFunction, UserParams

_UNILATERAL = PCOnActions()


def _half(wealth: FloatND) -> FloatND:
    return wealth / 2


# The former names, spelled in pieces so that no rename of the current names reaches
# them.
_FORMER_MODULE = "lcm." + "ex" + "its"


@pytest.mark.parametrize(
    "name",
    ["Unilateral", "Renegotiate", "Settl" + "ement", "Ex" + "it", "Ex" + "itTarget"],
)
def test_the_rule_classes_carry_the_literatures_names_only(name: str) -> None:
    """The rules are named by the literature's terms; the former names are gone."""
    for public in (
        "PCOnActions",
        "PCOnWeight",
        "MutualConsent",
        "Transfer",
        "Consent",
        "CompositionAction",
        "Destination",
    ):
        assert hasattr(lcm, public)
    assert not hasattr(lcm, name)
    with pytest.raises(ImportError):
        exec(f"from lcm.composition_actions import {name}", {})  # noqa: S102
    with pytest.raises(ImportError):
        exec(f"import {_FORMER_MODULE}", {})  # noqa: S102


def test_participation_constraints_on_the_pareto_weight_carry_the_state() -> None:
    """`PCOnWeight` holds the adjusted state or none; other rules hold no choice."""
    weight_rule = lcm.PCOnWeight(state="mu", tie="leave")
    assert (weight_rule.state, weight_rule.tie) == ("mu", "leave")
    assert lcm.PCOnWeight().state is None
    assert not hasattr(lcm.PCOnActions(), "compares")
    assert not hasattr(lcm.PCOnActions(), "state")
    assert "compares" not in lcm.MutualConsent.__dataclass_fields__


def test_composition_action_targets_snapshot_the_authored_projection() -> None:
    """A composition action keeps the declared projection if caller mappings change."""
    projection = {"wealth": _half}
    target = Destination(regime="single", projection=projection)
    targets = {"self": target}
    declaration = CompositionAction(destinations=targets, rule=PCOnActions())
    projection.clear()
    targets.clear()
    actual_projection = declaration.destinations["self"].projection
    assert isinstance(actual_projection, Mapping)
    assert actual_projection["wealth"] is _half
    with pytest.raises(FrozenInstanceError):
        target.regime = "other"  # ty: ignore[invalid-assignment] - mutation refusal


def _utility() -> float:
    return 1.0


def test_regime_replacement_preserves_its_composition_action() -> None:
    """Replacing an economic helper keeps the authored composition action."""
    declaration = CompositionAction(
        destinations={"self": Destination(regime="single", projection={})},
        rule=PCOnActions(),
    )
    regime = lcm.Regime(functions={"utility": _utility}, composition_action=declaration)
    assert regime.replace(description="another label").composition_action is declaration


def test_phased_projection_snapshots_both_transfers() -> None:
    """Caller edits cannot alter either phase's destination transfer."""
    solve = {"wealth": _half}
    simulate = {"wealth": _half}
    target = Destination(
        regime="single", projection=lcm.Phased(solve=solve, simulate=simulate)
    )
    solve.clear()
    simulate.clear()
    assert isinstance(target.projection, lcm.Phased)
    assert target.projection.solve["wealth"] is _half
    assert target.projection.simulate["wealth"] is _half


@pytest.mark.parametrize(
    "kind", ["empty", "observes", "target_name", "consent_payment"]
)
def test_malformed_composition_action_declarations_fail_at_construction(
    kind: str,
) -> None:
    """Empty targets, invalid names, duplicate observations and a paid formation."""
    target = Destination
    declaration = CompositionAction
    with pytest.raises(RegimeInitializationError):  # noqa: PT012 - malformed constructor variants
        if kind == "empty":
            declaration(destinations={}, rule=PCOnActions())
        elif kind == "observes":
            declaration(
                destinations={"self": target(regime="single", projection={})},
                rule=PCOnActions(),
                observes=("offer", "offer"),
            )
        elif kind == "consent_payment":
            declaration(
                destinations={
                    "self": target(
                        regime="couple",
                        role="f",
                        projection={},
                        utility_adjustment=_cost,
                    )
                },
                rule=Consent(
                    counterparty=Counterparty(regime="single", role="m", projection={})
                ),
            )
        else:
            target(regime="bad__regime", projection={})


def test_collective_composition_action_names_every_source_member() -> None:
    """A couple cannot silently omit a member's destination benchmark."""
    with pytest.raises(RegimeInitializationError, match="destination"):
        lcm.Regime(
            functions={
                "utility": lcm.CollectiveUtility(
                    utilities={"f": _utility, "m": _utility}
                )
            },
            composition_action=CompositionAction(
                destinations={"f": Destination(regime="single", projection={})},
                rule=PCOnActions(),
            ),
        )


def _transfer(*, wealth: FloatND, share: FloatND) -> FloatND:
    return wealth * share


def _realized_transfer(*, wealth: FloatND, realized_share: FloatND) -> FloatND:
    return wealth * realized_share


def _available(*, wealth: FloatND, threshold: FloatND) -> BoolND:
    return wealth > threshold


def _cost(cost: FloatND) -> FloatND:
    return -cost


def _wealth_utility(wealth: FloatND) -> FloatND:
    return wealth


def _declared_model(
    *,
    fixed_params: UserParams | None = None,
    rule: CompositionRule = _UNILATERAL,
    grid: Grid | None = None,
) -> Model:
    from lcm import InitialNodes, LinSpacedGrid, Model, Phased, Regime  # noqa: PLC0415
    from lcm.typing import ScalarInt  # noqa: PLC0415

    @lcm.categorical(ordered=False)
    class RegimeId:
        source: ScalarInt
        target: ScalarInt

    return Model(
        fixed_params={} if fixed_params is None else fixed_params,
        regimes={
            "source": Regime(
                functions={"utility": _utility},
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="target",
                            projection=Phased(
                                solve={"wealth": _transfer},
                                simulate={"wealth": _realized_transfer},
                            ),
                            utility_adjustment=_cost,
                        )
                    },
                    rule=rule,
                    available=_available,
                ),
            ),
            "target": Regime(functions={"utility": _wealth_utility}),
        },
        states={
            "wealth": LinSpacedGrid(start=0, stop=2, n_points=3)
            if grid is None
            else grid
        },
        n_periods=1,
        regime_id_class=RegimeId,
        initial_nodes=InitialNodes(by_period={0: ("source", "target")}),
        edges={},
    )


def test_composition_action_only_state_and_phase_parameters_remain_declared() -> None:
    """Both transfer phases and their free inputs remain in the template."""
    model = _declared_model()
    assert model.get_params_template()["source"]["composition_action"] == {
        "available": {"threshold": "FloatND"},
        "destinations": {
            "self": {
                "projection": {
                    "wealth": {"share": "FloatND", "realized_share": "FloatND"}
                },
                "utility_adjustment": {"cost": "FloatND"},
            }
        },
    }
    assert "wealth" in model.user_regimes["source"].states


def test_bound_composition_action_model_cannot_silently_run_a_stay_only_problem() -> (
    None
):
    """The public declaration's superior outside option changes its Full value."""
    solution = _declared_model().solve(
        params={"share": 1.0, "realized_share": 1.0, "threshold": 0.0, "cost": 0.0},
        log_level="off",
    )
    np.testing.assert_array_equal(solution.values[0]["source"][..., 0], [1.0, 1.0, 2.0])


def _switch_wealth(wealth: FloatND) -> FloatND:
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp.where(wealth > 15, 10.0, 20.0)


def test_transfer_reads_physical_grid_values_and_returns_target_codes() -> None:
    """A transfer decodes source values and encodes destination codes."""
    import jax.numpy as jnp  # noqa: PLC0415

    from lcm import DiscreteGrid, InitialNodes, Model, Regime  # noqa: PLC0415
    from lcm.typing import ScalarInt  # noqa: PLC0415

    @lcm.categorical(ordered=False)
    class RegimeId:
        source: ScalarInt
        target: ScalarInt

    model = Model(
        regimes={
            "source": Regime(
                functions={"utility": _utility},
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="target", projection={"wealth": _switch_wealth}
                        )
                    },
                    rule=PCOnActions(),
                ),
            ),
            "target": Regime(functions={"utility": _wealth_utility}),
        },
        states={"wealth": DiscreteGrid(values=jnp.asarray([10.0, 20.0]))},
        n_periods=1,
        regime_id_class=RegimeId,
        initial_nodes=InitialNodes(by_period={0: ("source", "target")}),
        edges={},
    )
    projection = (
        cast("CompositionAction", model.user_regimes["source"].composition_action)
        .destinations["self"]
        .projection
    )
    assert isinstance(projection, Mapping)
    assert projection["wealth"](wealth=jnp.int32(1)) == 0


@pytest.mark.parametrize("kind", ["unknown", "missing_role", "bad_role", "projection"])
def test_composition_action_destination_must_be_a_complete_declared_policy_context(
    kind: str,
) -> None:
    """Destination names, collective roles and projected states are model-owned."""
    from lcm import InitialNodes, LinSpacedGrid, Model, Regime  # noqa: PLC0415
    from lcm.typing import ScalarInt  # noqa: PLC0415

    @lcm.categorical(ordered=False)
    class RegimeId:
        source: ScalarInt
        target: ScalarInt

    target = Destination(
        regime="missing" if kind == "unknown" else "target",
        projection={} if kind == "projection" else {"wealth": _half},
        role="other" if kind == "bad_role" else None,
    )
    with pytest.raises(RegimeInitializationError, match=r"[Cc]omposition action"):
        Model(
            regimes={
                "source": Regime(
                    functions={"utility": _wealth_utility},
                    composition_action=CompositionAction(
                        destinations={"self": target}, rule=PCOnActions()
                    ),
                ),
                "target": Regime(
                    functions={
                        "utility": lcm.CollectiveUtility(
                            utilities={"f": _wealth_utility, "m": _wealth_utility}
                        )
                    }
                    if kind in {"missing_role", "bad_role"}
                    else {"utility": _wealth_utility}
                ),
            },
            states={"wealth": LinSpacedGrid(start=0, stop=2, n_points=3)},
            n_periods=1,
            regime_id_class=RegimeId,
            initial_nodes=InitialNodes(by_period={0: ("source", "target")}),
            edges={},
        )


def test_fixed_composition_action_parameter_is_removed_from_the_runtime_template() -> (
    None
):
    """A fixed adjustment uses its own exact declaration slot."""
    model = _declared_model(
        fixed_params={
            "source": {
                "composition_action": {
                    "destinations": {"self": {"utility_adjustment": {"cost": 0.0}}}
                }
            }
        }
    )
    template = model.get_params_template()["source"]["composition_action"]
    assert template == {
        "available": {"threshold": "FloatND"},
        "destinations": {
            "self": {
                "projection": {
                    "wealth": {"share": "FloatND", "realized_share": "FloatND"}
                },
            }
        },
    }


def test_consent_snapshots_its_counterparty_projection() -> None:
    """Formation retains the declared counterparty Raw state mapping."""
    projection = {"wealth": _half}
    consent = Consent(
        counterparty=Counterparty(regime="single", role="m", projection=projection)
    )
    projection.clear()
    assert isinstance(consent.counterparty.projection, Mapping)
    assert consent.counterparty.projection["wealth"] is _half


@pytest.mark.parametrize(
    "kind", ["unknown", "own_role", "unknown_role", "projection", "inactive"]
)
def test_consent_counterparty_is_a_coherent_declared_policy_context(kind: str) -> None:
    """Consent binds the other destination member to a complete Raw stay context."""
    from lcm.typing import ScalarInt  # noqa: PLC0415

    @lcm.categorical(ordered=False)
    class RegimeId:
        source: ScalarInt
        other: ScalarInt
        couple: ScalarInt

    counterparty = Counterparty(
        regime="missing" if kind == "unknown" else "other",
        role="f"
        if kind == "own_role"
        else "missing"
        if kind == "unknown_role"
        else "m",
        projection={} if kind == "projection" else {"wealth": _half},
    )
    error = (
        ModelInitializationError if kind == "inactive" else RegimeInitializationError
    )
    message = (
        r"a Raw simulation policy query requires 'other' at age 0, .*"
        r"no simulate edge out of 'other' at that age"
        if kind == "inactive"
        else r"[Cc]ounterparty"
    )
    with pytest.raises(error, match=message):
        Model(
            regimes={
                "source": lcm.Regime(
                    functions={"utility": _wealth_utility},
                    composition_action=CompositionAction(
                        destinations={
                            "self": Destination(
                                regime="couple",
                                role="f",
                                projection={"wealth": _half},
                            )
                        },
                        rule=Consent(counterparty=counterparty),
                    ),
                ),
                "other": lcm.Regime(
                    functions={"utility": _wealth_utility},
                    state_transitions={"wealth": lcm.fixed_transition("wealth")}
                    if kind == "inactive"
                    else {},
                ),
                "couple": lcm.Regime(
                    functions={
                        "utility": lcm.CollectiveUtility(
                            utilities={"f": _wealth_utility, "m": _wealth_utility}
                        )
                    }
                ),
            },
            states={"wealth": lcm.LinSpacedGrid(start=0, stop=2, n_points=3)},
            n_periods=3 if kind == "inactive" else 1,
            regime_id_class=RegimeId,
            initial_nodes=lcm.InitialNodes(
                by_period={0: ("source", "couple"), 1: "other"}
            )
            if kind == "inactive"
            else lcm.InitialNodes(by_period={0: ("source", "other", "couple")}),
            edges={"other": {"source": lcm.Periods(values=(1,))}}
            if kind == "inactive"
            else {},
        )


def test_composition_action_transfer_phase_variation_is_reported() -> None:
    """Different solve and realised transfers prohibit invariant replay."""
    from _lcm.regime_building.phases import phase_variation_paths  # noqa: PLC0415

    model = _declared_model()
    paths = phase_variation_paths(
        user_regime=model.user_regimes["source"], law=model.graph.laws["source"]
    )
    assert "composition_action.destinations['self'].projection" in paths


def _composition_action_signal(*, eps: FloatND, signal_scale: FloatND) -> FloatND:
    return eps * signal_scale


def _signal_available(
    *, composition_action_signal: FloatND, threshold: FloatND
) -> BoolND:
    return composition_action_signal > threshold


def _signal_adjustment(
    *, eps: FloatND, composition_action_signal: FloatND, fee: FloatND
) -> FloatND:
    return eps + composition_action_signal - fee


def test_composition_action_observed_shock_and_helper_reads_are_not_parameters() -> (
    None
):
    """Composition action callback ancestors keep physical shock operands.

    They also keep their helper slots.
    """
    from lcm.shocks import GaussHermite, Normal, Product, ShockGroup  # noqa: PLC0415
    from lcm.typing import ScalarInt  # noqa: PLC0415

    @lcm.categorical(ordered=False)
    class RegimeId:
        source: ScalarInt
        target: ScalarInt

    model = Model(
        regimes={
            "source": lcm.Regime(
                functions={
                    "utility": _utility,
                    "composition_action_signal": _composition_action_signal,
                },
                shocks=("innovation",),
                composition_action=CompositionAction(
                    destinations={
                        "self": Destination(
                            regime="target",
                            projection={},
                            utility_adjustment=_signal_adjustment,
                        )
                    },
                    rule=PCOnActions(),
                    available=_signal_available,
                ),
            ),
            "target": lcm.Regime(functions={"utility": _utility}),
        },
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Normal(mu=0, sigma=1)},
                nodes=Product(eps=GaussHermite(n=3)),
            )
        },
        n_periods=1,
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: ("source", "target")}),
        edges={},
    )
    source = model.get_params_template()["source"]
    assert source["composition_action_signal"] == {"signal_scale": "FloatND"}
    assert source["composition_action"] == {
        "available": {"threshold": "FloatND"},
        "destinations": {"self": {"utility_adjustment": {"fee": "FloatND"}}},
    }


def test_shared_projection_object_remains_phase_invariant() -> None:
    """Snapshotting one explicitly shared transfer preserves replay identity."""
    projection = {"wealth": _half}
    target = Destination(
        regime="single", projection=lcm.Phased(solve=projection, simulate=projection)
    )
    assert isinstance(target.projection, lcm.Phased)
    assert target.projection.solve is target.projection.simulate


@pytest.mark.parametrize("change", ["transfer", "information", "rule"])
def test_composition_action_declarations_enter_durable_model_identity(
    change: str,
) -> None:
    """Stored solutions identify each part of a composition action.

    The parts are its transfers, its information and its decision rule.
    """
    from dataclasses import replace  # noqa: PLC0415

    from _lcm.regime_law import bind_regime_law  # noqa: PLC0415
    from _lcm.solution.fingerprint import (  # noqa: PLC0415
        _project_user_regime_declaration,
        _semantic_fingerprint,
    )

    base = CompositionAction(
        destinations={
            "self": Destination(regime="single", projection={"wealth": _half})
        },
        rule=PCOnActions(),
    )
    if change == "transfer":
        changed = replace(
            base,
            destinations={
                "self": Destination(regime="single", projection={"wealth": _transfer})
            },
        )
    elif change == "information":
        changed = replace(base, observes=("offer",))
    else:
        changed = replace(base, rule=PCOnActions(tie="leave"))
    regime = lcm.Regime(functions={"utility": _utility}, composition_action=base)
    law = bind_regime_law(None)
    assert _semantic_fingerprint(
        _project_user_regime_declaration(regime, law=law)
    ) != _semantic_fingerprint(
        _project_user_regime_declaration(
            regime.replace(composition_action=changed), law=law
        )
    )


def _only_period_one(period: lcm.typing.Period) -> BoolND:
    return period == 1


def _never_available() -> bool:
    return False


@pytest.mark.parametrize("available", [_only_period_one, _never_available])
def test_composition_action_time_only_availability_can_exclude_inactive_contexts(
    available: lcm.typing.UserFunction,
) -> None:
    """An unavailable composition action needs no inactive destination policy."""
    from lcm.typing import ScalarInt  # noqa: PLC0415

    @lcm.categorical(ordered=False)
    class RegimeId:
        source: ScalarInt
        later: ScalarInt

    model = Model(
        regimes={
            "source": lcm.Regime(
                functions={"utility": _utility},
                composition_action=CompositionAction(
                    destinations={"self": Destination(regime="later", projection={})},
                    rule=PCOnActions(),
                    available=available,
                ),
            ),
            "later": lcm.Regime(functions={"utility": _utility}),
        },
        n_periods=3,
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "source", 1: "later"}),
        edges={"later": {"source": lcm.Periods(values=(1,))}},
    )
    assert model.get_params_template()["source"]["composition_action"] == {
        "available": {}
    }


def _kind(kind: IntND) -> IntND:
    return kind


def _double(wealth: FloatND) -> FloatND:
    return 2 * wealth


def test_without_pruned_states_returns_the_action_with_nothing_to_drop() -> None:
    """Where no destination prunes a projected state, the declaration itself returns."""
    declaration = CompositionAction(
        destinations={
            "self": Destination(
                regime="couple", role="f", projection={"wealth": _half, "kind": _kind}
            )
        },
        rule=Consent(
            counterparty=Counterparty(
                regime="single_m", role="m", projection={"wealth": _half}
            )
        ),
    )
    pruned = {"single_f": frozenset({"kind"}), "single_m": frozenset({"kind"})}
    assert composition_action_without_pruned_states(
        declaration=declaration, pruned=pruned
    ) is (declaration)


def test_without_pruned_states_drops_a_destination_s_pruned_state_only() -> None:
    """The projection onto the pruned `kind` goes; wealth and health stay in order."""
    declaration = CompositionAction(
        destinations={
            "self": Destination(
                regime="single",
                projection={"wealth": _half, "kind": _kind, "health": _double},
            )
        },
        rule=_UNILATERAL,
    )
    result = composition_action_without_pruned_states(
        declaration=declaration, pruned={"single": frozenset({"kind"})}
    )
    assert result is not None
    assert tuple(
        cast(
            "Mapping[StateName, UserFunction]", result.destinations["self"].projection
        ).items()
    ) == (
        ("wealth", _half),
        ("health", _double),
    )


def test_without_pruned_states_drops_by_each_destination_s_own_pruning() -> None:
    """The counterparty's single prunes `kind` and loses its projection onto it; the
    couple carries `kind` and keeps the target's projection onto the same name.
    """
    declaration = CompositionAction(
        destinations={
            "self": Destination(
                regime="couple", role="f", projection={"kind": _kind, "wealth": _half}
            )
        },
        rule=Consent(
            counterparty=Counterparty(
                regime="single_m", role="m", projection={"kind": _kind, "wealth": _half}
            )
        ),
    )
    result = composition_action_without_pruned_states(
        declaration=declaration,
        pruned={
            "single_f": frozenset({"kind"}),
            "single_m": frozenset({"kind"}),
            "couple": frozenset(),
        },
    )
    assert result is not None
    assert isinstance(result.rule, Consent)
    assert (
        tuple(
            cast(
                "Mapping[StateName, UserFunction]",
                result.destinations["self"].projection,
            )
        ),
        tuple(
            cast(
                "Mapping[StateName, UserFunction]", result.rule.counterparty.projection
            )
        ),
    ) == (("kind", "wealth"), ("wealth",))


def test_without_pruned_states_drops_a_phased_projection_by_phase() -> None:
    """Each transfer loses its projection onto the pruned `kind`, and keeps its own
    projection onto wealth, as a `Phased` projection.
    """
    declaration = CompositionAction(
        destinations={
            "self": Destination(
                regime="single",
                projection=lcm.Phased(
                    solve={"wealth": _half, "kind": _kind},
                    simulate={"wealth": _double, "kind": _kind},
                ),
            )
        },
        rule=_UNILATERAL,
    )
    result = composition_action_without_pruned_states(
        declaration=declaration, pruned={"single": frozenset({"kind"})}
    )
    assert result is not None
    projection = result.destinations["self"].projection
    assert isinstance(projection, lcm.Phased)
    assert (dict(projection.solve), dict(projection.simulate)) == (
        {"wealth": _half},
        {"wealth": _double},
    )


def _gain_utility(gain: FloatND) -> FloatND:
    return gain


def _other_gain_utility(gain: FloatND) -> FloatND:
    return 2.0 * gain


def _zero_utility() -> float:
    return 0.0


def _penalty(gain: FloatND) -> FloatND:
    return -gain


def _other_penalty(gain: FloatND) -> FloatND:
    return -2.0 * gain


def _charge(penalty: FloatND) -> FloatND:
    return penalty


def _pays_penalty(penalty: FloatND) -> FloatND:
    return penalty


def _pays_charge(charge: FloatND) -> FloatND:
    return charge


def _pays_kept(kept: FloatND) -> FloatND:
    return kept


def _pays_next_gain(next_gain: FloatND) -> FloatND:
    return next_gain


def _pays_division(division: FloatND) -> FloatND:
    return division


def _open_at_penalty(penalty: FloatND) -> BoolND:
    return penalty < 0.0


def _lands_division(division: FloatND) -> FloatND:
    return division


def _lands_other_division(division: FloatND) -> FloatND:
    return 2.0 * division


def _keep_gain(gain: FloatND) -> FloatND:
    return gain


def _halve_gain(gain: FloatND) -> FloatND:
    return 0.5 * gain


_DIFFERS = ", which differs between solve and simulation"
_PAYMENT = (
    "The payment of the composition action from regime 'couple' to destination "
    "'f_alone' reads "
)
_AVAILABLE = (
    "The availability condition of the composition action from regime 'couple' reads "
)
_DEFAULT = "The transfer default of the composition action from regime 'couple' reads "
_PROJECTION = (
    "The projection of the composition action from regime 'couple' to destination "
    "'f_alone' is declared `Phased` and reads the transfer 'division'"
)


@pytest.mark.parametrize("differs", [False, True])
@pytest.mark.parametrize(
    ("form", "message"),
    [
        ("payment", f"{_PAYMENT}function 'penalty'{_DIFFERS}"),
        ("helper", f"{_PAYMENT}function 'penalty'{_DIFFERS}"),
        ("carried", f"{_PAYMENT}state 'kept'{_DIFFERS}"),
        ("law", f"{_PAYMENT}function 'next_gain'{_DIFFERS}"),
        ("available", f"{_AVAILABLE}function 'penalty'{_DIFFERS}"),
        ("default", f"{_DEFAULT}function 'penalty'{_DIFFERS}"),
        ("projection", _PROJECTION),
    ],
    ids=["payment", "helper", "carried", "law", "available", "default", "projection"],
)
def test_composition_action_payments_and_conditions_are_common_to_solve_and_simulation(
    *, form: str, message: str, differs: bool
) -> None:
    """A payment, default or availability reading a phase-dependent name is refused.

    So is a `Phased` projection reading the transfer. The same declaration builds
    when the name it reads is the same in both phases, beside a member utility that
    differs, and so does a `Phased` projection reading no transfer.
    """
    from lcm import (  # noqa: PLC0415
        AgeGrid,
        CollectiveUtility,
        LinSpacedGrid,
        MutualConsent,
        Phased,
        Regime,
        Transfer,
        fixed_transition,
    )
    from lcm.regime import (  # noqa: PLC0415
        FunctionEntry,
        StateEntry,
        StateTransitionEntry,
    )
    from lcm.typing import ScalarInt  # noqa: PLC0415

    @lcm.categorical(ordered=False)
    class RegimeId:
        couple: ScalarInt
        couple_done: ScalarInt
        f_alone: ScalarInt
        m_alone: ScalarInt

    grid = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)
    penalty = Phased(solve=_penalty, simulate=_other_penalty) if differs else _penalty
    payment = {
        "payment": _pays_penalty,
        "helper": _pays_charge,
        "carried": _pays_kept,
        "law": _pays_next_gain,
        "default": _pays_division,
        "projection": _pays_division,
    }.get(form)
    projection: Mapping[StateName, UserFunction] | Phased = {}
    if form == "projection":
        projection = Phased(
            solve={"gain": _lands_division if differs else _keep_gain},
            simulate={"gain": _lands_other_division if differs else _halve_gain},
        )
    states: dict[str, StateEntry] = {"gain": grid}
    laws: dict[str, StateTransitionEntry] = {"gain": fixed_transition("gain")}
    if form == "carried":
        states["kept"] = Phased(solve=_keep_gain, simulate=grid) if differs else grid
        laws["kept"] = fixed_transition("kept")
    if form == "law" and differs:
        laws["gain"] = Phased(solve=_keep_gain, simulate=_halve_gain)
    helpers: dict[str, FunctionEntry] = {}
    if form not in {"carried", "law", "projection"}:
        helpers["penalty"] = penalty
    if form == "helper":
        helpers["charge"] = _charge

    def build() -> Model:
        return Model(
            regimes={
                "couple": Regime(
                    states=states,
                    state_transitions=laws,
                    functions={
                        "utility": CollectiveUtility(
                            utilities={
                                "f": Phased(
                                    solve=_gain_utility, simulate=_other_gain_utility
                                ),
                                "m": _gain_utility,
                            }
                        ),
                        **helpers,
                    },
                    composition_action=CompositionAction(
                        destinations={
                            "f": Destination(
                                regime="f_alone",
                                projection=projection,
                                utility_adjustment=payment,
                            ),
                            "m": Destination(regime="m_alone", projection={}),
                        },
                        rule=MutualConsent(
                            transfer=Transfer(
                                name="division",
                                candidates=grid,
                                default=_pays_penalty
                                if form == "default"
                                else _keep_gain,
                            )
                        )
                        if form in {"default", "projection"}
                        else PCOnActions(),
                        available=_open_at_penalty if form == "available" else None,
                    ),
                ),
                "couple_done": Regime(
                    states={"gain": grid},
                    functions={
                        "utility": CollectiveUtility(
                            utilities={"f": _gain_utility, "m": _gain_utility}
                        )
                    },
                ),
                "f_alone": Regime(
                    states={"gain": grid}, functions={"utility": _gain_utility}
                )
                if form == "projection"
                else Regime(functions={"utility": _zero_utility}),
                "m_alone": Regime(functions={"utility": _zero_utility}),
            },
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            regime_id_class=RegimeId,
            initial_nodes={0: "couple"},
            edges={"couple": {"couple_done": 0}},
        )

    if not differs:
        build()
        return
    with pytest.raises(RegimeInitializationError, match=message):
        build()
