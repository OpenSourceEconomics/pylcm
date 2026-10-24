"""Public exits declare current-period choices and immutable mapped targets."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass
from typing import ClassVar

import pytest

import lcm
from _lcm.grids import Grid
from lcm import Consent, Counterparty, Exit, ExitTarget, Model, RuleResult
from lcm.exceptions import RegimeInitializationError, UnsupportedOperationError
from lcm.exits import ExitRule
from lcm.typing import BoolND, FloatND, IntND, UserParams


def _half(wealth: FloatND) -> FloatND:
    return wealth / 2


def test_exit_targets_snapshot_the_authored_projection() -> None:
    """An exit keeps its declared projection when caller mappings change."""
    projection = {"wealth": _half}
    target = ExitTarget(regime="single", projection=projection)
    targets = {"self": target}
    declaration = Exit(targets=targets, rule="unilateral")
    projection.clear()
    targets.clear()
    actual_projection = declaration.targets["self"].projection
    assert isinstance(actual_projection, Mapping)
    assert actual_projection["wealth"] is _half
    with pytest.raises(FrozenInstanceError):
        target.regime = "other"  # ty: ignore[invalid-assignment] - mutation refusal


def _utility() -> float:
    return 1.0


def test_regime_replacement_preserves_its_named_exits() -> None:
    """Replacing an economic helper retains its authored exit declaration."""
    declaration = Exit(
        targets={"self": ExitTarget(regime="single", projection={})},
        rule="unilateral",
    )
    regime = lcm.Regime(functions={"utility": _utility}, exits={"leave": declaration})
    assert regime.replace(description="another label").exits["leave"] is declaration


def test_phased_projection_snapshots_both_settlements() -> None:
    """Caller edits cannot alter either phase's destination settlement."""
    solve = {"wealth": _half}
    simulate = {"wealth": _half}
    target = ExitTarget(
        regime="single", projection=lcm.Phased(solve=solve, simulate=simulate)
    )
    solve.clear()
    simulate.clear()
    assert isinstance(target.projection, lcm.Phased)
    assert target.projection.solve["wealth"] is _half
    assert target.projection.simulate["wealth"] is _half


def test_custom_exit_rule_requires_outside_value_declaration() -> None:
    """A callable cannot hide whether it changes its source participation DAG."""
    with pytest.raises(RegimeInitializationError, match="reads_outside_values"):
        Exit(
            targets={"self": ExitTarget(regime="single", projection={})},
            rule=lambda: RuleResult(action=0),
        )


@pytest.mark.parametrize("kind", ["empty", "observes", "target_name", "projection"])
def test_malformed_exit_declarations_fail_at_construction(kind: str) -> None:
    """Invalid names, duplicate observations and noncallable settlements fail early."""
    target = ExitTarget
    declaration = Exit
    with pytest.raises(RegimeInitializationError):  # noqa: PT012 - malformed constructor variants
        if kind == "empty":
            declaration(targets={}, rule="unilateral")
        elif kind == "observes":
            declaration(
                targets={"self": target(regime="single", projection={})},
                rule="unilateral",
                observes=("offer", "offer"),
            )
        elif kind == "target_name":
            target(regime="bad__regime", projection={})
        else:
            target(regime="single", projection={"wealth": 1.0})  # ty: ignore[invalid-argument-type] - invalid input


def test_collective_exit_names_every_source_member() -> None:
    """A couple cannot silently omit a member's destination benchmark."""
    with pytest.raises(RegimeInitializationError, match="targets"):
        lcm.Regime(
            functions={
                "utility": lcm.CollectiveUtility(
                    utilities={"f": _utility, "m": _utility}
                )
            },
            exits={
                "divorce": Exit(
                    targets={"f": ExitTarget(regime="single", projection={})},
                    rule="unilateral",
                )
            },
        )


def test_multiple_builtin_exits_need_an_explicit_custom_rule() -> None:
    """An implicit built-in rule cannot choose which outside benchmark to apply."""
    declaration = Exit(
        targets={"self": ExitTarget(regime="single", projection={})},
        rule="unilateral",
    )
    with pytest.raises(RegimeInitializationError, match="built-in"):
        lcm.Regime(
            functions={"utility": _utility},
            exits={"one": declaration, "two": declaration},
        )


def _settlement(*, wealth: FloatND, share: FloatND) -> FloatND:
    return wealth * share


def _realized_settlement(*, wealth: FloatND, realized_share: FloatND) -> FloatND:
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
    rule: ExitRule = "unilateral",
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
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target",
                                projection=Phased(
                                    solve={"wealth": _settlement},
                                    simulate={"wealth": _realized_settlement},
                                ),
                                utility_adjustment=_cost,
                            )
                        },
                        rule=rule,
                        available=_available,
                    )
                },
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


def test_exit_only_state_and_phase_parameters_remain_declared() -> None:
    """Both settlement phases and their free inputs remain in the template."""
    model = _declared_model()
    assert model.get_params_template()["source"]["exits"] == {
        "leave": {
            "available": {"threshold": "FloatND"},
            "targets": {
                "self": {
                    "projection": {
                        "wealth": {"share": "FloatND", "realized_share": "FloatND"}
                    },
                    "utility_adjustment": {"cost": "FloatND"},
                }
            },
        }
    }
    assert "wealth" in model.user_regimes["source"].states


def test_unlowered_exit_model_cannot_silently_run_a_stay_only_problem() -> None:
    """An unlowered declaration cannot silently run a different economic problem."""
    with pytest.raises(UnsupportedOperationError, match=r"exit.*lowering"):
        _declared_model().solve(
            params={"share": 0.5, "realized_share": 0.5, "threshold": 0.0, "cost": 0.0},
            log_level="off",
        )


def _switch_wealth(wealth: FloatND) -> FloatND:
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp.where(wealth > 15, 10.0, 20.0)


def test_exit_settlement_reads_physical_grid_values_and_returns_target_codes() -> None:
    """A settlement decodes source values and encodes destination codes."""
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
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target", projection={"wealth": _switch_wealth}
                            )
                        },
                        rule="unilateral",
                    )
                },
            ),
            "target": Regime(functions={"utility": _wealth_utility}),
        },
        states={"wealth": DiscreteGrid(values=jnp.asarray([10.0, 20.0]))},
        n_periods=1,
        regime_id_class=RegimeId,
        initial_nodes=InitialNodes(by_period={0: ("source", "target")}),
        edges={},
    )
    projection = model.user_regimes["source"].exits["leave"].targets["self"].projection
    assert isinstance(projection, Mapping)
    assert projection["wealth"](wealth=jnp.int32(1)) == 0


@pytest.mark.parametrize("kind", ["unknown", "missing_role", "bad_role", "projection"])
def test_exit_destination_must_be_a_complete_declared_policy_context(kind: str) -> None:
    """Destination names, collective roles and projected states are model-owned."""
    from lcm import InitialNodes, LinSpacedGrid, Model, Regime  # noqa: PLC0415
    from lcm.typing import ScalarInt  # noqa: PLC0415

    @lcm.categorical(ordered=False)
    class RegimeId:
        source: ScalarInt
        target: ScalarInt

    target = ExitTarget(
        regime="missing" if kind == "unknown" else "target",
        projection={} if kind == "projection" else {"wealth": _half},
        role="other" if kind == "bad_role" else None,
    )
    with pytest.raises(RegimeInitializationError, match=r"[Ee]xit"):
        Model(
            regimes={
                "source": Regime(
                    functions={"utility": _wealth_utility},
                    exits={"leave": Exit(targets={"self": target}, rule="unilateral")},
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


def test_phase_normalization_binds_the_corresponding_exit_settlement() -> None:
    """Canonical solve/simulate specs retain their own explicit settlement helper."""
    from _lcm.regime_building.phases import normalize_regime_phases  # noqa: PLC0415

    model = _declared_model()
    phases = normalize_regime_phases(
        model.user_regimes["source"], law=model.graph.laws["source"]
    )
    solve = phases.solution.exits["leave"].targets["self"].projection
    simulate = phases.simulation.exits["leave"].targets["self"].projection
    assert isinstance(solve, Mapping)
    assert isinstance(simulate, Mapping)
    assert solve["wealth"] is _settlement
    assert simulate["wealth"] is _realized_settlement


def test_fixed_exit_parameter_is_removed_from_the_runtime_template() -> None:
    """A fixed adjustment uses its own exact declaration slot."""
    model = _declared_model(
        fixed_params={
            "source": {
                "exits": {
                    "leave": {
                        "targets": {"self": {"utility_adjustment": {"cost": 0.0}}}
                    }
                }
            }
        }
    )
    template = model.get_params_template()["source"]["exits"]
    assert template == {
        "leave": {
            "available": {"threshold": "FloatND"},
            "targets": {
                "self": {
                    "projection": {
                        "wealth": {"share": "FloatND", "realized_share": "FloatND"}
                    },
                }
            },
        }
    }


def test_rule_result_transports_a_dynamic_id_through_jit() -> None:
    """A traceable custom rule returns its stable action ID as an ordinary pytree."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    result = jax.jit(lambda action: RuleResult(action=action))(jnp.int32(7))
    assert int(result.action) == 7


def test_consent_snapshots_its_counterparty_projection() -> None:
    """Formation retains the declared counterparty Raw state mapping."""
    projection = {"wealth": _half}
    consent = Consent(
        counterparty=Counterparty(regime="single", role="m", projection=projection)
    )
    projection.clear()
    assert isinstance(consent.counterparty.projection, Mapping)
    assert consent.counterparty.projection["wealth"] is _half


def test_exit_name_is_a_single_parameter_path_segment() -> None:
    """Exit keys cannot be confused with nested parameter namespaces."""
    declaration = Exit(
        targets={"self": ExitTarget(regime="single", projection={})}, rule="unilateral"
    )
    with pytest.raises(RegimeInitializationError, match=r"[Ee]xit"):
        lcm.Regime(functions={"utility": _utility}, exits={"bad__name": declaration})


@dataclass(frozen=True, slots=True)
class _CustomRule:
    reads_outside_values: ClassVar[bool] = True

    def __call__(
        self,
        *,
        wealth: FloatND,
        Q_self: FloatND,  # noqa: ARG002 - exercise internal operand binding
        exit_payoffs: Mapping[str, Mapping[str, FloatND]],  # noqa: ARG002
        availability: Mapping[str, BoolND],  # noqa: ARG002
        alternative_ids: IntND,
        pareto_weights: Mapping[str, FloatND],  # noqa: ARG002
        cutoff: FloatND,
    ) -> RuleResult:
        import jax.numpy as jnp  # noqa: PLC0415

        return RuleResult(
            action=jnp.where(wealth > cutoff, alternative_ids[-1], alternative_ids[0])
        )


def test_custom_rule_channels_are_not_runtime_parameters() -> None:
    """Only authored scalar inputs enter a custom rule's declaration namespace."""
    template = _declared_model(rule=_CustomRule()).get_params_template()["source"][
        "exits"
    ]
    assert isinstance(template, dict)
    assert isinstance(template["leave"], dict)
    assert template["leave"]["rule"] == {"cutoff": "FloatND"}


def test_custom_rule_metadata_survives_categorical_adaptation() -> None:
    """A class-owned outside-value declaration remains valid after code adaptation."""
    import jax.numpy as jnp  # noqa: PLC0415

    model = _declared_model(
        rule=_CustomRule(), grid=lcm.DiscreteGrid(values=jnp.asarray([10.0, 20.0]))
    )
    rule = model.user_regimes["source"].exits["leave"].rule
    assert callable(rule)
    result = rule(
        wealth=jnp.int32(1),
        Q_self=jnp.zeros(1),
        exit_payoffs={},
        availability={},
        alternative_ids=jnp.asarray([0, 7]),
        pareto_weights={"self": jnp.asarray(1.0)},
        cutoff=jnp.asarray(15.0),
    )
    assert isinstance(result, RuleResult)
    assert int(result.action) == 7


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
    with pytest.raises(RegimeInitializationError, match=r"[Cc]ounterparty"):
        Model(
            regimes={
                "source": lcm.Regime(
                    functions={"utility": _wealth_utility},
                    exits={
                        "marry": Exit(
                            targets={
                                "self": ExitTarget(
                                    regime="couple",
                                    role="f",
                                    projection={"wealth": _half},
                                )
                            },
                            rule=Consent(counterparty=counterparty),
                        )
                    },
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


def test_fixed_custom_rule_input_uses_its_declared_slot() -> None:
    """A fixed rule cutoff binds independently of internal value channels."""
    model = _declared_model(
        rule=_CustomRule(),
        fixed_params={"source": {"exits": {"leave": {"rule": {"cutoff": 1.0}}}}},
    )
    exits = model.get_params_template()["source"]["exits"]
    assert isinstance(exits, dict)
    assert isinstance(exits["leave"], dict)
    assert "rule" not in exits["leave"]


def test_exit_settlement_phase_variation_is_reported() -> None:
    """Different solve and realised settlements prohibit invariant replay."""
    from _lcm.regime_building.phases import phase_variation_paths  # noqa: PLC0415

    model = _declared_model()
    paths = phase_variation_paths(
        user_regime=model.user_regimes["source"], law=model.graph.laws["source"]
    )
    assert "exits['leave'].targets['self'].projection" in paths


def _exit_signal(*, eps: FloatND, signal_scale: FloatND) -> FloatND:
    return eps * signal_scale


def _signal_available(*, exit_signal: FloatND, threshold: FloatND) -> BoolND:
    return exit_signal > threshold


def _signal_adjustment(*, eps: FloatND, exit_signal: FloatND, fee: FloatND) -> FloatND:
    return eps + exit_signal - fee


def test_exit_observed_shock_and_helper_reads_are_not_parameters() -> None:
    """Exit callback ancestors retain physical shock operands and helper slots."""
    from lcm.shocks import GaussHermite, Normal, Product, ShockGroup  # noqa: PLC0415
    from lcm.typing import ScalarInt  # noqa: PLC0415

    @lcm.categorical(ordered=False)
    class RegimeId:
        source: ScalarInt
        target: ScalarInt

    model = Model(
        regimes={
            "source": lcm.Regime(
                functions={"utility": _utility, "exit_signal": _exit_signal},
                shocks=("innovation",),
                exits={
                    "leave": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="target",
                                projection={},
                                utility_adjustment=_signal_adjustment,
                            )
                        },
                        rule="unilateral",
                        available=_signal_available,
                    )
                },
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
    assert source["exit_signal"] == {"signal_scale": "FloatND"}
    assert source["exits"] == {
        "leave": {
            "available": {"threshold": "FloatND"},
            "targets": {"self": {"utility_adjustment": {"fee": "FloatND"}}},
        }
    }


def test_shared_projection_object_remains_phase_invariant() -> None:
    """Snapshotting one explicitly shared settlement preserves replay identity."""
    projection = {"wealth": _half}
    target = ExitTarget(
        regime="single", projection=lcm.Phased(solve=projection, simulate=projection)
    )
    assert isinstance(target.projection, lcm.Phased)
    assert target.projection.solve is target.projection.simulate


@pytest.mark.parametrize("change", ["settlement", "information", "rule"])
def test_exit_declarations_enter_durable_model_identity(change: str) -> None:
    """Stored solutions identify exit settlements, information and decision rules."""
    from dataclasses import replace  # noqa: PLC0415

    from _lcm.regime_law import bind_regime_law  # noqa: PLC0415
    from _lcm.solution.fingerprint import (  # noqa: PLC0415
        _project_user_regime_declaration,
        _semantic_fingerprint,
    )

    base = Exit(
        targets={"self": ExitTarget(regime="single", projection={"wealth": _half})},
        rule="unilateral",
    )
    if change == "settlement":
        changed = replace(
            base,
            targets={
                "self": ExitTarget(regime="single", projection={"wealth": _settlement})
            },
        )
    elif change == "information":
        changed = replace(base, observes=("offer",))
    else:
        changed = replace(base, rule="mutual")
    regime = lcm.Regime(functions={"utility": _utility}, exits={"leave": base})
    law = bind_regime_law(None)
    assert _semantic_fingerprint(
        _project_user_regime_declaration(regime, law=law)
    ) != _semantic_fingerprint(
        _project_user_regime_declaration(
            regime.replace(exits={"leave": changed}), law=law
        )
    )


def _only_period_one(period: lcm.typing.Period) -> BoolND:
    return period == 1


def _never_available() -> bool:
    return False


@pytest.mark.parametrize("available", [_only_period_one, _never_available])
def test_exit_time_only_availability_can_exclude_inactive_contexts(
    available: lcm.typing.UserFunction,
) -> None:
    """An unavailable exit does not require an inactive destination policy."""
    from lcm.typing import ScalarInt  # noqa: PLC0415

    @lcm.categorical(ordered=False)
    class RegimeId:
        source: ScalarInt
        later: ScalarInt

    model = Model(
        regimes={
            "source": lcm.Regime(
                functions={"utility": _utility},
                exits={
                    "leave": Exit(
                        targets={"self": ExitTarget(regime="later", projection={})},
                        rule="unilateral",
                        available=available,
                    )
                },
            ),
            "later": lcm.Regime(functions={"utility": _utility}),
        },
        n_periods=3,
        regime_id_class=RegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "source", 1: "later"}),
        edges={"later": {"source": lcm.Periods(values=(1,))}},
    )
    assert model.get_params_template()["source"]["exits"] == {
        "leave": {"available": {}}
    }


def test_custom_rule_class_metadata_changes_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit outside-value declaration binds each model's decision DAG."""
    first = _declared_model(rule=_CustomRule())
    monkeypatch.setattr(_CustomRule, "reads_outside_values", False)
    second = _declared_model(rule=_CustomRule())
    assert first._model_structure_fingerprint != second._model_structure_fingerprint
