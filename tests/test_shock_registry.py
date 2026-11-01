"""Model shock declarations retain base identity without changing state layouts."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import cloudpickle
import jax.numpy as jnp
import pytest
from dags import with_signature

from _lcm.solution.fingerprint import _project_user_regime_declaration
from lcm import (
    CollectiveUtility,
    InitialNodes,
    JointNormalIIDProcess,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    NormalMixtureIIDProcess,
    Periods,
    Phased,
    Regime,
    Transition,
    categorical,
)
from lcm.exceptions import (
    InvalidParamsError,
    RegimeInitializationError,
    ShockDeclarationError,
    UnsupportedOperationError,
)
from lcm.shocks import GaussHermite, Normal, Product, ShockGroup
from lcm.typing import ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class Status:
    alive: ScalarInt


def _utility() -> float:
    return 1.0


def _model(*, shocks: Mapping[str, ShockGroup]) -> Model:
    return Model(
        n_periods=1,
        regimes={"alive": Regime(functions={"utility": _utility})},
        regime_id_class=Status,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks=shocks,
    )


def test_model_shocks_snapshot_mapping() -> None:
    """Registry declarations cannot be changed through the caller's mapping."""
    group = ShockGroup(
        shocks={"eps": Normal(mu=0.0, sigma=1.0)},
        nodes=Product(eps=GaussHermite(3)),
    )
    declarations = {"earnings": group}
    model = _model(shocks=declarations)
    declarations.clear()
    assert model.shocks == MappingProxyType({"earnings": group})


@pytest.mark.parametrize("group_name", ["alive", "edges", "a__b", "group_", "_group"])
def test_group_name_conflicts_are_refused(group_name: str) -> None:
    """Group namespaces cannot alias ordinary model vocabulary."""

    with pytest.raises(ShockDeclarationError, match="group name"):
        _model(shocks={group_name: _group()})


def _group(*, count: int = 3) -> ShockGroup:
    return ShockGroup(
        shocks={"eps": Normal(mu=0.0, sigma=None)},
        nodes=Product(eps=GaussHermite(count)),
    )


def _consumer(*, regime: Regime, group: ShockGroup) -> Model:
    return Model(
        n_periods=1,
        regimes={"alive": regime},
        regime_id_class=Status,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={"earnings": group},
    )


def test_base_identity_is_independent_of_node_rule() -> None:
    """Changing quadrature cardinality leaves the named innovation unchanged."""
    left = _model(shocks={"earnings": _group(count=3)})
    right = _model(shocks={"earnings": _group(count=5)})
    assert left._shock_registry.innovations == right._shock_registry.innovations


def test_node_rule_changes_durable_model_identity() -> None:
    """The numerical rule is retained separately from innovation identity."""
    left = _model(shocks={"earnings": _group(count=3)})
    right = _model(shocks={"earnings": _group(count=5)})
    assert left._model_structure_fingerprint != right._model_structure_fingerprint


def test_physical_name_cannot_have_competing_group_laws() -> None:
    """Competing groups cannot redefine a named physical innovation."""

    with pytest.raises(ShockDeclarationError, match="conflicting declarations"):
        _model(shocks={"first": _group(), "second": _group()})


@pytest.mark.parametrize("references", [("missing",), ("earnings", "earnings")])
def test_invalid_group_references_are_refused(references: tuple[str, ...]) -> None:
    """References must name existing groups without duplicates."""

    with pytest.raises((RegimeInitializationError, ShockDeclarationError)):
        _consumer(
            regime=Regime(functions={"utility": _utility}, shocks=references),
            group=_group(),
        )


def test_regime_replacement_retains_group_references() -> None:
    regime = Regime(functions={"utility": _utility}, shocks=("earnings",))
    assert regime.replace(description="report").shocks == ("earnings",)


def test_group_template_is_distinct_from_ordinary_model_parameters() -> None:
    """A runtime law slot stays local instead of becoming a utility parameter."""
    model = _consumer(
        regime=Regime(functions={"utility": _utility}, shocks=("earnings",)),
        group=_group(),
    )
    assert (
        dict(
            model._shock_registry.bindings["alive", "solve", "earnings"].params_template
        ),
        model.get_params_template(),
    ) == (
        {"eps__sigma": "float"},
        {"alive": {"utility": {}}, "earnings": {"eps__sigma": "float"}},
    )


def test_current_iid_execution_requires_declared_group_parameters() -> None:

    model = _consumer(
        regime=Regime(functions={"utility": _utility}, shocks=("earnings",)),
        group=_group(),
    )
    with pytest.raises(InvalidParamsError, match=r"missing.*eps__sigma"):
        model.solve(params={}, log_level="off")


def _shock_utility(eps: float) -> float:
    return eps


def test_unlowered_physical_draw_is_not_an_ordinary_parameter() -> None:

    with pytest.raises(UnsupportedOperationError, match="integrated shock lowering"):
        _consumer(
            regime=Regime(functions={"utility": _shock_utility}),
            group=_group(),
        )


def _state_scale(*, wealth: float, multiplier: float) -> FloatND:
    return jnp.asarray(wealth * multiplier)


def test_registry_keeps_law_state_domains_before_broadcast_pruning() -> None:
    """A law-only state dependency remains available to native shock lowering."""

    grid = LinSpacedGrid(start=1.0, stop=2.0, n_points=3)
    group = ShockGroup(
        shocks={"eps": Normal(mu=0.0, sigma=_state_scale)},
        nodes=Product(eps=GaussHermite(3)),
    )
    model = Model(
        n_periods=1,
        regime_id_class=Status,
        regimes={
            "alive": Regime(functions={"utility": _utility}, shocks=("earnings",))
        },
        states={"wealth": grid},
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={"earnings": group},
    )
    binding = model._shock_registry.bindings["alive", "solve", "earnings"]
    assert (binding.state_inputs, binding.state_domains, binding.params_template) == (
        frozenset({"wealth"}),
        {"wealth": grid},
        {"multiplier": "float"},
    )


@pytest.mark.parametrize(
    "argument", ["choice", "eps", "eps_code", "other", "Q", "next_wealth"]
)
def test_choice_and_realized_draw_dependencies_are_refused(argument: str) -> None:
    """Law parameters cannot read any action or group innovation."""

    @with_signature(kwargs=[argument])
    def law(**kwargs: float) -> FloatND:
        return jnp.asarray(next(iter(kwargs.values())))

    group = ShockGroup(
        shocks={"eps": Normal(mu=0.0, sigma=law)},
        nodes=Product(eps=GaussHermite(3)),
    )
    other = ShockGroup(
        shocks={"other": Normal(mu=0.0, sigma=1.0)},
        nodes=Product(other=GaussHermite(3)),
    )
    with pytest.raises(ShockDeclarationError, match="cannot read"):
        Model(
            n_periods=1,
            regimes={
                "alive": Regime(
                    functions={"utility": _utility},
                    actions={"choice": LinSpacedGrid(start=0, stop=1, n_points=2)},
                    shocks=("earnings",),
                )
            },
            regime_id_class=Status,
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={},
            shocks={"earnings": group, "secondary": other},
        )


def test_legacy_phase_specific_laws_remain_distinct_records() -> None:

    solve = NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)
    simulate = NormalIIDProcess(n_points=3, gauss_hermite=True, mu=2.0, sigma=1.0)
    model = Model(
        n_periods=1,
        regime_id_class=Status,
        regimes={
            "alive": Regime(
                states={"eps": Phased(solve=solve, simulate=simulate)},
                functions={"utility": _shock_utility},
            )
        },
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
    )
    assert tuple(
        record.declaration for record in model._shock_registry.legacy.values()
    ) == (solve, simulate)


def test_legacy_joint_declaration_stays_joint_in_registry() -> None:

    joint = JointNormalIIDProcess(
        states=("female", "male"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=((1.0, 0.5), (0.5, 1.0)),
    )
    model = Model(
        n_periods=1,
        regime_id_class=Status,
        regimes={
            "alive": Regime(
                states={"pair": joint}, functions={"utility": _joint_utility}
            )
        },
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
    )
    record = model._shock_registry.legacy["alive", "solve", "pair"]
    assert (record.declaration, record.physical_names) == (joint, ("female", "male"))


def _joint_utility(*, female: ContinuousState, male: ContinuousState) -> FloatND:
    return female + male


def test_registry_survives_model_roundtrip() -> None:
    model = _consumer(
        regime=Regime(functions={"utility": _utility}, shocks=("earnings",)),
        group=_group(),
    )
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    assert (
        restored.shocks["earnings"].fingerprint,
        restored.user_regimes["alive"].shocks,
    ) == (model.shocks["earnings"].fingerprint, ("earnings",))


def test_empty_shocks_preserves_legacy_declaration_projection() -> None:
    model = _model(shocks={})
    declaration = _project_user_regime_declaration(
        model.user_regimes["alive"], law=model.graph.laws["alive"]
    )
    assert "shocks" not in declaration["fields"]


def test_state_law_draw_is_not_an_ordinary_parameter() -> None:
    with pytest.raises(UnsupportedOperationError, match="integrated shock lowering"):
        Model(
            n_periods=2,
            regimes={
                "alive": Regime(
                    states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=3)},
                    state_transitions={"wealth": _shock_utility},
                    functions={"utility": _utility},
                )
            },
            regime_id_class=Status,
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={"alive": {"alive": Periods(values=(0,))}},
            shocks={"earnings": _group()},
        )


def test_transitive_action_dependency_is_refused() -> None:
    def current_scale(choice: FloatND) -> FloatND:
        return choice + 1.0

    def sigma(current_scale: FloatND) -> FloatND:
        return current_scale

    with pytest.raises(ShockDeclarationError, match="choice"):
        _consumer(
            regime=Regime(
                functions={"utility": _utility, "current_scale": current_scale},
                actions={"choice": LinSpacedGrid(start=0, stop=1, n_points=2)},
                shocks=("earnings",),
            ),
            group=ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=sigma)},
                nodes=Product(eps=GaussHermite(3)),
            ),
        )


def test_legacy_mixture_retains_its_physical_grid_rule() -> None:
    process = NormalMixtureIIDProcess(
        n_points=3, n_std=3.0, p1=0.4, mu1=-1, sigma1=0.5, mu2=2, sigma2=1
    )
    model = Model(
        n_periods=1,
        regimes={
            "alive": Regime(
                states={"eps": process}, functions={"utility": _shock_utility}
            )
        },
        regime_id_class=Status,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
    )
    assert model._shock_registry.legacy["alive", "solve", "eps"].declaration is process


@dataclass(frozen=True)
class _CallableUtility:
    offset: float = 0.0

    def __call__(self, eps: float) -> float:
        return eps + self.offset


def test_callable_dataclass_draw_is_not_a_free_parameter() -> None:
    with pytest.raises(UnsupportedOperationError, match="integrated shock lowering"):
        _consumer(
            regime=Regime(functions={"utility": _CallableUtility()}), group=_group()
        )


def test_base_address_identity_survives_regrouping() -> None:
    left = _model(shocks={"earnings": _group()})
    right = _model(shocks={"income": _group()})
    assert (
        left._shock_registry.innovations["eps"].identity
        == right._shock_registry.innovations["eps"].identity
    )


def test_group_namespace_can_share_a_real_state_name() -> None:
    model = Model(
        n_periods=1,
        regimes={
            "alive": Regime(
                functions={"utility": _wealth_utility},
                states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=3)},
            )
        },
        regime_id_class=Status,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={},
        shocks={"wealth": _group()},
    )
    assert model.shocks["wealth"].has_same_laws_as(_group())


def _wealth_utility(wealth: ContinuousState) -> ContinuousState:
    return wealth


@pytest.mark.parametrize("group_name", ["utility", "period", "next_wage"])
def test_group_namespace_is_separate_from_function_and_time_inputs(
    group_name: str,
) -> None:
    assert (
        _model(shocks={group_name: _group()})
        .shocks[group_name]
        .has_same_laws_as(_group())
    )


def test_physical_name_cannot_shadow_generated_collective_function() -> None:
    group = ShockGroup(
        shocks={"utility_f": Normal(mu=0.0, sigma=1.0)},
        nodes=Product(utility_f=GaussHermite(3)),
    )
    with pytest.raises(ShockDeclarationError, match="conflicting declarations"):
        _consumer(
            regime=Regime(
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _utility, "m": _utility}
                    )
                }
            ),
            group=group,
        )


def _shock_regime_transition(eps: FloatND) -> ScalarInt:
    return jnp.asarray(0.0 * eps, dtype=jnp.int32)


def test_regime_law_draw_is_not_an_ordinary_parameter() -> None:
    """A source's ordinary transition cannot read an unbound physical draw."""
    with pytest.raises(UnsupportedOperationError, match="integrated shock lowering"):
        Model(
            n_periods=2,
            regimes={"alive": Regime(functions={"utility": _utility})},
            regime_id_class=Status,
            initial_nodes=InitialNodes(by_period={0: "alive"}),
            edges={
                "alive": Transition(
                    targets={"alive": Periods(values=(0,))},
                    law=_shock_regime_transition,
                )
            },
            shocks={"earnings": _group()},
        )
