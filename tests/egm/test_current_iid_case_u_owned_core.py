"""Genuine Model-generation source frontier for the private Case-U core.

The authentic constructor currently installs the resources-only subclass.
This exercises actual ownership/materialization and explicit missing installation,
not acceptance of general Case-U classification, publication or forward replay.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Never

import jax
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluation
from _lcm.execution.core_program import CoreBuildContext
from _lcm.execution.workspace_planning import WorkspaceBudgetError
from _lcm.solution.current_iid_case_u_build import (
    CASE_U_ECONOMIC_CORE,
    build_owned_case_u_program,
    plan_private_case_u_workspace,
)
from _lcm.typing import PytreeValue
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    InitialNodes,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
)
from lcm.exceptions import UnsupportedOperationError
from lcm.model import _ModelConstructorInputs
from lcm.shocks import Normal, ShockGroup, UserNodes
from lcm.solvers import DCEGM, ExactEnvelope
from lcm.typing import UserParams
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_parent_demands import (
    _current_resources,
    _identity,
    _inverse,
    _landing,
    _RegimeId,
    _resources,
    _savings,
    _terminal,
    _utility,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _actual_declarations() -> _ModelConstructorInputs:
    grid = IrregSpacedGrid(points=np.array([1.0, 4.0]))
    child_grid = IrregSpacedGrid(points=np.array([1.0, 12.5]))
    child_savings = IrregSpacedGrid(
        points=np.array(
            [
                0.25,
                0.5,
                0.75,
                1.0,
                1.25,
                1.375,
                1.5,
                1.75,
                1.875,
                2.0,
                2.25,
                2.5,
                4.25,
                4.75,
                6.25,
                6.75,
                12.5,
            ]
        )
    )
    liquid = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    functions = {
        "utility": _utility,
        "resources": _resources,
        "savings": _savings,
        "inverse_marginal_utility": _inverse,
    }
    parent = ConsumptionSavingsRegime(
        states={"wealth": grid},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.25, 2.0]))},
        functions=functions,
        state_transitions={"wealth": {"child": _landing}},
        liquid=liquid,
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.5, 0.75, 1.0, 4.0])),
            envelope=ExactEnvelope(),
        ),
    )
    child = ConsumptionSavingsRegime(
        states={"wealth": child_grid},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.25, 2.0]))},
        functions={**functions, "resources": _current_resources},
        state_transitions={"wealth": {"done": _identity}},
        liquid=liquid,
        solver=DCEGM(savings_grid=child_savings, envelope=ExactEnvelope()),
        shocks=("income",),
    )
    return _ModelConstructorInputs(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={"wealth": child_savings}, functions={"utility": _terminal}
            ),
        },
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={0: "parent"}),
        edges={"parent": {"child": (0,)}, "child": {"done": (1,)}},
        regime_id_class=_RegimeId,
        shocks={
            "income": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.array([[0.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )


def _params() -> UserParams:
    return {
        "discount_factor": 1.0,
        "parent": {"next_wealth": {"slope": 2.0}},
        "child": {"next_wealth": {"slope": 1.0}},
    }


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_owned_case_u_materializer_uses_actual_model_preparation_and_leaf_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An authentic intermediate preserves authority and refuses invented admission."""
    model = Model._from_call_owned_params(
        declarations=_actual_declarations(), params=_params()
    )
    flat_params, generation = model._prepare_call(params=_params())
    preparation = generation.solver_preparation(regime_name="child", period=1)
    preparation.validate_owner()
    spec = generation.execution._regimes["done"].solution.continuation_spec
    assert spec is not None
    child_template = spec.template_at(period=2)
    assert isinstance(child_template, EGMCarry)
    context = CoreBuildContext(
        state_action_space=preparation.context.state_action_space,
        next_regime_to_V_arr=MappingProxyType({}),
        next_regime_to_continuation=MappingProxyType({"done": child_template}),
        flat_params=flat_params,
        period=1,
        ages=preparation.context.ages,
    )
    owned = build_owned_case_u_program(
        generation=generation, regime_name="child", period=1, context=context
    )
    assert owned.preparation.step is preparation.step
    assert owned.preparation.context is preparation.context
    assert isinstance(owned.template.output, CurrentResourcesEvaluation)
    assert context.state_action_space is not None
    assert (
        owned.template.output.value.shape
        == context.state_action_space.states["wealth"].shape
    )
    assert owned.program.requirements.axes == ()
    assert owned.program.requirements.host_buffers == {}
    assert owned.program.donation_candidates == ()
    materialized = owned.materialize(context=context)
    assert materialized.name == CASE_U_ECONOMIC_CORE
    children = materialized.arguments["next_regime_to_continuation"]
    assert isinstance(children, Mapping)
    assert children["done"] is child_template
    assert {
        read.target.leaf_path for read in owned.program.requirements.value_reads
    } == set(child_template.leaves())
    for read in owned.program.requirements.value_reads:
        assert read.target.period == 2
        assert read.source.source_period == 1
        assert read.source.path[0] == "done"
        assert read.source.core_key == CASE_U_ECONOMIC_CORE
    with pytest.raises(UnsupportedOperationError, match="not installed"):
        owned.require_model_installation()
    with pytest.raises(TypeError, match="producer date"):
        owned.materialize(context=replace(context, period=0))

    def forbid_compile(_function: Callable[..., PytreeValue]) -> Never:
        raise AssertionError(
            "Real retained roots must exhaust the budget before compilation."
        )

    monkeypatch.setattr(jax, "jit", forbid_compile)
    with pytest.raises(WorkspaceBudgetError, match="resident"):
        plan_private_case_u_workspace(owned=owned, context=context, budget_bytes=1)
