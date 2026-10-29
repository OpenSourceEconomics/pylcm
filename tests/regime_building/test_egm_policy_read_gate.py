"""Build-time gates for the off-grid simulation policy read.

`SimulationPhase.egm_policy_read` is set only where the published rows sit on
the coordinates a subject is read at; every other configuration keeps the
grid-argmax simulate path.
"""

import dataclasses

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DeterministicTransition,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Transition,
    fixed_transition,
)
from lcm.solvers import FUESEnvelope
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from lcm_examples.iskhakov_et_al_2017 import WEALTH_GRID, next_wealth_from_savings
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.envelope_configs import EnvelopeName, envelope_config
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    DCEGM_SOLVER,
    LIQUID_MARGIN,
    dcegm_retirement,
    dead,
)
from tests.test_models.ds2024_housing import build_model

_PORTABLE_DCEGM_SOLVER = dataclasses.replace(
    DCEGM_SOLVER,
    envelope=FUESEnvelope(),
)
_PORTABLE_DCEGM_RETIREMENT = dcegm_retirement.replace(solver=_PORTABLE_DCEGM_SOLVER)


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_negm_regime_does_not_qualify_for_the_policy_read():
    """A NEGM regime keeps the grid path: its published policy is keeper-only.

    The nested period kernel maximizes the value over keeper and adjuster
    candidates but publishes only the keeper's inner consumption function, so
    replaying it would pair an adjuster-won value with the keeper's policy.
    """
    model = build_model(variant="negm", n_grid=12, n_periods=3)
    assert model._regimes["alive"].simulation.egm_policy_read is None


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "exact",
            marks=pytest.mark.requires_exact_affine_kernel(
                reason=EXACT_KERNEL_SKIP_REASON
            ),
        ),
        "fues",
        "rfc",
        "ltm",
    ],
)
def test_every_envelope_backend_qualifies_for_the_policy_read(backend: EnvelopeName):
    """Plain DC-EGM simulates policy-first under every upper-envelope backend.

    The read needs no envelope certificate: a subject whose read leaves its
    row support, or fails the budget or feasibility check, takes the grid
    argmax instead.
    """
    model = _retirement_model_with_backend(backend)
    assert model._regimes["retirement"].simulation.egm_policy_read is not None


def test_grid_replay_keeps_the_grid_argmax():
    """`DCEGM(replay="grid")` opts a plain DC-EGM regime out of the policy read."""
    solver = dataclasses.replace(DCEGM_SOLVER, replay="grid")
    model = _model_from_alive(alive=dcegm_retirement.replace(solver=solver))
    assert model._regimes["retirement"].simulation.egm_policy_read is None


def test_process_state_regime_does_not_qualify_for_the_policy_read():
    """A regime with a continuous stochastic-process state keeps the grid path.

    A process state is stored as a node-valued row axis, but its simulation
    transition draws a genuinely continuous value that need not land on a
    process node. Selecting one row by nearest node would read the policy of
    the wrong process-state conditional; interpolating across process rows is
    the tracked follow-up.
    """
    model = _model_from_alive(
        alive=_PORTABLE_DCEGM_RETIREMENT.replace(
            liquid=dataclasses.replace(LIQUID_MARGIN, resources="resources"),
            states={
                "wealth": WEALTH_GRID,
                "wage_shock": NormalIIDProcess(
                    n_points=5, gauss_hermite=True, mu=0.0, sigma=0.3
                ),
            },
            functions={
                **dict(dcegm_retirement.functions),
                "resources": _resources_with_shock,
                "savings": _savings_from_resources,
            },
        )
    )
    assert model._regimes["retirement"].simulation.egm_policy_read is None


def test_asset_row_regime_does_not_qualify_for_the_policy_read():
    """An asset-row DCEGM regime keeps the grid path: rows are pointwise optima.

    When a savings-stage function reads the current Euler state, DC-EGM solves
    per exogenous asset node and publishes one optimal point per node rather
    than a crossing-complete resources-space row. Interpolating across nodes
    would mix two endogenous branches wherever the winning branch changes
    between adjacent nodes. Here the regime transition law is the savings-stage
    function reading wealth, so the source gets two outgoing edges at age 40.
    """
    model = Model(
        regimes={"retirement": _PORTABLE_DCEGM_RETIREMENT, "dead": dead},
        edges={
            "retirement": Transition(
                targets={"retirement": 40, "dead": (40, 50)},
                law=DeterministicTransition(func=_next_regime_reads_wealth),
            )
        },
        ages=AgeGrid(start=40, inclusive_stop=60, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
        initial_nodes={40: "retirement"},
    )
    assert model._regimes["retirement"].simulation.egm_policy_read is None


def test_passive_state_regime_does_not_qualify_for_the_policy_read():
    """A regime with a passive continuous state keeps the grid path.

    Each published row is already the upper-envelope policy conditional on one
    passive-state node. When the winning endogenous branch differs between two
    passive nodes, linearly blending their policies yields an action from
    neither branch; the passive read returns once conditional-value
    re-decision across the passive axis is available.
    """
    skill_grid = LinSpacedGrid(start=0.5, stop=1.5, n_points=5)
    alive = _PORTABLE_DCEGM_RETIREMENT.replace(
        states={"wealth": WEALTH_GRID, "skill": skill_grid},
        state_transitions={
            "wealth": next_wealth_from_savings,
            "skill": fixed_transition("skill"),
        },
        functions={**dict(dcegm_retirement.functions), "utility": _skill_alive_utility},
    )
    dead_regime = dead.replace(
        states={"wealth": WEALTH_GRID, "skill": skill_grid},
        functions={"utility": _skill_bequest_utility},
    )
    model = Model(
        regimes={"retirement": alive, "dead": dead_regime},
        edges={"retirement": {"dead": 40}},
        ages=AgeGrid(start=40, inclusive_stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
        initial_nodes={40: "retirement"},
    )
    assert model._regimes["retirement"].simulation.egm_policy_read is None


def _retirement_model_with_backend(backend: EnvelopeName) -> Model:
    solver = dataclasses.replace(DCEGM_SOLVER, envelope=envelope_config(backend))
    return _model_from_alive(
        alive=dcegm_retirement.replace(
            solver=solver,
        )
    )


def _model_from_alive(*, alive, dead_states=None) -> Model:
    dead_regime = dead if dead_states is None else dead.replace(states=dead_states)
    return Model(
        regimes={"retirement": alive, "dead": dead_regime},
        edges={"retirement": {"dead": 40}},
        ages=AgeGrid(start=40, inclusive_stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
        initial_nodes={40: "retirement"},
    )


def _resources_with_shock(
    *, wealth: ContinuousState, wage_shock: ContinuousState
) -> FloatND:
    return wealth + wage_shock


def _savings_from_resources(
    *, resources: FloatND, consumption: ContinuousAction
) -> FloatND:
    return resources - consumption


def _skill_alive_utility(*, consumption: FloatND, skill: ContinuousState) -> FloatND:
    # `skill` rides along as a passive state; the flow utility keeps it in the
    # DAG (0.0 * skill) without changing the consumption optimum.
    return jnp.log(consumption) + 0.0 * skill


def _skill_bequest_utility(
    *, wealth: ContinuousState, skill: ContinuousState
) -> FloatND:
    return skill * jnp.log(wealth)


def _next_regime_reads_wealth(
    *, age: int, wealth: ContinuousState, final_age_alive: float
) -> ScalarInt:
    alive = (age < final_age_alive) & (wealth > -1.0e30)
    return jnp.where(
        alive,
        retirement_only.RetirementOnlyRegimeId.retirement,
        retirement_only.RetirementOnlyRegimeId.dead,
    ).astype("int32")
