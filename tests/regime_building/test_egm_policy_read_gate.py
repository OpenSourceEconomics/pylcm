"""Build-time gates for the off-grid simulation policy read.

`SimulationPhase.egm_policy_read` is set only where replaying the solve-phase
policy is valid; every other configuration keeps the grid-argmax simulate path.
"""

import dataclasses

import pytest

from lcm import AgeGrid, Model
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    DCEGM_SOLVER,
    dcegm_retirement,
    dead,
)
from tests.test_models.ds2024_housing import build_model


def test_negm_regime_does_not_qualify_for_the_policy_read():
    """A NEGM regime keeps the grid path: its published policy is keeper-only.

    The nested period kernel maximizes the value over keeper and adjuster
    candidates but publishes only the keeper's inner consumption function, so
    replaying it would pair an adjuster-won value with the keeper's policy.
    """
    model = build_model(variant="negm", n_grid=12, n_periods=3)
    assert model._regimes["alive"].simulation.egm_policy_read is None


@pytest.mark.parametrize("backend", ["rfc", "ltm"])
def test_non_crossing_envelope_backends_do_not_qualify(backend: str):
    """RFC/LTM rows keep the grid path: they publish no crossing topology.

    Those backends leave the envelope switch between two retained nodes, so
    linear policy interpolation across the switch would mix two branch
    policies into an action belonging to neither.
    """
    model = _retirement_model_with_backend(backend)
    assert model._regimes["retirement"].simulation.egm_policy_read is None


@pytest.mark.parametrize("backend", ["fues", "mss"])
def test_crossing_inserting_envelope_backends_qualify(backend: str):
    """FUES/MSS rows qualify: duplicated crossing abscissae carry the switch."""
    model = _retirement_model_with_backend(backend)
    assert model._regimes["retirement"].simulation.egm_policy_read is not None


def _retirement_model_with_backend(backend: str) -> Model:
    solver = dataclasses.replace(DCEGM_SOLVER, upper_envelope=backend)
    return Model(
        regimes={
            "retirement": dcegm_retirement.replace(
                active=lambda age: age < 50, solver=solver
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=40, stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )
