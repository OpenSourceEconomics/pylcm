"""Continuation demand and representation travel through one explicit contract."""

from collections.abc import Mapping
from typing import cast

import jax.numpy as jnp
import pytest

from _lcm import engine
from _lcm.continuation import (
    ContinuationSpec,
    EGMContinuationLayout,
    EGMContinuationSpec,
)
from _lcm.egm.carry import EGMCarry
from _lcm.reachability import build_phase_reachability
from _lcm.regime_building.processing import _continuation_demands
from _lcm.solution import contract
from _lcm.typing import RegimeName
from lcm.grids import LinSpacedGrid
from lcm.regime import Regime as UserRegime
from lcm.solver_api import EGM_CONTINUATION, ArtifactKey
from lcm.solvers import EGM, GridSearch
from tests.mock_regime import MockRegime


def _template() -> EGMCarry:
    """Return a finite one-row continuation template."""
    row = jnp.asarray([[0.0, 1.0]])
    return EGMCarry(
        endog_grid=row,
        value=row,
        marginal_utility=jnp.ones_like(row),
        taste_shock_scale=jnp.asarray(0.0),
    )


def test_continuation_payload_alias_has_one_engine_definition():
    """The engine and solver contract import the same continuation alias."""
    assert engine.ContinuationPayload is contract.ContinuationPayload


def test_solution_kernels_bundle_template_with_its_layout():
    """A continuation template cannot be published without its interpretation."""
    layout = EGMContinuationLayout(
        retains_discrete_action_rows=False,
        rows_share_state_grid=True,
        n_stacked_candidates=3,
    )
    spec = EGMContinuationSpec(template=_template(), layout=layout)
    kernels = contract.SolutionKernels(period_kernels={}, continuation_spec=spec)

    assert kernels.continuation_template is spec.template
    assert cast("EGMContinuationSpec", kernels.continuation_spec).layout is layout


def test_grid_search_declares_the_layout_of_engine_produced_carries():
    """A brute-force child has shared-grid rows and no discrete-action rows."""
    assert GridSearch().egm_continuation_layout == EGMContinuationLayout(
        retains_discrete_action_rows=False,
        rows_share_state_grid=True,
    )


def test_only_reachable_targets_of_continuation_readers_publish_carries():
    """Unreachable and value-only targets do not receive an EGM carry artifact."""
    reachability = build_phase_reachability(
        n_periods=2,
        active_periods_by_regime={
            "reader": (0,),
            "value_only": (0,),
            "needed": (1,),
            "unused": (1,),
        },
        candidate_targets_by_source={
            "reader": ("needed",),
            "value_only": ("unused",),
            "needed": (),
            "unused": (),
        },
        terminal_regimes=("needed", "unused"),
    )
    continuation_reader = EGM(
        savings_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=4)
    )
    regimes: Mapping[RegimeName, UserRegime] = {
        "reader": MockRegime(solver=continuation_reader),
        "value_only": MockRegime(solver=GridSearch()),
        "needed": MockRegime(solver=GridSearch()),
        "unused": MockRegime(solver=GridSearch()),
    }

    demands = _continuation_demands(
        user_regimes=regimes,
        phase_reachability=reachability,
    )
    assert frozenset(target for _source, target, _key in demands) == frozenset(
        {"needed"}
    )


def test_a_spec_accepts_a_template_that_carries_the_declared_key():
    """Template and declaration agreeing is what makes a spec constructible."""
    spec = ContinuationSpec(template=_template(), artifact_key=EGM_CONTINUATION)

    assert spec.artifact_key == EGM_CONTINUATION


def test_a_template_carrying_another_key_than_the_declared_one_is_refused():
    """Constructing the spec names the template's key and the declared one."""
    declared = ArtifactKey(type_id="example_solver.euler_residuals", schema_version=2)

    with pytest.raises(ValueError, match="not the declared") as excinfo:
        ContinuationSpec(template=_template(), artifact_key=declared)

    message = str(excinfo.value)
    assert "pylcm.egm.continuation" in message
    assert "example_solver.euler_residuals" in message
