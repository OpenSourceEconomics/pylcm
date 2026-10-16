"""Literal combo masks and actual-state predicates publish distinct DC records."""

from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import QueryNodeKind
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from lcm import IrregSpacedGrid, Model
from lcm.typing import (
    Age,
    BoolND,
    ContinuousState,
    DiscreteAction,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dc_predicate_admission import (
    _assert_unconditional_control,
    _quarter_child_liquid,
)
from tests.egm.test_dynamic_dcegm_consumers import (
    _capture_with_checks_off,
    _run_installed,
)
from tests.egm.test_dynamic_dcegm_passive_consumers import RegimeId
from tests.egm.test_query_point_discrete_feasibility import _model_and_params

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _never() -> BoolND:
    return jnp.asarray(a=False)


def _first_job(*, job: DiscreteAction) -> BoolND:
    return job == 0


def _selected_job(*, job: DiscreteAction, allowed_job: float) -> BoolND:
    return job == allowed_job


def _target_clock(*, job: DiscreteAction, period: ScalarInt, age: Age) -> BoolND:
    return (job == 0) | ((period == 1) & (age == 41.0))


def _eligible(*, job: DiscreteAction, state_eligible: BoolND) -> BoolND:
    return (job == 0) | state_eligible


def _passive_island(*, z: ContinuousState) -> BoolND:
    return (z > 0.4) & (z < 0.6)


def _euler_island(*, liquid: ContinuousState) -> BoolND:
    return (liquid > 0.2) & (liquid < 0.3)


def _legacy_model_and_params(
    *, mask: Literal["constant", "discrete", "parameter", "clock"]
) -> tuple[Model, UserParams]:
    """Add one combo-only predicate to the literal two-by-three public model."""
    seed, params = _model_and_params(predicate="none", query_point=0.5, literal=True)
    child = seed.user_regimes["child"]
    predicate = {
        "constant": _never,
        "discrete": _first_job,
        "parameter": _selected_job,
        "clock": _target_clock,
    }[mask]
    model = Model(
        regimes={
            **seed.user_regimes,
            "child": child.replace(
                constraints={**child.constraints, "eligibility": predicate}
            ),
        },
        ages=seed.ages,
        edges=seed.edges,
        initial_nodes={40: "parent"},
        regime_id_class=RegimeId,
    )
    return model, {
        **params,
        "child": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "resources": {"shift": 0.0},
            **({"eligibility": {"allowed_job": 0.0}} if mask == "parameter" else {}),
        },
    }


def _hidden_model_and_params(
    *, axis: Literal["passive", "euler"]
) -> tuple[Model, UserParams]:
    """Hide the interpolated state behind the same Boolean predicate signature."""
    seed, params = _model_and_params(predicate="none", query_point=0.5, literal=True)
    child = seed.user_regimes["child"]
    parent = seed.user_regimes["parent"]
    if axis == "euler":
        parent = parent.replace(
            state_transitions={
                **parent.state_transitions,
                "liquid": _quarter_child_liquid,
            }
        )
    return Model(
        regimes={
            **seed.user_regimes,
            "parent": parent,
            "child": child.replace(
                functions={
                    **child.functions,
                    "state_eligible": (
                        _passive_island if axis == "passive" else _euler_island
                    ),
                },
                constraints={**child.constraints, "eligibility": _eligible},
            ),
        },
        ages=seed.ages,
        edges=seed.edges,
        initial_nodes={40: "parent"},
        regime_id_class=RegimeId,
    ), params


def test_literal_no_predicate_publication_control(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    "The unconstrained literal child publishes four leaves and parent value thirteen."
    model, params = _model_and_params(predicate="none", query_point=0.5, literal=True)
    _assert_unconditional_control(
        model=model,
        params=params,
        directory=tmp_path,
        monkeypatch=monkeypatch,
        parent_value=13.0,
        literal=True,
    )


@pytest.mark.parametrize(
    ("mask", "child_value", "parent_value"),
    [
        ("constant", -np.inf, -np.inf),
        ("discrete", 0.0, 3.0),
        ("parameter", 0.0, 3.0),
        ("clock", 10.0, 13.0),
    ],
    ids=("constant", "discrete", "parameter", "clock"),
)
def test_literal_combo_masks_preserve_four_leaf_publication(
    *,
    mask: Literal["constant", "discrete", "parameter", "clock"],
    child_value: float,
    parent_value: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Masks constant across interpolation coordinates retain the literal carry."""
    model, params = _legacy_model_and_params(mask=mask)
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    assert isinstance(
        model._regimes["child"].solution.period_kernels[1], _DCEGMPeriodKernel
    )
    np.testing.assert_array_almost_equal(
        context["next_regime_to_V_arr"]["child"][0],
        np.full((2, 3), child_value),
        decimal=DECIMAL_PRECISION,
    )
    output = _run_installed(kernel=kernel, context=context)
    np.testing.assert_array_almost_equal(
        output.value[1],
        np.full((2, 3), parent_value),
        decimal=DECIMAL_PRECISION,
    )
    carry = context["next_regime_to_continuation"]["child"]
    assert isinstance(carry, EGMCarry)
    assert carry.query_data_for_transport() is None
    assert len(carry.leaves()) == 4
    assert model._regimes["child"].solution.query_authority is None


@pytest.mark.parametrize(
    ("axis", "parent_value"),
    [("passive", 10.0), ("euler", 13.25)],
    ids=("passive", "euler"),
)
def test_hidden_state_predicates_retain_false_at_saved_nodes_branch(
    *,
    axis: Literal["passive", "euler"],
    parent_value: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hidden state island retains a branch absent from every saved state node."""
    model, params = _hidden_model_and_params(axis=axis)
    state_grid = model.user_regimes["child"].states[
        "z" if axis == "passive" else "liquid"
    ]
    assert isinstance(state_grid, IrregSpacedGrid)
    np.testing.assert_array_equal(
        state_grid.to_jax(),
        [0.0, 1.0] if axis == "passive" else [0.0, 0.5, 1.5],
    )
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    assert isinstance(
        model._regimes["child"].solution.period_kernels[1], _DCEGMPeriodKernel
    )
    np.testing.assert_array_almost_equal(
        context["next_regime_to_V_arr"]["child"][0],
        np.zeros((2, 3)),
        decimal=DECIMAL_PRECISION,
    )
    carry = context["next_regime_to_continuation"]["child"]
    assert isinstance(carry, EGMCarry)
    data = carry.query_data_for_transport()
    assert data is not None
    authority = model._regimes["child"].solution.query_authority
    assert authority is not None
    assert authority.target_periods == (1,)
    assert authority.state_names == ("z", "w")
    assert authority.action_names == ("job",)
    assert ("z" if axis == "passive" else "liquid") in dict(authority.evaluator_inputs)[
        "predicate:eligibility"
    ]
    np.testing.assert_array_equal(data.floor_used, 0.0)
    assert data.corner_continuation is not None
    np.testing.assert_array_equal(data.corner_continuation, 0.0)
    assert data.node_kind.shape[:3] == (2, 3, 2)
    np.testing.assert_array_equal(
        np.any(np.asarray(data.node_kind)[:, :, 1] == QueryNodeKind.FINITE, axis=-1),
        np.ones((2, 3), dtype=bool),
    )
    output = _run_installed(kernel=kernel, context=context)
    np.testing.assert_array_almost_equal(
        output.value[1],
        np.full((2, 3), parent_value),
        decimal=DECIMAL_PRECISION,
    )
