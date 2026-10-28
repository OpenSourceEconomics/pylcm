"""Actual runtime Euler nodes satisfy the original contract before dated producers."""

from dataclasses import replace
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.regime_building.processing as processing_module
from _lcm.egm.current_iid_euler_grid import resolve_dated_euler_grid
from lcm import ConsumptionSavingsRegime, IrregSpacedGrid, Model
from lcm.exceptions import (
    GridInitializationError,
    ModelInitializationError,
    UnsupportedOperationError,
)
from lcm.typing import FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_call_owned_generation import _declarations, _params

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _wrong_inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 2.0 / marginal_continuation


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_runtime_euler_support_keeps_each_actual_generation_validation() -> None:
    """Each canonical support owns its validation and actual dated nodes."""
    declarations = _declarations(runtime_child_grid=True)
    first_params = _params(child_points=np.array([1.0, 12.5]))
    second_params = _params(child_points=np.array([1.25, 12.75]))
    model = Model._from_call_owned_params(
        declarations=declarations, params=first_params
    )
    _, first = model._prepare_call(params=first_params)
    _, second = model._prepare_call(params=second_params)
    records = []
    for generation, points in ((first, [1.0, 12.5]), (second, [1.25, 12.75])):
        dated = generation.execution._dated_generation
        assert dated is not None
        numerical = dated.preparation.numerical
        declaration = generation.execution._declaration_preparation
        assert declaration is not None
        obligations = declaration.euler_grid_obligations
        numerical.validate_euler_grids(expected=obligations)
        record = numerical.euler_grid_validations["child", 1]
        owner = generation.solver_preparation(regime_name="child", period=1)
        owner.validate_owner()
        assert record.call.state_action_space is not None
        assert record.nodes is record.call.state_action_space.states["wealth"]
        assert record.obligation.grid.points is None
        assert (
            owner.context.regime_to_v_interpolation_info["child"].continuous_states[
                "wealth"
            ]
            is record.obligation.grid
        )
        date = owner.declarations["child"].date_at(period=1)
        assert not bool(date.template.valid)
        assert int(date.template.valid_length) == 0
        assert float(date.template.domain_lower) == points[0]
        assert float(date.template.domain_upper) == points[-1]
        np.testing.assert_array_equal(record.nodes, points)
        records.append(record)
    assert records[0] is not records[1]
    assert records[0].nodes is not records[1].nodes


@pytest.mark.parametrize("points", [np.array([12.5, 1.0]), np.array([1.0, 1.0])])
def test_original_runtime_geometry_refuses_before_projected_producers(
    *, points: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime nodes retain the original strict irregular-grid geometry rule."""

    def forbidden(
        **kwargs: object,  # noqa: PAN001 - refusal sentinel never reads operands
    ) -> None:
        del kwargs
        pytest.fail(
            "A dated producer preceded original runtime Euler geometry validation."
        )

    monkeypatch.setattr(processing_module, "projected_producer_declarations", forbidden)
    with pytest.raises(GridInitializationError, match="strictly ascending"):
        Model._from_call_owned_params(
            declarations=_declarations(runtime_child_grid=True),
            params=_params(child_points=points),
        )


@pytest.mark.parametrize("defect", ["savings-coverage", "inverse"])
def test_runtime_nodes_do_not_waive_original_economic_checks(
    *, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Original savings coverage and inverse-MU checks run before any dated bank."""
    declarations = _declarations(runtime_child_grid=True)
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    if defect == "inverse":
        declarations = replace(
            declarations,
            regimes={
                **declarations.regimes,
                "child": replace(
                    child,
                    functions={
                        **child.functions,
                        "inverse_marginal_utility": _wrong_inverse,
                    },
                ),
            },
        )
    points = np.array([1.0, 13.0 if defect == "savings-coverage" else 12.5])

    def forbidden(
        **kwargs: object,  # noqa: PAN001 - refusal sentinel never reads operands
    ) -> None:
        del kwargs
        pytest.fail("A dated producer preceded the original Euler contract checks.")

    monkeypatch.setattr(processing_module, "projected_producer_declarations", forbidden)
    with pytest.raises(
        ModelInitializationError, match=r"below the upper end|inverse_marginal_utility"
    ):
        Model._from_call_owned_params(
            declarations=declarations, params=_params(child_points=points)
        )


@pytest.mark.parametrize("defect", ["date", "extent", "dtype", "nan"])
def test_actual_call_context_cannot_substitute_invalid_euler_nodes(
    *, defect: str
) -> None:
    """Validation requires actual canonical support with the original geometry."""
    model = Model._from_call_owned_params(
        declarations=_declarations(runtime_child_grid=True),
        params=_params(child_points=np.array([1.0, 12.5])),
    )
    dated = model._dated_generation
    assert dated is not None
    original = dated.preparation.numerical.euler_grid_validations["child", 1]
    call = original.call
    assert call.state_action_space is not None
    if defect == "date":
        call = replace(call, period=0)
    else:
        nodes = (
            jnp.array([1.0])
            if defect == "extent"
            else jnp.array([1, 12], dtype=jnp.int32)
            if defect == "dtype"
            else jnp.array([1.0, jnp.nan])
        )
        call = replace(
            call,
            state_action_space=call.state_action_space.replace(
                states=MappingProxyType(
                    {**call.state_action_space.states, "wealth": nodes}
                )
            ),
        )
    expected = GridInitializationError if defect == "nan" else UnsupportedOperationError
    with pytest.raises(expected):
        resolve_dated_euler_grid(obligation=original.obligation, call=call)


def test_ordinary_model_retains_runtime_euler_grid_refusal() -> None:
    """Only a retained dated constructor can defer this contract.

    A model without a current shock is not dated, so its constructor refuses.
    """
    declarations = _declarations(runtime_child_grid=True)
    child = declarations.regimes["child"]
    with pytest.raises(
        ModelInitializationError, match="supplies its points only at runtime"
    ):
        Model(
            regimes={**declarations.regimes, "child": replace(child, shocks=())},
            ages=declarations.ages,
            initial_nodes=declarations.initial_nodes,
            edges=declarations.edges,
            regime_id_class=declarations.regime_id_class,
            shocks={},
        )


@pytest.mark.parametrize("defect", ["missing-date", "wrong-grid", "wrong-solver"])
def test_incomplete_runtime_binding_refuses_before_any_projected_factory(
    *, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine built Model cannot lend another date or original grid authority."""
    model = Model._from_call_owned_params(
        declarations=_declarations(runtime_child_grid=True),
        params=_params(child_points=np.array([1.0, 12.5])),
    )
    metadata = model._dated_parameters
    dated = model._dated_generation
    assert metadata is not None
    assert dated is not None
    numerical = dated.preparation.numerical
    records = numerical.euler_grid_validations
    original = metadata.euler_grid_obligations["child"]
    obligations = dict(metadata.euler_grid_obligations)
    calls = {key: record.call for key, record in records.items()}
    if defect == "missing-date":
        calls.pop(("child", 1))
    elif defect == "wrong-grid":
        obligations["child"] = replace(original, grid=IrregSpacedGrid(n_points=2))
    else:
        obligations["child"] = replace(
            original, user_regime=model.user_regimes["parent"]
        )

    def forbidden(
        **kwargs: object,  # noqa: PAN001 - refusal sentinel never reads operands
    ) -> None:
        del kwargs
        pytest.fail("An unresolved runtime Euler binding reached a projected factory.")

    monkeypatch.setattr(processing_module, "projected_producer_declarations", forbidden)
    with pytest.raises(UnsupportedOperationError, match="Runtime Euler"):
        metadata.builder.prepare_balanced_regimes(
            call_contexts=MappingProxyType(calls),
            fixed_flat_params=metadata.fixed.granular,
            euler_grid_obligations=MappingProxyType(obligations),
            state_utility_obligations=metadata.state_utility_obligations,
        )
