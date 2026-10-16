"""Only actual NEGM inner producers enable current binding publication."""

import dataclasses
import functools
from collections.abc import Callable
from typing import NotRequired, Protocol, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import step, step_core
from _lcm.egm.carry import EGMCarry
from _lcm.egm.step_core import _EgmKernelPieces
from _lcm.execution.core_program import core_program_graph
from _lcm.solution import fingerprint
from _lcm.solution.contract import (
    PeriodKernel,
    _BoundLiquidMargin,
    _BoundOuterContinuousMargin,
)
from _lcm.solution.dcegm import _BoundDCEGM
from _lcm.solution.negm import _BoundNEGM, _NEGMPeriodKernel, _with_outer_post_decision
from _lcm.typing import DataclassInstance, HostArray
from lcm import (
    AgeGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    Regime,
    outer_unchanged,
)
from lcm.solver_api import EGM_CONTINUATION, KernelOutput
from lcm.solvers import DCEGM, NEGM
from lcm.typing import BoolND, ContinuousState, Float1D, FloatND, ScalarFloat, ScalarInt
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution import test_nested_dynamic_borrowing as original
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.solution.test_negm_age_specialized_function import _build_model, _specialized
from tests.solution.test_negm_core_program import _fixed_flat_params

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_FIELD = "_negm_current_publication"


class _NamedField(Protocol):
    """The exact metadata accessed while preserving the original field type."""

    @property
    def name(self) -> str: ...


class _ConstrainedArguments(TypedDict):
    first_endogenous_point: ScalarFloat
    publish_resources: FloatND
    borrowing_limit: ScalarFloat
    n_constrained: int
    constrained_ratio: float
    utility_of_action: Callable[[ScalarFloat], ScalarFloat]
    discounted_expected_value_at_limit: ScalarFloat


class _PublicationArguments(TypedDict):
    refined_grid: Float1D
    refined_policy: Float1D
    refined_value: Float1D
    n_kept: ScalarInt
    n_pad: int
    publish_resources: FloatND
    borrowing_limit: ScalarFloat
    utility_of_action: Callable[[ScalarFloat], ScalarFloat]
    discounted_expected_value_at_limit: ScalarFloat
    _negm_current_publication: NotRequired[bool]
    _binding_query: NotRequired[BoolND | None]


def _pieces(*, kernel: PeriodKernel) -> _EgmKernelPieces:
    function = core_program_graph(kernel=kernel)["replay"].function
    seen = set()
    while not hasattr(function, "pieces"):
        assert id(function) not in seen
        seen.add(id(function))
        if isinstance(function, functools.partial):
            function = function.func
        else:
            assert hasattr(function, "__wrapped__")
            function = function.__wrapped__
    assert isinstance(function.pieces, _EgmKernelPieces)
    return function.pieces


def test_standalone_default_is_absent_from_the_public_solver_and_fingerprint(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The private false default hashes like its absence; true binds identity."""
    assert _FIELD not in {field.name for field in dataclasses.fields(DCEGM)}
    bound = _BoundDCEGM(
        savings_grid=original._SAVINGS,
        continuous_state="liquid",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
    )
    fields = {field.name: field for field in dataclasses.fields(bound)}
    assert _FIELD in fields, "The adopted private bound-inner policy is missing"
    assert getattr(bound, _FIELD) is False
    assert fields[_FIELD].default is False
    assert fields[_FIELD].metadata.get("fingerprint_omit_if_default") is True
    default_key = fingerprint._semantic_fingerprint(bound)
    enabled = dataclasses.replace(bound, **{_FIELD: True})
    assert fingerprint._semantic_fingerprint(enabled) != default_key
    actual_fields = dataclasses.fields

    def filter_fields[FieldT: _NamedField](
        function: Callable[
            [DataclassInstance | type[DataclassInstance]], tuple[FieldT, ...]
        ],
    ) -> Callable[[DataclassInstance | type[DataclassInstance]], tuple[FieldT, ...]]:
        def without_default_field(
            value: DataclassInstance | type[DataclassInstance],
        ) -> tuple[FieldT, ...]:
            declared = function(value)
            return (
                tuple(field for field in declared if field.name != _FIELD)
                if value is bound
                else declared
            )

        return without_default_field

    # Same nominal actual class and actual fingerprint implementation. This
    # models the field's absence only, without inventing a solver stand-in.
    with monkeypatch.context() as patch:
        patch.setattr(fingerprint.dataclasses, "fields", filter_fields(actual_fields))
        assert fingerprint._semantic_fingerprint(bound) == default_key


def test_grouped_age_specialized_and_fixed_bound_inners_keep_the_private_policy() -> (
    None
):
    """Every actual keeper/adjuster build and its fixed-param copy stay enabled."""
    # Reuse the existing admitted age-specialization/fixed-param fixture unchanged.
    model = _build_model(helper_name="credited", override=_specialized("credited"))
    kernels = model._regimes["alive"].solution.period_kernels
    assert len(kernels) >= 2
    for kernel in kernels.values():
        assert isinstance(kernel, _NEGMPeriodKernel)
        for active in (
            kernel,
            kernel.with_fixed_params(fixed_flat_params=_fixed_flat_params()),
        ):
            for inner in (active.keeper_kernel, active.adjuster_kernel):
                assert getattr(_pieces(kernel=inner), _FIELD) is True


def test_active_private_policy_changes_the_actual_lowering_program_identity(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same public model cannot reuse an executable with another policy."""
    active = original._model(family="negm", lower=0.0)
    bind = NEGM._with_margins

    # keyword-only-exempt: library-callback=NEGM._with_margins
    def disabled_binding(
        self: NEGM, *, liquid: _BoundLiquidMargin, outer: _BoundOuterContinuousMargin
    ) -> _BoundNEGM:
        bound = bind(self, liquid=liquid, outer=outer)
        assert getattr(bound.inner, _FIELD) is True
        return dataclasses.replace(
            bound, inner=dataclasses.replace(bound.inner, **{_FIELD: False})
        )

    # A private test-only identity counterfactual, not a public option, guard
    # bypass, numerical acceptance fixture or alternate source of authority.
    with monkeypatch.context() as patch:
        patch.setattr(NEGM, "_with_margins", disabled_binding)
        disabled = original._model(family="negm", lower=0.0)
    active_params = active._process_params(original._PARAMS, runtime_checks=False)
    disabled_params = disabled._process_params(original._PARAMS, runtime_checks=False)
    assert active._program_fingerprint(
        flat_params=active_params
    ) != disabled._program_fingerprint(flat_params=disabled_params)


def slope_resources_before_cost(*, liquid: ContinuousState) -> FloatND:
    return 2.0 * liquid + 1.25


def _slope_model() -> Model:
    """Original finite fixture grids and family, with slope2 in actual resources."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources=NetOfAdjustmentCost(
            output="resources",
            before_cost="resources_before_outer_cost",
            cost="credited",
        ),
        post_decision_state="savings",
    )
    alive = NestedConsumptionSavingsRegime(
        states={"liquid": original._LIQUID, "illiquid": original._STOCK},
        actions={
            "consumption": original._CONSUMPTION,
            "investment": original._INVESTMENT,
        },
        state_transitions={
            "liquid": original.next_liquid,
            "illiquid": original.next_illiquid,
        },
        functions={
            "utility": original.utility,
            "service": original.service,
            "savings": original.savings,
            "new_illiquid": original.new_illiquid,
            "inverse_marginal_utility": original.inverse_marginal_utility,
            "resources_before_outer_cost": slope_resources_before_cost,
            "credited": original.credited,
        },
        solver=NEGM(
            inner=DCEGM(savings_grid=original._SAVINGS), outer_grid=original._OUTER
        ),
        liquid=margin,
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="investment",
            post_decision_state="new_illiquid",
            no_adjustment=outer_unchanged,
        ),
    )
    return Model(
        regimes={
            "alive": alive,
            "done": Regime(
                states={"liquid": original._LIQUID, "illiquid": original._STOCK},
                functions={"utility": original.terminal},
            ),
        },
        ages=AgeGrid(exact_values=(10, 20)),
        edges={"alive": {"done": (10,)}},
        initial_nodes={10: "alive"},
        regime_id_class=original.RegimeId,
    )


def test_actual_keeper_and_adjusters_use_own_cost_and_nonzero_continuation(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual source-owned candidates use betaW(0), their own target and cost."""
    declarations = []
    build = step._build_kernel_pieces

    def observe_build[**P](
        function: Callable[P, _EgmKernelPieces],
    ) -> Callable[P, _EgmKernelPieces]:
        def observe(*args: P.args, **kwargs: P.kwargs) -> _EgmKernelPieces:
            pieces = function(*args, **kwargs)
            declarations.append(
                (getattr(kwargs["solver"], _FIELD), getattr(pieces, _FIELD))
            )
            return pieces

        return observe

    with monkeypatch.context() as patch:
        patch.setattr(step, "_build_kernel_pieces", observe_build(build))
        model = _slope_model()
    assert len(declarations) >= 2
    assert all(row == (True, True) for row in declarations)
    source_points, chain = [], []
    constrained = step_core._compute_constrained_candidates
    publish = step_core._publish_V_and_carry_rows

    def observe_source(
        **kwargs: Unpack[_ConstrainedArguments],
    ) -> tuple[FloatND, FloatND]:
        output = constrained(**kwargs)

        # keyword-only-exempt: library-callback=jax.debug.callback
        def record(c: HostArray, b: HostArray) -> None:
            source_points.append((float(c), float(b)))

        jax.debug.callback(record, output[0][0], kwargs["borrowing_limit"])
        return output

    def observe_publication(
        **kwargs: Unpack[_PublicationArguments],
    ) -> tuple[FloatND, Float1D, Float1D]:
        output = publish(**kwargs)
        flags = kwargs.get("_binding_query")
        if kwargs.get("_negm_current_publication", False) and flags is not None:
            resources = kwargs["publish_resources"]
            # Test-only local AD of current publication at the SAME certified
            # query/row. It performs no native refine, expectation or carry read.
            _, derivative = jax.jvp(
                lambda r: publish(
                    **cast("_PublicationArguments", {**kwargs, "publish_resources": r})
                )[0],
                (resources,),
                (jnp.full_like(resources, 2.0),),
            )

            # keyword-only-exempt: library-callback=jax.debug.callback
            def record(r: HostArray, binding: HostArray, d: HostArray) -> None:
                chain.append((float(r), bool(binding), float(d)))

            jax.debug.callback(record, resources[0], flags[0], derivative[0])
        return output

    monkeypatch.setattr(step_core, "_compute_constrained_candidates", observe_source)
    monkeypatch.setattr(step_core, "_publish_V_and_carry_rows", observe_publication)
    kernel, context = ride_along_kernel(
        model=model, params={"discount_factor": 0.5}, period=0
    )
    assert isinstance(kernel, _NEGMPeriodKernel)
    published = original._inner(kernel=kernel, context=context)
    keeper = original._inner(kernel=kernel.keeper_kernel, context=context)
    adjusters = tuple(
        original._inner(
            kernel=kernel.adjuster_kernel,
            context={
                **context,
                "flat_params": _with_outer_post_decision(
                    flat_params=context["flat_params"],
                    regime_name="alive",
                    outer_post_decision="new_illiquid",
                    value=node,
                ),
            },
        )
        for node in original._OUTER.to_jax()
    )
    outputs = (keeper, *adjusters)
    jax.effects_barrier()
    _assert_source_candidates(
        source_points=source_points, published=published, outputs=outputs
    )
    _assert_binding_chain_and_values(chain=chain, outputs=outputs)


def _assert_source_candidates(
    *,
    source_points: list[tuple[float, float]],
    published: KernelOutput,
    outputs: tuple[KernelOutput, ...],
) -> None:
    """Assert source ownership, marginal values and the unchanged carried bank."""
    assert source_points
    assert all(bound == 0.0 for _, bound in source_points)
    source_c = source_points[0][0]
    assert source_c > 0.0
    assert all(c == source_c for c, _ in source_points)
    original._assert_published_bank(published=published, outputs=outputs, family="negm")
    for output in (published, *outputs):
        carry = output.continuations[EGM_CONTINUATION]
        assert isinstance(carry, EGMCarry)
        assert carry.query_data_for_transport() is None
        assert set(carry.leaves()) == {
            ("endog_grid",),
            ("value",),
            ("marginal_utility",),
            ("taste_shock_scale",),
        }
    for output in outputs:
        carry = output.continuations[EGM_CONTINUATION]
        assert isinstance(carry, EGMCarry)
        leaves = carry.leaves()
        # The actual first constrained candidate is observed at its source
        # constructor; its ownership is not inferred from R-C or carry values.
        for stock_index in range(3):
            grid = np.asarray(leaves[("endog_grid",)])[stock_index]
            hits = np.flatnonzero(grid == source_c)
            assert hits.size == 1
            actual_m = np.asarray(leaves[("marginal_utility",)])[stock_index, hits[0]]
            np.testing.assert_array_almost_equal(
                actual_m,
                jnp.asarray(1.0) / jnp.asarray(source_c),
                decimal=DECIMAL_PRECISION,
            )


def _assert_binding_chain_and_values(
    *, chain: list[tuple[float, bool, float]], outputs: tuple[KernelOutput, ...]
) -> None:
    """Assert the publication derivative and each candidate's own economic value."""
    valid_chain = [(r, d) for r, binding, d in chain if binding and r > 0.0]
    assert valid_chain
    assert any(r == 1.25 for r, d in valid_chain)
    for resource, derivative in valid_chain:
        np.testing.assert_array_almost_equal(
            derivative, 2.0 / resource, decimal=DECIMAL_PRECISION
        )
    # At the first two original liquid nodes every legal candidate is binding:
    # beta=.5, terminal W(A,target)=A+target gives the unconstrained C*=2.
    # Own R <=1.75 before costs, so C=R, A=0 and betaW(0,target)=target/2.
    for liquid_index, liquid in enumerate((0.0, 0.25)):
        for stock_index, stock in enumerate((0.0, 1.0, 2.0)):
            for output, target in zip(outputs, (stock, 0.0, 1.0, 2.0), strict=True):
                resource = 2.0 * liquid + 1.25 - (target - stock) ** 2 / 2.0
                expected = (
                    -np.inf
                    if resource <= 0
                    else np.log(resource) + target * np.log(2.0) + target / 2.0
                )
                np.testing.assert_array_almost_equal(
                    np.asarray(output.value)[liquid_index, stock_index],
                    expected,
                    decimal=DECIMAL_PRECISION,
                )
                if liquid_index == 0 and resource > 0:
                    actual_c = original._consumption_at_zero(
                        output=output, stock_index=stock_index, resource=resource
                    )
                    np.testing.assert_array_almost_equal(
                        actual_c, resource, decimal=DECIMAL_PRECISION
                    )
