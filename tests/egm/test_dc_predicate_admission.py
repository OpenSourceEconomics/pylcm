"""Eligible predicates retain conditional DC records independently of floor syntax."""

from pathlib import Path
from typing import Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import continuation as continuation_module
from _lcm.egm.carry import EGMCarry
from _lcm.egm.dcegm_query import DCEGMQuery, DCEGMQueryResult
from _lcm.egm.query_data import QueryNodeKind
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.feasibility import BoundedResult, FeasibilityKernelOutput
from _lcm.typing import HostArray
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    post_decision_lower_bound,
)
from lcm.solvers import DCEGM
from lcm.transition import fixed_transition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarFloat,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_consumers import (
    _capture_with_checks_off,
    _run_installed,
)
from tests.egm.test_dynamic_dcegm_passive_consumers import (
    Job,
    RegimeId,
    _AggregateInputs,
    _assert_clean_installed_statuses,
    _ChoiceInputs,
    _inverse,
    _parent_resources,
    _savings,
    _source_lower,
)
from tests.egm.test_query_point_discrete_feasibility import (
    _child_utility,
    _child_w,
    _child_z,
    _model_and_params,
    _next_liquid,
    _parent_utility,
    _resources,
    _terminal,
)
from tests.solution._nbegm_direct_oracle import OracleContext

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


type HostFloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float32 | np.float64]]


def _euler_predicate(
    *, job: DiscreteAction, liquid: ContinuousState, cutoff: float
) -> BoolND:
    return (job == 0) | (liquid >= cutoff)


def _quarter_child_liquid(savings: FloatND) -> ContinuousState:
    return 4.0 * savings - 0.75


def _euler_model_and_params(*, predicate: bool) -> tuple[Model, UserParams]:
    """Use an exact represented Euler tuple at actual child liquid one quarter."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    passive = {
        "z": IrregSpacedGrid(points=(0.0, 1.0)),
        "w": IrregSpacedGrid(points=(0.0, 1.0, 2.0)),
    }
    child = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=(0.0, 0.5, 1.5)), **passive},
        actions={
            "job": DiscreteGrid(category_class=Job),
            "consumption": IrregSpacedGrid(points=(0.025, 0.125, 0.5, 1.0, 4.0)),
        },
        functions={
            "utility": _child_utility,
            "resources": _resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=_source_lower),
            **({"eligibility": _euler_predicate} if predicate else {}),
        },
        state_transitions={
            "liquid": _next_liquid,
            "z": fixed_transition("z"),
            "w": fixed_transition("w"),
        },
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=(0.0, 0.5, 1.5)),
            n_constrained_points=4,
        ),
    )
    parent = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=(0.375, 0.5)), **passive},
        actions={
            "route": DiscreteGrid(category_class=Job),
            "consumption": IrregSpacedGrid(points=(0.025, 0.125, 0.25)),
        },
        functions={
            "utility": _parent_utility,
            "resources": _parent_resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse,
        },
        constraints={"borrowing": post_decision_lower_bound(margin=margin, lower=0.25)},
        state_transitions={
            "liquid": _quarter_child_liquid,
            "z": _child_z,
            "w": _child_w,
        },
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=(0.25, 0.375, 0.5)),
            n_constrained_points=4,
        ),
    )
    model = Model(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={"liquid": IrregSpacedGrid(points=(0.0, 16.0)), **passive},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=40, inclusive_stop=42, step="Y"),
        edges={"parent": {"child": 40}, "child": {"done": 41}},
        initial_nodes={40: "parent"},
        regime_id_class=RegimeId,
    )
    return model, {
        "parent": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "next_z": {"query_point": 0.5},
        },
        "child": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "resources": {"shift": 0.0},
            "borrowing_lower": {"floor": 0.0},
            **({"eligibility": {"cutoff": 1.5}} if predicate else {}),
        },
        "done": {},
    }


def _assert_unconditional_control(
    *,
    model: Model,
    params: UserParams,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_value: float,
    literal: bool,
) -> None:
    """Check the same physical model's ordinary publication.

    Observe the control before adding a predicate.
    """
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=directory,
        monkeypatch=monkeypatch,
    )
    np.testing.assert_array_almost_equal(
        context["next_regime_to_V_arr"]["child"][0],
        np.full((2, 3), 10.0),
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
    if literal:
        assert carry.query_data_for_transport() is None
        assert len(jax.tree.leaves(carry)) == 4
        assert model._regimes["child"].solution.query_authority is None


def _observe_predicate_consumer(
    *,
    kernel: _DCEGMPeriodKernel,
    context: OracleContext,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    FeasibilityKernelOutput,
    list[tuple[HostArray, HostArray, HostArray, HostArray]],
    list[tuple[HostArray, HostArray]],
]:
    """Observe untouched installed query results without requiring a callable floor."""
    choices: list[tuple[HostArray, HostArray, HostArray, HostArray]] = []
    aggregates: list[tuple[HostArray, HostArray]] = []
    read_choices = DCEGMQuery.read_choices
    aggregate = continuation_module._aggregate_child_choices

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record_choices(
        facts: HostArray, values: HostArray, marginals: HostArray, savings: HostArray
    ) -> None:
        choices.append((facts, values, marginals, savings))

    # keyword-only-exempt: library-callback=jax.debug.callback
    def record_aggregate(facts: HostArray, pair: HostArray) -> None:
        aggregates.append((facts, pair))

    def observe_choices(
        self: DCEGMQuery, **kwargs: Unpack[_ChoiceInputs]
    ) -> DCEGMQueryResult:
        result = read_choices(self, **kwargs)
        assert isinstance(result, DCEGMQueryResult)
        pool = kwargs["pool"]
        facts = jnp.stack(
            (
                *(
                    cast("ScalarFloat | ScalarInt", pool[name])
                    for name in (
                        "liquid",
                        "z",
                        "w",
                        "period",
                        "age",
                        "resources__shift",
                        "koopmans_aggregator__discount_factor",
                        "eligibility__cutoff",
                    )
                ),
                kwargs["euler_gradient"],
            )
        )
        jax.debug.callback(
            record_choices, facts, result.value, result.marginal, result.savings
        )
        return result

    def observe_aggregate(
        **kwargs: Unpack[_AggregateInputs],
    ) -> (
        tuple[ScalarFloat, ScalarFloat] | BoundedResult[tuple[ScalarFloat, ScalarFloat]]
    ):
        result = aggregate(**kwargs)
        if kwargs.get("query") is not None:
            pool = kwargs.get("query_pool")
            assert pool is not None
            assert isinstance(result, BoundedResult)
            facts = jnp.stack(
                tuple(
                    cast("ScalarFloat | ScalarInt", pool[name])
                    for name in ("liquid", "z", "w")
                )
            )
            jax.debug.callback(record_aggregate, facts, jnp.stack(result.output))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(DCEGMQuery, "read_choices", observe_choices)
        patch.setattr(
            continuation_module, "_aggregate_child_choices", observe_aggregate
        )
        output = _run_installed(kernel=kernel, context=context)
        jax.block_until_ready(output.value)
        jax.effects_barrier()
    assert isinstance(output, FeasibilityKernelOutput)
    return output, choices, aggregates


def _assert_real_query(
    *,
    model: Model,
    params: UserParams,
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    liquid: float,
    cutoff: float,
    child_publication: np.ndarray,
    conditional_value: float,
    savings: float,
    parent_value: float,
) -> None:
    """Require the real conditional bank and actual selected tuple before its rank."""
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=directory,
        monkeypatch=monkeypatch,
    )
    assert isinstance(kernel, _DCEGMPeriodKernel)
    np.testing.assert_array_almost_equal(
        context["next_regime_to_V_arr"]["child"][0],
        child_publication,
        decimal=DECIMAL_PRECISION,
    )
    actual_cutoff = jnp.asarray(context["flat_params"]["child"]["eligibility__cutoff"])
    canonical_cutoff = jnp.asarray(cutoff, dtype=jnp.asarray(0.0).dtype)
    assert actual_cutoff.dtype == canonical_cutoff.dtype
    np.testing.assert_array_equal(actual_cutoff, canonical_cutoff)
    carry = context["next_regime_to_continuation"]["child"]
    assert isinstance(carry, EGMCarry)
    data = carry.query_data_for_transport()
    assert data is not None, "An eligible predicate needs a real conditional query bank"
    authority = model._regimes["child"].solution.query_authority
    assert authority is not None, (
        "An eligible predicate needs an installed query authority"
    )
    assert authority.target_periods == (1,)
    assert authority.state_names == ("z", "w")
    assert authority.action_names == ("job",)
    np.testing.assert_array_equal(data.floor_used, 0.0)
    assert data.corner_continuation is not None
    np.testing.assert_array_equal(data.corner_continuation, 0.0)
    assert data.node_kind.shape[:3] == (2, 3, 2)
    assert np.all(
        np.any(np.asarray(data.node_kind)[:, :, 1] == QueryNodeKind.FINITE, axis=-1)
    )
    output, choices, aggregate = _observe_predicate_consumer(
        kernel=kernel,
        context=context,
        monkeypatch=monkeypatch,
    )
    expected_target = [liquid, 0.5, 0.5]
    selected = [
        item for item in choices if np.array_equal(item[0][:3], expected_target)
    ]
    selected_pairs = [
        pair for facts, pair in aggregate if np.array_equal(facts, expected_target)
    ]
    assert selected, "The actual conditional query was not reached"
    assert selected_pairs, "The actual conditional query was not aggregated"
    for facts, values, marginals, chosen_savings in selected:
        np.testing.assert_array_equal(
            facts[:7], [liquid, 0.5, 0.5, 1.0, 41.0, 0.0, 1.0]
        )
        np.testing.assert_array_equal(facts[7], canonical_cutoff)
        np.testing.assert_array_equal(facts[8], 4.0)
        np.testing.assert_array_equal(chosen_savings, [savings, savings])
        np.testing.assert_array_almost_equal(
            cast("HostFloatArray", values),
            [conditional_value, -np.inf],
            decimal=DECIMAL_PRECISION,
        )
        np.testing.assert_array_equal(marginals, [4.0, 0.0])
    for pair in selected_pairs:
        np.testing.assert_array_almost_equal(
            cast("HostFloatArray", pair),
            [conditional_value, 4.0],
            decimal=DECIMAL_PRECISION,
        )
    np.testing.assert_array_almost_equal(
        output.value[1],
        np.full((2, 3), parent_value),
        decimal=DECIMAL_PRECISION,
    )
    _assert_clean_installed_statuses(model=model, output=output)


def test_literal_floor_predicate_installs_a_conditional_query(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A literal floor preserves eligible query predicates.

    Predicate queries are independent of bound syntax.
    """
    control, control_params = _model_and_params(
        predicate="none", query_point=0.5, literal=True
    )
    _assert_unconditional_control(
        model=control,
        params=control_params,
        directory=tmp_path / "ordinary",
        monkeypatch=monkeypatch,
        parent_value=13.0,
        literal=True,
    )
    model, params = _model_and_params(
        predicate="threshold", query_point=0.5, literal=True
    )
    _assert_real_query(
        model=model,
        params=params,
        directory=tmp_path / "predicate",
        monkeypatch=monkeypatch,
        liquid=0.0,
        cutoff=0.6,
        child_publication=np.broadcast_to(np.asarray([0.0, 10.0])[:, None], (2, 3)),
        conditional_value=0.0,
        savings=0.0,
        parent_value=3.0,
    )


def test_euler_state_predicate_is_evaluated_after_inversion(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eligible liquid-state predicates govern publication.

    The same predicates govern actual conditional reads.
    """
    control, control_params = _euler_model_and_params(predicate=False)
    _assert_unconditional_control(
        model=control,
        params=control_params,
        directory=tmp_path / "ordinary",
        monkeypatch=monkeypatch,
        parent_value=13.25,
        literal=False,
    )
    model, params = _euler_model_and_params(predicate=True)
    _assert_real_query(
        model=model,
        params=params,
        directory=tmp_path / "predicate",
        monkeypatch=monkeypatch,
        liquid=0.25,
        cutoff=1.5,
        child_publication=np.zeros((2, 3)),
        conditional_value=0.25,
        savings=0.5,
        parent_value=3.25,
    )
