"""The ride-along NB-EGM period kernel agrees with a direct scalar oracle.

The oracle (`tests/solution/_nbegm_direct_oracle.py`) solves one regime-period in
NumPy scalars from the model's declarations alone; the production kernel must
publish the same value array, carry rows, and consumption on every ride-along
route the test models exercise.
"""

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tests.test_models as test_models_package
from _lcm.egm import continuation as continuation_module
from _lcm.egm.upper_envelope import query as query_module
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from lcm.solvers import NBEGM
from tests.conftest import DECIMAL_PRECISION
from tests.solution import _nbegm_oracle_routes as routes
from tests.solution import test_nbegm_epstein_zin as epstein_zin_model
from tests.solution._nbegm_direct_oracle import (
    direct_oracle_period,
    nnbegm_inner_contexts,
    ride_along_kernel,
    run_production_kernel,
)
from tests.test_models import (
    n_nbegm_discrete_toy,
    n_nbegm_toy,
    nbegm_brute_child_toy,
    nbegm_ces_utility_toy,
    nbegm_continuous_ride_along_toy,
    nbegm_derived_var_toy,
    nbegm_indexed_threshold_toy,
    nbegm_jump_ride_along_toy,
    nbegm_mappingleaf_threshold_toy,
    nbegm_multi_discrete_toy,
    nbegm_multi_source_jump_toy,
    nbegm_multi_source_toy,
    nbegm_multi_target_toy,
    nbegm_next_asset_cliff_toy,
    nbegm_ride_along_toy,
    nbegm_ride_discrete_toy,
    nbegm_stochastic_node_toy,
)


@dataclass(frozen=True, kw_only=True)
class _Route:
    """One ride-along route: how to build its model and reach its kernel."""

    name: str
    build_model: Callable[[], Any]
    build_params: Callable[[], Any]
    regime_name: str = "alive"
    period: int | None = None


_SMALL: dict[str, Any] = {"n_liquid": 12, "n_savings": 16, "n_consumption": 24}

_ROUTES = (
    _Route(
        name="ride_along",
        build_model=lambda: nbegm_ride_along_toy.build_model(
            variant="nbegm", n_periods=3, **_SMALL
        ),
        build_params=nbegm_ride_along_toy.build_params,
    ),
    _Route(
        name="ride_along_per_kind_crra",
        build_model=lambda: nbegm_ride_along_toy.build_model(
            variant="nbegm", n_periods=3, per_kind_crra=True, **_SMALL
        ),
        build_params=lambda: nbegm_ride_along_toy.build_params(per_kind_crra=True),
    ),
    _Route(
        name="ride_along_per_kind_discount",
        build_model=lambda: nbegm_ride_along_toy.build_model(
            variant="nbegm", n_periods=3, per_kind_discount=True, **_SMALL
        ),
        build_params=lambda: nbegm_ride_along_toy.build_params(per_kind_discount=True),
    ),
    _Route(
        name="ride_along_distributed_kind",
        build_model=lambda: nbegm_ride_along_toy.build_model(
            variant="nbegm", n_periods=3, distributed_kind=True, **_SMALL
        ),
        build_params=nbegm_ride_along_toy.build_params,
    ),
    _Route(
        name="derived_var",
        build_model=lambda: nbegm_derived_var_toy.build_model(
            variant="nbegm", n_periods=3, **_SMALL
        ),
        build_params=nbegm_derived_var_toy.build_params,
    ),
    _Route(
        name="multi_source",
        build_model=lambda: nbegm_multi_source_toy.build_model(
            variant="nbegm", n_periods=3, **_SMALL
        ),
        build_params=nbegm_multi_source_toy.build_params,
    ),
    _Route(
        name="multi_target",
        build_model=lambda: nbegm_multi_target_toy.build_model(
            variant="nbegm", n_periods=4, **_SMALL
        ),
        build_params=nbegm_multi_target_toy.build_params,
        regime_name="alive_a",
        period=1,
    ),
    _Route(
        name="stochastic_node_kink",
        build_model=lambda: nbegm_stochastic_node_toy.build_model(
            variant="nbegm", n_periods=3, tax_kind="kink", **_SMALL
        ),
        build_params=nbegm_stochastic_node_toy.build_params,
        period=0,
    ),
    _Route(
        name="stochastic_node_kink_with_kind",
        build_model=lambda: nbegm_stochastic_node_toy.build_model(
            variant="nbegm", n_periods=3, tax_kind="kink", with_kind=True, **_SMALL
        ),
        build_params=lambda: nbegm_stochastic_node_toy.build_params(with_kind=True),
        period=0,
    ),
    _Route(
        name="brute_child",
        build_model=lambda: nbegm_brute_child_toy.build_model(
            young_variant="nbegm", **_SMALL
        ),
        build_params=nbegm_brute_child_toy.build_params,
        regime_name="young",
        period=0,
    ),
    _Route(
        name="ces_utility_kink",
        build_model=lambda: nbegm_ces_utility_toy.build_model(
            variant="nbegm",
            breakpoint_kind="continuous_kink",
            n_periods=3,
            n_wage=3,
            **_SMALL,
        ),
        build_params=lambda: nbegm_ces_utility_toy.build_params(
            breakpoint_kind="continuous_kink"
        ),
    ),
    _Route(
        name="ces_utility_jump",
        build_model=lambda: nbegm_ces_utility_toy.build_model(
            variant="nbegm", breakpoint_kind="jump", n_periods=3, n_wage=3, **_SMALL
        ),
        build_params=lambda: nbegm_ces_utility_toy.build_params(breakpoint_kind="jump"),
    ),
    _Route(
        name="jump_ride_along",
        build_model=lambda: nbegm_jump_ride_along_toy.build_model(
            variant="nbegm", n_periods=3, **_SMALL
        ),
        build_params=nbegm_jump_ride_along_toy.build_params,
    ),
    _Route(
        name="jump_ride_along_bridged",
        build_model=lambda: nbegm_jump_ride_along_toy.build_model(
            variant="nbegm", n_periods=3, jump_read="bridged", **_SMALL
        ),
        build_params=nbegm_jump_ride_along_toy.build_params,
    ),
    _Route(
        name="continuous_ride_along",
        build_model=lambda: nbegm_continuous_ride_along_toy.build_model(
            variant="nbegm", n_periods=3, n_wage=3, **_SMALL
        ),
        build_params=nbegm_continuous_ride_along_toy.build_params,
    ),
    _Route(
        name="indexed_threshold",
        build_model=lambda: nbegm_indexed_threshold_toy.build_model(
            variant="nbegm", n_periods=3, **_SMALL
        ),
        build_params=nbegm_indexed_threshold_toy.build_params,
    ),
    _Route(
        name="mappingleaf_threshold",
        build_model=lambda: nbegm_mappingleaf_threshold_toy.build_model(
            variant="nbegm", n_periods=3, **_SMALL
        ),
        build_params=nbegm_mappingleaf_threshold_toy.build_params,
    ),
    _Route(
        name="multi_source_jump",
        build_model=lambda: nbegm_multi_source_jump_toy.build_model(
            variant="nbegm", n_periods=3, **_SMALL
        ),
        build_params=nbegm_multi_source_jump_toy.build_params,
    ),
    _Route(
        name="stochastic_node_jump",
        build_model=lambda: nbegm_stochastic_node_toy.build_model(
            variant="nbegm", n_periods=3, tax_kind="jump", **_SMALL
        ),
        build_params=lambda: nbegm_stochastic_node_toy.build_params(tax_lump=1.0),
        period=0,
    ),
    _Route(
        name="ride_discrete",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, **_SMALL
        ),
        build_params=nbegm_ride_discrete_toy.build_params,
    ),
    _Route(
        name="ride_discrete_action_in_costate",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, action_in_costate=True, **_SMALL
        ),
        build_params=nbegm_ride_discrete_toy.build_params,
    ),
    _Route(
        name="ride_discrete_action_in_utility",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, action_in_utility=True, **_SMALL
        ),
        build_params=nbegm_ride_discrete_toy.build_params,
    ),
    _Route(
        name="ride_discrete_action_in_regime_transition",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, action_in_regime_transition=True, **_SMALL
        ),
        build_params=nbegm_ride_discrete_toy.build_params,
    ),
    _Route(
        name="ride_discrete_jump_schedule",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, jump_schedule=True, **_SMALL
        ),
        build_params=lambda: nbegm_ride_discrete_toy.build_params(jump_schedule=True),
    ),
    _Route(
        name="ride_discrete_action_in_liquid_law",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, action_in_liquid_law=True, **_SMALL
        ),
        build_params=lambda: nbegm_ride_discrete_toy.build_params(
            action_in_liquid_law=True
        ),
    ),
    _Route(
        name="ride_discrete_action_in_schedule_variable",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, action_in_schedule_variable=True, **_SMALL
        ),
        build_params=lambda: nbegm_ride_discrete_toy.build_params(
            action_in_schedule_variable=True
        ),
    ),
    _Route(
        name="ride_discrete_costate_reads_liquid_piecewise",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm",
            n_periods=3,
            costate_reads_liquid=True,
            costate_smooth=False,
            **_SMALL,
        ),
        build_params=nbegm_ride_discrete_toy.build_params,
    ),
    _Route(
        name="ride_discrete_transition_reads_liquid",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, transition_reads_liquid=True, **_SMALL
        ),
        build_params=nbegm_ride_discrete_toy.build_params,
    ),
    _Route(
        name="ride_discrete_schedule_variable_with_interval_continuation",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm",
            n_periods=3,
            action_in_schedule_variable=True,
            costate_reads_liquid=True,
            **_SMALL,
        ),
        build_params=lambda: nbegm_ride_discrete_toy.build_params(
            action_in_schedule_variable=True
        ),
    ),
    _Route(
        name="ride_discrete_action_in_costate_with_jump_schedule",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm",
            n_periods=3,
            action_in_costate=True,
            jump_schedule=True,
            **_SMALL,
        ),
        build_params=lambda: nbegm_ride_discrete_toy.build_params(jump_schedule=True),
    ),
    _Route(
        name="ride_discrete_action_in_health_transition",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, action_in_health_transition=True, **_SMALL
        ),
        build_params=nbegm_ride_discrete_toy.build_params,
        period=0,
    ),
    _Route(
        name="ride_discrete_action_in_discount",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, action_in_discount=True, **_SMALL
        ),
        build_params=lambda: nbegm_ride_discrete_toy.build_params(
            action_in_discount=True
        ),
        period=0,
    ),
    _Route(
        name="ride_discrete_action_in_all_channels",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm",
            n_periods=3,
            action_in_costate=True,
            action_in_liquid_law=True,
            action_in_utility=True,
            **_SMALL,
        ),
        build_params=lambda: nbegm_ride_discrete_toy.build_params(
            action_in_liquid_law=True
        ),
    ),
    _Route(
        name="multi_discrete",
        build_model=lambda: nbegm_multi_discrete_toy.build_model(
            variant="nbegm", n_actions=2, n_periods=3, **_SMALL
        ),
        build_params=lambda: nbegm_multi_discrete_toy.build_params(n_actions=2),
    ),
    _Route(
        name="next_asset_cliff",
        build_model=lambda: nbegm_next_asset_cliff_toy.build_model(
            variant="nbegm", n_periods=3, **_SMALL
        ),
        build_params=nbegm_next_asset_cliff_toy.build_params,
    ),
    _Route(
        name="epstein_zin",
        build_model=lambda: epstein_zin_model._build_model(
            solver=NBEGM(
                savings_grid=epstein_zin_model._SAVINGS_GRID,
                envelope_arithmetic="ordinary",
            )
        ),
        build_params=lambda: epstein_zin_model._PARAMS,
        period=0,
    ),
)

_NNBEGM_ROUTES = (
    _Route(
        name="n_nbegm",
        build_model=lambda: n_nbegm_toy.build_model(variant="n_nbegm"),
        build_params=lambda: {"discount_factor": 0.95},
    ),
    _Route(
        name="n_nbegm_discrete",
        build_model=lambda: n_nbegm_discrete_toy.build_model(variant="n_nbegm"),
        build_params=lambda: {"discount_factor": 0.95, "alive": {"premium": 1.0}},
    ),
)


def _tolerance() -> float:
    """Relative tolerance for a quantity computed by two different routes."""
    return 100.0 * 10.0**-DECIMAL_PRECISION


def _assert_agrees(*, got: Any, expected: Any, label: str) -> None:
    got_arr = np.asarray(got, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    assert got_arr.shape == expected_arr.shape, label
    scale = np.max(np.abs(expected_arr[np.isfinite(expected_arr)]), initial=1.0)
    np.testing.assert_allclose(
        got_arr,
        expected_arr,
        rtol=_tolerance(),
        atol=_tolerance() * scale,
        err_msg=label,
    )


@pytest.mark.parametrize("route", _ROUTES, ids=lambda route: route.name)
def test_direct_oracle_matches_the_tiled_core(route: _Route) -> None:
    """Value, carry rows, and consumption agree with the scalar oracle."""
    kernel, context = ride_along_kernel(
        model=route.build_model(),
        params=route.build_params(),
        regime_name=route.regime_name,
        period=route.period,
    )
    assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
    _assert_kernel_agrees_with_oracle(kernel=kernel, context=context)


@pytest.mark.parametrize("route", _NNBEGM_ROUTES, ids=lambda route: route.name)
def test_direct_oracle_covers_the_nnbegm_inner_contexts(route: _Route) -> None:
    """The nested solver's keeper and adjuster kernels agree with the oracle.

    Both inner kernels are ride-along NB-EGM kernels; the adjuster is checked
    with the outer post-decision value bound at the first and the middle outer
    node, exactly as the nested solver binds it.
    """
    contexts = nnbegm_inner_contexts(
        model=route.build_model(),
        params=route.build_params(),
        regime_name=route.regime_name,
        period=route.period,
    )
    assert {label.split("@")[0] for label, _, _ in contexts} == {"keeper", "adjuster"}
    for label, kernel, context in contexts:
        assert isinstance(kernel, _RideAlongNBEGMPeriodKernel), label
        _assert_kernel_agrees_with_oracle(kernel=kernel, context=context)


def _assert_agrees_up_to_ties(
    *, got: Any, expected: np.ndarray, alternatives: np.ndarray, label: str
) -> None:
    """A winner-dependent channel agrees, or is that of a candidate tied in value.

    Two candidates whose values differ by less than the value tolerance are a
    discrete decision the working format cannot separate, so at such a state the
    kernel may publish either candidate's consumption and marginal.
    """
    got_arr = np.asarray(got, dtype=np.float64)
    assert got_arr.shape == expected.shape, label
    tolerance = _tolerance() * max(1.0, float(np.max(np.abs(expected))))
    mismatched = np.argwhere(np.abs(got_arr - expected) > tolerance)
    for state in map(tuple, mismatched):
        tied = alternatives[state]
        assert np.any(np.abs(tied - got_arr[state]) <= tolerance), (
            label,
            state,
            got_arr[state],
            tied,
        )


def _assert_kernel_agrees_with_oracle(*, kernel: Any, context: dict[str, Any]) -> None:
    outputs = run_production_kernel(kernel=kernel, context=context)
    value, carry, policy, *banks = outputs
    oracle = direct_oracle_period(
        kernel=kernel,
        context=context,
        tie_tolerance=_tolerance() * max(1.0, float(np.max(np.abs(np.asarray(value))))),
    )
    _assert_agrees(got=value, expected=oracle.value, label="value")
    _assert_agrees_up_to_ties(
        got=policy,
        expected=oracle.policy,
        alternatives=oracle.policy_alternatives,
        label="policy",
    )
    _assert_agrees(
        got=carry.endog_grid, expected=oracle.carry.endog_grid, label="endog"
    )
    _assert_agrees(got=carry.value, expected=oracle.carry.value, label="carry value")
    _assert_agrees_up_to_ties(
        got=carry.marginal_utility,
        expected=oracle.carry.marginal_utility,
        alternatives=oracle.carry_marginal_alternatives,
        label="carry marginal",
    )
    assert (carry.breakpoints is None) == (oracle.carry.breakpoints is None)
    if carry.breakpoints is not None:
        _assert_agrees(
            got=carry.breakpoints,
            expected=oracle.carry.breakpoints,
            label="carry breakpoints",
        )
    assert (carry.policy is None) == (oracle.carry.policy is None)
    if carry.policy is not None:
        _assert_agrees(
            got=carry.policy, expected=oracle.carry.policy, label="carry policy"
        )
    assert bool(banks) == (oracle.branch_value is not None)
    if banks:
        branch_value, branch_inner_action = banks
        _assert_agrees(
            got=branch_value, expected=oracle.branch_value, label="branch value"
        )
        assert oracle.branch_inner_action is not None
        assert oracle.branch_inner_action_alternatives is not None
        _assert_agrees_up_to_ties(
            got=branch_inner_action,
            expected=oracle.branch_inner_action,
            alternatives=oracle.branch_inner_action_alternatives,
            label="branch inner action",
        )


def test_direct_oracle_is_independent_of_the_production_expectation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oracle runs with the production continuation read and envelope disabled."""

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        msg = "the oracle must not reach production solver code"
        raise AssertionError(msg)

    kernel, context = ride_along_kernel(
        model=nbegm_ride_along_toy.build_model(variant="nbegm", n_periods=3, **_SMALL),
        params=nbegm_ride_along_toy.build_params(),
    )
    monkeypatch.setattr(continuation_module, "bind_continuation", refuse)
    monkeypatch.setattr(query_module, "envelope_at_query", refuse)
    oracle = direct_oracle_period(kernel=kernel, context=context)
    assert np.all(np.isfinite(oracle.value))


@pytest.mark.parametrize("dropped", ["target", "stochastic_node", "cliff_family"])
def test_direct_oracle_detects_a_dropped_target_and_a_dropped_stochastic_node(
    dropped: str,
) -> None:
    """Removing one term or candidate family from the oracle makes it disagree.

    The positive control for the agreement test: the oracle is sensitive to the
    parts of the expectation and of the candidate set it claims to check.
    """
    if dropped == "cliff_family":
        # A savings grid coarse enough that no node sits near the child's cliff,
        # so the save-to-cliff targets carry a payoff the nodes alone forfeit.
        model = nbegm_jump_ride_along_toy.build_model(
            variant="nbegm", n_liquid=24, n_savings=12, n_consumption=8
        )
        params = nbegm_jump_ride_along_toy.build_params()
        kernel, context = ride_along_kernel(model=model, params=params, period=1)
        assert kernel.cliff_candidates
        mutated_kernel = replace(kernel, cliff_candidates=False)
        value, *_rest = run_production_kernel(kernel=kernel, context=context)
        oracle = direct_oracle_period(kernel=mutated_kernel, context=context)
        with pytest.raises(AssertionError):
            _assert_agrees(got=value, expected=oracle.value, label="value")
        return
    if dropped == "target":
        model = nbegm_multi_target_toy.build_model(
            variant="nbegm", n_periods=4, **_SMALL
        )
        params = nbegm_multi_target_toy.build_params()
        kernel, context = ride_along_kernel(
            model=model, params=params, regime_name="alive_a", period=1
        )
        plan = kernel.continuation_plan
        assert len(plan.stateful_targets) > 1
        mutated_plan = replace(plan, stateful_targets=plan.stateful_targets[:1])
    else:
        model = nbegm_stochastic_node_toy.build_model(
            variant="nbegm", n_periods=3, tax_kind="kink", **_SMALL
        )
        params = nbegm_stochastic_node_toy.build_params()
        kernel, context = ride_along_kernel(model=model, params=params, period=0)
        plan = kernel.continuation_plan
        (target,) = plan.stateful_targets
        read = plan.child_reads[target]
        assert read.stochastic_node_values
        mutated_read = replace(
            read,
            stochastic_node_values=tuple(
                values[:-1] for values in read.stochastic_node_values
            ),
        )
        mutated_plan = replace(plan, child_reads={target: mutated_read})
    mutated_kernel = replace(kernel, continuation_plan=mutated_plan)
    value, *_rest = run_production_kernel(kernel=kernel, context=context)
    oracle = direct_oracle_period(kernel=mutated_kernel, context=context)
    with pytest.raises(AssertionError):
        _assert_agrees(got=value, expected=oracle.value, label="value")


_SOURCE = Path(__file__).read_text()
_TABLES = _SOURCE[_SOURCE.index("_ROUTES = (") : _SOURCE.index("def _tolerance")]


def _declared_routes(*, source: str = _SOURCE) -> dict[str, routes.RouteIdentity]:
    return {
        **routes.declared_route_identities(
            source=source, table_name="_ROUTES", context=routes.RIDE_ALONG
        ),
        **routes.declared_route_identities(
            source=source, table_name="_NNBEGM_ROUTES", context=routes.NNBEGM_INNER
        ),
    }


def test_the_route_tables_declare_exactly_the_supported_route_identities() -> None:
    """Every supported route is declared once, at its identity, and nothing else.

    A route's identity is its test-model module, semantic model and parameter
    keywords, regime, period, and context; grid sizes are not part of it. A
    missing, extra, drifted, or duplicated route is a census failure.
    """
    discrepancies = routes.census_discrepancies(
        declared=_declared_routes(), supported=routes.SUPPORTED_ROUTES
    )

    assert discrepancies == ()


def test_the_declared_route_names_are_the_route_tables_names() -> None:
    """The identities are read from the same tables the oracle parametrizes over."""
    declared = _declared_routes()

    assert set(declared) == {route.name for route in (*_ROUTES, *_NNBEGM_ROUTES)}


@pytest.mark.parametrize(
    ("mutation", "expected_kind"),
    [
        (
            (
                """    _Route(
        name="ride_discrete_action_in_costate",
        build_model=lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, action_in_costate=True, **_SMALL
        ),
        build_params=nbegm_ride_discrete_toy.build_params,
    ),
""",
                "",
            ),
            "missing route 'ride_discrete_action_in_costate'",
        ),
        (
            (
                "n_periods=3, action_in_costate=True, **_SMALL",
                "n_periods=3, action_in_utility=True, **_SMALL",
            ),
            "drifted route 'ride_discrete_action_in_costate'",
        ),
        (
            (
                "n_periods=3, action_in_costate=True, **_SMALL",
                "n_periods=3, action_in_utility=True, **_SMALL",
            ),
            "duplicate identity",
        ),
        (
            (
                '        regime_name="alive_a",\n        period=1,\n',
                '        regime_name="alive_a",\n        period=2,\n',
            ),
            "drifted route 'multi_target'",
        ),
        (
            (
                'name="ride_discrete_jump_schedule",',
                'name="ride_discrete_jump_schedule_renamed",',
            ),
            "unsupported route 'ride_discrete_jump_schedule_renamed'",
        ),
    ],
    ids=["deleted", "reflagged", "duplicated", "reperioded", "renamed"],
)
def test_the_route_census_detects_a_mutated_route_table(
    *, mutation: tuple[str, str], expected_kind: str
) -> None:
    """Deleting, re-flagging, duplicating, re-perioding, or renaming a route fails.

    The census must notice a deleted route even while another route of the same
    module remains, and a changed semantic flag even when the name is unchanged.
    """
    old, new = mutation
    assert _TABLES.count(old) == 1, "the mutation must hit exactly one table site"
    mutated = _SOURCE.replace(_TABLES, _TABLES.replace(old, new))

    discrepancies = routes.census_discrepancies(
        declared=_declared_routes(source=mutated), supported=routes.SUPPORTED_ROUTES
    )

    assert any(expected_kind in item for item in discrepancies), discrepancies


def test_an_unreadable_route_table_is_a_census_failure_not_an_empty_census() -> None:
    """A route whose builder is hidden from the census fails loudly."""
    old = (
        "build_model=lambda: nbegm_ride_discrete_toy.build_model(\n"
        '            variant="nbegm", n_periods=3, action_in_costate=True, **_SMALL\n'
        "        ),"
    )
    assert _TABLES.count(old) == 1
    mutated = _SOURCE.replace(
        _TABLES, _TABLES.replace(old, "build_model=_hidden_builder,")
    )

    with pytest.raises(TypeError, match="statically visible builder"):
        _declared_routes(source=mutated)


@pytest.mark.parametrize(
    ("relpath", "function", "flags"),
    routes.POSITIVE_WITNESSES,
    ids=[relpath.rsplit("/", 1)[-1] for relpath, _, _ in routes.POSITIVE_WITNESSES],
)
def test_every_positive_ride_discrete_witness_has_a_route_with_its_flags(
    *, relpath: str, function: str, flags: Mapping[str, object]
) -> None:
    """A production-path test of the ride-discrete toy is covered by the oracle.

    The witness test's builder call must pass exactly the flags recorded for it,
    and a supported ride-along route of the ride-discrete toy must carry exactly
    those flags beyond the variant and period.
    """
    source = (Path(__file__).parents[2] / relpath).read_text()
    built = {
        key: value
        for key, value in routes.witness_flags(source=source, function=function).items()
        if key not in {"variant", "n_periods"}
    }
    supported_flag_sets = {
        frozenset(
            (key, value)
            for key, value in identity.model_kwargs
            if key not in {"variant", "n_periods"}
        )
        for identity in routes.SUPPORTED_ROUTES.values()
        if identity.module == "nbegm_ride_discrete_toy"
    }

    assert built == dict(flags)
    assert frozenset(flags.items()) in supported_flag_sets


def test_every_test_model_module_that_builds_a_ride_along_kernel_is_routed() -> None:
    """A new toy that reaches the ride-along kernel must be given an oracle route.

    Every `nbegm_*` test-model module is built at the oracle's grid sizes; a
    build error propagates rather than reading as "no route needed".
    """
    routed_modules = {identity.module for identity in routes.SUPPORTED_ROUTES.values()}
    ride_along_modules = {
        module_info.name
        for module_info in pkgutil.iter_modules(test_models_package.__path__)
        if module_info.name.startswith("nbegm_")
        and _builds_a_ride_along_kernel(
            importlib.import_module(f"tests.test_models.{module_info.name}")
        )
    }

    assert ride_along_modules <= routed_modules, sorted(
        ride_along_modules - routed_modules
    )


def _builds_a_ride_along_kernel(module: Any) -> bool:
    build_model = getattr(module, "build_model", None)
    if build_model is None:
        return False
    parameters = inspect.signature(build_model).parameters
    kwargs: dict[str, Any] = {
        name: size for name, size in _SMALL.items() if name in parameters
    }
    if "variant" in parameters:
        kwargs["variant"] = "nbegm"
    if "young_variant" in parameters:
        kwargs["young_variant"] = "nbegm"
    model = build_model(**kwargs)
    return any(
        isinstance(kernel, _RideAlongNBEGMPeriodKernel)
        for regime in model._regimes.values()
        for kernel in regime.solution.period_kernels.values()
    )
