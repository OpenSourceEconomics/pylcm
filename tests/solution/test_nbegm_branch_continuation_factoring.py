"""The ride-along core reads one continuation per continuation-equivalence class.

A discrete action's branches share a continuation whenever the action reaches
nothing the continuation read consumes. The channels that can make the read
branch-dependent are the regime transition, a stateful target's law of motion
(including the regime's own law the save-to-cliff targets invert), a child's
resources, the discount factor, and, on a per-interval read, a schedule variable
the breakpoints are derived from. Branches that agree on every action reaching
one of those channels form one class; the core evaluates the continuation once
per class and gathers it per branch, so an action that only shifts the current
budget and utility costs one continuation read however many branches it
declares. The class partition is a build-time static, visible on the kernel, and
the factored read agrees with the scalar oracle on every branch route.

The routes are read at the first period of the ride-along toy, where both the
continuing and the terminal target are reachable; at the last active period only
the terminal target is, and a co-state law toward the continuing target is not a
channel the read consumes.
"""

import inspect
from collections.abc import Callable
from typing import Any

import pytest

from _lcm.solution import nbegm as nbegm_module
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.solution.test_nbegm_direct_oracle import (
    _assert_kernel_agrees_with_oracle,
)
from tests.test_models import nbegm_multi_discrete_toy, nbegm_ride_discrete_toy

_SMALL: dict[str, Any] = {"n_liquid": 12, "n_savings": 16, "n_consumption": 24}
_PERIOD = 0

_Route = tuple[Callable[[], Any], Callable[[], Any]]

_PARAMS_FLAGS = frozenset(
    {"jump_schedule", "action_in_liquid_law", "action_in_schedule_variable"}
)


def _ride_discrete(**variant: bool) -> _Route:
    """Model and params builders for one variant of the ride-along discrete toy."""
    params_flags = {name: on for name, on in variant.items() if name in _PARAMS_FLAGS}
    return (
        lambda: nbegm_ride_discrete_toy.build_model(
            variant="nbegm", n_periods=3, **_SMALL, **variant
        ),
        lambda: nbegm_ride_discrete_toy.build_params(**params_flags),
    )


_BUDGET_ONLY: dict[str, _Route] = {
    "ride_discrete": _ride_discrete(),
    "action_in_utility": _ride_discrete(action_in_utility=True),
    "jump_schedule": _ride_discrete(jump_schedule=True),
    "action_in_schedule_variable": _ride_discrete(action_in_schedule_variable=True),
    "multi_discrete": (
        lambda: nbegm_multi_discrete_toy.build_model(
            variant="nbegm", n_actions=2, n_periods=3, **_SMALL
        ),
        lambda: nbegm_multi_discrete_toy.build_params(n_actions=2),
    ),
}

_CONTINUATION_FEEDING: dict[str, _Route] = {
    "action_in_regime_transition": _ride_discrete(action_in_regime_transition=True),
    "action_in_costate": _ride_discrete(action_in_costate=True),
    "action_in_liquid_law": _ride_discrete(action_in_liquid_law=True),
    "action_in_interval_schedule_variable": _ride_discrete(
        action_in_schedule_variable=True, costate_reads_liquid=True
    ),
}


def _kernel(route: _Route) -> tuple[Any, Any]:
    build_model, build_params = route
    kernel, context = ride_along_kernel(
        model=build_model(),
        params=build_params(),
        regime_name="alive",
        period=_PERIOD,
    )
    assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
    return kernel, context


@pytest.mark.parametrize("route", list(_BUDGET_ONLY))
def test_budget_only_actions_share_one_continuation_class(*, route: str):
    kernel, _ = _kernel(_BUDGET_ONLY[route])
    statics = kernel.statics

    assert statics.n_action_branches > 1
    assert statics.continuation_action_names == ()
    assert statics.n_continuation_classes == 1
    assert statics.continuation_representatives == (0,)
    assert statics.continuation_class_of_branch == (0,) * statics.n_action_branches


@pytest.mark.parametrize("route", list(_CONTINUATION_FEEDING))
def test_an_action_reaching_the_continuation_splits_the_classes(*, route: str):
    kernel, _ = _kernel(_CONTINUATION_FEEDING[route])
    statics = kernel.statics

    assert statics.continuation_action_names == ("buy_private",)
    assert statics.n_continuation_classes == statics.n_action_branches == 2
    assert statics.continuation_representatives == (0, 1)
    assert statics.continuation_class_of_branch == (0, 1)


def _reads(*names: str) -> Callable[..., Any]:
    """A callable whose signature names exactly `names`, keyword-only."""
    parameters = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in names
    ]

    def func(**kwargs: Any) -> Any:
        return kwargs

    func.__signature__ = inspect.Signature(parameters)  # ty: ignore[unresolved-attribute]
    return func


_QUIET: dict[str, Any] = {
    "regime_transition": _reads("age"),
    "target_laws": {"alive": _reads("savings", "kind"), "dead": _reads("savings")},
    "target_resources_arg_names": {
        "alive": frozenset({"liquid", "income"}),
        "dead": frozenset({"liquid"}),
    },
    "discount_factor_dag": _reads("kind"),
    "interval_schedule_dags": (_reads("liquid", "kind"),),
}

_CHANNEL_CASES: dict[str, dict[str, Any]] = {
    "regime_transition": {"regime_transition": _reads("age", "work")},
    "target_law": {
        "target_laws": {
            **_QUIET["target_laws"],
            "dead": _reads("savings", "work"),
        },
    },
    "cliff_target_map": {
        "target_laws": {
            **_QUIET["target_laws"],
            "alive": _reads("savings", "kind", "work"),
        },
    },
    "child_resources": {
        "target_resources_arg_names": {
            **_QUIET["target_resources_arg_names"],
            "alive": frozenset({"liquid", "income", "work"}),
        },
    },
    "discount_factor": {"discount_factor_dag": _reads("kind", "work")},
    "interval_schedule_variable": {
        "interval_schedule_dags": (_reads("liquid", "kind", "work"),),
    },
}


def _classify(overrides: dict[str, Any]) -> tuple[str, ...]:
    return nbegm_module._continuation_action_names(
        **{**_QUIET, **overrides}, action_names=("work", "claim")
    )


def test_an_action_reaching_no_channel_is_not_a_continuation_action():
    assert _classify({}) == ()


@pytest.mark.parametrize("channel", list(_CHANNEL_CASES))
def test_each_channel_makes_the_action_a_continuation_action(*, channel: str):
    assert _classify(_CHANNEL_CASES[channel]) == ("work",)


_PAYOFF_BEARING_ROUTES = (
    "action_in_regime_transition",
    "action_in_costate",
    "action_in_liquid_law",
)


@pytest.mark.parametrize("route", _PAYOFF_BEARING_ROUTES)
def test_ignoring_the_channels_makes_the_core_disagree_with_the_oracle(
    *, route: str, monkeypatch: pytest.MonkeyPatch
):
    """Positive control: the factoring is load-bearing on every feeding route.

    The classification is a reachability statement, so it also splits the
    per-interval schedule route; there the toy's tracker law switches at a liquid
    level both branches' interval midpoints sit on the same side of, so the rows
    the two branches read coincide and sharing them is not observable. The
    routes here carry the action into the payoff: the survival probability, the
    streak the utility pays on, and the next liquid the terminal value reads.
    """
    monkeypatch.setattr(
        nbegm_module, "_continuation_action_names", lambda **_kwargs: ()
    )
    kernel, context = _kernel(_CONTINUATION_FEEDING[route])
    assert kernel.statics.n_continuation_classes == 1

    with pytest.raises(AssertionError):
        _assert_kernel_agrees_with_oracle(kernel=kernel, context=context)


@pytest.mark.parametrize("route", [*_BUDGET_ONLY, *_CONTINUATION_FEEDING])
def test_the_factored_read_matches_the_direct_oracle_on_every_branch_route(
    *, route: str
):
    kernel, context = _kernel({**_BUDGET_ONLY, **_CONTINUATION_FEEDING}[route])

    _assert_kernel_agrees_with_oracle(kernel=kernel, context=context)


def test_the_class_partition_is_derived_from_the_continuation_actions_alone():
    """The partition keys each branch by its continuation actions' codes only."""
    kernel, _ = _kernel(_BUDGET_ONLY["multi_discrete"])
    statics = kernel.statics
    continuation_positions = tuple(
        statics.discrete_action_names.index(name)
        for name in statics.continuation_action_names
    )
    keys = [
        tuple(codes[position] for position in continuation_positions)
        for codes in statics.discrete_action_codes
    ]
    distinct = list(dict.fromkeys(keys))

    assert statics.n_continuation_classes == len(distinct)
    assert statics.continuation_class_of_branch == tuple(
        distinct.index(key) for key in keys
    )
    assert statics.continuation_representatives == tuple(
        keys.index(key) for key in distinct
    )
