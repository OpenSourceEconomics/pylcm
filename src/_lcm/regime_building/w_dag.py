"""Construction of `W_kwargs` for the Koopmans aggregator.

A regime's `koopmans_aggregator` — `W(utility, CE, discount_factor)` by
default, or any user-supplied one — may declare parameters that are not
states/actions/user-params but are outputs of regime functions registered
under the same name (e.g. a `discount_factor` DAG function that indexes a
per-type Series by a `pref_type` state).

This module exposes:

- `_get_build_W_kwargs`: factory that returns a closure computing
  `W_kwargs` from `states_actions_params` at runtime. Used by Q_and_F.
- `get_dag_targets_consumed_by_W`: names of regime functions whose
  outputs W consumes. Used by `_validate_all_variables_used` as
  reachability targets so states consumed only via W's DAG
  dependencies count as "used".
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, no_type_check

from dags import concatenate_functions

from _lcm.typing import FunctionName
from _lcm.utils.functools import get_union_of_args
from lcm.typing import UserFunction


def get_dag_targets_consumed_by_W(
    *,
    functions: Mapping[FunctionName, Callable[..., Any]],
    koopmans_aggregator: UserFunction | None,
) -> frozenset[FunctionName]:
    """Return names of regime functions whose outputs W consumes.

    These are W's signature parameters that are also regime functions,
    minus `utility` (wired directly from `U_and_F`) and `feasibility`
    (never a legitimate W input). Empty in terminal regimes, which have
    no aggregator.

    Args:
        functions: Mapping of regime function names to callables.
        koopmans_aggregator: The regime's Bellman aggregator, or `None` in a
            terminal regime.

    Returns:
        Frozenset of regime function names whose outputs are routed
        into W at runtime.

    """
    if koopmans_aggregator is None:
        return frozenset()
    W_accepted_params = _accepted_params(koopmans_aggregator)
    return W_accepted_params & set(functions) - {"utility", "feasibility"}


def _get_build_W_kwargs(
    *,
    functions: Mapping[FunctionName, Callable[..., Any]],
    koopmans_aggregator: UserFunction,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Return a closure that builds `W_kwargs` from `states_actions_params`.

    W's signature parameters come from two disjoint pools:

    1. `states_actions_params` — states, actions, and flat user params
       — passed through verbatim for names W accepts directly.
    2. Outputs of regime functions whose names match W's params —
       computed at runtime via a compiled DAG.

    The returned closure assembles both into a single dict for
    `**W_kwargs` at the Bellman step.

    Args:
        functions: Regime functions (user and generated).
        koopmans_aggregator: The regime's Bellman aggregator.

    Returns:
        Callable mapping `states_actions_params` to the complete
        `W_kwargs` dict.

    """
    W_accepted_params = _accepted_params(koopmans_aggregator)
    dag_targets = get_dag_targets_consumed_by_W(
        functions=functions, koopmans_aggregator=koopmans_aggregator
    )
    passthrough = W_accepted_params - dag_targets

    if not dag_targets:
        return _BuildWKwargs(passthrough=passthrough, dag_func=None)

    dag_func = concatenate_functions(
        functions=dict(functions),
        targets=sorted(dag_targets),
        return_type="dict",
        enforce_signature=False,
    )
    return _BuildWKwargs(passthrough=passthrough, dag_func=dag_func)


@dataclass(frozen=True, kw_only=True, eq=False)
class _BuildWKwargs:
    """Assemble `W_kwargs` from `states_actions_params` at the Bellman step."""

    passthrough: frozenset[FunctionName]
    """Names W accepts directly from states, actions, and flat user params."""
    dag_func: Callable[..., dict[str, Any]] | None
    """Compiled DAG computing W's regime-function inputs, or `None` if it has none."""

    # The kernel is traced with whatever leaves its caller supplies -- tracers,
    # Python scalars, arrays of either integer width -- so its annotations
    # document the contract and are not enforced at call time.
    @no_type_check
    def __call__(self, states_actions_params: Mapping[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in states_actions_params.items() if k in self.passthrough}
        if self.dag_func is not None:
            out |= self.dag_func(**states_actions_params)
        return out


def _accepted_params(koopmans_aggregator: UserFunction) -> frozenset[FunctionName]:
    """Return W's signature parameters other than the two it is always handed."""
    return frozenset(get_union_of_args([koopmans_aggregator]) - {"utility", "CE"})
