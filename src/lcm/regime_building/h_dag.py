"""Construction of `H_kwargs` for the Bellman aggregator.

The default Bellman aggregator `H(utility, E_next_V, discount_factor)` —
and any user-supplied H — may declare parameters that are not
states/actions/user-params but are outputs of regime functions
registered under the same name (e.g. a `discount_factor` DAG function
that indexes a per-type Series by a `pref_type` state).

This module exposes:

- `_get_build_H_kwargs`: factory that returns a closure computing
  `H_kwargs` from `states_actions_params` at runtime. Used by Q_and_F.
- `get_dag_targets_consumed_by_H`: names of regime functions whose
  outputs H consumes. Used by `_validate_all_variables_used` as
  reachability targets so states consumed only via H's DAG
  dependencies count as "used".
"""

from collections.abc import Callable, Mapping
from typing import Any

from dags import concatenate_functions

from lcm.utils.functools import get_union_of_args


def get_dag_targets_consumed_by_H(
    functions: Mapping[str, Callable[..., Any]],
) -> frozenset[str]:
    """Return names of regime functions whose outputs H consumes.

    These are H's signature parameters that are also regime functions,
    minus `H`, `utility`, `feasibility` (H cannot consume its own
    output; `utility` is wired directly from `U_and_F`; `feasibility`
    is never a legitimate H input). Empty in terminal regimes, which
    have no `H`.

    Args:
        functions: Mapping of regime function names to callables.

    Returns:
        Frozenset of regime function names whose outputs are routed
        into H at runtime.

    """
    H = functions.get("H")
    if H is None:
        return frozenset()
    H_accepted_params = frozenset(get_union_of_args([H]) - {"utility", "E_next_V"})
    return H_accepted_params & set(functions) - {"H", "utility", "feasibility"}


def _get_build_H_kwargs(
    functions: Mapping[str, Callable[..., Any]],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Return a closure that builds `H_kwargs` from `states_actions_params`.

    H's signature parameters come from two disjoint pools:

    1. `states_actions_params` — states, actions, and flat user params
       — passed through verbatim for names H accepts directly.
    2. Outputs of regime functions whose names match H's params —
       computed at runtime via a compiled DAG.

    The returned closure assembles both into a single dict for
    `**H_kwargs` at the Bellman step.

    Args:
        functions: Regime functions (user and generated), including `H`.

    Returns:
        Callable mapping `states_actions_params` to the complete
        `H_kwargs` dict.

    """
    H = functions["H"]
    H_accepted_params = frozenset(get_union_of_args([H]) - {"utility", "E_next_V"})
    dag_targets = get_dag_targets_consumed_by_H(functions)
    passthrough = H_accepted_params - dag_targets

    if not dag_targets:

        def build(states_actions_params: Mapping[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in states_actions_params.items() if k in passthrough}

        return build

    dag_func = concatenate_functions(
        functions={k: v for k, v in functions.items() if k != "H"},
        targets=sorted(dag_targets),
        return_type="dict",
        enforce_signature=False,
    )

    def build(states_actions_params: Mapping[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in states_actions_params.items() if k in passthrough}
        out |= dag_func(**states_actions_params)
        return out

    return build
