"""H's DAG-target bookkeeping, shared between runtime and validation.

The default Bellman aggregator `H(utility, E_next_V, discount_factor)` —
and any user-supplied H — may declare parameters that are not
states/actions/user-params but are outputs of regime functions
registered under the same name (e.g. a `discount_factor` DAG function
that indexes a per-type Series by a `discount_type` state).

This module exposes:

- `get_h_accepted_params`: H's signature minus `utility` / `E_next_V`.
- `get_h_dag_target_names`: those H parameters that are *also* regime
  functions. Q_and_F compiles these into a runtime DAG;
  `_validate_all_variables_used` uses them as reachability targets so
  states consumed only via H's DAG dependencies count as "used".
"""

from collections.abc import Callable, Mapping
from typing import Any

from lcm.utils.functools import get_union_of_args


def get_h_accepted_params(
    functions: Mapping[str, Callable[..., Any]],
) -> frozenset[str]:
    """H's signature parameters, minus `utility` and `E_next_V`.

    Empty when the regime has no `H` (terminal regimes).

    Args:
        functions: Mapping of regime function names to callables (user
            and generated).

    Returns:
        Frozenset of parameter names H accepts beyond `utility` / `E_next_V`.

    """
    h_func = functions.get("H")
    if h_func is None:
        return frozenset()
    return frozenset(get_union_of_args([h_func]) - {"utility", "E_next_V"})


def get_h_dag_target_names(
    *,
    functions: Mapping[str, Callable[..., Any]],
    h_accepted_params: frozenset[str],
) -> frozenset[str]:
    """Names of regime functions whose outputs H consumes via the DAG.

    These are H's signature parameters that are also regime functions,
    minus `H`, `utility`, `feasibility` (H cannot consume its own
    output; `utility` is wired directly from `U_and_F`; `feasibility`
    is never a legitimate H input).

    Args:
        functions: Mapping of regime function names to callables (user
            and generated).
        h_accepted_params: Names H accepts beyond `utility` / `E_next_V`
            (typically the output of `get_h_accepted_params`).

    Returns:
        Frozenset of regime function names whose outputs are routed
        into H at runtime.

    """
    return frozenset(h_accepted_params) & set(functions) - {
        "H",
        "utility",
        "feasibility",
    }
