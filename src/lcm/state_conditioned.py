"""`StateConditioned`: a process parameter that varies with a discrete state."""

from collections.abc import Mapping
from dataclasses import dataclass

from lcm.typing import StateName


@dataclass(frozen=True, kw_only=True)
class StateConditioned:
    r"""A shock parameter conditioned on a discrete state of the same regime.

    Write it in place of the parameter it conditions. Use it when the size of a shock
    depends on where the subject currently is — the variance of the earnings innovation
    differing by employment status, with
    $\sigma_\text{employed} < \sigma_\text{unemployed}$:

        NormalIIDProcess(
            n_points=7,
            gauss_hermite=False,
            mu=0.0,
            n_std=3.0,
            sigma=StateConditioned(
                on="employment_status",
                by={"employed": 0.2, "unemployed": 0.5},
            ),
        )

    Writing $s_t$ for the time-$t$ value of `on` and $\sigma_{s_t}$ for `by[s_t]`, an
    AR(1) process transitions as

    ```{math}
    y_{t+1} \mid y_t, s_t \sim N(\mu + \rho y_t,\ \sigma_{s_t}^2),
    ```

    an IID process dropping the $\rho y_t$ term. The row is binned on one set of nodes,
    placed from the widest value in `by` — only $\sigma_{s_t}$ varies with $s_t$.
    Standing where the scalar would go, it says which parameter is conditioned and
    leaves no way to give that parameter twice.

    The conditioning value is dated $t$, so the variance of the innovation realized
    between $t$ and $t+1$ is set by where the subject is at $t$. The mapping it is read
    against is the *target* regime's declaration, so a process declared with different
    values in the source and target regimes takes the target's.

    `on` must name a `DiscreteGrid` state the regime carries. Build that grid from the
    same `@categorical` class in every regime that carries it: the per-category value is
    selected by the category's integer code, so regimes that encode the same categories
    in a different order would draw each other's values.

    All of it is fixed at build time. Neither `by` nor the process's own parameters
    reach the params template, so none of them can be estimated.

    Defined in this leaf module so the engine's process base class can annotate the
    field without an import cycle; re-exported from `lcm.processes`.
    """

    on: StateName
    """Name of the `DiscreteGrid` state whose time-$t$ value selects the parameter."""

    by: Mapping[str, float]
    """Mapping of that state's category names to the value used for each."""
