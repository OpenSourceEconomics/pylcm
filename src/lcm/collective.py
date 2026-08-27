"""Declarations a collective regime makes about its household decision."""

from collections.abc import Mapping
from dataclasses import dataclass

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm.typing import UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ParetoObjective:
    r"""The household's Pareto scalarization of its stakeholders' action values.

    A collective regime takes one action for everybody, and this declares how
    the stakeholders' action values are traded off in taking it:

    ```{math}
        a^*(x) = \arg\max_{a\,:\,F(x,a)} \sum_s \lambda_s(x)\, Q^s(x, a),
        \qquad V^s(x) = Q^s(x, a^*(x)).
    ```

    Declaring it rather than writing the sum as an ordinary function is what
    lets the engine own what a Pareto weight means: one per stakeholder, finite
    and non-negative, with a strictly positive total, normalized cell by cell,
    and multiplied in zero-safely — so a stakeholder carrying no weight cannot
    decide the household's choice through an admissible `-inf` of her own.

    Omit it (the regime's default) for equal weights.
    """

    weights: Mapping[str, UserFunction | float]
    r"""One weight $\lambda_s$ per stakeholder, keyed by stakeholder name.

    A `float` is a constant. A callable is an ordinary DAG function of the
    regime's states, its other functions, and free parameters — the parameters
    surface in `get_params_template()` under the regime's `pareto_objective`
    key, so a weight is estimated like anything else. A weight may not read an
    action: a weight that varies with the choice states a different objective
    per candidate, whose maximizer is a Pareto optimum of no fixed weighting.
    """

    normalization: str = "pointwise"
    """How the declared weights are turned into the weights actually used.

    - `"pointwise"` (the default) divides by the total at each cell, so the
      weights sum to one wherever the objective is evaluated and a
      state-dependent declaration keeps one scale across the grid.
    - `"none"` uses the declared weights as they stand. The scalarization is
      then not on the stakeholders' own scale, and comparing values across
      cells whose totals differ compares different objectives.
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "weights", ensure_containers_are_immutable(self.weights)
        )
        if self.normalization not in {"pointwise", "none"}:
            msg = (
                f"`ParetoObjective.normalization` is {self.normalization!r}, "
                'which is neither "pointwise" nor "none".'
            )
            raise ValueError(msg)
