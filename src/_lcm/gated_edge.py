"""The gated edge a `ValueDependentTransition` decomposes into.

A model author declares a value-dependent transition in the regime's own
`transition` slot and reads the result back as `regime.gated_edges`. This is
that result: the engine's form of one edge, carrying the gate, the routes, the
references the gate reads and the off-grid contract. It lives here rather than
in the public package because nothing constructs it — it is derived.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm.collective import ProjectedRegimeValue, StakeholderRoute
from lcm.typing import UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class GatedEdge:
    """A gated edge routing a source regime's continuation into a target.

    The construct that unlocks MIXED singleton/collective
    regime topologies: a singleton regime may reach a collective regime (mutual
    consent marriage) and a collective regime may route per-stakeholder to
    singleton regimes (dissolution) — but only THROUGH a declared gated edge. Direct
    raw transitions between different-stakeholder regimes stay rejected.

    A source regime declares `gated_edges` as a mapping of TARGET regime name
    to `GatedEdge`. **The key is always the GATE-OPEN target** — the regime a
    row enters when the gate is true — and each leg's `fallback` is where the
    gate-false branch sends it. A dissolution edge is therefore keyed by the
    CONTINUING collective regime under `gate = ~D_target`, with each partner's
    single regime as that partner's leg fallback; keying it by one partner's
    single regime would send both partners there whenever the couple stays
    together.

    A leg thus owns four facts about where its rows go, and simulation carries
    all four: the open branch's regime (the key) and role
    (`StakeholderRoute.target_stakeholder`), and the closed branch's regime and role
    (`fallback.regime` and `fallback.stakeholder`). A row's own role —
    `initial_conditions["own_stakeholder"]`, published in the simulated frame —
    is what picks its leg, and it is updated to the branch's role as the row
    moves; a row landing in a singleton regime carries none.

    At the end of each period's solve, the engine folds, for
    each declared edge and each source stakeholder `s`, a gated continuation
    object on the target regime's grid:

        Wbar^s(x) = jnp.where(gate(x), V_target^{leg_s}(x), V_fallback^s(pi_s(x)))

    The source's continuation then reads `Wbar` in place of the raw target V,
    threaded through the ordinary transition machinery.

    - `gate` — a BOOLEAN user function evaluated pointwise on the target
      regime's grid. It may read the target regime's per-stakeholder value
      components under the names `V_target_<s>` (one per target stakeholder),
      the target's dissolution flag `D_target` (a collective target only), each
      key of `gate_refs` (a same-period reference value at its projection), and
      ordinary target states / params, plus the target fold's engine context
      (`period` / `age`). Mutual consent to marriage is the strict, unanimous gate
      `gate = (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)`;
      "no dissolution this period" is `gate = ~D_target`. Stochastic gates —
      a gate returning a probability in `(0, 1)` rather than a boolean — are
      not supported.
    - `legs` — one `StakeholderRoute` per SOURCE stakeholder (keyed by source
      stakeholder name; a singleton source declares exactly one leg under any
      key).
    - `gate_refs` — extra same-period reference values the `gate` reads,
      exactly like a regime's `same_period_refs` but projected from the TARGET
      regime's grid.
    - `off_grid` — what the edge promises about a landing point between the
      target's grid nodes. Both phases read the operands there and gate them
      there, so the value the source maximizes is one a branch really pays;
      `"reject"` additionally refuses, at model build, a target on whose grid
      such a point can occur at all.
    """

    gate: UserFunction
    """Boolean gate on the target grid, evaluated in the target fold's context."""

    legs: Mapping[str, StakeholderRoute]
    """One `StakeholderRoute` per source stakeholder.

    A singleton source declares exactly one, under any key.
    """

    gate_refs: Mapping[str, ProjectedRegimeValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Same-period reference values the `gate` reads (projected from the target
    grid)."""

    off_grid: Literal["pointwise", "reject"] = "pointwise"
    """How a landing point between the target's grid nodes is treated.

    - `"pointwise"` (the default) reads every operand at the landing point and
      applies the gate there, in both phases. The operands are interpolated, so
      the value carries the ordinary interpolation error of any continuation —
      but it is a value one branch really delivers, and the branch the solve
      priced is the branch simulation routes down.
    - `"reject"` demands that no such point exists: the model refuses to build
      unless the target regime's grid is reached exactly, i.e. it carries no
      continuous state. Declare it where a straddled gate would be an economic
      error rather than an approximation.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "legs", ensure_containers_are_immutable(self.legs))
        object.__setattr__(
            self, "gate_refs", ensure_containers_are_immutable(self.gate_refs)
        )
