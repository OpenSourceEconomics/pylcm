"""Executable spec for the collective-regimes extension (DRAFT / WIP).

A collective regime declares a tuple of `stakeholders` and carries one
`utility_<s>` per stakeholder, so a single household decision is scored by
several value functions at once. The extension is only partly built, and this
file is where the line between what a collective regime already does and what it
must eventually do is written down. The tests split into two groups:

* Pinning tests (pass today) — the construction-time contract: the
  `Regime.stakeholders` API surface exists, a regime that declares stakeholders
  is validated against its per-stakeholder utilities, and the singleton default
  (`stakeholders=None`) is untouched by any of it.

* Target-behavior tests (`xfail`, `strict=False`) — what a collective regime
  must do once the numerics land. They are written against the real construction
  API and fail today because the result APIs they read — per-stakeholder value
  arrays, the consent gate, the simulation value router — do not exist yet.
  `strict=False` so a test that starts passing before it is un-xfailed does not
  turn the suite red.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import (
    ContinuousAction,
    DiscreteAction,
    FloatND,
    IntND,
    ScalarInt,
)

# Shared building blocks: a stripped-down couples problem, in which the two
# stakeholders differ only in their disutility of work.


@categorical(ordered=True)
class LaborSupply:
    do_not_work: ScalarInt
    work: ScalarInt


def _utility_f(
    consumption: ContinuousAction,
    labor_supply_f: DiscreteAction,
    match_quality: float,
) -> FloatND:
    """Wife's per-stakeholder felicity (illustrative)."""
    return (
        jnp.log(consumption)
        - 0.3 * (labor_supply_f == LaborSupply.work)
        + match_quality
    )


def _utility_m(
    consumption: ContinuousAction,
    labor_supply_m: DiscreteAction,
    match_quality: float,
) -> FloatND:
    """Husband's per-stakeholder felicity (illustrative)."""
    return (
        jnp.log(consumption)
        - 0.5 * (labor_supply_m == LaborSupply.work)
        + match_quality
    )


_WEALTH = LinSpacedGrid(start=1, stop=10, n_points=5)
_CONSUMPTION = LinSpacedGrid(start=1, stop=5, n_points=5)


@categorical(ordered=False)
class _CoupleRegimeId:
    """Regime ids of the minimal two-regime couples model."""

    married: ScalarInt
    widowed: ScalarInt


def _next_regime_widowed(age: FloatND) -> IntND:
    """The married household enters `widowed` next period."""
    return jnp.full_like(age, _CoupleRegimeId.widowed, dtype=jnp.int32)


def _build_married_regime() -> Regime:
    """Construct a two-stakeholder `married` regime via the real API surface.

    How the household scalarization `O({Q^s})` and the value-aware mask are
    declared is still an open question, so the two per-stakeholder utilities are
    supplied as named functions and `stakeholders` names the value axis. Every
    test that builds this regime is `xfail`: each reads a result API the
    extension does not expose yet.
    """
    return Regime(
        transition=None,  # terminal, keeps the spec minimal
        stakeholders=("f", "m"),
        states={"wealth": _WEALTH},
        actions={
            "labor_supply_f": DiscreteGrid(LaborSupply),
            "labor_supply_m": DiscreteGrid(LaborSupply),
            "consumption": _CONSUMPTION,
        },
        functions={
            # Per-stakeholder utilities. One `utility` today; a stakeholder
            # axis tomorrow.
            "utility_f": _utility_f,
            "utility_m": _utility_m,
        },
    )


# Pinning tests: the collective-regime contract that holds today.


def test_declaring_non_terminal_stakeholders_constructs():
    """A non-terminal collective regime constructs.

    Declaring `stakeholders` alongside a regime transition is accepted: the
    per-stakeholder contract is checked at construction, and the constructed
    regime keeps both its stakeholder tuple and its non-terminal status. What a
    collective regime is allowed to route to, and how it solves, is pinned in
    `tests/regime_building/test_nonterminal_collective_solve.py`.
    """

    def _some_transition(_age: float) -> int:
        return 0

    regime = Regime(
        transition=_some_transition,
        stakeholders=("f", "m"),
        states={"wealth": _WEALTH},
        actions={"labor_supply_f": DiscreteGrid(LaborSupply)},
        state_transitions={"wealth": lambda wealth: wealth},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
    )
    assert regime.stakeholders == ("f", "m")
    assert not regime.terminal


def test_terminal_stakeholders_without_per_stakeholder_utility_is_rejected():
    """A collective regime must carry a `utility_<s>` for every stakeholder.

    Supplying a single `utility` where `utility_f` and `utility_m` are required
    is a configuration error. Completeness is a property of the merged regime —
    a bare `Regime` may still receive functions from a model-level slot — so it
    is reported when the model finalizes its regimes.
    """
    married = Regime(
        transition=_next_regime_widowed,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={
            "labor_supply_f": DiscreteGrid(LaborSupply),
            "labor_supply_m": DiscreteGrid(LaborSupply),
            "consumption": _CONSUMPTION,
        },
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
    )
    widowed = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wealth": _WEALTH},
        actions={
            "labor_supply_f": DiscreteGrid(LaborSupply),
            "consumption": _CONSUMPTION,
        },
        functions={"utility": _utility_f},
    )

    with pytest.raises(RegimeInitializationError, match="per-stakeholder utility"):
        Model(
            regimes={"married": married, "widowed": widowed},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=_CoupleRegimeId,
        )


def test_singleton_default_is_untouched():
    """A regime that declares no stakeholders is a plain singleton regime.

    Omitting `stakeholders` leaves the field `None` and construction takes the
    ordinary single-value path — the collective branch is reached only by an
    explicit declaration, never by default.
    """

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    regime = Regime(
        transition=None,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        functions={"utility": utility},
    )
    assert regime.stakeholders is None


def test_stakeholders_field_default_is_none():
    """`stakeholders` defaults to `None` (the singleton) when omitted."""
    field = Regime.__dataclass_fields__["stakeholders"]
    assert field.default is None


# Target-behavior tests: what a collective regime must do once the numerics land.


@pytest.mark.xfail(
    reason="collective regimes not yet implemented",
    strict=False,
)
def test_two_stakeholder_values_differ():
    """A two-stakeholder regime yields two distinct value arrays.

    At the common household argmax, the wife's and husband's per-stakeholder
    values are read off separately (`V^s = Q^s(x, a*)`). Because their felicities
    differ (different disutility of work), the two value arrays must not be
    identical.
    """
    regime = _build_married_regime()
    # Target API: the solved regime exposes one value array per stakeholder.
    values = regime.solve_period_values()  # ty: ignore[unresolved-attribute]
    assert not jnp.allclose(values["f"], values["m"])


@pytest.mark.xfail(
    reason="collective regimes not yet implemented",
    strict=False,
)
def test_value_aware_feasibility_reads_reference_value():
    """The action mask compares Q^s against a same-period reference value.

    The married participation set is `Q^j(x, a) >= V^j(outside_j) - Delta_j`, so
    the mask reads a *same-period* single-regime reference value at the matched
    shock realization — a quantity that only exists once Q has been formed, and
    so cannot be computed ahead of it. The solve must also expose an explicit
    dissolution flag `D = 1[mask empty]`, distinct from a numeric -inf value.
    """
    regime = _build_married_regime()
    result = regime.solve_period_values()  # ty: ignore[unresolved-attribute]
    # Target API: a boolean dissolution flag alongside the per-stakeholder values,
    # never inferred from V == -inf.
    assert result.dissolution_flag.dtype == jnp.bool_


@pytest.mark.xfail(
    reason="collective regimes not yet implemented",
    strict=False,
)
def test_mutual_consent_gate():
    """The singles->married edge forms a marriage only by mutual consent.

    The gated edge object folds `E_eps[ kappa*V_married + (1-kappa)*V_single ]`
    where the consent gate `kappa` is `1` iff `V^{jM}_{t+1} > V^j_{t+1}` for BOTH
    stakeholders (strict, no slack). A candidate marriage that clears only one
    partner's outside option must NOT form.
    """
    regime = _build_married_regime()
    # Target API: the edge gate is a callable reading both stakeholders' values
    # and returning a per-node acceptance indicator.
    gate = regime.consent_gate  # ty: ignore[unresolved-attribute]
    v_married = {"f": jnp.array([1.0, 0.0]), "m": jnp.array([1.0, 1.0])}
    v_single = {"f": jnp.array([0.5, 0.5]), "m": jnp.array([0.5, 0.5])}
    accepted = gate(v_married=v_married, v_single=v_single)
    # Node 0: both clear -> accept. Node 1: wife does not clear -> reject.
    assert accepted[0]
    assert not accepted[1]


@pytest.mark.xfail(
    reason="collective regimes not yet implemented",
    strict=False,
)
def test_simulate_value_router_routes_on_realized_values():
    """The simulator routes regimes by recomputed values, not by Phi(x,a).

    At simulation, the router draws candidate realizations, recomputes the
    candidate regimes' per-stakeholder values at the realized point, evaluates
    the same mutual-consent gates the solve applies, then routes and discards
    the losing candidate.
    """
    regime = _build_married_regime()
    router = regime.simulate_value_router  # ty: ignore[unresolved-attribute]
    assert callable(router)
