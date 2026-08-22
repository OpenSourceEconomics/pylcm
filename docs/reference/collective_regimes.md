---
title: Collective regimes
---

# Collective regimes

A collective `Regime` carries several stakeholder value functions but chooses one shared
action by a weighted household objective.

## Shared decision

Set `stakeholders=("f", "m")` and provide one utility function per stakeholder:

```python
couple = lcm.Regime(
    transition=...,
    stakeholders=("f", "m"),
    weights={"f": 0.5, "m": 0.5},
    functions={
        "utility_f": utility_f,
        "utility_m": utility_m,
    },
)
```

The solver maximizes

$$
O(x,a) = \sum_s \lambda_s Q^s(x,a)
$$

and stores every stakeholder's own value `Q^s` at the shared maximizing action. Omit
`weights` for equal weights. Collective regimes currently require `GridSearch` and do
not support EV1 taste shocks, nonlinear certainty equivalents, or folded shock states.

## Participation constraints

`value_constraints` run after stakeholder action values have been formed. They may read
`Q_<stakeholder>` and names declared in `same_period_refs`.

`SamePeriodRef(regime, projection, stakeholder=None)` reads another regime's value in
the same period at projected state coordinates. The reference regime is solved first;
cycles are rejected. Set `stakeholder` only when the referenced regime is collective.

A participation constraint can therefore compare a partner's value inside the household
with the value of being single in the same period. A state cell with no jointly feasible
action publishes a separate dissolution flag.

## Gated transitions across stakeholder layouts

Direct transitions between singleton and collective layouts are rejected. Declare a
`GatedEdge` on the source regime:

- `gate` is a Boolean evaluated on the target grid;
- `gate_refs` supplies any same-period outside-option values needed by the gate;
- `legs` contains one `EdgeLeg` per source stakeholder;
- each leg identifies the target stakeholder when the gate is open and a `SamePeriodRef`
  fallback when it is closed.

This represents mutual-consent entry or stakeholder-specific dissolution without mixing
values from different household topologies.

A same-period or fallback projection is evaluated against the referenced regime's grid
for the period being folded. This matters for `AgeSpecializedGrid`: equal array shapes
at two ages do not imply equal coordinates. A fallback projection supplies every
simulation-state coordinate of the fallback regime; a gate reference supplies the
solve-state coordinates of its reference value function.

## Solve and simulate

Request dissolution flags when an edge gate reads `D_target`:

```python
solution, flags = model.solve(
    params=params,
    log_level="debug",
    return_dissolution_flags=True,
)

result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=solution,
    period_to_regime_to_dissolution_flags=flags,
    own_stakeholder="f",
    log_level="debug",
)
```

Simulation carries one fixed-size cohort. Dissolution does not split one row into two
linked people. `own_stakeholder` selects which stakeholder's fallback leg governs the
entire simulated cohort; simulate separate cohorts for separate roles. Off-grid gates
interpolate already-maximized target values rather than re-solving the target household
problem at the realized point.

See the progressive [collective-regimes example](../examples/collective_regimes.md).
