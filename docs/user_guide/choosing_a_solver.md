---
title: Choosing a solver
---

# Choosing a solver

pylcm ships several solvers for the continuous part of a regime's period problem. They
are not interchangeable: each is fastest — or only correct — for a particular problem
structure. Pick one in two passes. The **feasibility** tree narrows to the solvers that
are *correct* for your problem's structure; the **hardware / speed** tree picks the
fastest among those on your machine.

The guiding principle is conservative. **`GridSearch` (brute force) is the default, and
it is often the right answer.** On a GPU it is a dense map-reduce with static shapes and
perfect chunkability, and it is exact to its action grid. An endogenous-grid method has
lower arithmetic complexity, but that only pays off if it does not materialize large
transients, carry long sequential scans, or compile many shape variants. So the rule is:
**adopt a structure-specific solver only after benchmarking it against `GridSearch` on
your target hardware** — on peak memory, compile time, and wall-time. (Near
institutional cliffs, `GridSearch` smooths across the discontinuity, so there it is a
speed baseline and a diagnostic, not the accuracy reference — see
[the NB-EGM solver](nbegm.md).) See [Performance and Memory Tuning](tuning.md) and
[Benchmarking](benchmarking.md).

## Decision tree by feasibility

Which solvers are *correct* for your problem's structure.

```{mermaid}
flowchart TD
    q0(["How many continuous states carry an Euler equation?"])
    q0 -->|"None"| gs0["GridSearch"]
    q0 -->|"1"| qbp{"Declared institutional breakpoints? (asset tests, brackets, notches, floors)"}
    q0 -->|"2"| q2d{"Genuinely coupled 2-D first-order-condition system?"}

    q2d -->|"Yes"| twodim["TwoDimEGM (G2EGM)"]
    q2d -->|"No — clean inner nest (liquid + durable/illiquid)"| negm["NEGM"]

    qbp -->|"Yes"| nbegm["NBEGM"]
    qbp -->|"No"| qdc{"Discrete choice induces non-concavity (secondary kinks)?"}

    qdc -->|"Yes"| dcegm["DCEGM (or NBEGM — see hardware tree)"]
    qdc -->|"No — smooth and concave"| egm["OneAssetEGM or GridSearch"]
```

At the secondary-kink leaf, both `DCEGM` and `NBEGM` are correct. `DCEGM` is the natural
choice for a plain discrete–continuous problem with no institutional breakpoints;
`NBEGM` handles the same secondary kinks (via its discrete-branch envelope and
Euler-path fold-splitting) and is the choice once the model *also* carries declared
cliffs. Which is faster is a hardware question — the next tree.

## Decision tree by hardware and speed

Among the feasible solvers, which is fastest.

```{mermaid}
flowchart TD
    h0(["Target hardware?"])
    h0 -->|"GPU"| g1{"Action grid modest for the required accuracy?"}
    h0 -->|"CPU"| c1{"Branchy discrete–continuous envelope?"}

    g1 -->|"Yes"| gs["GridSearch — dense map-reduce usually wins"]
    g1 -->|"No — fine grid, or cliffs"| g2{"Full-row envelope is the memory wall?"}
    g2 -->|"Yes"| gq["Query-side segmented envelope: NBEGM, or DCEGM(upper_envelope='ltm')"]
    g2 -->|"No"| ge["EGM-family: OneAssetEGM / NEGM / TwoDimEGM / NBEGM"]

    c1 -->|"Yes"| cd["DCEGM — FUES / RFC / LTM / MSS all viable on CPU"]
    c1 -->|"No — smooth"| ce["OneAssetEGM, or GridSearch"]
```

Two cross-cutting factors:

- **GPU parallelism.** A GPU favours dense, static-shape map-reduces — `GridSearch`, and
  the query-side upper envelope used by `NBEGM` (and available to `DCEGM` via
  `upper_envelope="ltm"`). A CPU tolerates the sequential, topology-discovering envelope
  scans (`DCEGM`'s default FUES backend) that a GPU runs poorly. So at the
  secondary-kink leaf, prefer `NBEGM`'s query-side envelope on a GPU and `DCEGM`'s FUES
  on a CPU.
- **Compile-shape explosion.** Many static shapes — long age grids, per-period target
  splits, branch axes — multiply compiled programs. When that dominates, fall back to
  `GridSearch` or a simple EGM.

## Solvers at a glance

| Solver        | Use when                                                                                                                                              | Key constructor arguments                                                              |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `GridSearch`  | The default. Any regime, especially with a modest continuous-action grid on a GPU.                                                                    | *(none)*                                                                               |
| `OneAssetEGM` | Smooth, concave one-asset consumption–saving problem where a fine action grid would otherwise be needed.                                              | `savings_grid`                                                                         |
| `DCEGM`       | One liquid asset with a discrete choice that makes the value function non-concave (secondary kinks).                                                  | `continuous_state`, `continuous_action`, `resources`, `savings_grid`, `upper_envelope` |
| `NEGM`        | Two continuous choices with a clean nest: an inner 1-D EGM consumption solve inside an outer deterministic search over a durable/illiquid post-state. | `inner`, `outer_action`, `outer_post_decision`, `outer_grid`                           |
| `TwoDimEGM`   | Two continuous assets whose first-order conditions are genuinely coupled (the G2EGM setting).                                                         | `a_grid`, `b_grid`, `consumption_grid`, `threshold`                                    |
| `NBEGM`       | One liquid asset with **declared** institutional kinks and cliffs. See [the NB-EGM solver](nbegm.md).                                                 | `savings_grid`, `jump_read`                                                            |

`DCEGM`'s upper-envelope backend is selectable via `upper_envelope=` (`"fues"`, `"rfc"`,
`"ltm"`, `"mss"`). `"fues"` is a topology-discovering scan (CPU-friendly); `"ltm"` is a
query-side segment evaluator (GPU-friendly). Switch only under a benchmark.

## A note on current-state dependence

Standard EGM's speed comes from *amortization*: invert the Euler equation once per
post-decision savings node and read the resulting policy at every current state. That
only works when the Euler right-hand side depends on savings alone after conditioning on
discrete states and smooth branches. If an institutional rule leaves the right-hand side
depending on the *current* liquid state even after conditioning, the amortization is
lost. The exact fallback is one EGM problem per current-state node (`DCEGM`'s asset-row
mode), which forfeits EGM's advantage — at which point `GridSearch` is usually the
better choice. `NBEGM`'s per-interval continuation path is the structured escape hatch:
when the dependence is piecewise-constant on declared breakpoints, one curve per
interval restores exactness without per-node replication.
