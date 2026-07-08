---
title: Choosing a solver
---

# Choosing a solver

pylcm ships several solvers for the continuous part of a regime's period problem. They
are not interchangeable: each is fastest — or only correct — for a particular problem
structure. This page helps you pick one.

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

## Decision tree

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

    qdc -->|"Yes"| dcegm["DCEGM (or NBEGM — see below)"]
    qdc -->|"No — smooth and concave"| egm["OneAssetEGM or GridSearch"]
```

## Reading the tree

The tree branches on problem **structure**, which fixes the set of solvers that are
*correct*. Two things then decide which correct solver is *fastest*:

- **Hardware is a second axis over every leaf.** A GPU favours dense, static-shape
  map-reduces — `GridSearch`, and the query-side upper envelope used by `NBEGM` (and
  available to `DCEGM`). A CPU tolerates the sequential, topology-discovering envelope
  scans (`DCEGM`'s default FUES backend) that a GPU runs poorly. On a GPU with a modest
  action grid, `GridSearch` frequently wins outright, so benchmark against it whatever
  the leaf.
- **`DCEGM` vs. `NBEGM` at the secondary-kink leaf.** Both resolve the non-concavity a
  discrete choice induces. `DCEGM` is the natural choice for a plain discrete–continuous
  problem with no institutional breakpoints, and its FUES envelope is competitive on
  CPU. `NBEGM` handles the same secondary kinks (via its discrete-branch envelope and
  Euler-path fold-splitting) with a GPU-parallel query-side envelope, and is the choice
  once the model *also* carries declared cliffs. See [the NB-EGM solver](nbegm.md).

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
`"ltm"`, `"mss"`). `"fues"` is a topology-discovering scan; the others trade off
differently on GPU versus CPU. The default is a reasonable starting point — switch only
under a benchmark.

## By model feature

| Feature                                                          | Recommended solver                               |
| ---------------------------------------------------------------- | ------------------------------------------------ |
| Smooth one-asset problem                                         | `OneAssetEGM`, or `GridSearch` after a benchmark |
| Smooth one-asset problem with a modest action grid               | `GridSearch` (often wins on GPU)                 |
| One-asset discrete–continuous non-concavity                      | `DCEGM`                                          |
| Finite institutional thresholds (kinks, cliffs, notches, floors) | `NBEGM`                                          |
| Current-state dependence survives all case conditioning          | `GridSearch`, or `DCEGM` asset-row mode          |
| Two continuous choices with a clean inner nest                   | `NEGM`                                           |
| Genuinely coupled 2-D first-order-condition system               | `TwoDimEGM`                                      |
| Dense, regular, low-dimensional action grid                      | `GridSearch`                                     |

## By hardware

| Hardware condition                              | Solver tendency                                                                     |
| ----------------------------------------------- | ----------------------------------------------------------------------------------- |
| GPU, high bandwidth, moderate action grid       | `GridSearch`                                                                        |
| GPU, memory-bound full-row envelopes            | Query-side segmented EGM (`DCEGM` / `NBEGM`)                                        |
| GPU, high action resolution needed for accuracy | EGM-family (`OneAssetEGM` / `NEGM` / `TwoDimEGM` / `NBEGM`) if the topology streams |
| CPU, small rows, branchy envelopes              | `DCEGM` envelope backends are all viable                                            |
| Many static shapes / age variants               | `GridSearch` or a simple EGM, to avoid compile-shape explosion                      |

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
