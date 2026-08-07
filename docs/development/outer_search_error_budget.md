# Outer-search numerical error budget

Companion to the [continuous-outer ADR](adr_continuous_outer_inference.md). A central
finite difference of the moment map decomposes as

```text
Ĝ(h) − G  =  O(h²)                       parameter truncation
           + O(ε_outer  / h)             outer search (mesh + bracket)
           + O(ε_inner  / h)             inner NB-EGM solve
           + O(ε_interp / h)             outer interpolation
           + O(ε_sim    / h)             finite simulation
```

Every term must have an *observable* estimate before a Jacobian is accepted; "the steps
agree with each other" alone is consistency, not accuracy — three steps can share a bias
regime and agree while uniformly wrong.

| Error source         | Measured by                                             |
| -------------------- | ------------------------------------------------------- |
| parameter truncation | comparison of h, h/2, h/4 central differences           |
| outer search         | tighter mesh and bracket tolerances                     |
| inner NB-EGM         | tighter savings-grid/envelope settings and Euler errors |
| interpolation        | exact midpoint validations (interpolant vs exact solve) |
| simulation           | independent scrambles and larger N                      |
| branch ties          | population mass near keeper/adjuster and local-max ties |
| moment nonsmoothness | median atoms, empty groups, exact-equality mass         |

All seven diagnostics are stored in the Jacobian manifest; inference refuses a manifest
whose diagnostics are missing or whose configuration does not match the run being
reported.

## Mesh freezing for derivatives

Adaptive node insertion is discrete, so the mesh must not differ arbitrarily between θ+h
and θ−h. Derivative protocol: adapt at baseline θ and at pilot perturbations θ ±
h_max·e_j; take the per-period union of all resulting nodes; sort, deduplicate, freeze;
rerun baseline and every perturbation on the frozen mesh; verify the frozen mesh meets
the interpolation-error criterion for every run; on any failure, augment the mesh and
restart the entire derivative batch. The frozen nodes are stored in the Jacobian
manifest.

## Simulation variance

The simulation allowance added to the empirical moment variance is the variance of **one
production-sized simulated moment vector**, estimated from independent
scrambles/replications; it is *not* divided by the number of auxiliary replications (the
estimator's objective uses one simulation, not their average).
