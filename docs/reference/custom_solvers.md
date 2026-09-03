---
title: Custom solvers
---

# Custom solvers

A stable out-of-tree solver extension contract is not available yet.

`Solver`, `OneMarginSolver`, `TwoMarginSolver`, `SolverBuildContext`, and
`SolutionKernels` are publicly importable because the engine and in-tree solvers share
them, but parts of a complete custom implementation still depend on private types,
helpers, and concrete dispatch. Treat these names as contributor-facing contracts, not
as a supported plugin API.

`lcm.solver_api.KernelOutput` is the dependency-safe producer envelope used by plain
EGM. It defines the transport side of the planned solver seam; the solver build,
continuation, layout, and replay contracts and the conformance suite are not yet public.
The interface therefore remains experimental and is not a supported out-of-tree plugin
API.

Use a shipped solver from `lcm.solvers`. If none represents the economic problem, use
`GridSearch` where feasible or discuss the missing solver class with the maintainers. Do
not copy the old internal “Adding a Solver” guide into application code: it described
pylcm implementation details and could produce an extension that built successfully but
missed required engine behavior.

This page will become an implementation guide after
[issue #422](https://github.com/OpenSourceEconomics/pylcm/issues/422) establishes and
tests a public extension seam. Until then, the authoritative contributor view is the
[architecture chapter](../explanations/architecture.md) and the source tree.
