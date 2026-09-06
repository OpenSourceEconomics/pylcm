---
title: Declared non-convex budgets
---

# Declared non-convex budgets

Taxes, transfers, asset tests, floors, and adjustment rules can make liquid resources
piecewise smooth or discontinuous. Hiding those rules inside `where`, `maximum`, or
Python branching gives a general solver executable arithmetic, but it does not give a
structure-specific solver the boundaries and one-sided ownership it needs.

pylcm therefore separates the **economic function** from its **structural declaration**.
The decorated functions remain ordinary callables and still work under `GridSearch`.

## Two declaration forms

### Case pieces

Use a case boundary when one supported binary predicate selects between two smooth
formulas for the same output. The declaration identifies:

- the liquid variable;
- the threshold parameter;
- which side owns equality;
- whether the boundary is a continuous kink, jump, or hard constraint;
- the `when` and `otherwise` formula for the output.

This form is intentionally narrow. `NBEGM` validates that the declared split matches a
supported route instead of guessing at arbitrary branch code.

### Piecewise-affine schedules

Use `piecewise_affine` when one output is affine in the liquid variable between a
sequence of declared breakpoints. Each `affine_breakpoint` supplies the threshold and
kind. Indexed thresholds can describe schedules that vary with a categorical state.

This covers bracketed taxes, floors, and mixed kink/jump schedules without inventing a
separate case function for every interval.

## What the solver does with the structure

Within each smooth run, `NBEGM` performs a one-dimensional endogenous-grid solve. It
then augments candidates at declared boundaries, applies exact side ownership, and takes
a branch-aware upper envelope. A parent can read a jump one-sidedly or use a faster
bridged finite-grid representation according to `jump_read`.

The period is solved tile-locally: each block of ride-along cells reads its
transition-aware continuation and runs its envelope solve in savings space before the
next block, so the expected continuation over every cell never exists as one array. A
next-period law that reads the current liquid state through declared cliffs makes the
continuation piecewise-constant across the cliff intervals; the read is then bound per
interval and each interval is solved as its own case. A discrete action that only shifts
the budget and utility shares one continuation read across its branches; an action that
reaches the continuation (through the regime transition, a law of motion,
stochastic-state transition weights, a child's resources, the discount factor, or a
schedule variable on a per-interval read) is read once per class of branches agreeing on
it.

The structural declaration does not make an unsupported model supported. The solver
still checks its scope: margin count, action structure, budget form, differentiability,
and which names are available at each candidate stage. A retained `Condition` has the
same rule: structure enables a proof or compiler only when one exists.

## Correctness and brute-force comparisons

Grid search is a useful diagnostic because the same functions execute without the
specialized metadata. It is not automatically an accuracy reference at a cliff: a fixed
action grid can miss a threshold or interpolate across it. Validation should compare
values and decisions away from and at the declared boundaries, use one-sided checks
where appropriate, and refine the brute grid until remaining differences have an
economic interpretation.

Exact decorator contracts are split into [Case pieces](../reference/case_pieces.md) and
[Piecewise-affine schedules](../reference/piecewise_affine.md). Feasibility boundaries
belong to [Constraints and structured Conditions](../reference/conditions.md), not to
either budget declaration.

## Related pages

- **Guide:** [Choosing a solver](../user_guide/choosing_a_solver.md) and
  [Authoring for EGM-family solvers](../user_guide/authoring_specialized_solvers.md)
- **Methods:** [Solver families](solver_families.md) and
  [The endogenous-grid method](egm_foundations.md)
- **Example:** [Mahler & Yum (2024)](../examples/mahler_yum_2024.md)
- **Reference:** [Case pieces](../reference/case_pieces.md),
  [Piecewise-affine schedules](../reference/piecewise_affine.md), and
  [Solvers and capabilities](../reference/solvers.md)
