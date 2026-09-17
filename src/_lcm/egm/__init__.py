"""Endogenous Grid Method (EGM) building blocks.

The package is large; each module's own docstring is the authority on what it
does. The families it groups into:

- **The core step.** `step_core` is the textbook single-post-state DC-EGM solve
  (Euler inversion → constrained candidates → upper envelope → publish V and
  carry); `euler`, `budget`, `carry`, `continuation`, `terminal`, `interp` and
  `step` are the pieces it composes, and `asset_row` maps it over the asset grid
  for Euler-state-dependent savings stages.
- **Upper envelopes.** `upper_envelope/` refines the candidate correspondence,
  with the exact-arithmetic route underneath it.
- **NB-EGM.** `nbegm*` covers the non-convex-budget family: case-piece metadata,
  breakpoint geometry, constraint boundaries, segmentation, the CRRA felicity
  the steps assume, the routes the case-piece solvers walk, and validation.
- **Outer search.** `outer_*` covers the nested continuous-outer family:
  candidate banks, the cash-on-hand outer envelope, interpolation across the
  outer-node axis, inversion, safeguarded refinement, and what a simulation
  replay may assume about one regime-period's outer margin.
- **Arithmetic and preferences.** `framed_arithmetic`,
  `comparison_arithmetic`, `numeric_inverse`, `ez_kernel` and `preferences`.
- **What the solve publishes.** `published_policy` and
  `nested_published_policy` for simulation, `euler_errors` for accuracy.
- **Build-time checks.** `validation`, `negm_validation`, `nnbegm_validation`,
  `kernel_scope`, `regime_introspection`, `declared_law`, `case_conditions`.
"""
