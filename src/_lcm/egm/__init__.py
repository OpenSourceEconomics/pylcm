"""Endogenous Grid Method (EGM) building blocks.

Submodules:

- `_lcm.egm.validation`: DC-EGM model-contract validation.
- `_lcm.egm.regime_introspection`: pure spec-introspection of regimes and carry
  targets, shared by the kernel build, continuation, and scope checks.
- `_lcm.egm.kernel_scope`: build-time checks naming features outside the kernel's
  current scope (the source of the raising-step message).
- `_lcm.egm.continuation`: the expected next-period value and marginal over the
  regime's targets (multi-target carry, passive blend, taste shocks, stochastic
  nodes) that the EGM step consumes per savings node.
- `_lcm.egm.budget`: resources and post-decision (budget) evaluation.
- `_lcm.egm.euler`: Euler-equation inversion to candidate consumption.
- `_lcm.egm.carry`: the per-period marginal-utility carry passed between periods.
- `_lcm.egm.step`: the backward-induction EGM step tying the pieces together.
- `_lcm.egm.terminal`: the terminal-period carry producer.
- `_lcm.egm.upper_envelope`: upper-envelope refinement of EGM candidates.
- `_lcm.egm.interp`: interpolation on the NaN-padded grids the refinement produces.
- `_lcm.egm.published_policy`: the simulation-facing policy emitted by the solve.
"""
