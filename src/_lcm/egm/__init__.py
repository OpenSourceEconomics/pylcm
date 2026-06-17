"""Endogenous Grid Method (EGM) building blocks.

Submodules:

- `_lcm.egm.validation`: DC-EGM model-contract validation.
- `_lcm.egm.budget`: resources and post-decision (budget) evaluation.
- `_lcm.egm.euler`: Euler-equation inversion to candidate consumption.
- `_lcm.egm.carry`: the per-period marginal-utility carry passed between periods.
- `_lcm.egm.step`: the backward-induction EGM step tying the pieces together.
- `_lcm.egm.terminal`: the terminal-period carry producer.
- `_lcm.egm.upper_envelope`: upper-envelope refinement of EGM candidates.
- `_lcm.egm.interp`: interpolation on the NaN-padded grids the refinement produces.
- `_lcm.egm.published_policy`: the simulation-facing policy emitted by the solve.
"""
