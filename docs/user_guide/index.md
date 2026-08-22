---
title: User Guide
---

# User Guide

The User Guide answers “how do I build and run my model?” It gives short workflows and
links outward when you need numerical reasoning or an exact contract.

If you have not solved the tiny model yet, start with
[Getting Started](../getting_started/index.md). In particular, make the
[solver-choice checkpoint](../getting_started/next_steps.md) before adapting it.

## Build the model

- [Write economics, not glue code](write_economics.ipynb) introduces named-function
  composition.
- [Defining models](defining_models.md) assembles the model-wide objects.
- [Regimes](regimes.ipynb) covers qualitatively different decision problems.
- [Grids](grids.md) defines numerical outcome spaces.
- [Transitions](transitions.ipynb) covers deterministic, stochastic, joint, and regime
  transitions.
- [Age-specialized functions and grids](age_specialized.md) changes declarations over
  the lifecycle.
- [Continuous stochastic processes](continuous_stochastic_processes.md) discretizes and
  conditions shocks.
- [Parameters](parameters.md) fills the model-generated parameter template.

## Solve, simulate, and inspect

- [Choosing a solver](choosing_a_solver.md) starts from economic structure and then
  compares scaling and hardware.
- [Solving and simulating](solving_and_simulating.md) runs backward induction and
  forward simulation.
- [Working with DataFrames and Series](pandas_interop.md) moves between pylcm and data
  workflows.

## Diagnose and scale

- [Performance and memory tuning](tuning.md) diagnoses a model on its target hardware.
- [Debugging](debugging.md) uses validation levels, snapshots, and smaller
  reproductions.

For explanations of equations and algorithms, use
[Concepts & Methods](../methods/index.md). For exact signatures and capability tables,
use the [Reference](../reference/index.md).
