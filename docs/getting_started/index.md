---
title: Getting Started
---

# Getting Started

This chapter is the shortest path from a Python environment to a model you can change.
It assumes basic Python and the idea of a finite-horizon dynamic program, but no JAX,
DAG, or pylcm knowledge.

Follow the pages in order:

1. [Install pylcm](../user_guide/installation.md).
1. [Solve the tiny model](../user_guide/tiny_example.ipynb).
1. [Learn the model vocabulary](model_vocabulary.md).
1. [Choose the right starting declaration](next_steps.md) before adapting the example to
   your own economics.

The tiny model deliberately uses the broad `Regime` + `GridSearch` route. It teaches the
common workflow; it is not a universal template. If you expect to use an endogenous-grid
solver, write the model as a `ConsumptionSavingsRegime` or
`NestedConsumptionSavingsRegime` immediately. Retrofitting solver-specific economic
roles after a model has grown is avoidable work.

For a refresher on dynamic programming, see the
[QuantEcon Dynamic Programming book](https://dp.quantecon.org/).
