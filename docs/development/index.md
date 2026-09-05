---
title: Development
---

# Development

This chapter is for people changing pylcm itself. Public model semantics and numerical
methods live elsewhere; private `_lcm` representations and contributor workflows live
here.

- [Development setup](setup.md)
- [Conventions](conventions.md)
- [Architecture](../explanations/architecture.md)
- [Dispatchers](../explanations/dispatchers.ipynb)
- [Continuous integration](continuous_integration.md)
- [Package benchmarks](benchmarking.md)
- [Resource contract for collective and gated models](collective_resource_contract.md)
- [Architecture transition ledger](architecture_transition_ledger.md)

A solver can be written against the public names alone, and
[Custom solvers](../reference/custom_solvers.md) is the guide to what one owes the
engine. That page defines the exact-version persistence, replay, and conformance
contract, which is why it, rather than these implementation notes, is the reference for
an out-of-tree author.
