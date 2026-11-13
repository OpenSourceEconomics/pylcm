---
title: Joint dependency facts for regime analysis
---

# Joint dependency facts for regime analysis

The internal fibre analysis separates original-source inspection from dependency closure
and from choosing a layout. These modules do not run a solver, simulate subjects, search
over layouts or edit a model.

Given an existing original Model capture, inspect its admitted cells as follows:

```python
from pathlib import Path

from _lcm_regime_finder.cli import load_problem
from _lcm_regime_finder.fibre_facts import calculate_fibre_liveness
from _lcm_regime_finder.fibre_source import FibreSource

original = load_problem(Path("model-capture.json"))
source = FibreSource(original)
facts = source.get_cell_facts()
live = calculate_fibre_liveness(facts)
```

These are internal interfaces. The public model-analysis interface is documented in
[Check a regime graph](../user_guide/regime_checker.md).

On Linux, an installed checker can export these facts without running a layout search:

```console
pixi run python -m _lcm_regime_finder source-facts model-capture.json --output source-facts
```

The output directory contains `report.json` and the worker's `resource.json`. Inputs and
their stored companions retain the normal protected-delivery checks. Resource monitoring
has no default time or memory cutoff; `--seconds` and `--host-gib` set optional limits.
The report includes joint dependencies, least cell liveness, proved categorical support,
whole-domain action exclusions and unresolved source obligations. Unknown payloads keep
their original domains. `SOURCE_FACTS_BUILT` does not authorize a model change or
certify that a layout is optimal.

## Joint read certificates

A primitive's read set must be sufficient when **all** state coordinates outside that
set are free together. Testing each coordinate while fixing the others is insufficient.
For example, with three binary statuses, a majority payoff can be unchanged after each
individual change at `(0, 0, 0)`, yet differ at `(1, 1, 1)`.

`get_joint_read_set` attempts status removal in its supplied order. Each attempt fixes
only statuses that remain in the candidate set. Other state coordinates, actions, shocks
and runtime parameters remain free. The frozen result records the selected coordinates,
removal order and removed statuses. The source adapter resolves category-code arguments
through the original state and function DAG. Unsupported source retains the full
source-state set.

Read certificates are chosen once, before dependency closure or grouping. Different
sufficient read sets can be incomparable. Reselecting them while successor dependencies
change would not define the same monotone closure.

## Original primitive channels

`FibreSource` normalizes original declaration metadata and independently replays its
source pointers and capture inventories. An external producer's smaller read set is not
authority. Inspection covers solution, decisions during simulation and realized
simulation, including declared outputs and registered shock laws.

The source and target of a transition, its primitive phase and any joint producers
determine its dependency context. A target's process law must not be replaced by the
source's similarly named law. During solution a carried-only state uses its declared
imputation; its simulation grid, including a singleton grid, cannot replace that
function. Rational age labels are interpreted as exact model ages. A period model does
not acquire a biological age argument.

Action exclusion requires proof on the entire requested status rectangle. It includes
points owned by no reachable fibre. Every other action, ordinary state and runtime
parameter remains free. An unknown constraint retains the action value. Excluded results
are category codes, with their original numerical values kept distinct. A scalar zero
returned by a stochastic probability function is not a deterministic target-state code.

Categorical codes and values may be inline or stored in a checked structural payload.
The adapter reads the original scalar sequence or one-dimensional array, including its
byte order, offset and stride. It does not rebuild a grid or execute array callbacks.
The resulting code/value indices are immutable. A fixed category keeps its supplied
nominal code even when other categories have the same numeric value; an update known
only by value retains every matching code. Unsupported scalar storage remains an
explicit qualification and cannot justify removing values.

## Dependency closure and its boundary

`calculate_fibre_liveness` computes the least fixed point **for the supplied read
certificates**. A source keeps its direct reads and the inputs to every update of a live
successor coordinate. Next-period transitions and within-period value reads both
contribute. The immutable links retain their phases and original incidence IDs. Missing
producers cannot be replaced with an assumed identity.

`get_cell_facts` supplies a conservative baseline without status specialization. It
combines joint and ordinary producers of one transition while preserving distinct
projected value reads. Gates and imputations are computations rather than artificial
recursive value links. A query-only owner outside the admitted cell set remains a
qualification and conservatively protects source coordinates.

Successful source inspection is not a certificate of a proposed native reduction, an
emission rule, or optimality. The API does not replace existing source, preservation or
adoption checks. Fibre reachability, block admissibility, complete action-domain facts
and layout costs require their own checks; the unspecialized cell baseline alone does
not establish those results.
