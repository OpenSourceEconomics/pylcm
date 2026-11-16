---
title: Exact implicit fibre facts
---

# Exact implicit fibre facts

The internal implicit compiler stores the original fibre facts as separate decision
diagrams. Every original status code and fibre address remains available. Equal facts
share storage; this does not merge economic states, change a Model or choose a graph.
The compiler reads an existing declaration capture and its structural companions. It
never executes an economic callback, solver or simulation.

This interface uses the original **point-table** query order and fact inventory.
Equality means agreement with the installed evaluator's pointwise program. A capture
from an earlier pylcm version is replayed by checking out that version.

## Produce and independently check facts

Create `axes.json` with every original regime as a key and its status coordinate names
as an array. Other state coordinates stay free. Original declaration order governs
certificate selection, even if the names in this file use a different order.

```json
{"working": ["married", "children"], "retired": ["married"]}
```

```python
import json
from pathlib import Path

from _lcm_regime_finder.implicit.implicit_native import compile_table
from _lcm_regime_finder.implicit.verify_native import verify

packet = compile_table(snapshot=Path("model.json"), axes_file=Path("axes.json"))
Path("facts.json").write_text(json.dumps(packet))
receipt = verify(
    snapshot=Path("model.json"),
    axes_file=Path("axes.json"),
    artifact=Path("facts.json"),
)
```

The producer returns `PRODUCED_UNVERIFIED`. The separate verifier checks every original
fact on the full original rectangle, including unkept points. It uses the original
source API and an independent decoder. Only a finished replay with unchanged inputs
returns `COMPLETE_FACT_EQUALITY`. This receipt certifies fact equality; it does not
certify a graph reduction.

The compiler retains conservative unknown results, actual transition producers,
registered laws, phases, joint read certificates and all source qualifications. If
binding-cache order independence cannot be proved, it follows the exact original query
order with one shared original cache.

## Check a supplied projection

A removal proposal is a set of `(original regime, status coordinate)` pairs. An empty
set checks the identity projection. Removing a status classification makes that variable
an ordinary coordinate; it does not delete it from the Model.

```python
from _lcm_regime_finder.implicit.check_projection import check_projection

result = check_projection(
    snapshot=Path("model.json"),
    axes_file=Path("axes.json"),
    artifact=Path("facts.json"),
    removed=frozenset({("working", "married"), ("retired", "married")}),
)
```

This call repeats independent full-domain fact verification, reconstructs original
support incidences, computes possible-support closure and fixed-certificate liveness,
and checks all eight finite premises, I1–I5 and a–c. Final guards bind the complete
transaction. A refuted premise retains its witness. An interrupted or invalid
transaction returns no new certificate.

The diagnostic box adapter can expand exponentially, even for independent identity
updates. Separate diagrams and support factors remain the canonical representation.
Neither compiler nor verifier has a polynomial complexity guarantee. Exhausted resources
are incomplete work, not proof that the original graph cannot be represented.

`FINITE_PREMISES_PROVED` retains the original source and proof qualifications. It does
not authorize a native model edit, prove maximal removals, choose an attractive graph or
prove global optimality. Actions are retained; forced-action savings are not measured.
The existing witness-dependent search is unchanged because a compact counterexample can
select a different first witness.
