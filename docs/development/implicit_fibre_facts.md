---
title: Exact implicit fibre facts
---

# Exact implicit fibre facts

The internal implicit compiler stores the original fibre facts as separate decision
diagrams. Every original status code and fibre address remains available. Equal facts
share storage; this does not merge economic states, change a Model or choose a graph.
The compiler reads an existing declaration capture and its structural companions. It
never executes an economic callback, solver or simulation.

This interface uses the original **point-table** query order and fact inventory. Each
packet names the selected evaluator version and its exact source map. Equality means
agreement with that evaluator's pointwise program, not equivalence between different
evaluator versions. This contract differs from the public `source-facts` path; their
read inventories must not be interchanged.

`source639-builtin-int-v1` identifies the evaluator with bounded builtin-int predicate
reads and early-return continuation analysis. Its descriptor binds the complete source
at `596ceb54b347b918b60cd3dcc284f1f975ee011b`. The historical `source587` and
`source645` descriptors retain their original programs and native declaration dialects;
they do not acquire the new evaluator semantics. The selector does not swap imported
modules. The installed evaluator must match the selected source inventory and module
origins exactly. Only a bundled version with a separately reviewed qualification receipt
for the installed algorithm bytes can run an ordinary producer or verifier. Candidates
and unknown IDs fail before loading the model capture.

The typed public-process evaluator in this branch is `source645-builtin-int-v1`, bound
to `39a79831aa573ef23fb08d77da89ee965da6bff8`. Use this version with its public process
declarations. Its complete source map and qualification are separate from the private
process evaluator `source639-builtin-int-v1`.

## Produce and independently check facts

Create `axes.json` with every original regime as a key and its status coordinate names
as an array. Other state coordinates stay free. Original declaration order governs
certificate selection, even if the names in this file use a different order.

```json
{"working": ["married", "children"], "retired": ["married"]}
```

From a public checkout with its Pixi environment:

```console
pixi run -e tests-cpu python -m _lcm_regime_finder.implicit.implicit_native model.json axes.json facts.json --source-version source645-builtin-int-v1
pixi run -e tests-cpu python -m _lcm_regime_finder.implicit.verify_native model.json axes.json facts.json fact-receipt.json --source-version source645-builtin-int-v1
```

On local Linux development hosts, apply the documented `cap` launcher around each
command. Keep the original capture and all its stored companions together.

The producer returns `PRODUCED_UNVERIFIED`. The separate verifier checks every original
fact on the full original rectangle, including unkept points. It uses the original
source API and an independent decoder. Only a finished replay with unchanged source and
input guards returns `COMPLETE_FACT_EQUALITY`. This receipt certifies fact equality; it
does not certify a graph reduction.

The compiler retains conservative unknown results, actual transition producers,
registered laws, phases, joint read certificates and all source qualifications. If
binding-cache order independence cannot be proved, it follows the exact original query
order with one shared original cache.

## Check a supplied projection

A removal proposal is an array of `[original regime, status coordinate]` pairs. An empty
array checks the identity projection. Removing a status classification makes that
variable an ordinary coordinate; it does not delete it from the Model.

```json
[["working", "married"], ["retired", "married"]]
```

```console
pixi run -e tests-cpu python -m _lcm_regime_finder.implicit.check_projection model.json axes.json facts.json removed.json projection-receipt.json --source-version source645-builtin-int-v1 --expand-boxes
```

This command repeats independent full-domain fact verification, reconstructs original
support incidences, computes possible-support closure and fixed-certificate liveness,
and checks all eight finite premises, I1–I5 and a–c. Final guards bind the complete
transaction. A refuted premise retains its witness. An interrupted or invalid
transaction returns no new certificate.

The `--expand-boxes` switch is required because this diagnostic adapter can expand
exponentially, even for independent identity updates. Separate diagrams and support
factors remain the canonical representation. Neither compiler nor verifier has a
polynomial complexity guarantee. Exhausted resources are incomplete work, not proof that
the original graph cannot be represented.

`FINITE_PREMISES_PROVED` retains the original source and proof qualifications. It does
not authorize a native model edit, prove maximal removals, choose an attractive graph or
prove global optimality. Actions are retained; forced-action savings are not measured.
The existing witness-dependent search is unchanged because a compact counterexample can
select a different first witness.

## Qualify an evaluator version

Semantic descriptors contain the complete evaluator source map, query protocol,
evaluator identity and capture dialect. Admission records separately bind the descriptor
digest, current compiler/verifier/bridge/screen/registry code digest and reviewed
qualification receipt. Receipt and admission data are excluded from that code digest, so
their identities do not depend on themselves. Each transaction separately guards the
descriptor, admission table and receipt bytes.

The metadata-only qualification tool belongs to the checkout's test suite:

```console
pixi run -e tests-cpu python -m tests.regime_finder.implicit_version_qualification --source-version source645-builtin-int-v1 --cases cases.json --evidence qualification
```

Run it in a fresh process of the actual merged checkout. The new evidence directory must
not exist. The cases file selects unchanged metadata inputs; it cannot change the
mandatory controls:

```json
{"schema": "implicit-qualification-cases/v1", "cases": []}
```

The tool runs fixed differential, namespace, source-mutation, decoder and finite
projection controls and a genuine tiny native model fixture. Additional cases name
`name`, `snapshot` and `axes_file`; optional `removed` pairs specify a proposal. All
facts are independently replayed at every original address before the supplied
projection is checked. The tool returns neutral qualification evidence and never changes
the admission table or issues a production fact-equality receipt for an unqualified
version. An independent review must approve the exact evidence before a maintainer pins
its receipt. Changing algorithm bytes requires requalification.

Keep historical captures unchanged. Incompatible native type spellings are rejected, not
renamed. Historical v1 packets require their sealed matching implementation at
`8bc9edd17a776329f76fb593ef10287332e941d6`; modifying the algorithm changes its byte
identity even when the evaluator map is unchanged. A v2 replay can create a separate new
receipt from an unchanged compatible capture. It does not replace the old packet or its
receipt, or give an incompatible capture a native guarantee.
