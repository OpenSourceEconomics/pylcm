# Execution, audit and external-owner routing

The authoritative planning repository for this checkout is
`/home/hmg/sciebo/pro-audits/pylcm-architecture-performance-plan`.
Read its `19-CURRENT-STATUS.md` to resolve current decisions and evidence before working
on the active architecture effort. Preserve all full documents 01–19 and the review
folders in place. If unavailable, recover the authoritative source or record the blocker;
do not reconstruct a contract from an old summary.

## Required contracts by surface

Read the relevant full contracts, not only the checkpoint's success narrative:

- Ownership or memory lifetime: `01-ARCHITECTURE-AND-OWNERSHIP.md`,
  `03-CROSS-SOLVER-EXECUTION.md`, and the adopted slice plan named by current status.
- Public solver, solution, persistence or artifact boundaries:
  `02-PUBLIC-SOLVER-AND-SOLUTION-API.md`, plus the affected public reference pages.
- Execution, sharding, transfers or admission: `03-CROSS-SOLVER-EXECUTION.md`,
  `08-M1-DESIGN-SPEC.md` and the applicable `15-M1-SLICE-3-PLAN.md`,
  `16-M1-SLICE-4-PLAN.md` or `18-M1-SLICE-5-PLAN.md`. Follow their decision references.
  Also read the repository's `docs/development/compiler_allocation_admission.md`,
  `docs/development/collective_resource_contract.md` and
  `docs/development/feasibility_admission_evidence.md` when making admission claims.
- GridSearch or NB-EGM algorithm changes: respectively `04-GRIDSEARCH-IMPROVEMENTS.md`
  or `05-NBEGM-IMPROVEMENTS.md`, plus the adopted implementation contract and math/JAX
  routes. Other solver work starts from `06-OTHER-SOLVERS.md`.
- Sequencing and next implementation choice: `07-IMPLEMENTATION-ORDER.md` and the
  current decisions in `19-CURRENT-STATUS.md`.

A route is an entry point, not permission to ignore dependent consumers or a protocol's
required reading. An audit requiring all 01–19 still reads all of them.

## Audit intake and packaging

Select the applicable audit skill by the actual scope. Use its canonical builder,
transport and intake procedures and complete required reading. Record the protocol
version and source pin for the round; reconcile version changes explicitly. Verify
transport and content identity before ingesting a repeated delivery, retaining evidence
of the delivery even when the findings are duplicates. Keep source conclusions,
reference checks, supplied native receipts and independently executed acceptance distinct.

## ACA and cluster ownership

Consult current status and its linked owner protocol/request before any job action.
Preserve the current allocation, owner, cohort, grids, source pin and acceptance gates.
Collect terminal evidence through the existing authorized owner/mechanism; an inherited
job ID is not live status or authority to restart, repin or cancel it. Mark an unknown
owner or stale observation explicitly. Never send a message or launch a new allocation
without authorization. Existing authorization remains valid; do not ask for it again.

Read the memory skill before heavy work. Local batteries share one aggregate cap budget;
serialize them. Independent Marvin allocations can run concurrently only with explicit
CPU/memory limits and the applicable owner contract. Never cancel by shared account.
