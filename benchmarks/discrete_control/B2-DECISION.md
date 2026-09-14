# B2: retained original contract, unresolved exact-pin/applicability decision

B2 is not invented by the scaling follow-up. The immutable round1 profile contains:

- `measurement_protocol.comparison`: “Same harness, economic inputs/parameters/grids,
  device/precision/toolchain, allocator/cache, retention and seeds. Separate
  parent->head, handoff->head, main->head.”
- `repair_review_triggers.comparison`: “Same-control parent->head and separately
  main->head, outside observed noise. These are review triggers, not universal
  acceptance limits or permission for numerical changes.”

Authority is bundle-round1/protocol/profile-contract.json in canonical25, retained
unchanged as contract/profile-contract.json in this packet. Policy digest:
9949d584371c98f758dbc1f4200f6265e07186e5c2f552b8ff08f270cb71b4c4.
The required profile is balanced. The implementation-delta request explicitly says
“Review accepted22 implementation under unchanged profile” with this digest.
Returned audit-result.json resource_report B2 is main→head, not_run, with no exact
main pin or matched resource series. NEXT-EVIDENCE.md requires any main→head series
required by that resource contract to remain separately pinned and labelled.

These clauses establish why main/head remains in the original resource disposition.
They do not select a valid main commit, prove harness compatibility, authorize a job,
or require this B1 control to run through an incompatible historical API. “Main” at a
moving branch tip is not a scientific pin. No main revision is invented in this packet.
A prior local note that one explored main lacked the required interface is not a scope
waiver and is not used as an executable control here.

Exact outstanding maintainer decision:

1. Retain full balanced resource acceptance as the target and supply an exact main SHA
   plus a scientifically matched supported control/compatibility contract for B2; then
   prepare its own bounded owner work order, without relabelling B1 as main/head; or
2. Explicitly narrow/defer resource acceptance with B2 remaining open, or record a named
   B2-specific waiver/applicability disposition and its rationale in the canonical
   acceptance state. An unqualified clean/full-resource label cannot silently drop B2.

B1 preparation and existing continuous job27534935 proceed only in their already
specified scopes. This B2 decision need not block preparation or harvest of them,
but their completion cannot settle B2. No ACA work or new execution follows from
this document. Integration owner retains top-level07/19 and the T6 acceptance record.
