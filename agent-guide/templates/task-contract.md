# TASK-<ID>: <one verifiable outcome>

Status: proposed | approved | running | blocked | accepted
Baseline: <full commit SHA; dirty-tree rule>
Authority: <adopted decision + version/hash>
Implementation owner: <role>
Test / cluster owner: <existing authorized role>

## Outcome and invariants
<What changes; what must not change. Include exact numerical/resource/RNG/API
obligations where relevant. Do not substitute “tests pass” for the contract.>

## Scope and entry points
<Verified paths and symbols; affected consumers; allowed changes; non-goals.
A path list is a starting boundary, not permission to ignore an actual dependency.>

## Decided design
<The approved mechanism and its rationale. Identify remaining implementation
freedom and any unresolved decision that prevents implementation.>

## Oracle and test fixture contract
<Independent expected result or reproducer; genuine metadata/fixture classes;
identity/key conventions; dtype, shape, placement and ownership distinctions.>

## Acceptance sequence
1. <Collection/type/fixture smoke; command or source-verified selector.>
2. <Representative positive and negative cases.>
3. <Required precision/device/public-API matrix.>
4. <Integration, certificate, performance and resource gates.>

For each gate record: source head, environment, owner, command, exit code,
raw evidence path/hash, result, and what the gate does NOT establish.
Record exact conditions under which existing evidence may be reused.

## Resource and authority limits
<Permitted queues/resources, population/grids, walltime/watchdog, memory bound,
active-job ownership, approvals, sole-committer rule. No implied new allocation.>

## Escalation / external dependencies
<What would require a new decision; which receipt/job is pending; freshness time.>
Repeated approach without new evidence -> bounded escalation, not silent acceptance.

## Completion
<Required patch/commit, evidence and checkpoint update. Any remaining gate stays
explicitly open. A source-design review is not a native execution receipt.>
