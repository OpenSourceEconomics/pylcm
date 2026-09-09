# Compiler allocation admission

Budgeted admission enforces both the raw compiler peak and the represented allocation
storage on each device. For a complete record, the reservation is
`max(peak, argument + output - alias + temporary)`. Existing external retained residency
is added afterward. Each record's categories stay paired before device maxima are taken.
`compiler_peak_bytes` and published `peak_bytes` remain raw compiler peaks.

The supported report contains nonnegative integral peak, argument, output, alias, and
preallocated temporary counts, plus explicit host argument/output/alias/temporary
counts. Aliases cannot exceed either their argument or output category. Missing,
malformed, empty, inconsistent, and unpaired columnar reports refuse budgeted admission.
The raw peak accessor retains its independent normalization contract, including columnar
peaks. An absent allocation counter never becomes zero and a small raw peak cannot
override represented temporary storage.

Nonzero host allocation counts refuse this default-memory reservation because the pinned
peak calculation does not reliably separate memory spaces. Generated-code metadata is
excluded from this allocation contract and alone causes no host-space refusal. The
existing offload path moves arrays to an actual CPU device and compiles separate CPU
assembly. Its host storage remains excluded from a GPU ceiling; selected CPU execution
accounts for its own CPU storage. Unbudgeted execution is unchanged.

The reservation covers represented requirements. Generated code, omitted runtime
workspace and constants, thread stacks, allocator overhead, and executable-cache storage
remain outside a complete runtime bound. The CPU evidence below establishes this
allocation contract on the captured executable. It establishes no GPU capacity claim.

## Captured CPU witness

The implementation base is `1f554a30c49ad6ddfaa8ead541ad40431e7d7235`. The public
witness uses two actions, two subjects, a sorting feasibility constraint, a 16,384-byte
budget, Python 3.14.7, JAX/JAXLIB 0.11.1, CPU, and float64.

| Captured counter                         |   Bytes |
| ---------------------------------------- | ------: |
| Raw peak                                 |      90 |
| Arguments                                |      56 |
| Outputs                                  |       2 |
| Aliases                                  |       0 |
| Preallocated temporary allocations       | 131,072 |
| Each host allocation counter             |       0 |
| Default and host generated-code counters |       0 |
| Computed represented reservation         | 131,130 |

The original regression failed when the admitted executable reached the numerical
dispatch boundary. A guard intercepted that call before executing the feasibility
producer. After the central change, the same regression passes with unchanged counters
and no dispatch attempt. Both runs retain the complete scalar report, serialized buffer
assignment, and HLO text at the compile seam.

The saved assignment contains 31 logical buffers, 30 allocations, and zero heap traces
or events. Allocation 29 has size 131,072, color zero, and none of the entry-parameter,
live-out, constant, or thread-local flags. Its logical buffers 44 and 45 share offset
zero. Static charges reproduce 90 bytes exactly: 56 argument bytes, 2 output bytes, and
32 constant bytes. The temporary allocation receives no heap-event charge in this
executable. The represented reservation removes sharing within the allocation through
the compiler's own allocation count.

The raw peak includes those 32 constant bytes; the allocation expression excludes them.
Taking the maximum of the two reports leaves simultaneous coverage of the constants and
temporary allocation unestablished. This is one concrete limit on the represented-buffer
reservation's scope.

The interpretation follows the wheel's recorded JAX revision
`2d66622450e2c8633cda2307688ef7aa294bd6eb` and its pinned XLA revision
`dcf304bc5dca1932b99f740b911dbd73631a1a69`:

- [Allocation category accounting](https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/compiled_memory_stats.cc#L34).
- [CPU report construction](https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/pjrt/cpu/cpu_client.cc#L1177).
- [Static roots and heap-event peak calculation](https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/buffer_assignment.cc#L3459).
- [Serialized allocation and trace schema](https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/hlo.proto#L644).

The schema SHA-256 is
`ef83e18f357ef5e9f75948a64293988b166abf73ad5090af2f93e012ed760f2a`. The captured
assignment SHA-256 is
`35c58cd22ea00baaacd553b47b9c57c295b1bf563d05169a09a31b95c6cfaf51`. The bounded decoder
encountered no unknown fields. Exact instruction-ID mapping remains limited by the
absence of a serialized HLO module; the retained text contains the sorting operation.

## Previously captured GPU record

The separately authorized
[IAME diagnostic](https://github.com/OpenSourceEconomics/pylcm/actions/runs/34390034490)
captured a complete GPU report on source `c710f9b522ec6041830211d9eaf2536f1c842826`,
fp64, JAX 0.11.1, and a Tesla V100. Its arguments, outputs, aliases, and temporary bytes
were 7,093,116, 1,772,928, 0, and 5,803,032,504. Their allocation expression is
5,811,898,548 bytes; the raw peak is 5,811,906,304 bytes and remains the computed
reservation. Host allocation counters were zero. Generated-code metadata was 31,512
bytes, which leaves the allocation contract unchanged.

That diagnostic failed a 5,803,032,576-byte contiguous allocation with a
4,294,967,296-byte largest reusable block and 5,620,367,360 bytes of remaining
allocator-pool growth. A represented-buffer reservation does not establish contiguous
allocation feasibility or resolve pool fragmentation. This reused record is arithmetic
evidence from the prior diagnostic; no GPU execution of this repair was performed, and
the original asynchronous failure site remains outside its conclusions.

## Migration inventory

| Budgeted consumer                                            | Reservation and ownership behavior                                                                                                                                                                  |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `execution/workspace_planning.py`                            | Computes complete compiler records, ranks widths by reservation plus external residency, returns separate raw peak and reservation metadata.                                                        |
| `solution/backward_induction.py`                             | Caches complete reports by lowering, pairs each donation variant with its own residency, compares reservations in compilation waves and final selection, logs raw peaks under their existing label. |
| `simulation/runtime.py`                                      | Reads the exact executable's complete report and rechecks call-owned original, placed, and compiler-eliminated arguments through the existing residency inventory.                                  |
| `simulation/host_operations.py`                              | Caches complete accounting with executable code; preserves `keep_unused=True` and fresh physical-buffer residency for each call.                                                                    |
| `simulation/solution_copies.py`                              | Uses complete cached accounting while retaining original and earlier private copies.                                                                                                                |
| `simulation/process_grids.py`                                | Uses complete accounting for the supported uniform-grid and staged normal-support producers, including cumulative call-owned support and temporary buffers.                                         |
| `simulation/chunk_planning.py`                               | Adds stage reservations to actual retained and future output storage; retains raw stage peaks separately and preserves separate CPU assembly.                                                       |
| Feasibility, action-grid, padding, and diagnostic operations | Reach the central planner directly or through the profiled-operation owner, preserving refusal before dispatch.                                                                                     |

Compiler aliases only remove represented argument/output overlap. Existing physical
buffer unioning and compiler-input metadata determine which caller-owned spans can be
excluded from external residency. Eliminated inputs, uncovered source spans, and prior
outputs stay charged. Source ownership and per-device transfer semantics are unchanged.

## Executable controls

`tests/execution/test_compiler_allocation_reservation.py` covers temporary-dominant and
peak-dominant reports, valid and invalid aliases, heterogeneous device records, complete
counter requirements, explicit host-space refusal, generated-code metadata, cached
lookups, external residency, and unbudgeted behavior.

`tests/simulation/test_compiler_allocation_admission.py` retains the original public
refusal witness. Existing feasibility tests check sufficient-budget values, selected
actions, caller arrays, and diagnostics. The two downstream diagnostic-pressure cases
use cheap predicates so their intended gathers and per-constraint producers can be
reached before refusal.

The gather-pressure test still requires exactly two cohort gathers, for summary and
serial replay, followed by refusal before the diagnostic-subset gather dispatches. The
predicate-pressure test still requires exactly three gathers and refusal of the
individual `_batched_feasibility_check` producer. Both retain their invalid-result
predicate, original budgets, population size, and dispatch guards; the latter retains
eight additional diagnostic predicates. Replacing only the sorting predicate with its
existing parameter-only fixture lets these downstream assertions run under the stricter
admission rule. The separate public witness retains the expensive sorting predicate.

The certificate preserves its seven existing populations totaling 538 controls. Fourteen
separately named `allocation_reservation:` controls cover central arithmetic, malformed
fields, host space, per-device reduction, raw metadata, cached operations, solve waves,
donation variants, and chunk consumption. Production-source coverage remains 116 paths;
test, certificate, and evidence binding manifests have a separate larger population.
