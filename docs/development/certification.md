---
title: Certification and preflight
---

# Certification and preflight

An edit to a file under `src/` can turn a test red for a reason that has nothing to do
with whether the edit is correct. This page is about the machinery that does that, why
it exists, and how to re-anchor it honestly.

It also disambiguates one word. "Preflight" names three unrelated things in this
repository, and only the first is a runtime concept:

- **simulation preflight** — the action grids and gated-edge cohorts a simulation
  resolves before it runs; `tests/simulation/test_preflight_contract.py` and its
  ordering, state-reuse and buffer siblings belong here;
- **benchmark preflight** — `benchmarks/preflight.py`, reached through the pixi task
  `asv-preflight`;
- there is **no CI preflight stage**. The thing that behaves like one is the pre-commit
  hook `candidate-certificate-seals`.

## What the candidate certificate claims

`tests/candidate_certificate/direct_flow.py` proves route-local candidate-array flow
from `Q_and_F` to the full reducers, along nine named corridors — singleton, collective
and taste-shock, each in a dense solve, a streamed reference and a simulate variant. Its
own module docstring is the authoritative statement of the claim and of its boundary.

The boundary matters as much as the claim. The certificate says nothing about the
*economics*: user DAGs, constraints, transitions, continuation values, interpolation and
fold weights are an explicit semantic boundary and are not re-proved. What it proves is
that the arrays a route builds are the arrays that route reduces, with no intervening
candidate-changing expression, and that the program which was resolved is the program
which was lowered, compiled and dispatched.

The proof is deliberately strict: a new statement inside a certified transport corridor
is not assumed harmless. It has to be added to the explicit, independently checked
representation allowlist. That strictness is the whole value, and it is also why an
innocent refactor can turn the certificate red.

## Three kinds of pin

Only the first two are machine-repairable, and each by its own tool.

**Byte seals.** Every certified source is pinned by whole-file SHA-256 in two places:
the generated inventory `tests/candidate_certificate/sources.json` and the
hand-maintained `_SOURCE_SEALS` map in `direct_flow.py`. The inventory itself is derived
from the certificate's AST: `generate_sources.py` collects the repository-relative path
literal of every `_parse` obligation in
`tests/test_grid_search_candidate_certificate.py`, so the set of certified sources
cannot drift from the set the certificate actually parses. Editing any certified source
invalidates both. `check_seals.py --fix` repairs both.

**AST corridor pins.** For each certified source, `direct_flow.py` pins a *module
transport surface* digest and a per-callable AST digest for every callable the corridor
depends on, written as `"<Class>.<method>": "<sha256>"` entries. These are computed from
the parsed source with docstrings stripped, so they move when the code moves and not
when a comment does. `repin_corridors.py` recomputes them.

**Reviewable declarations.** Exact field tuples of the transport dataclasses, verbatim
enum bodies, module-level binding counts (`expected_binding_counts`), and
`relevant_import_names` / `expected_imports` lists. These are prose a reviewer reads. No
tool rewrites them; adding a field to `CoreExecutionRequirements` means adding the field
line to its pinned tuple in the same commit, and adding a module-level name means adding
its binding count.

On top of the three, the mutation matrices are sealed *by name*:
`EXPECTED_DIRECT_FLOW_MUTATION_NAMES_SHA256` and its supplemental and uniform-process
siblings hash the sorted mutation-name lists, with a count beside each. A mutation that
silently disappears is therefore a failure, not a smaller matrix.

### Anchors are named by search string, not by line number

Line numbers inside `direct_flow.py` move whenever a seal is added, and the file is over
ten thousand lines long. Every anchor is found by searching for the literal string — the
digest, the `class_name="..."` argument, the field line, the enum body. Any procedure,
note or review comment that identifies an anchor by line number is already stale. Cite
the string.

## The pre-commit gate

`.pre-commit-config.yaml` runs `candidate-certificate-seals` on any change under `src/`,
to `tests/candidate_certificate/`, or to the certificate module itself. It takes well
under a second, and it exists because a stale seal fails the certificate tests on every
CI platform and nowhere earlier.

It is in the CI `skip:` list, together with `keyword-only-convention`, `ty`,
`pixi-lock-check`, `pre-push-hooks-installed` and `notebook-cell-source-format`. So the
seals hook runs in developer clones and nowhere else: `prek run --all-files` locally is
the only place it fires. A clone that never installed the hooks gets no warning until CI
runs the certificate batteries.

## Re-anchoring after an intended edit

Run these from the repository root, in this order. Clear the certificate's `__pycache__`
between steps: each tool loads `direct_flow.py` from the tree under repair, and a stale
bytecode cache makes a repaired file look unrepaired.

```console
rm -rf tests/candidate_certificate/__pycache__
pixi run python tests/candidate_certificate/repin_corridors.py \
    --changed-source src/_lcm/execution/value_transfer.py \
    --changed-source src/_lcm/solution/backward_induction.py
rm -rf tests/candidate_certificate/__pycache__
pixi run python tests/candidate_certificate/check_seals.py --fix
rm -rf tests/candidate_certificate/__pycache__
pixi run python tests/candidate_certificate/verify.py --repo-root .
pixi run python tests/candidate_certificate/verify.py --repo-root . --self-test
```

Both `verify.py` runs must exit 0. The second one re-runs the mutation controls and is
the check that the certificate can still *fail*: a control that no longer fires is a
failure, not a pass.

`repin_corridors.py` is deliberately narrow.

- It recomputes digests with `direct_flow.py`'s own helpers, never a local
  reimplementation, so a change in how the certificate hashes a callable cannot disagree
  with how the tool re-pins it.
- It rewrites only pins owned by the sources named with `--changed-source`. If a pin
  owned by any *other* certified source drifted, an unintended edit reached that source
  and the run is refused without writing anything.
- It refuses a name pinned to two different digests, and any pin it cannot attribute to
  exactly one source, rather than guessing.
- It never touches the reviewable declarations. When the verifier still reports errors
  after a re-pin, the tool prints them: those are field tuples, enum bodies and binding
  counts for a reviewer to edit.
- `--check` reports drift and exits non-zero without writing, so it is safe to run on a
  tree you do not intend to change.

The governing rule, which the architecture transition ledger also states:

> Any callable move, bridge deletion, or source-route change deliberately regenerates
> and reviews the source inventory. Never refresh a seal merely to make a changed route
> green.

Widening an allowlist, deleting an anchor, or resealing past an error you did not
understand all convert a proof into a formality. If an anchor's failure is not
explicable in one sentence, it is a regression until shown otherwise.

## Stale seal, or real regression?

`check_seals.py` distinguishes the two by exit code.

- **0** — the inventory, the seal map and the corridor pins all agree with the tree.
  Nothing to do.
- **1** — a *byte* seal drifted and nothing else. A certified source's bytes changed but
  every corridor pin still matches, so the change moved no certified transport: a
  comment, a docstring, an unrelated helper. `check_seals.py --fix` is the whole repair.
- **2** — the verifier reports an error that is not a seal. This is the interesting
  case, and it splits again:
  - the error names a **callable digest or module surface** ⇒ certified transport moved.
    If you moved it on purpose, `repin_corridors.py --changed-source <path>` re-anchors
    it. If you did not, you have found a regression: some edit changed a corridor you
    were not editing, and the tool will refuse to write for exactly that reason.
  - the error names a **field, enum member, binding count or import** ⇒ the declaration
    and the source disagree. Decide which is right and edit the losing one by hand.
  - the error names a **corridor statement** — an expression that is not in the
    representation allowlist — ⇒ a new statement entered a proved route. This is never
    repaired by re-pinning. Either the statement is candidate-preserving and belongs in
    the allowlist with a reviewed justification, or the route no longer proves what the
    certificate claims.

One exit-2 failure is neither of the three. If the message is a plain
`source seal mismatch` that survived a `--fix` which reported success, the reseal
skipped that entry rather than the entry being unrepairable: `check_seals.py` finds seal
lines with a regex over the constant's name, so a name class that misses a legitimate
spelling leaves exactly one seal stale and dresses it as a corridor violation. Before
hunting a regression, re-run `--fix` and check whether the same source is still named.

A useful sanity question: does the failure mention a source you edited? If not, treat it
as a regression first and a stale anchor second.

## Mutation controls and the self-test

`verify.py --self-test` runs the mutation matrices: deliberate defects seeded into the
certified routes, each of which the certificate must reject. The counts and the hash of
the sorted mutation names are themselves pinned in `direct_flow.py`, so the matrix
cannot shrink quietly.

`tests/candidate_certificate/controls/` holds five standalone controls (`mt5`–`mt9`)
covering the source set binding, multi-feasible simulate, policy anchor binding, rank
neighbourhood and direct-flow route. They are cheap individually and exhaustive in
aggregate; the exhaustive population is why the source-contract lane exists rather than
running everywhere.

The point of all of it is that a green certificate means something. A probe that reports
a negative has to be shown capable of reporting a positive in the same run, and the
mutation matrix is that demonstration.

## Where certificates run in CI

`.github/workflows/cpu.yml` gives the certificate work three homes.

- **`tests-source-contract`** runs
  `tests/test_simulation_candidate_program_certificate.py` and
  `tests/test_uniform_process_grid_certificate.py` on one fp64 Linux lane only. They are
  source-only campaigns: they parse and hash, they run no model, and so they carry no
  platform signal worth paying for four times.
- **`tests-certified`** runs the slow certified numerical batteries, including
  `tests/solution/test_nbegm_multi_discrete_agreement.py`.
- Every general lane keeps `tests/test_source_certificate_portability.py`, whose four
  controls are what makes "the exhaustive population runs on one lane" safe: they prove
  on each platform that the certificate's file reading and hashing are
  checkout-independent, so a one-lane exhaustive run generalizes.

`tests/ci/test_ci_workloads_manifest.py` asserts that the two source-only modules are
selected by the source-contract lane and by nothing else.

## The CI workload manifest

`tests/ci/ci-workloads.json` records every pytest invocation `cpu.yml` makes — lane, OS,
precision, worker count, isolation kind, selection expression, JUnit filename — together
with the test files each invocation selects. `tests/ci/ci_workloads.py` reads it back;
nothing reads it to decide what to run at CI time, so it is a *contract*, checked
against the workflow rather than driving it.

Facts worth knowing before you touch it:

- **`frozen_head`** names the commit the per-file weights were measured at, not the
  commit the manifest was last edited at. It moves when the weights are re-measured and
  at no other time; registering a new test file leaves it alone.
- **Weights are per leg.** `leg_weights` holds one OS/precision combination each
  (`fp64-linux`, `fp64-macos`, `fp64-windows`, `fp32-linux`, `fp64-solution`,
  `fp32-solution`). The cross-leg totals in `file_weights` sum four general legs, so
  dividing them by the leg count would over-provision the cheap platform and
  under-provision the expensive one.
- **A file with no measurement is never weighted zero.** It is listed in
  `unweighted_files`, which is a visible gap rather than a free pass.
- **Shards are derived, not hand-assigned.** `tests/ci/shard_test_files.py` partitions
  the recorded `general_shard_universe` by decreasing per-leg weight into the currently
  lightest bin — a longest-processing-time partition whose result depends only on its
  inputs. Under `--dist loadfile` a whole file is one indivisible atom, so the largest
  file is a lower bound on its bin no matter how many bins there are.
- **Guardrail budgets** (`tests/ci/test_ci_workloads_guardrails.py`): 24 minutes of
  payload and 30 minutes total per job, and 30 seconds for an ordinary `notslow` test.
  These are operating budgets, enforced only against files that have an observed weight.

### Registering a new test file

Adding a file under `tests/` without recording it here fails
`test_every_test_file_is_in_the_manifest_or_explicitly_excluded` in CI and nowhere
earlier — the file is selected by no invocation, so it runs on no lane and still looks
green.

```console
pixi run -e tests-cpu python -m tests.ci.generate_ci_workloads
pixi run -e tests-cpu python -m tests.ci.generate_ci_workloads --check
```

The first call registers every unaccounted test file in `general_shard_universe` and
`unweighted_files` and re-derives the `files` list of every general shard. The second
exits non-zero whenever the committed manifest differs from regeneration, which is how
you check a manifest you did not just write. Both leave `frozen_head`, the observed
weights, the exclusion list, the shard counts and every non-general lane untouched.

A new `cpu.yml` invocation is a different change and stays a reviewed hand edit, checked
by `tests/ci/test_cpu_workflow_contract.py`.

## Pinned test files, and how to move one

Several test files are pinned **by path and by classname** outside the file itself.
Moving or renaming one silently breaks things that stay green:

| Pin site                                                                        | What it pins                                                                                                                                                                             |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cpu.yml`, `--deselect=` lines in the `tests` and `tests-fp32` jobs             | Three timing-witness node IDs in `tests/simulation/test_compile_requests.py` and `tests/simulation/test_preflight_contract.py`, kept off the general lanes so they never run under xdist |
| `cpu.yml`, the `tests-timing` job                                               | The same three node IDs, selected positively                                                                                                                                             |
| `cpu.yml`, the `tests-source-contract` job                                      | `tests/test_simulation_candidate_program_certificate.py` and `tests/test_uniform_process_grid_certificate.py` by path                                                                    |
| `cpu.yml`, the `tests-certified` job                                            | `tests/solution/test_nbegm_multi_discrete_agreement.py` by path                                                                                                                          |
| `cpu.yml`, the topology jobs                                                    | Every four-device and eight-device witness, each invocation naming one file by path                                                                                                      |
| `tests/ci/check_simulation_timing_report.py` (`EXPECTED_CASES`)                 | Two module *classnames* × three functions × two witnesses = six expected rows                                                                                                            |
| `tests/ci/test_simulation_timing_report.py`                                     | The same classnames, in fixtures and in mutation cases                                                                                                                                   |
| `tests/ci/test_timing_population.py`                                            | The same two sibling modules                                                                                                                                                             |
| `tests/ci/ci-workloads.json`                                                    | The file paths, the deselected node IDs, `file_weights`, and `per_test_ceiling_backlog`                                                                                                  |
| `tests/candidate_certificate/generate_sources.py`                               | `tests/test_grid_search_candidate_certificate.py`, the certificate the inventory is derived from                                                                                         |
| `tests/candidate_certificate/check_seals.py`, `verify.py`, `repin_corridors.py` | `tests/candidate_certificate/direct_flow.py`                                                                                                                                             |
| `.pre-commit-config.yaml`                                                       | The `candidate-certificate-seals` `files:` regex, and the `name-tests-test` exclusion regex naming `^tests/ci/`, `^tests/candidate_certificate/` and `^tests/conformance_solver/`        |

To move a pinned test file:

1. Update the `cpu.yml` deselections and the `tests-timing` node ID list.
1. Update the classnames in `tests/ci/check_simulation_timing_report.py`,
   `tests/ci/test_simulation_timing_report.py` and `tests/ci/test_timing_population.py`.
1. Regenerate `tests/ci/ci-workloads.json` and hand-edit any node ID that names the old
   path.
1. Check the `.pre-commit-config.yaml` regexes.
1. Run the whole of `tests/ci/` — several contract tests pin workflow properties, so
   running a single file proves nothing.

A deselection that no longer matches anything does not fail. It silently stops
deselecting, and the timing witness then runs under xdist on the general lanes, where
its measurement is meaningless.

## The timing-witness lane

`tests-timing` runs the three timing witnesses in a lane of their own, serially, on
Linux fp64, Linux fp32 and Windows fp64. macOS is deliberately absent: its hosted
runners never satisfy the steadiness precondition, and since a declined row counts as a
failure the lane could only ever be red. Each witness compares a host-time ratio between
two log levels, which is a statement about pylcm's runtime validation and carries that
meaning only when the machine served both legs comparably.

When it did not, the row **declines to measure**: it skips with a reason prefixed
`TIMING_UNSTABLE_HOST` (`tests/ci/simulation_timings.py`), and
`check_simulation_timing_report.py` reads that as a measurement not taken. Any *other*
skip is refused exactly as a failure is, because it means the bar did not run and
nothing said why. The steadiness ceiling is a relative interquartile range of 0.15 on
the control leg.

The checker also verifies the complete six-row population, not just the verdicts: a lane
that produced four rows and no failures has not passed.

## Test policy and markers

`pixi run test` is the policy launcher (`tests/ci/policy_launcher.py`); see
[Continuous integration](continuous_integration.md) for the policy tiers, the hardware
profiles, and the four markers a test uses to declare its contract.

## Coverage

Every lane that runs under coverage uploads a `coverage-*` artifact. A single `coverage`
job downloads all of them, runs `tests/ci/check_coverage_manifest.py` against the
manifest's `coverage_contributors` list — 16 lanes — and uploads one combined report
under the `cpu-python` flag.

The checker is the gate, not Codecov: `codecov.yml` sets `after_n_builds: 1`, so Codecov
publishes as soon as the single upload arrives. What guarantees completeness is that the
combine step refuses to proceed unless every recorded lane delivered a non-empty report,
and refuses an *unexpected* artifact too — an extra lane means somebody added coverage
without anyone deciding whether it belongs in the published number.

Adding or removing a coverage-producing lane therefore means editing
`coverage_contributors` in the manifest in the same change.
