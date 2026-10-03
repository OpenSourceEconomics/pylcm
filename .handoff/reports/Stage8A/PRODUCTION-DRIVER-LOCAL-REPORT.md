# Post-port driver verification on current 0019

The actual target worktree is
`/home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified`,
HEAD 0019faab26766a0bbffb24d786f873e05a16aa67. No candidate/source files were
changed by this verifier. Parent prose changes are separate.

All 124 cases selected from the transferred `.handoff/drivers/stage8a` directory
passed, with 0 failures/errors/skips, exact JUnit **0:00:29.634**:
`focused-post-port.xml`, full verbose log, exit0. This uses actual public tiny
CPU archives plus explicit binding/runtime recorders, not full ACA/native
acceptance. No195/600/full numerical suite, install or native job ran.

Direct managed Ruff 0.16.6 and Ty 0.0.79 pass for the18 explicit new Python
targets. Ruff format confirms18 files; maintained keyword-only exits0. Both
sbatch files pass bash-n. Scripts and raw exit/log receipts preserve exact
commands and tests/* versus benchmarks/* established Ruff policy mappings.
The Ty command uses actual own type-checking prefix, explicit source targets,
no-respect-ignore-files/no-force-exclude and frozen ACA source search roots.
No test skip, broad metric/private ignore or no-files acceptance.

The actual current helper was also explicitly checked against
`git show0019faab:.handoff/drivers/stage5b/stage3_arms.py`. Ruff baseline and
candidate each have43 diagnostics and exit1; Ty each has2 diagnostics and
exit1. Exact diagnostic message multisets match, zero new diagnostics:
`helper-diagnostic-comparison.json`. Ty retains unresolved CPU
nvidia.cuda_runtime and inherited np.savez kwargs typing. These are baseline
comparison results, not clean whole-helper gates. No helper was overwritten.

`imports.json` authenticates driver and builder imports from THIS target:

- DriverSHA 2105aa053942da89d69e64d35fae3e34fdc49fb05c9c8334ea6ddc513d7fdf5a.
- HelperSHAfecf0ee4f27470dfc92992ae86b3735d8ff534377909a7a62a15ca361cb54247.
- Actual tests-cpu Python/prefix is under this target's own `.pixi/envs`.
- PYTHONPATH contains only currenttarget/src:currenttarget, never olddriverroot.

`identity-before.json` and `identity-after.json` exactly match: all 320 protected
core files match the published postcommit proof, nativeCPU READY,
JAX/JAXlib 0.11.1, CPU/fp32 and installed/generated
0.0.2.dev257+g0019faab2. Both proof calls exit0; exact before/after cmp exits0.
`final-source.sha256` equals the tested Python/helper/sbatch hash list
(`source-after-matches-tested.exit0`). Current source selection is explicit;
hidden-file exclusion did not determine test or Ty coverage.

Owner selector/restore fixtures are authenticated from frozen
ACA-modelad38653696ec366e318ac61b9a81b597a4ecb700 via the existing archived
STAGE8A_ACA_MODEL_GIT configuration. Diagnostic Slurmb650 source search roots
remain diagnostic, not selected production compatibility or construction
proof. Full native reference/N3×8 mlgpu jobs remain unsubmitted.

Execution checked hostnamehmg-office and cap.slice Units0/Tasks0 before
starting. Commands were serialized capped zsh/Pixi --as-is, pytest-v-n0 with
absoluteJUnit and 30-second blocking waits. All processes finished; observed
cap.slice Units0/Tasks0 at release. Parent was notified promptly. No commits,
pushes, network or source/code mutations by this verifier.

Read `final-junit-summary.json` for parsed exact counts/time. The copied
final-proof script's printed final-selector list is empty because that print
filter names earlier evidence; its complete summary JSON correctly includes
focused-post-port.xml. It is not used as a missing-selection acceptance claim.
