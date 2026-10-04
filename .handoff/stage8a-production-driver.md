# Stage 8A production driver

The component engine and production driver are published in draft PR497.
The original worker qualification below freezes
`0019faab26766a0bbffb24d786f873e05a16aa67`. This branch has
locally qualified production routing, persistence, comparison and refusal
checks; its publication checks are recorded separately from this worker receipt.
Native ACA construction and the
required simultaneous three-node × eight-GPU run on **mlgpu_short** remain pending.

The exact partition binding is `mlgpu_short` in both owner recipes and collected
worker admission. Local partition detectors first failed three cases in
0:00:03.515, then the bounded affected-protocol set passed37/37 with zero
fail/error/skip in 0:00:03.846. Literal `mlgpu`, unknown and `sgpu_short`
worker receipts remain refused. Direct driver Ruff/Ty and both recipe syntax
checks pass. Recorded receipt fixtures establish protocol behavior, not physical
GPU admission or measured fit. Raw receipts are in
`stage8a-driver/partition-short/`; core source and certificates are unchanged.

Both output filesystem guards select `findmnt --types lustre`. Recorded stacked
autofs/Lustre mounts first failed both recipe flows with exit98 in 0:00:00.052;
the affected recipe module then passed21/21, zero fail/error/skip, in
0:00:00.286. Plain ext4, unknown and absent mounts return98 before any `srun`.
Direct driver Ruff/Ty and recipe syntax checks pass. Receipts are in
`stage8a-driver/lustre-mount/`; this is shell admission proof, not physical I/O evidence.

Post-port verification on this actual branch passed124/124, zero fail/error/skip,
exact JUnit0:00:29.634. All18 explicit Python gates and both shell syntax checks
passed; inherited helper diagnostic messages remain equal to the0019baseline.
Driver/helper imports resolve to this branch, all320 protected core hashes and
source/installed version/nativeCPU READY match before/after. The parent carried
only the optional helper keyword delta, preserving its current rounding and CI
content. Receipts are in `stage8a-driver/post-port-0019/`; these are local checks,
not nativeCUDA or submitted-job evidence.

See [the driver guide](drivers/stage8a/README.md) for the current CLI, complete
frozen economic workload, separate matched one-node reference, immutable
planning/reference receipt binding, owner installation and native gates.

The final bounded CPU test set passed **124 cases**, with zero failures,
errors or skips, in exact JUnit time **0:00:29.697**. It combines actual tiny
numeric/public-persistence checks with explicitly recorded runtime/CLI
bindings. It does not prove full ACA construction, native CUDA admission,
memory fit, physical exclusivity or actual distributed overlap.

Direct managed Ruff 0.16.6 and Ty 0.0.79 pass for all 18 new Python files;
format, maintained keyword-only convention and both shell syntax checks pass.
The changed inherited builder is checked separately: baseline and candidate
both report43 mapped Ruff diagnostics and 2 Ty diagnostics, with no new
message-level diagnostics. These inherited exit1 checks are not clean gates.
Only its optional preconstruction execution-config keyword and its typing
change; preserve the current parent's helper when carrying this minimal delta.

The authenticated test prefix/source is the committed component engine above,
version `0.0.2.dev257+g0019faab2`; all 320 source hashes remain unchanged.
The frozen ACA model is `ad38653696ec366e318ac61b9a81b597a4ecb700`.
Slurm `b650981593437799ae3235d1d912ea2cd9d5feda` was diagnostic source only;
a compatible production Slurm pin still needs verification. The local full
construction diagnostic refused unsupported hardware before Model construction.
There is no crop, alternate hardware fallback or construction-success claim.

Task evidence is under `stage8a-driver/completion/next-grant/`, including raw
XML/logs, ordered RED/GREEN observations, explicit tool commands, inherited
baseline comparison and final source identity. Historical receipt failures
remain unchanged. Neither reference nor production has been submitted.
