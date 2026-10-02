# Stage 3 rerun at ec882386, reduced3 fp32 (checkpoint)

The Marvin subagent wrote this as its hand-back; the parent saved it, condensed but complete in substance. Evidence in this directory (`gpu/`, `analysis/`, `queue-status.txt`), `SHA256SUMS` over 74 files; 14 large npz/arrow files stay on Marvin under `~/pylcm-inv-jobs/stage3/out-2802415{5,6}`, indexed in `large-files.{sha256,sizes}`.

## Identity
- Clone `~/pylcm-inv-s3h` (own benchmarks-cuda12 env), detached at `ec882386ca8fe0717ed878b8eb8e7a74173c3df4`, clean, HEAD matched in both jobs. `pixi.lock`/`pyproject.toml` unchanged from 9f03c34e.
- Job 28024155 (sgpu018, 1×A100-SXM4-80GB): reduced3 fp32 A1 B1 B2 A2, WARM=3, driver `stage3_arms.py` (sha `b7b6f8e5…`), same sbatch as 28022853. All rc=0.
- Job 28024156 (sgpu026): `JAX_LOG_COMPILES=1`, cold only, B then A. Both rc=0.

## Parity (`compare_rerun.py`; comparator self-check passes)
- All 6 pairs among A1 A2 B1 B2 bitwise for cold, warm_same_1, warm_changed: 181/181, max ULP 0, equal non-finite masks; simulate panels sha256 equal.
- Positive control: 0/181 equal, same vs changed params, every arm.
- Stage 0 reduced3 A1 (job 28020721, f2b97fbb) vs A1 and A2 at ec882386: bitwise, panels included.

## Compile requests
| Call | A1 / A2 | B1 / B2 | B at 9f03c34e |
|---|---|---|---|
| Cold compiles | 181 / 181 | 196 / 196 | 206 |
| Cold traces | 153 | 168 | 178 |
| warm_same_1 | 25 / 25 | 25 / 25 | 25 |
| later warm | 0 | 0 | 0 |
| simulate cold / warm | 266 / 0 | 266 / 0 | 266 / 0 |

Logged-compile diff B vs A (`analysis/logc-ec882386-diff.txt`): only +8 `jit(_select_view_blocks)` and +7 `jit(_write_block)` (338 vs 323 "Compiling" lines). `_tile_block` (7) and `slice` (2) are gone.

## Timing (h:mm:ss, 1×A100)
| Call | A1 / A2 ec882386 | B1 / B2 ec882386 | A / B at 9f03c34e (28022853) |
|---|---|---|---|
| Cold | 0:04:09.5 / 0:04:08.0 | 0:04:10.9 / 0:04:12.0 | 0:03:57–0:03:58 / 0:03:58–0:03:59 |
| warm_same_1 | 0:00:09.6 / 0:00:09.7 | 0:00:11.9 / 0:00:11.9 | 0:00:09.6–9.7 / 0:00:12.0–12.1 |
| Steady warm_same_2–3 | 0:00:08.1–0:00:08.4, median 0:00:08.3 | 0:00:10.4–0:00:10.6, median 0:00:10.5 | A median 0:00:08.3 / B 0:00:10.6–10.9, median 0:00:10.7 |
| warm_changed | 0:00:09.0 / 0:00:09.0 | 0:00:11.3 / 0:00:11.3 | 0:00:09.0 / 0:00:11.5 |
| simulate_cold | 0:02:53.1 / 0:02:53.0 | 0:02:53.9 / 0:02:53.9 | 0:02:08–0:02:09 all arms |
| simulate_warm | 0:00:19.9 / 0:00:19.9 | 0:00:20.6 / 0:00:20.5 | 0:00:19.9–20.0 / 0:00:20.5–20.6 |

Different nodes (sgpu018 vs sgpu003). A's steady warm range is identical on both; simulate_cold moved equally in A and B (node effect). B/A steady-warm ratio ≈1.27 (was ≈1.29); ranges do not overlap.

Phase ledger, warm_same_2 (s):
| Phase | A | B ec882386 | B 9f03c34e |
|---|---|---|---|
| params_validation | 2.84 | 2.75 | 2.78 |
| structural_resolution | 0.46 | 1.26 | 1.38 |
| residency_inventory | 0.23 | 0.47 | 0.48 |
| compilation_waves | 0.59 | 0.85 | 0.85 |
| workspace_selection | 0.07 | 0.19 | 0.19 |
| backward_induction | 3.89 | 4.85 | 4.92 |

## Memory (`peak_bytes_in_use`: solves / after warm_changed+simulate)
| Arm | Solves | After |
|---|---|---|
| A1, A2 | 800,288,512 | 868,026,112 (same as Stage 0 and 9f03c34e) |
| B1 | 749,662,976 | 819,727,104 (same as 9f03c34e) |
| B2 | 749,747,712 | 819,885,824 |

B ≈ 6.3% below A. B2 is 84,736 B (+0.01%) above B1 with bitwise-identical values; at 9f03c34e B1 and B2 were identical.

## Other jobs (none cancelled)
| Job | Content | State | Est. start |
|---|---|---|---|
| 28022854 | 4×A100 assets partition, s3g at 9f03c34e | PENDING | 2026-10-02 23:25 |
| 28024079 | Production pair at 9f03c34e, s3p (released) | PENDING | 2026-10-03 03:25 |
| 28020582 | Stage 0 production A/A | PENDING | 2026-10-02 20:10 |
