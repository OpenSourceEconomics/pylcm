# Stage 3 rerun at 9f03c34e on Marvin (checkpoint)

The Marvin subagent wrote this as its hand-back message; the parent saved it, condensed but complete in substance. Evidence is in this directory (`gpu/`, `analysis/`, `drivers/`); the top-level `SHA256SUMS` covers 408 files. 39 large npz/arrow files stay on Marvin, indexed in `large-files.{sha256,sizes}`.

## Summary
- Blocked (B) values equal unblocked (A) bitwise at every precision and workload run, and A at 9f03c34e equals Stage 0's A.
- B's cold solve issues 24–25 more compile requests than A, at both 2 and 3 types: block-assembly and view-selection helpers compiled once per distinct shape. Warm compile counts are equal. Plan §12 counts unexpected compile growth as a structural failure.
- B's steady warm solve is 20–29% slower than A, outside the A/A range.
- Production pair job 28024079 (A1 then B1, 4×A100 fp32, WARM=2, clone `~/pylcm-inv-s3p`) was submitted held.
- 4×A100 assets-partition job 28022854 is pending.

## Jobs (single A100-SXM4-80GB unless noted; every run rc=0, HEAD matched)
| Job | Content | Result |
|---|---|---|
| 28022846 | Reproducer reduced3 B0 fp32 cold, plus debug DBGB reduced3/reduced2 | Passes, cold 0:04:09; V equals Stage 0 A1, 181/181, max ULP 0 |
| 28022853 | WARM=3 arms: reduced2 fp32 A1 B1 B2 A2; reduced3 fp32 A1 B1 B2 A2; reduced3 fp64 A1 B1 | All 10 rc=0 |
| 28024017 | reduced2 A vs B cold with `JAX_LOG_COMPILES=1` | rc=0 |
| 28022854 (4×A100) | `--sharded-assets`: debug A, B; then SHA1 SHB1 SHB2 SHA2 reduced3 fp32 | Pending |
| 28024079 (4×A100) | Production A1 then B1 fp32 | Held |

## Parity
Bitwise for cold, warm_same_1 and warm_changed (181/181 arrays, max ULP 0, equal non-finite masks); simulate_cold and simulate_warm panel sha256 equal.
- reduced2 fp32 and reduced3 fp32: all 6 pairs among A1 A2 B1 B2.
- reduced3 fp64: A1 vs B1.
- Positive control: warm_same_1 vs warm_changed (bequest_shifter ×1.01) gives 0/181 equal in every arm.
- A vs Stage 0: reduced2/3 fp32 A1 and A2 bitwise against Stage 0 A1, panels included; reduced3 fp64 A1 bitwise against Stage 0 C64. Compile counts and fp32 peaks identical to Stage 0.

## Compile requests
| Workload | Cold A | Cold B | warm_same_1 | later warm | simulate cold / warm |
|---|---|---|---|---|---|
| reduced2 fp32 | 182 | 206 | 25 / 25 | 0 | 266 / 0 |
| reduced3 fp32 | 181 | 206 | 25 / 25 | 0 | 266 / 0 |
| reduced3 fp64 | 181 | 206 | 25 / 25 | 0 | 266 / 0 |

Traces go 153 → 178 cold (reduced3 fp32). Extra compiles in B cold at reduced2 (`analysis/logc-extra-compiles-LCB.txt`): 8 × `jit(_select_view_blocks)`, 7 × `jit(_write_block)`, 7 × `jit(_tile_block)`, 2 × `jit(slice)`. The count does not grow with type count.

## Timing (h:mm:ss, same job and node per workload, predeclared order)
| Call | r2 fp32 A1/A2 | r2 fp32 B1/B2 | r3 fp32 A1/A2 | r3 fp32 B1/B2 | r3 fp64 A1 | r3 fp64 B1 |
|---|---|---|---|---|---|---|
| Construction | 0:00:34 / 0:00:33 | 0:00:33 / 0:00:34 | 0:00:31 / 0:00:31 | 0:00:31 / 0:00:32 | 0:00:34 | 0:00:34 |
| Cold | 0:04:09.5 / 0:04:04.5 | 0:04:06.6 / 0:04:08.7 | 0:03:57.7 / 0:03:56.6 | 0:03:58.5 / 0:03:59.0 | 0:04:15.5 | 0:04:15.2 |
| warm_same_1 | 0:00:09.0 / 0:00:08.8 | 0:00:10.2 / 0:00:10.5 | 0:00:09.7 / 0:00:09.6 | 0:00:12.1 / 0:00:12.0 | 0:00:10.5 | 0:00:13.1 |
| Steady warm 2–3 | 0:00:07.3–0:00:07.7 | 0:00:08.7–0:00:09.3 | 0:00:08.1–0:00:08.4 | 0:00:10.6–0:00:10.9 | 0:00:09.1–0:00:09.3 | 0:00:11.7–0:00:11.9 |
| warm_changed | 0:00:08.4 / 0:00:08.2 | 0:00:09.6 / 0:00:09.9 | 0:00:09.0 / 0:00:09.0 | 0:00:11.5 / 0:00:11.5 | 0:00:10.0 | 0:00:12.5 |
| simulate_cold | 0:02:56.7 / 0:02:54.2 | 0:02:56.1 / 0:02:56.8 | 0:02:09.1 / 0:02:08.1 | 0:02:08.6 / 0:02:08.9 | 0:02:27.5 | 0:02:28.0 |
| simulate_warm | 0:00:23.4 / 0:00:23.3 | 0:00:23.7 / 0:00:23.8 | 0:00:19.9 / 0:00:20.0 | 0:00:20.6 / 0:00:20.5 | 0:00:20.4 | 0:00:20.8 |

Steady warm: reduced2 about +20%, reduced3 about +29% (median 0:00:08.3 → 0:00:10.7), fp64 about +28%. Phase ledger, reduced3 fp32 warm_same_2, A → B: backward_induction 3.9 → 4.9 s; structural_resolution 0.46 → 1.37 s; residency_inventory 0.23 → 0.48 s; workspace_selection 0.07 → 0.19 s; params_validation unchanged. Cold and simulate are inside the A/A spread. Reduced seconds on one GPU; no production inference.

## Memory (`peak_bytes_in_use` after solves / after warm_changed+simulate; identical within A1/A2 and B1/B2)
| Workload | A | B | Change |
|---|---|---|---|
| reduced2 fp32 | 760,399,104 / 805,555,200 | 738,361,600 / 783,968,000 | −2.9% |
| reduced3 fp32 | 800,288,512 / 868,026,112 | 749,662,976 / 819,727,104 | −6.3% |
| reduced3 fp64 | 596,628,224 / 732,075,520 | 371,027,456 / 511,860,736 | −37.8% |

## Plan records (untimed debug, one GPU; `analysis/plan-summary-1gpu.txt`)
| | A (1650805e) | B (9f03c34e) |
|---|---|---|
| reduced3 records total / blocked | 181 / 0 | 455 / 411 |
| Blocked regimes | — | 18, in 7 block shapes |
| Cell width, p0 `retiree_nomc_inelig_canwork` | 166212 | 55404 = 166212/3, full block extent, dense |
| `compiler_reservation_bytes`, same program | 48,286,900 | 31,123,820 |
| Largest `compiler_reservation_bytes` | 768,510,848 | 713,745,688 |
| `transfer_workspace_bytes` | 0 | up to 1,772,940; 443,244 per code at p0 |
| `active_replica_bytes` | 0 | 0 (nothing replicated on one GPU) |

At reduced grids both routes take the full extent, so fixed-width and reselected experiments coincide. Replica ≈ C/K needs job 28022854 (reduced assets grid has 3 points → legal 3-device mesh on 4 GPUs; driver `stage3_arms_v2.py` adds `--sharded-assets`).

## Task C
Job 28020582 still pending (est. 2026-10-02 20:10), untouched. 28021011 analysis unchanged.

## Clones
`~/pylcm-inv-s3g` detached at 9f03c34e, clean; new `~/pylcm-inv-s3p` at 9f03c34e with its own benchmarks-cuda12 env so concurrent jobs never share a pixi lock.

## Open
1. Decision on the 24–25 cold helper compiles; release or cancel 28024079.
2. Steady warm 20–29% slower (backward_induction, structural_resolution).
3. 28022854 pending.
4. 28020582 pending.
