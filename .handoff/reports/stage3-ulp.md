# Stage 3 blocked vs unblocked: ULP differences from FMA contraction

Investigated on 2026-10-02 at 1da79f5, CPU only (AVX-512 with FMA, jax/jaxlib 0.11.1).
No code was changed. Evidence is in `stage3-ulp/`.

## Finding
On 5A's life-cycle model at fp64, blocked (`invariant_block_widths={"pref_type": 1}`)
and unblocked solves differ:
- **`work` value cells:** 5, 7, 4 and 2 of 36 in periods 0–3, at most 2 ULP.
- **`dead` regime:** bitwise equal.
- **Type 0:** never differs.
- **Type-free-dead variant:** also differs, 2 cells at fp64 and 1 at fp32.
- **Simulated panels:** only `value` differs (at most 2 ULP). States, actions and
  regime paths are bitwise equal.

## Root cause
The bound per-code program (`pref_type` grid `s32[1]`) and the full program (`s32[3]`)
are differently shaped modules, so XLA fuses them differently. Inside each fused kernel,
LLVM contracts multiply→add pairs into FMA, and the pairs it contracts differ between
the two routes. The arithmetic graph itself is identical: same operations and operand
order, same widths, same reduction order and dtype, same hard-max merge.

| XLA_FLAGS (compile cache off) | differing cells, fp64 / fp32 |
|---|---|
| default, `max_isa=AVX2/AVX512` | 20 / 1 |
| `max_isa=AVX` or `SSE4_2` (no FMA) | 0 / 0 |
| `xla_backend_optimization_level=0` | 0 / 0 |

`fma_counts_p3.txt` shows the per-fusion FMA counts differing between the routes.
The two contraction sites, found by adding `optimization_barrier` in scratch copies:
1. **Engine-owned:** the interpolation corner sum in `ndimage.py` (`_sum_all`).
   This accounts for periods 0–2 and the fp32 cell.
2. **User-owned:** the model's utility, `weight[pref_type]*log(c) + 0.2*health`.
   This accounts for period 3.

**Width selection is ruled out.** With widths pinned, the routes still differ. The
unblocked route also differs from itself by up to 2 ULP when only its widths change,
which is the documented "across batch size, not bit for bit" behaviour.

## Why it can't be fixed without approximation or a redesign
- jaxlib has no flag to turn off floating-point contraction. `xla_cpu_max_isa` would
  change every model's numbers and speed, works only on CPU, and does nothing on GPU.
- Barriers in the engine fix only site 1, change unblocked numbers and cost fusion.
  They can't reach user functions (site 2).
- Making the bound program the same shape as the full one defeats blocking.

## Proposed acceptance contract (pending user decision)
- **Blocked vs unblocked values:** `assert_agrees_to_ulp(n_ulp=8)` on every published
  value array at fp64 and fp32, with non-finite values compared exactly. 8 ULP matches
  the state-sharding tests. The observed maximum is 2 ULP.
- **Structural outputs stay exact:** argmax/policies, regime paths, simulated states
  and actions.
- **Same compiled per-code programs stay bitwise:** 5B block-major vs period-major,
  and 5A grouped vs ungrouped simulation on the same solution.
- **`test_blocked_solve_equals_the_unblocked_solve_bitwise`:** it passes on this CPU
  but nothing guarantees it. Either keep it as a sentinel or move it to the ULP
  comparison.
- **ACA:** measure and report the maximum ULP over the full horizon rather than
  assuming 8.

Tests at 1da79f5: `tests/solution/test_invariant_blocking.py` passes 35, fails 0,
errors 0 and skips 4 at both precisions. The 4 skips are the pending compile-count
xfail parametrizations.
