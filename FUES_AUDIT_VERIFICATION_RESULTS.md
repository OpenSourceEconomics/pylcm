# Results: FUES audit findings F3 / F4 / F5

Verification of the three code-level findings from the external adversarial
audit against the current `refine_envelope`
(`src/_lcm/egm/upper_envelope/fues.py`, branch `docs/dcegm-example`).

## Method

Each finding is judged on two axes:

1. **Reproduces?** — does the named mechanism produce the predicted output when
   the counterexample is run through `refine_envelope`?
2. **Reachable?** — can pylcm's EGM step actually *produce* that input for a
   target model? The candidate array fed to `refine_envelope` is, per combo,
   `concat([borrowing_limit + constrained_actions, savings_i + c_i])`
   (`step.py:1030-1034`): one Euler point per savings node plus the closed-form
   constrained segment.

The **ground truth** is the *interpolated envelope function*, not the raw kept
array: the refined `(grid, policy, value)` rows exist to be read by
`interp_on_padded_grid` downstream (the EGM carry read and off-grid
simulation), so a finding only matters if it corrupts that interpolated
function. Element-wise inspection of the kept array is used only to *explain*
behavior, never as the verdict — `refine_envelope` legitimately emits points a
naive pointwise-max would not (double-inserted segment-crossing kinks).

Reproductions: `refine_envelope` + `interp_on_padded_grid` on the audit's exact
inputs (x64). Reachability: a small IJRS worker-model DC-EGM solve
(`get_full_model("dcegm", n_periods)`), with `refine_envelope` wrapped in a
`jax.debug.callback` to capture every pre-refine candidate array, then measured
host-side.

---

## F3 — duplicate abscissae can retain a dominated point

**Verdict: `confirmed_bug` (array level) · no practical impact in pylcm.**

**Reproduces — yes.** On `endog_grid=[0,0,1], policy=[0,1,1], value=[0,1,2]`
the kept envelope is

```
(0.0, 0.0, 0.0)
(0.0, 1.0, 1.0)
(1.0, 1.0, 2.0)
```

The strictly-dominated `(x=0, v=0)` survives, exactly as predicted. Mechanism
confirmed: `_slope` returns `0.0` on a coincident abscissa
(`fues.py:469-472`), so `_has_policy_jump` reports no jump and the
`value_i < value_j` drop (`step.py:208`) never fires for the *first* of the two
`x=0` points (it is compared against itself in the carry, not against its
higher-valued duplicate). The sort is `jnp.argsort` (stable), so the dominated
point is retained in input order.

**Function-level impact — none.** Interpolating the kept arrays the way the EGM
step does gives, at query points `[-0.5, 0, 0.25, 0.5, 1.0]`:

```
value:  [1.0, 1.0, 1.25, 1.5, 2.0]   (analytic envelope 1 + max(x,0): identical)
policy: [1.0, 1.0, 1.0,  1.0, 1.0]
```

`interp_on_padded_grid` handles duplicated abscissae explicitly
(`interp.py:28-39, 80-89`): `searchsorted(side="right")` skips past the
left/lower-index duplicate, so the dominated `(0,0)` is never selected as a
bracket endpoint for any in-range query. The interpolated function is exact.

**Reachable — no (in-range).** Across the captured solve rows, exact-duplicate
abscissae *do* occur (~16 distinct per non-empty row) but every one is at a
saturated out-of-range abscissa (~`4.5e15`, vs the wealth grid's max of 400)
and is an **exact value tie** (same policy, same value) — not the
same-`x`/different-`v` pathology F3 needs. Count of in-range duplicate
abscissae with differing values: **0**. The construction
`concat([borrowing_limit + constrained_actions, savings_i + c_i])` makes an
exact float collision between two *differently-valued* candidates a
measure-zero event in the queried range.

**Recommendation.** Low priority. The latent bug is real and order-dependent
(if a future model fed the dominated duplicate at the *higher* index,
`interp_on_padded_grid` would return it for queries at/above the abscissa), so
a cheap defense-in-depth guard is worth a regression test: after the stable
sort, collapse equal-`x` candidates to the max-value representative before the
scan. Not urgent — no current target model reaches it. Regression test added
(xfail) so a fix is forced to flip it.

---

## F4 — bounded scan can keep a whole run of suboptimal points

**Verdict: `confirmed_bug` at production defaults · reachability model-dependent,
not exercised by IJRS.**

**Reproduces — yes, at the default `n_points_to_scan=10`.** The audit's input
(upper segment `A: {(0,0),(0.1,0.1),(12,12)}`, `p=x`; eleven lower segment-`B`
points `(i, i-0.5)` for `i=1..11`, `p=x-100`; `jump_thresh=2`) keeps **all 14
candidates**, and the interpolated value function sits uniformly **0.5 below**
the true upper envelope `A(x)=x`:

```
interp value(0..12): [0, 0.5, 1.5, 2.5, ..., 10.5, 12.0]
analytic A(x)=x:     [0, 1.0, 2.0, 3.0, ..., 11.0, 12.0]   max|dev| = 0.5
```

The bounded forward scan (`_find_same_segment_point`, window of
`n_points_to_scan` candidates, `fues.py:400`) cannot see segment-`A`'s
continuation at `x=12` past the eleven interleaved `B` points, so each `B`
point is wrongly accepted.

**Boundary of the claim — exact.** Widening the window to cover all candidates
removes the error completely:

| `n_points_to_scan` | `n_kept` | `max|dev|` from `A(x)=x` |
| ------------------ | -------- | ------------------------ |
| 10 (default)       | 14       | 0.50                     |
| 11                 | 3        | 0.00                     |
| 14                 | 3        | 0.00                     |

So this is a genuine correctness bug at the shipped default, not merely a
documented approximation knob — the default silently under-scans.

**Reachable — not in IJRS; cannot be excluded for richer models.** Every
non-empty candidate row captured from the IJRS worker solve is **fully
concave**: `segment_jumps = 0` and the nearest same-segment successor is always
the immediate next point (`max same-segment-successor offset = 0`,
`0 / N` rows with offset ≥ 10), at `n_periods ∈ {3, 6}`. The canonical IJRS
DC-EGM example never produces the multi-segment, finely-interleaved input the
bug needs. Triggering F4 requires (a) genuine multi-segment non-concavity *and*
(b) more than `n_points_to_scan` points of one segment falling between two
consecutive points of another in sorted-grid order — i.e. a strong asymmetry in
per-segment grid density inside an overlap region. With the 200-node cubically
clustered savings grid this asymmetry is plausible for strongly non-concave
target models (ACA's discrete-choice structure, the Laibson liquid/illiquid
model), so it should not be assumed safe there.

**Recommendation.** Open a pylcm issue. Either (a) add a correctness-critical
mode that scans exhaustively (`n_points_to_scan = n_candidates`), or (b) carry
explicit segment/bracket ids and jump directly to the next same-segment point
(see F5). As an immediate mitigation, raising the default `n_points_to_scan`
(`solvers.py:102`) only pushes the boundary out — it does not close the gap.
Regression test added (xfail, strict) asserting the interpolated envelope
equals `A(x)=x` at the default; a real fix will flip it.

---

## F5 — no bracket-index switch (paper's Algorithm 1)

**Verdict: `confirmed` as a missing-capability/interface claim · out of scope
for pylcm's current model class.**

**Confirmed by inspection.** (a) `refine_envelope`'s signature
(`fues.py:34-42`) has no segment/bracket-id parameter. (b) `_has_policy_jump`
(`fues.py:423-453`) keys *only* on the implied-savings slope
`|Δ(R−c)/ΔR| > jump_thresh` (`savings_slope`, `fues.py:447`); there is no
alternative trigger. (c) Therefore two bracket-constrained segments with
locally flat/equal policy on both sides of a notch (so the savings slope does
not exceed `jump_thresh`) but different bracket labels cannot be distinguished —
and no argument to the current signature can force a switch there.

**Relevance gate — out of scope (currently).** pylcm's non-concavity arises from
*discrete choices* in the continuation, whose segment boundaries do produce a
savings-slope jump and are detected by the existing test. pylcm does not
currently feed bracket-constrained / tax-notch problems with locally-flat
policy through `refine_envelope`. If a target model with explicit tax notches
(some ACA tax/transfer structures) is brought into DC-EGM, this gap becomes
live.

**Recommendation.** No code change now; record the limitation. The audit's
smallest repair is the right shape if it becomes relevant: add an optional
`segment_id=None` argument and switch on `policy_jump OR segment_id change` when
labels are provided. No executable regression test in this pass — there is no
breaking array to assert against without the label argument; this is a
write-up-only finding.

---

## Summary

| Finding | Reproduces | Reachable in pylcm | Verdict | Action |
| ------- | ---------- | ------------------ | ------- | ------ |
| F3 | yes (array level) | no (in-range; ties only) | `confirmed_bug`, no practical impact | low-pri guard + xfail test |
| F4 | yes (at default `n_points_to_scan`) | not in IJRS; plausible for rich non-concave models | `confirmed_bug` | issue + xfail test |
| F5 | yes (by inspection) | out of scope (no notches fed) | `confirmed` missing capability | record; revisit if notches enter |

No fixes were applied. Regression tests for F3 and F4 are committed as `xfail`
(`strict=True`) in `tests/solution/test_fues_upper_envelope_audit.py`, so any
repair is forced to remove the marker.
