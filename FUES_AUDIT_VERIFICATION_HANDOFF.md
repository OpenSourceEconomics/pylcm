# Handoff: verify FUES code findings F3 / F4 / F5 against `refine_envelope`

## What this is

An external adversarial correctness audit (ChatGPT 5.5 Pro, via the
`pro-comp-method-audit` skill) of the Dobrescu & Shanker FUES paper *and* this
repo's FUES implementation returned a verdict of **`serious_gap`**. Three of its
"serious" findings are concrete claims about **bugs in our code**
(`src/_lcm/egm/upper_envelope/fues.py`), each with a small, runnable
counterexample. Findings F1, F2, F6–F9 are about the *paper* (scope / proof
generality) and are **out of scope for this task**.

Your job: **reproduce each counterexample against the current
`refine_envelope`, decide whether each finding is a real bug or a reviewer
misread, and write up the verdict.** Do not fix anything yet — separate
diagnosis from repair. A successful repair must not erase the original
diagnosis.

The audit's line numbers (below) are from the PR #383 branch
(`feat/dcegm-simulate`) the bundle was built from. The current checkout
(`docs/dcegm-example`) is structurally identical but lines differ slightly —
trust the function/logic references, re-locate exact lines yourself.

## Target file

`src/_lcm/egm/upper_envelope/fues.py` — public entry point:

```python
refine_envelope(*, endog_grid, policy, value, n_refined,
                jump_thresh=2.0, n_points_to_scan=10)
    -> (refined_grid, refined_policy, refined_value, n_kept)
```

Outputs are static length `n_refined`, NaN-padded; `n_kept` is the true count
(`n_kept > n_refined` = overflow). Each counterexample below gives explicit
inputs; run them through `refine_envelope` and inspect the kept prefix
(`refined_*[:n_kept]`).

## How to run a counterexample

```python
import jax.numpy as jnp
from _lcm.egm.upper_envelope.fues import refine_envelope

g, p, v, n = refine_envelope(
    endog_grid=jnp.array([...]),
    policy=jnp.array([...]),
    value=jnp.array([...]),
    n_refined=10,
)
kept = int(n)
print(list(zip(g[:kept].tolist(), p[:kept].tolist(), v[:kept].tolist())))
```

Use this repo's environment (`pixi run python ...` or the project's test env).
For each finding also build the **brute-force pointwise upper envelope** of the
input candidates yourself (for each distinct `x`, the max-value candidate; on a
crossing, the max over segment lines) and compare — that is the ground truth the
audit is measuring against.

---

## F3 — duplicate endogenous-grid abscissae can retain a dominated point

**Audit severity:** serious · hinge · `algorithm_incorrect`
**Code locus (PR #383 lines):** `fues.py:74-83, 177-210, 456-472`
**Mechanism claimed:** `_slope` returns `0.0` when two candidates share the
same `x` (coincident-abscissa guard, current `fues.py:456-472`). When two
candidates have the same `endog_grid` value but different value/policy, the
lower-valued one can be emitted before the higher-valued duplicate is processed,
so a strictly dominated point reaches the output. At a shared abscissa the upper
envelope should keep only the max value (or duplicate only a genuine equal-value
kink).

**Counterexample to reproduce:**

```python
endog_grid = [0, 0, 1]
policy     = [0, 1, 1]
value      = [0, 1, 2]
n_refined  = 10
```

**Audit-predicted (buggy) output:** kept values `[0, 1, 2]` at grids
`[0, 0, 1]` — i.e. the point `(x=0, v=0)` survives even though `(x=0, v=1)`
dominates it at the same `x`.

**Decide:** Does the point `(0, 0)` actually survive? Is keeping both `x=0`
points ever correct here (it is not — same abscissa, strictly lower value)? Note
the sort is `argsort` (NaN-stable) — check whether the duplicate ordering is
even deterministic.

**Smallest repair the audit suggests (do NOT apply yet, just record):** after
sorting, collapse equal-`x` candidates to the max value per segment/kink before
the scan.

---

## F4 — bounded forward/backward scan can keep a whole run of suboptimal points

**Audit severity:** serious · hinge · `algorithm_incorrect`
**Code locus (PR #383 lines):** `fues.py:40-41, 188-203, 214-240, 364-420`
**Mechanism claimed:** the paper says scan "until a same-segment point is
found"; the code's `_find_same_segment_point` only inspects
`n_points_to_scan` candidates (default **10**, current `fues.py:364-420`). The
proof gives no bound that the nearest same-segment witness lies within 10
candidates after sorting. With > `n_points_to_scan` interleaved off-segment
candidates, the witness is missed and dominated points are accepted.

**Counterexample to reproduce:** upper segment A = `{(0,0), (0.1,0.1),
(12,12)}` with policy `p = x`; plus eleven lower segment-B points
`(i, i-0.5)` for `i = 1..11` with separated policy `p = x - 100`; with
`jump_thresh=2`, `n_points_to_scan=10`. Build the arrays:

```python
import numpy as np
A_x = [0.0, 0.1, 12.0];  A_p = [0.0, 0.1, 12.0];  A_v = [0.0, 0.1, 12.0]
B_x = [float(i) for i in range(1, 12)]
B_p = [x - 100.0 for x in B_x]
B_v = [i - 0.5 for i in range(1, 12)]
endog_grid = A_x + B_x
policy     = A_p + B_p
value      = A_v + B_v
# refine_envelope(..., n_refined=64, jump_thresh=2.0, n_points_to_scan=10)
```

**Audit-predicted (buggy) output:** all eleven B points are returned even
though each lies 0.5 below the true upper line `A(x) = x`.

**Decide:** Are the B points actually kept? Confirm the true envelope is the A
line (`v = x`) over `[0,12]`. Then probe the **boundary of the claim**: does the
failure disappear at `n_points_to_scan = 11` / `= len(candidates)`? Is this a
genuine correctness bug or a documented approximation knob? (The audit's
"smallest repair" is to run the scan exhaustively in a correctness-critical
mode, or store segment/bracket ids and jump to the next same-segment point
directly — see F5.)

---

## F5 — routine omits the bracket-index switch the paper's Algorithm 1 uses

**Audit severity:** serious · hinge · `algorithm_incorrect`
**Code locus (PR #383 lines):** `fues.py:34-42, 177-184, 423-453`
**Mechanism claimed:** the paper's Algorithm 1 triggers a segment switch when
the bracket label `z` changes (`z_{l+1} != z_{i_k}`); Application 3 explicitly
replaces pure jump detection with a policy-slope test **OR** a bracket-index
difference. Our `_has_policy_jump` (current `fues.py:423-453`) keys **only** on
the implied-savings slope `|Δ(R−c)/ΔR| > jump_thresh`, and `refine_envelope` has
**no bracket/segment-label input at all**. So two bracket-constrained segments
with locally flat/equal policy but different bracket labels cannot be
distinguished.

**Nature of this finding:** this is a *missing-capability / interface* claim, not
a single breaking array — there is no label argument to pass. Verify it by
**code inspection**: confirm (a) `refine_envelope`'s signature has no
segment/bracket id parameter, (b) `_has_policy_jump` uses only the savings
slope, and (c) construct the conceptual failure — two segments straddling a tax
notch where the constrained policy is locally flat on both sides (so the savings
slope does not exceed `jump_thresh`) but the bracket differs. Show that no input
to the *current* signature can force a switch there.

**Relevance gate (important):** decide whether this matters for pylcm's intended
use. Does pylcm's EGM step ever feed bracket-constrained / notch problems into
`refine_envelope`, or is the jump-threshold path sufficient for the model class
we target? If brackets are out of scope for pylcm, downgrade accordingly and say
so. **Smallest repair the audit suggests (record, don't apply):** add an
optional `segment_id=None` arg; switch on `policy_jump OR segment_id change` when
labels are provided.

---

## Deliverable

Write `FUES_AUDIT_VERIFICATION_RESULTS.md` next to this file with, per finding:

1. **Verdict** — `confirmed_bug` / `confirmed_but_low_impact` / `misread` /
   `cannot_reproduce`, with the actual `refine_envelope` output vs. the
   brute-force upper envelope.
2. **Exact current line references** in `src/_lcm/egm/upper_envelope/fues.py`.
3. For F4/F5, the **boundary of the claim** (when it triggers, when it doesn't,
   whether it matters for pylcm's model class).
4. A recommendation: open a pylcm issue / add a regression test / narrow a
   docstring claim / no action — but **do not implement fixes in this pass**.

## Provenance / pointers

- Full audit JSON (all 9 findings, counterexamples, failed-attack log):
  `/home/hmg/thunder/fues-audit-bundle/findings.json`
- Human-readable checklist: `/home/hmg/thunder/fues-audit-bundle/CHECKLIST.md`
- Audited paper: Dobrescu & Shanker, "A fast upper envelope scan method for
  discrete-continuous dynamic programming", current draft March 4 2026
  (`/home/hmg/thunder/fues-audit-bundle/`).
- The audit ran against this repo's FUES code as bundled from PR #383
  (`feat/dcegm-simulate`). Independent verification already done for the sibling
  RFC audit confirmed that finding's headline code bug was real — i.e. these
  Pro-constructed counterexamples have a track record of being concrete, but
  each still needs to be reproduced before acting.
