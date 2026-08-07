# KV2014: the two NNBEGM-simulate blockers — self-contained brief

**For:** whoever picks up the (A)/(B) items on `feat/continuous-outer` / `feat/nb-egm`.
**From:** the KV2014 (Kaplan & Violante) replication agent.
**Why this file:** the full history is Replies 47–52 of
`HANDOFF-continuous-outer-kinked-EZ-surface-KV2014.md` (2535 lines). This is the actionable
subset, with the code locations **re-verified 2026-07-28** — the ones quoted in Reply 48 have
since gone stale.

## Status

Both items were root-caused and design-closed on 2026-07-22, then **deliberately held** pending
the #407 r3 Pro verdict (the fix touches `query.py`, which the r3-green suite rides on). Nothing
has moved since. Neither item blocks the KV *deliverable* (Table V ships via GridSearch +
`policy_eval`, PR #7, CI green); they block only the **native continuous-outer** Table V, which
is the only route that could reproduce the paper's Table-V *level*.

## ⚠️ Corrected code location (Reply 48's line numbers are stale)

Reply 48 cites `query.py:176` and `:327`. As of the current branch tip those lines are a
double-double division helper. The real location today:

**Use grep anchors, not line numbers.** Numbers here have drifted twice in a week under the
#407 agent's own commits (Reply 55). The symbol names are stable; the offsets are not:

```
cd <pylcm checkout>
grep -n "ok = any_bracket"  src/_lcm/egm/upper_envelope/query.py   # the gate
grep -n "jnp.nan"           src/_lcm/egm/upper_envelope/query.py   # every emission site
```

Positions as of 2026-07-28 (file is `src/_lcm/egm/upper_envelope/query.py`, **not**
`src/_lcm/solution/query.py`; the `:176`/`:327` in handoff Reply 48 are long stale):

```
  :771   any_bracket = jnp.any(terms.brackets, axis=1)
  :781   ok = any_bracket & resolved & ~poisoned      <-- the gate
  :783/:786/:789   env_value/policy/marginal = jnp.where(ok, ..., jnp.nan)   (blocked path)
  :1047-:1049      no-block early return
  :1055-:1057      same three, dense path   (was :1041-:1043 before round-7 added ~30 lines)
```

## (A) Below-support state emits a *propagating* NaN

**Mechanism.** A wealth *state* below the feasible support (KV: `m_min = −0.3 < savings_start = 0`
in the retire regime) is bracketed by no live segment → `any_bracket = False` → `ok = False` →
**NaN**. Note it is the *state* that is off-support; the floored `coh` is irrelevant. The NaN then
poisons by linear interpolation: a continuation query just above 0 interpolates between the NaN
`m_min` node (idx 0) and its finite neighbour.

**Measured propagation (KV census).** p230 retire ≈ 2% NaN → grows backward through the
cross-regime continuation gather → p0 work ≈ 50% NaN. Wholesale, and invisible to a worst-cell
mesh metric, which is computed over the finite region only.

**Design input already supplied (Reply 49), measured off the *clean GridSearch* simulate — not
the divergent nbegm one:**
- retire-**entry** with `assets_liquid < 0`: **0.860%** (43/5000)
- any sustained retire person-period `< 0`: **0.011%**

Small but **not** strictly zero. A retiree at `m_min` has `coh ≈ −0.3 + pension < 0`, i.e.
genuinely infeasible — so a competes-and-loses `-inf` sentinel is defensible on the economics.
The binding requirement is only that it must **not propagate NaN into the reachable
`wealth ≥ 0` continuation**.

**The nuance to resolve first.** The NaN at `:783` is *intentional* — the in-code comment says a
bracketed-but-unrankable query "fails loud with NaN (requirement 3)", and there is now an explicit
`poisoned` flag to keep dense and blocked paths on the same rule. So (A) is **not** simply "stop
emitting NaN": it must separate
- *bracketed but unrankable / poisoned* → keep failing loud (requirement 3), from
- *below support, no bracket at all* → non-propagating sentinel.

Options the #407 agent listed: (i) don't emit a propagating NaN below support; (ii) mask those
cells out of the cross-regime continuation gather; (iii) a competes-and-loses sentinel the
feasible neighbour overrides. They intended to TDD against a minimal `NestedNBEGM` with
`wealth < savings_start` rather than blind-patch. Note `-inf` also poisons linear interp, and
`map_coordinates` clips the index, not the weight.

## (B) Outer-durable feasibility is not honoured in `Model.simulate`

**Independent of (A), and survives it.** Proven by A/B on the two builds:

| build | consumption | illiquid |
|---|---|---|
| NaN-dead (soft-floor removed) | pinned at `cons_grid[0]=1e-3` (all-NaN-Q argmax) | diverges to −1838 |
| finite V (HEAD soft-floor) | **interior** (median 0.046, max 4.0) | **still** diverges to −1838 |

So (A) fixes the consumption pinning and leaves the durable divergence untouched.

**Mechanism.** `illiquid_investment` spans `[−a_max, a_max]`, so simulate's Q re-optimisation can
pick `next_illiquid < 0`; off-grid V′ extrapolation below the `illiquid = 0` boundary rewards that
corner; and the outer feasibility constraint is not honoured in `Model.simulate` — a
`feasible_illiquid` mask gave **byte-identical** solve+sim, i.e. the same F1 outer-durable-mask
gap, now on the simulate side. `wealth` then goes NaN downstream (200676 cells) from the illiquid
blow-up.

**Needs:** `a′ ∈ [0, a_max]` enforced in simulate.

## How to verify a fix (KV side, I'll run it)

On a landed `feat/continuous-outer` SHA:
1. `pixi update pylcm` in `~/lcm-reps-kv-native` (worktree pins the branch).
2. `scratch_dump_worstcell.py` (SMOKE) — worst-cell profile smooth, no 3.98→1.94 cliff.
3. **Census V finiteness** — `isnan(V).mean()` per regime/period, not a worst-cell metric.
4. `kv_stage2_mpc.py` — MPC must stop sign-flipping under a 10× rebate scale (a smooth policy's
   MPC is ~invariant to perturbation size; strong size-dependence = non-differentiable policy).

Scripts (verified present 2026-07-28): `gb10:~/lcm-reps-kv-native/kv_stage1_validate.py`,
`kv_stage1_calibrate_beta.py`, `kv_stage2_mpc.py`, and
`~/econ/dev-pylcm/lcm-reps-kv-native/scratch_dump_worstcell.py` locally.

**Ping me on the handoff file when a build lands** — I have a watcher armed on it.
