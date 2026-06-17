# NEGM / multidim DC-EGM — Phase-0 risk-gate findings

Date: 2026-06-17. Branch: `feat/dcegm-negm` (off `feat/dcegm` @ `5f67a5de`).
All solves on gpu-01 (V100 16 GB, env `tests-cuda12`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`).

## Gate recommendation: **BUILD for the general capability — but the necessity premise is refuted; reframe the justification**

The single sharpest Phase-0 finding **contradicts the plan's premise**: brute force is
*not* the bottleneck. A two-continuous-state (liquid + illiquid) + two-continuous-action
(C, Iᶻ) + income-shock lifecycle solves on a single V100 in **~4–9 s using < 600 MB even at
2.5 billion state-action combinations per period**, and the authors' own paper solves by
**numerical backward induction** (not EGM). So:

- **Brute replicates the paper's model class** (it *is* the authors' method) and is
  single-solve-feasible at fine grids in pylcm today. The Laibson replication does **not
  require** NEGM/EGM for feasibility.
- **The real case for NEGM is accuracy + estimation throughput + generality**, not
  single-solve feasibility: brute restricts the continuous policy to the action grid,
  biasing exactly the consumption/savings Euler moments the paper identifies discount
  parameters from; MSM estimation re-solves under numerical gradients (the paper calls each
  moment computation "numerically costly"), so per-solve cost and smooth/unbiased moments
  compound.

Recommendation: **`build_but_resequence`, leaning toward the general-capability rationale**
the user already chose — but **drop "brute is infeasible at Laibson scale" from the
plan's motivation** (it is false on a V100) and replace it with the accuracy/throughput/
generality argument. NEGM is justified as a *general pylcm 2-asset capability* and an
*accuracy* upgrade, not a feasibility rescue. The NEGM nesting holds analytically (below).

## 1. Brute-feasibility probe — results

Model: states `wealth X`, `illiquid Z` (LinSpaced) + `income` (Rouwenhorst AR(1)); actions
`consumption C`, `illiquid_investment Iᶻ` (LinSpaced); withdrawal penalty on `Iᶻ<0`;
liquid/illiquid floors; `u(C + ιZ)`. `BruteForce` solver. Script:
`negm_phase0/brute_feasibility_probe.py`.

| N_X | N_Z | N_C | N_Iᶻ | N_inc | T | per-period width | solve | GPU peak |
|----:|----:|----:|----:|----:|--:|---:|---:|---:|
| 20 | 20 | 20 | 20 | 5 | 10 | 0.8 M | 3.49 s | 1.0 MB |
| 30 | 30 | 30 | 30 | 5 | 10 | 4.05 M | 3.51 s | 4.5 MB |
| 40 | 40 | 40 | 40 | 5 | 10 | 12.8 M | 3.51 s | 17.4 MB |
| 50 | 50 | 50 | 50 | 5 | 10 | 31.25 M | 3.48 s | 34.5 MB |
| 60 | 60 | 60 | 60 | 5 | 16 | 64.8 M | 4.13 s | 69.2 MB |
| 80 | 80 | 80 | 80 | 5 | 16 | 204.8 M | 4.09 s | 138.1 MB |
| 100 | 100 | 100 | 100 | 5 | 16 | 500 M | 3.87 s | 274.5 MB |
| 120 | 120 | 120 | 120 | 5 | 16 | 1.04 B | 9.06 s | 312.0 MB |
| 150 | 150 | 150 | 150 | 5 | 16 | 2.53 B | 4.86 s | 550.7 MB |
| 60 | 60 | **150** | **150** | 7 | **60** | 567 M | 8.04 s | 280.4 MB |

Reading: solve time is essentially **flat (~3.5–9 s)** and GPU memory **tiny (≤ 0.55 GB)**
across a >3000× range of search width. The V100 is far from saturated — the solve is
latency/overhead-bound, not compute- or memory-bound, because pylcm's brute solver
vectorizes the search and the retained `V` arrays are small (`N_X·N_Z·N_inc` per period).
A full annual-frequency lifecycle (T=60) with fine action grids (150×150) solves in **8 s**.
No OOM was reached on a 16 GB card up to 2.5 B/period; the memory wall would sit around
`N≈200` per dimension (absurdly fine), and `savings_grid.batch_size` would push it further.

**Implication for estimation:** at ~8 s/solve, an MSM loop of even a few thousand solves
(parameter search + numerical-gradient SEs + multistart) is hours, not days — feasible.
So brute is a viable interim replication backend. EGM/NEGM's win is **accuracy** (no
action-grid restriction on the policy → unbiased Euler moments at coarse grids) and
**per-solve speed** (compounds over the estimation loop), plus **generality**.

## 2. Kinked toy + brute oracle (the NEGM parity target)

Script: `negm_phase0/kinked_toy_oracle.py`. Smallest model with all Laibson frictions:
credit-card rate kink at `a^X=0` (12% borrow / 3% save), withdrawal penalty (κ=0.10) at
`Iᶻ=0`, `Z≥0` floor, `u(C+ιZ)`, ι=0.05, CRRA(2), T=4, grids 12×12 states / 25×25 actions.
Brute-solved on gpu-01. **Parity target** (period 0, regime `alive`, shape (12,12)):

```
V[wealth=0,  illiquid=0 ] = -0.4002052501
V[wealth=0,  illiquid=6 ] = -0.2250119814
V[wealth=6,  illiquid=0 ] = -0.2334363467
V[wealth=6,  illiquid=6 ] = -0.1619449719
V[wealth=11, illiquid=11] = -0.1331192126
sum = -26.82990570   min = -0.40020525   max = -0.13311921
```

V is correctly monotone-increasing in both assets. A future NEGM prototype (outer search
over `a^Z` + inner 1-D EGM on `X`) must reproduce these to ~1e-4 (brute carries action-grid
discretization, so exact parity is not expected — assert the brute value is a *lower bound*
the off-grid NEGM weakly improves on, and match dense-brute as `N_C, N_Iᶻ → ∞`).

## 3. NEGM nesting derivation for the toy (the make-or-break analytical check)

**It nests.** Period-`t` problem, state `(X, Z, ζ)`:

```
V_t(X,Z,ζ) = max_{C, Iᶻ}  u(C + ιZ) + β·E[ V_{t+1}(X', Z', ζ') | ζ ]
   a^X = X + Y(ζ) − C − credited(Iᶻ),   credited(Iᶻ) = Iᶻ if Iᶻ≥0 else (1−κ)·Iᶻ
   X' = R^X(a^X)·a^X    (R^X kinked at a^X=0: borrow vs save rate)
   Z' = Z + Iᶻ ≥ 0      (so Iᶻ ≥ −Z)
   a^X ≥ −limit,  C > 0
```

Reparametrize the illiquid margin by the **outer post-state** `a^Z ≡ Z' = Z + Iᶻ`. Fix
`a^Z`; then `Iᶻ = a^Z − Z` is fixed, so `credited(·)` is a constant and `ιZ` (current `Z`)
is a constant. The **inner** problem is a standard 1-D DC-EGM consumption problem on the
liquid margin:

```
W_t(X,Z,ζ; a^Z) = max_C  u(C + ιZ) + β·E[ V_{t+1}(R^X(a^X)·a^X, a^Z, ζ') ]
   R_inner = X + Y(ζ) − credited(a^Z − Z)        (resources, fixed given a^Z, Z)
   C = R_inner − a^X                              (consumption recovery)
V_t(X,Z,ζ) = max_{a^Z ≥ 0}  W_t(X,Z,ζ; a^Z)       (OUTER deterministic max)
```

Why the inner solve is the *existing* kernel, with three small modifications:
- **Euler state = X, action = C, post-decision = a^X, resources = R_inner.** Endogenous
  grid lives in `R_inner`-space exactly as 1-D DC-EGM.
- **Inverse marginal utility carries a constant shift.** `u'(C+ιZ)=m ⇒ C=(u')⁻¹(m)−ιZ`.
  The scalar inversion is unchanged; the `−ιZ` is a constant offset. **Feasibility guard
  (P1): C>0 ⟺ (u')⁻¹(m) > ιZ.**
- **The credit-card rate kink lives in the inner continuation** (`R^X(a^X)` kinked at
  `a^X=0`). **Split the liquid/savings grid at `a^X=0` (P1)** so the post-decision return
  is piecewise-smooth and the envelope sees the kink.

**The outer problem is a deterministic max over `a^Z`, not an Euler inversion (B2).**
`W_t(·; a^Z)` is generally **non-concave in `a^Z`** (withdrawal-penalty kink at `a^Z=Z`,
the `Z'≥0` floor, the rate kink), which is *why* the outer margin is a grid+candidate max,
not a second inverse-Euler. **Mandatory outer candidates (P1):**
- `a^Z = Z` (i.e. `Iᶻ=0`, no adjustment) — the withdrawal-penalty kink; **current-state-
  specific**, so a fixed exogenous `a^Z` grid misses it unless inserted per node.
- `a^Z = 0` (full withdrawal, `Z'=0` floor corner).
- segment the outer grid into `[0, Z]` (withdrawal, penalized slope) and `[Z, Z_max]`
  (deposit, unpenalized slope); the kink at `a^Z=Z` separates the segments.

**P2 timing (no taste shock in the toy, but the contract):** with a genuine taste-shocked
discrete choice `d` (work/retire), the order is **`max` over `a^Z` (and `C`) inside each
`d`, then `logsumexp` over `d`** — deterministic search nested inside the random-utility
aggregation. `max_{a^Z} logsumexp_d ≠ logsumexp_d max_{a^Z}`; the toy omits taste shocks so
it is a pure nested max.

**Residual risk (NEGM→RFC slide):** the outer max over a finite `a^Z` grid + candidates
carries discretization error in the illiquid policy. If the optimal `a^Z` is sharply
sensitive near the withdrawal kink, a coarse outer grid mislocates it. The P6 convergence
study (NEGM vs dense brute over a grid sequence: policies, Euler residuals, moments) is the
gate. With the mandatory candidates capturing the kinks, NEGM should hold — consistent with
the ~60–70 % prior. If outer refinement is prohibitively dense, slide to RFC (2-D envelope).

## 4. Build-vs-buy

| Tool | Solves Laibson? | Verified how | Adopt vs reimplement |
|---|---|---|---|
| **Authors' replication code** | yes (it *is* the paper) | Paper §"Code Availability" (`laibson-paper.md:1045`): "replication code is available in the Harvard Dataverse." Method = **numerical backward induction**, annual frequency, nonuniform grid (`:463`, `:1474`, `:1518`). Did not inspect the archive. | Best for *pure paper reproduction*; their own (non-pylcm, likely MATLAB/Fortran) code. Use as an **oracle**. |
| **pylcm BruteForce (today)** | yes, with action-grid bias | This probe: T=60 fine-grid 2-asset solve in 8 s / 280 MB on a V100. | Viable **interim replication backend** inside pylcm — matches the authors' method class. EGM upgrades accuracy, not feasibility. |
| Druedahl **G2EGM** (`github.com/JeppeDruedahl/G2EGM`) | with work | Pro-cited; the closest published 2-asset (liquid+illiquid) EGM analog. Not re-verified here. | Strong **oracle** for the C2-RFC fallback; MATLAB/C++ research code, not a pylcm backend. |
| NumEconCopenhagen **ConSav** / Dobrescu–Shanker **InverseDCDP** | with work | Pro-cited algorithm references (NEGM/durables; RFC/inverse-Euler). Not re-verified here. | Algorithm/oracle references; not drop-in pylcm backends. |

Net: **brute (pylcm) or the authors' code reproduces the paper; build NEGM for the general
capability + accuracy + estimation throughput.** Use brute as the immediate oracle (already
working — §2) and G2EGM as the fallback-route oracle.

## 5. What this changes in the plan

- **Remove the false necessity premise.** Phase-0 #1 ("does brute clear the accuracy /
  where does it break") is answered: brute is single-solve-feasible at fine grids; it
  breaks (memory) only at `N≳200`/dim. The plan's "likely past brute comfort" is wrong for
  a single solve — correct it to "brute is feasible but action-grid-biased; NEGM is an
  accuracy/throughput/generality play."
- **NEGM nesting confirmed** analytically for the toy (§3) with the exact P1 kink-candidate
  handling. The make-or-break check passes on paper; the empirical V-parity gate is the
  next step (prototype outer-search vs the §2 oracle).
- **The kinked-toy oracle exists and is committed** as the parity target.
- Brute stays the **oracle of record** for C2 (cheap, already working) and a credible
  interim replication backend.

Concrete next step (C2 prototype): implement the outer `a^Z` search (deterministic axis,
B2) + inner 1-D EGM on `X` against a B1-tabulated continuation, with the mandatory
candidates `a^Z∈{0, Z}` and the liquid grid split at `a^X=0`, and assert convergence to the
§2 dense-brute oracle.
