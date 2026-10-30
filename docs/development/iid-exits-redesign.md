# IID integration, exits and addressed simulation

Implementation contract for
[issue 607](https://github.com/OpenSourceEconomics/pylcm/issues/607). The base is
`671129d7ebdf1e1af005a3c658e9e73f6556dac5`, the head of PR 604 when this work started.
The issue defines the public semantics; this page resolves its remaining numerical
choices and records the ordered acceptance gates.

The owner authorized choosing these contracts in the implementation conversation on
2026-10-23. They are implementation decisions, not measured accuracy claims. Passing a
standalone reference establishes its arithmetic; each engine route must also be
connected to that reference through the public model API.

## Speed-first amendment (2026-10-29)

The owner adopted an ordinary-first direction for the whole EGM implementation and the
remaining S10 plan, recorded on
[issue 618](https://github.com/OpenSourceEconomics/pylcm/issues/618). It supersedes the
clauses below that impose exact numerical choices, universal topology or ownership
guarantees, or an expanding family × derivative × precision × archive matrix on ordinary
EGM use. Each superseded clause carries a pointer to this section; its text stays as the
historical record, and existing receipts, adverse runs and conclusions are unchanged.

The existing balanced choice is the normal fast EGM mode; there is no new profile.
Numerical approximation, shifted decision boundaries and some differing discrete choices
are acceptable. Certification is optional and runs after solving. When no audit is
requested it adds no tracing, numerical work, allocations, retained witnesses or
compilation. NBEGM defaults to `envelope_arithmetic="ordinary"` and
`probe_schedule="never"`, and routes with an ordinary equivalent do not refuse ordinary
arithmetic.

Ordinary behavior that remains required:

- the economic model: shock timing and information, per-shock action selection before
  integration, constraints, and the association of value, action, savings and
  transitions; shock-specific policies are never averaged
- the economic marginal as a channel distinct from the derivative of the value
  interpolant, including the law and weight chain-rule terms of the Euler equation
- the stable numerical primitives already implemented, real allocation limits and the
  common generation, retention and archive lifecycle

A different native owner or kink side caused by approximation is not a failed ordinary
solve. A discrete owner, participation decision, kink side or exit id no longer "has no
tolerance" as a default acceptance rule; the model's strict and weak comparison
semantics remain, but approximate values can lead to different decisions. Ex-post
comparisons report value loss, policy differences and useful aggregate effects; they do
not promise the continuous optimum or a uniform domain error bound. Exact comparisons on
a fixed candidate bank do not establish convergence; check that separately, when
requested, with finer grids and quadrature.

Revised completion gates:

| Requirement                                                                                                                                               | Disposition                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Genuine supported DC/NB/NEGM/NNBEGM solve, required incoming reads, physical action association, multi-period recurrence, default retention and save/load | Ship; reuse existing implementations and shared tests                                                   |
| Stable CE and economic directions needed by an actually supported model                                                                                   | Ship ordinary functionality; do not extend every extreme arithmetic or AD class                         |
| Exact owner/side/germ agreement, certified topology, exact predicates used only for checking, universal accuracy and all-action regret                    | Optional ex post; zero default cost                                                                     |
| Full P1–P6 × C1–C6 × D1–D6 × dtype/device/lifecycle intersections, new cold or foreign-family capabilities, broad mutation and custody proofs             | Defer unless a named actual consumer requires the combination                                           |
| Q1 generic GridSearch-to-EGM exit adapter                                                                                                                 | Keep separate unless advertised or needed by a concrete caller; EKL's GridSearch path does not need it  |
| Actual device-memory budget, schema consistency and retained-result lifetime                                                                              | Keep in the common owner; do not duplicate proof machinery per family                                   |
| Source receipts                                                                                                                                           | Preserve historical evidence; one source SHA, command and raw result record per new batch               |
| S11 draw/override work                                                                                                                                    | Proceed independently; integrated checks depend on the supported consumers actually reached             |
| S12 legacy retirement                                                                                                                                     | Proceed once actual consumers have working replacements and integration checks pass; not held by audits |

This reduces unfinished acceptance coverage. It does not permit removing a working
advertised feature, reinterpreting a model equation or labelling an unsupported read as
supported. A concrete consumer need can bring back a deferred item without bringing back
the former matrix.

### Default simulation and replay

Default `simulate` decides from the solution rather than re-deriving or auditing it:

- a plain DC-EGM regime reads its solved policy at the realized state (policy-first);
  only subjects whose read fails the cheap acceptance checks — row support, finite value
  and action, the intrinsic budget, constraint feasibility — take the action-grid argmax
- a regime reading a named current-IID shock is a node-state regime (decision 65): the
  shock is an internal node state, and a realized draw between nodes reads the
  node-indexed rows of the retained continuation; no path rescores candidates against a
  grid baseline or retains owner, side or corner witnesses

The simulated native owner and kink side are therefore not guaranteed to match the
solve's. `DCEGM(replay="grid")` restores the action-grid argmax for plain DC-EGM
regimes. The [solver reference](../reference/solvers.md#api-dcegm) lists where
policy-first applies.

The native current-IID route that solved such shocks without the node axis (Case-R
resources rows and banks, Case-U candidate reconstruction, the projected continuation
and its private frontiers) is removed; the node-state axis is the only EGM route for
named shocks. A DC-EGM draw between nodes blends the two neighbouring node rows
linearly, which is exact because the shock enters only through resources: an
endogenous-grid regime whose utility reads an IID shock otherwise is refused at
`Model(...)` (decision 82). Continuous NBEGM replay between nodes is not yet on that
route.

A DC-EGM regime followed by a finite NNBEGM regime has no replay route and is refused at
`Model(...)`; see
[unsupported combinations](../reference/solvers.md#unsupported-solver-combinations).

### Batch 4 compact acceptance set

The S10 closing set reuses existing tests, one per distinct implementation mechanism.
`tests/ci/ci-workloads.json` lists them as `egm_acceptance_tests` and
`tests/conftest.py` marks them `egm_acceptance`, so `-m egm_acceptance` on their files
selects exactly this set. Each still runs in its ordinary CI leg; no lane selects by the
marker.

| Mechanism                                               | Tests                                                                                                                                                                                                               |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scalar current-IID integration and realized replay (DC) | `test_current_iid_consumer_parity.py::test_declared_health_cost_shock_reproduces_node_state_model`; `test_current_iid_off_node_replay.py::test_public_case_r_replay_matches_the_literal_policy_at_off_node_shocks`  |
| NB with a passive state carried into the terminal       | `test_current_iid_consumer_parity.py::test_nbegm_carries_passive_discrete_state_into_terminal`                                                                                                                      |
| NEGM keeper/adjuster selection per shock node           | `test_current_iid_consumer_parity.py::test_declared_shock_solves_nested_egm_on_the_node_axis`                                                                                                                       |
| Fresh-instance save/load                                | `test_current_iid_consumer_parity.py::test_saved_declared_solution_publishes_and_replays_integrated_values`; `test_builtin_policy_persistence.py::test_a_saved_egm_solution_simulates_identically_in_a_fresh_model` |

Action-to-transition association and information timing are asserted by the DC replay
test (parent action and child landing) and the NEGM test (per-node selection before
expectation).

## Accepted scope

On 2026-10-24 the owner confirmed that faithful EKL remains a required acceptance case,
while the engine also supports generic positive-cost formation and independent hidden
shocks. EKL's formation adjustment uses the mapped lagged marital state, so newlyweds
have no lagged divorce cost; both spouses' offers are observed/shared at marital choice.
Action-dependent next-period offer laws retain their dating. The generic positive-cost
and hidden-offer witnesses test additional engine behavior.

Partial observation of correlated shocks is deferred. Refuse a query requiring that
conditional distribution explicitly. Wholly hidden groups retain their joint law.
Partial correlated physical overrides are also deferred: require a complete physical
frame for that group or use named base-innovation overrides and the forward transform.
Independent physical overrides remain supported. These decisions supersede the earlier
general Gaussian conditional-query and partial-physical-completion plan.

Legacy compatibility is not an acceptance requirement. Migrate callers directly to the
new API. Use focused local correctness checks; long certificate and broad platform
campaigns run in CI rather than blocking every implementation change locally.

## Invariants

Current IID shocks have no axis in stored values, continuation carries or policies.
Resolve feasibility, participation, action selection, exit routing and EV1 readout per
node before integrating. Never average an action index or an endogenous-grid coordinate.
Persistent states and transition-local draws retain their timing.

Every policy query returns the values of one common selected action. `Raw`,
`Participating` and `Full` are distinct query identities, and only a regime's own
occupants and publication read `Full`. Exit targets read `Participating`. Same-period
query consumers belong to the query DAG and buffer lifetime graph.

Ordinary feasibility precedes participation. Dissolution participation is weak,
formation consent is strict, zero-weight members retain vetoes, and stable global action
identifiers decide exact objective ties. Utility adjustments change both the exit payoff
and participation benchmark, once each at their specified use.

Source-read shocks are observed; target-only shocks are hidden unless declared observed.
Target taste draws stay hidden at the source choice. Shared innovations retain identity
in base space and use the law at each evaluated state. A formation query cannot silently
use an unobserved participation shock.

Simulation uses immutable original subject rows and typed, versioned addresses. Each
period permits one same-period move, one final-regime action and one scheduled
transition, with no transition after the terminal period. Moves retain the row's shock
overrides. Regime order, chunking and group edits cannot move an address.

The RNG address version is `pylcm.random.v1`. Its domain, immutable scalar name and
nonnegative component number form compact UTF-8 JSON with the version string. SHA-256 of
those bytes is decoded into big-endian uint32 words and folded into the typed root,
followed by the original row's high and low uint32 words and the computational period.
Domains `shock`, `state`, `regime` and `taste` remain disjoint. Scalar shock identities
survive changes to their containing group; lowered joint-normal coordinates retain the
declared scalar shock names. The regime lottery uses the reserved name `regime`, and a
destination's taste stream uses its regime name. Consumer DAG bindings do not identify
the innovation.

A transition executed from period `t` uses address period `t + 1`, while its law still
receives period-`t` states and actions. Current-period draws and entry draws use their
current computational period. Original rows span uint64 independently of enabled
floating precision; computational periods fit uint32. Padded rows repeat the final real
row. Roots are immutable, and panels record their address version, PRNG implementations
and floating precision. Categorical draws use half-open inverse-CDF cells, preserve
literal zero masses, and close the CDF plateau from its final positive category without
renormalizing the supplied probabilities.

## EGM publication and refinement

The discovery reviews in `sciebo:pro-audits/pylcm-pr621-grid_search/`,
`sciebo:pro-audits/pylcm-pr621-egm/` and `sciebo:pro-audits/pylcm-pr621-ce/` support a
sampled balanced acceptance contract. Keep the numerical budgets below and the economic
invariants above. Whole-domain topology proofs, exhaustive feature intersections,
arbitrary-finite exact accumulation, whole-Model reverse and higher-order AD are
separate capabilities, not routine balanced release prerequisites. Certification remains
optional and must fail closed whenever it is actually claimed. Source seals and broad
certificate populations run offline or in CI; they do not replace independent economic
checks.

> Superseded: certification is optional and runs after solving at zero default cost; see
> the [speed-first amendment](#speed-first-amendment-2026-10-29).

A public balanced route must name its supported workload and serve every required
continuation, parent, replay and archive read within it. A refused required read is a
delivery gap. Private staged examples and exact samples do not establish public
admission or complete S10. Production cost requires a matched captured regime-period and
replay slice; fixture durations alone do not establish a speedup.

Use a piecewise-affine value representation between smooth neighboring projected nodes.
At served kinks and jumps, preserve the owning-side value, policy and economic marginal,
through side records or original candidate replay. Ownership follows the source
candidate's one-sided ownership and stable id. Do not interpolate across an
unrepresented discontinuity: refine or refuse that cell. Pull back all boundaries with
their owning sides; nonmonotone resource maps require all preimages. Certified boundary
inversion requires supported exact topology or a refusal; balanced numerical inversion
records bracket uncertainty. Preserve exact-node queries before applying interpolation.

Coverage may be local during private staged implementation. An explicit finite-demand
bank may retain exact declared and demanded coordinates while unproved open intervals
remain holes. Every required coordinate must be finite, inside the declared closed
domain and exactly retained after capacity admission. A nonexact query through a hole
fails before discounting or economic masking. Intact exact samples do not establish
whole-domain topology or smooth support. Keep complete declared-workload coverage
separate from sample integrity; an optional whole-domain certificate outside that
workload is not a balanced prerequisite. A private fixture bank does not install
arbitrary production parent-demand preparation or forward replay.

Exact-demand selection evidence retains the actual live-node owner, side and corner
tuple in original quadrature order, bound to its coordinate and original producing
row/date/law/context. Inactive quadrature rows are not corner selections. A common
partition index is only an address and cannot substitute for the economic roster. Keep
any fixture inspection evidence separate from durable carry axes; production evidence
retention must declare its own schema, native roles, reads and lifetime.

> Superseded for ordinary solves: approximate ownership, sides and kink locations are
> acceptable, and exact selection evidence is optional ex post; see the
> [speed-first amendment](#speed-first-amendment-2026-10-29).

The economic marginal is a separate, explicitly labelled channel. On smooth cells it
includes derivatives of weights, physical-law transforms and resources. In particular,
`dW/db = sum(dw/db * V + w * dV/db)`. The derivative of the projected value interpolant
is a different statistic and cannot substitute in an Euler equation. At a kink use the
selected candidate's one-sided economic marginal; at a jump, label the owning-side
marginal and retain jump metadata. Reject an unresolved owning side. A zero primal
weight can have a nonzero derivative, so differentiate weights before dropping zero-mass
nodes. Nonlinear CE channels use their own stable typed sufficient statistic and full
chain rule.

The private unit-risk CE direction rule keeps the original power-mean primal and
ordinary coefficient path. When its nonnegative exponent coefficient is outside the
ordinary normal range, first-order transport carries binary scale separately and uses an
explicit transpose. Forward and transpose each meet the same independent numerical
reference; they need not be bit-identical across compiler paths. The transport fixtures
compare normal results after normalization, and subnormal products within one final
subnormal ULP of the represented binary-coefficient product. This bound allows
significand-product rounding followed by final scaling; it is not a bound on arbitrary
exponential/logarithm approximation or the economic model. True zero is allowed;
nonfinite base or direction operands, an unretained nonzero final underflow and
unresolved signed-scale cancellation refuse. Coefficient-residual differentiation beyond
the first-order rule remains unsupported. This kernel transpose does not add reverse
mode to the forward-only native exact-affine reader. The owned primitive's original
state, bindings and execution registrations are part of semantic capture authority.

The shared anchor product rule retains `exp(d) * Da + CE * Dd`, including at equal
payoffs. Its second term uses the actual represented CE coefficient for safe ordinary
products and sums, with the same owned first-order transpose. Tiny or unretained
products and sums retain exceptional scaled transport. The default logarithmic primitive
keeps its strict span refusal; ordinary anchor directions do not inherit that refusal
merely because one immaterial term is much smaller than the other.

The private balanced CE direction also checks conditioning before publishing validity.
Keep an uncancelled magnitude envelope `S` from the original child, weight and exponent
contributions, including absolute inner mass before opposing signed reductions. Refuse
when `64 * eps * S` exceeds `max(1e-4, 8 * sqrt(eps)) * max(1, abs(direction))` in the
profile's final units. Empty directions and ordinary finite cancellation remain
supported. This is a conditioning diagnostic backed by sampled independent economic
references, not a certified transcendental error bound or universal exact accumulator.
Existing probability, input, span and final-status checks still apply.

The law envelope also retains uncertainty from original payoffs whose separate logs can
lose their contrast. With `L = max_needed(abs(log(value)))` and
`r_i = exp(p * d_i - log_moment)`, it includes
`2 * CE / Z * sum(abs(Dw_i) * max(1, r_i)) * (1 + 2 * L)` before the same `64 * eps`
floor. Each differentiated factor retains its own binary scale. This component is empty
for zero law directions or bit-identical needed payoffs, including derivative-only null
events. Its positive sum is bounded by the largest active term times the active count;
the combined contribution and uncertainty envelopes are bounded by twice their maximum.
An empty uncertainty envelope preserves the original contribution envelope exactly.
These positive bounds do not enter the signed numerical reductions or their span gates.
High-scale unresolved law directions may refuse; ordinary finite and exact-equality
controls must remain usable. This is a conservative admission diagnostic, not a sharp
error bound for library logarithms.

Refinement bisects failed smooth cells and inserts known transformed breakpoints.
Compare the coarse representation with quarter, midpoint and three-quarter queries from
the solver's candidate construction, exact endpoints and adjacent representable points
at boundaries, then repeat with the refined representation. Report acceptance for that
declared sample when every channel meets its budget and known discontinuities are not
bridged. Required reads with exhausted capacity or unresolved ownership fail loudly.
Refining a savings grid and refining this projection are distinct operations.

Define normalized discrepancy as `abs(actual - reference) / max(scale, abs(reference))`.
Use unit scale for fixture value/marginal, action-domain span for policy, and a declared
positive natural unit for each moment. Fix scales before measurement. The acceptance
budgets are:

| Quantity                          | Certified                  | Balanced                   |
| --------------------------------- | -------------------------- | -------------------------- |
| Projected value and CE level      | `max(1e-8, 64 * eps)`      | `max(1e-6, 64 * eps)`      |
| Economic marginal on smooth cells | `max(1e-6, 8 * sqrt(eps))` | `max(1e-4, 8 * sqrt(eps))` |
| Continuous policy level           | `max(1e-6, 8 * sqrt(eps))` | `max(1e-4, 8 * sqrt(eps))` |
| Declared population moment        | `max(1e-6, 64 * eps)`      | `max(1e-4, 64 * eps)`      |

Here `eps` belongs to the active floating dtype. These are refinement and fixture
acceptance budgets, not a universal theorem about arbitrary economic models. Moments
must identify the population, common base draws and normalization. A discrete owner,
participation decision, kink side or exit id has no tolerance. Assess candidate omission
for `replay="grid"` separately from projection error against the declared policy and
moment budgets. If that assessment is unavailable, report omission error as unknown; do
not claim an unrestricted optimum. Simulation Q regret is not stored-value error.
Sampled refinement validates its declared query population. A uniform domain bound
additionally requires an analytic interval bound, a justified curvature or Lipschitz
bound, or exhaustive piecewise-affine verification. Diagnostics distinguish sampled
validation from uniform bounds.

> Superseded: discrete owner, participation, side and exit ids are not a default
> acceptance rule; see the [speed-first amendment](#speed-first-amendment-2026-10-29).

## What certified means

> Superseded: this is the meaning of the optional ex-post check, not a default solve
> requirement; see the [speed-first amendment](#speed-first-amendment-2026-10-29).

Certification concerns represented operands, finite candidate banks, declared quadrature
and piecewise representations. It does not certify the continuous economic optimum, the
adequacy of calibration, or an integration rule's error against the underlying
probability law.

Ownership and equality of represented candidates require exact comparison or an explicit
unresolved error. Existing exact-affine native requirements remain. The certified
profile refuses `replay="grid"`; realised-shock replay reconstructs the same candidate
bank and selection as solve. At a solve node the action id and candidate ownership must
match exactly; levels obey the dtype's rounding contract.

For the added owned-resources-row versus borrowing-corner selector, compare the actual
canonical scores evaluated in the active dtype, backend and execution route. The row
wins equality under the canonical ordered comparison, without an ownership tolerance.
This compares represented scores, not ideal transcendental expressions, and preserves
the existing native exact-affine comparison requirements. Invalid candidate arithmetic,
unresolved source sides or a different original candidate roster fails the consumed
query. Same-point replay ownership needs the actual original roster or an applicable
arithmetic-transfer argument; a scalar receipt or a probe roster is insufficient.

Certified smooth links require proof for the actual represented arithmetic or an
applicable conservative enclosure. An ideal log bound alone does not enclose the target
JAX log program. Without that proof, keep unresolved intervals as holes and refuse their
consumption; exact finite-demand checks do not grant a uniform bound.

Measure quadrature, projection and candidate omission separately. The certified profile
does not combine them into a single tolerance or use a projection tolerance to excuse a
changed structural decision. Public-API oracle comparisons cover the relevant supported
execution routes of each declared workload, including its precision and device. A route
claimed as supported needs execution evidence; source inspection is not an execution
receipt. A full Cartesian product of optional solver, topology and transformation
features is not a balanced acceptance campaign.

Signed quadrature is linear-only. Propagate its provenance into stored channels and
reject every nonlinear downstream reader, including CE, Euler inversion, constraints,
availability, exit rules and simulation sampling.

## Information and hidden-query integration

For source EV1 Full choices, optimize continuous actions within each existing
discrete-action combination, then append one source taste component per exit in
declaration order. The source regime's existing taste address supplies that one roster;
continuous/global stay action IDs do not create additional independent tastes. Solve
takes the logsum over admitted stay-combination maxima and admitted exit payoffs.
Built-in dissolution (`unilateral` and `mutual`) admits its exit only when the kept stay
set is empty; an exit roster position is masked otherwise. Consent admits formation only
after its existing strict Raw comparisons pass. Simulation selects the route with the
same source draw and publishes the selected unshocked value. Target taste remains hidden
at the source, priced at its own logsum, then drawn at the destination address in its
acting pass. An arbitrary custom Full selector is not generally an argmax and therefore
has no canonical logsum integration; source EV1 with such a selector is unsupported and
explicitly refused. This does not prevent a custom source rule from reading a
canonically priced hidden target EV1 policy. Collective EV1 keeps its existing
capability refusal.

Keep target prescribed-policy information separate from caller pricing information.
Additional caller knowledge does not change the target's participation policy. Evaluate
that policy at each hidden node before pricing its selected action and stakeholder
tuple. Independent hidden shocks, including target taste shocks, are integrated before
the source decision and revealed only when the destination acts. A supplied hidden
override does not become an observed pricing operand.

For a wholly hidden group, use its original joint forward operator, preserving row
order, weights, duplicates and dependence. For independent observed/hidden coordinates,
fix the observed physical operands and project the same numerical rows onto the hidden
coordinates. Do not infer scientific dependence by filtering a finite node table or
replace a hidden correlated group by independent marginals. A finite rule has no general
tower identity; retained and recomputed execution must agree for the same chosen
operator.

If declared dependence connects observed and hidden coordinates, refuse the requested
partial observation instead of performing Gaussian conditional refactorization. Validate
runtime correlation at each invocation; a cached independent view cannot admit later
cross-correlation. Derivatives of an admitted independent view hold that independence
partition fixed. General correlated conditional moments and derivatives are deferred.

Ordinary equations consume the sealed rounded physical value. Retain original named base
innovations and producing-law context for shared-law replay and later revelation; they
are not extra observations available to utility, availability or participation.
Source/target bindings, observed names, target policy information, clock, dtype and rule
identity belong to the query identity. Keep this preparation within the admitted query
chunk and account for its outputs, transfers and failed-reader lifetimes.

Shared categorical observations retain the common-uniform interval semantics required by
the shared-discrete projection contract. Unchanged partitions preserve the category,
including a null solve category, without dividing by its mass. Changed partitions use
qualified common-CDF refinement; unresolved consumed derivatives fail separately from
primal validity. An actual zero-probability observation is an integrity failure. A null
solve node can still contribute a finite `dmass * value`; never drop that derivative.

Named base overrides use the admitted forward law. Independent physical overrides and
complete correlated physical frames require source-law inversion/support validation;
being complete is not a validation bypass. Refuse partial correlated physical frames and
mixed base/physical forms within a correlated component. Preserve producing context,
original row/address, path-versus-entry precedence and actual revelation timing. Full
frames, singular support and unresolved inverses retain their explicit integrity gates.

Acceptance covers independent hidden pricing, preserved wholly hidden joint dependence,
fully observed correlated inputs, explicit partial-observation/override refusals,
shared-U boundaries, original-row revelation and execution equality. Numerical levels
and structural action/exit decisions remain distinct acceptance channels.

## Runtime correlation admission

Each concrete solve, simulation and period replay validates the supplied correlation
matrix and its active-dtype representation on the host. The engine owns the exact
zero-pivot support mask and passes it beside dynamic correlation values to the native
program. The mask is not a user-supplied validation bypass. A later invocation validates
again; correlation values are not frozen by the first call.

An admitted zero pivot contributes an exact zero factor column. Select safe operands
before square root and division, rather than hiding an invalid inactive expression
afterward. An active pivot must be finite and strictly positive, and its factor entries
must remain finite. Unresolved active arithmetic returns a dedicated invalid-law status;
clipping an active pivot is not proof of factorization. Rounded residual equality is not
required at host-admitted zero pivots.

Fixed declarations obey the same active-pivot validity requirement: exact host PSD
admission alone does not certify rounded factor arithmetic.

Derivatives hold admitted support fixed. Interior positive-definite derivatives use the
dynamic factor arithmetic. At singular support, derivatives have the interpretation of
tangents within the declared PSD stratum; crossing rank has no derivative guarantee.
Primal validity and finiteness of consumed derivatives are separate obligations.

## Native singleton GridSearch integration

Each real state cell evaluates complete joint node rows, applies its phase-specific law,
and invokes the canonical feasible action reducer before the weighted sum. Hard maximum
and EV1 keep that order. Node coordinates are temporary operands, with exact resolved
support and algorithm identity; stored values retain only real state axes. Ordinary
state tiling, hard-max action streaming and state/subject sharding use the existing
native program and admission paths. Every retained period's support bank and published
physical simulation bank contributes to memory admission.

Group law parameters use direct `params[group][slot]` leaves, including fixed slots. Law
helpers use their transitive dependency closure. Shared economic or report helpers
retain their ordinary regime parameters independently. A native cell requires scalar
resolved marginal parameters, or a categorical probability vector; a helper may consume
an array and select a scalar. Standalone group transforms retain their cell-batching
contract. Solve laws use carried-state imputations; simulate laws receive actual carried
rows. Law callables receive categorical state-grid values or explicit state-code
arguments at their usual boundary.

Discrete physical shocks expose their category code through either the bare name or the
explicit `<shock>_code` alias. Continuous physical draws have no node-code alias.
Categorical carried states use their declared domain for both solve imputations and
actual simulation rows. Invalid imputed codes feed mandatory law integrity even when a
code-reading callable would otherwise return finite parameters.

Shared discrete projection admission consumes the demanded query's actual physical read
closure. An unused group declaration does not create an observed category. Admitted
reuse requires the same semantic law/helper bindings, parameter operand paths, ordered
categorical domains and effective law-input projections. Continuous identity requires an
effect-free direct-return JAXpr without equations or closed constants; finite discrete
identity may instead enumerate the complete code domain with matching codecs. A private
cardinality cap bounds exhaustive build-time work; direct-return proofs apply to large
domains without enumeration. A failed proof names the shock, query and projection input.
If an adopted EKL current law changes a relevant demanded input, the required joint CDF
refinement remains separate from this private admission. The verified transition-local
next-offer law is not itself a new current law.

Simulation draws fresh addressed base innovations before the decision. Ordinary utility,
reporting and transitions consume the same rounded physical realization, which survives
result save/load. Compiler barriers preserve that physical boundary; an ordinary
consumer cannot recover an unrepresented base coordinate by reassociation. Mandatory law
and factor integrity is enforced even with logging and optional checks disabled.

Weight factors retain support before their rounded product underflows, including factors
inside a joint categorical group. Finite rare-weight times large-value terms and finite
zero-primal weight derivatives use the shared scaled arithmetic. Active-dtype/backend
value cotangents can still underflow before a large upstream payoff derivative; this
extreme sensitivity is not an accuracy-qualified marginal channel. Later economic
marginals must weight their explicitly represented derivative values.

Uncomposed current-IID intersections fail explicitly: non-GridSearch decisions,
Decoupled objectives, action partitions, co-mapped or invariant state blocks,
bound/query reconstruction, and gated or same-period value readers. Refusals follow
actual owners and incoming readers, preserving unrelated legacy components. Policy
lookup awaits explicit observation/conditional semantics. Signed rule ancestry is
refused at the native decision boundary. These scoped holds do not retire legacy routes
or complete the later exit, query and signed-reader stages.

## Legacy migration

Migrate legacy IID-state and folded declarations directly to named shock groups with
explicit base rules. Retaining constructor spellings, physical-bin nodes or the former
finite quadrature operator is not an acceptance requirement. Do not introduce frozen
pullback anchors or a compatibility bridge to preserve those numerical choices.

Migration preserves the economic law, information available at choice, draw date and
joint dependence. A current observed shock and a post-action transition-local lottery
are distinct: moving the latter before choice changes the model. Persistent processes
remain real states, with their node-mode and code support retained. State-conditioned
laws must keep their actual conditioning date rather than merely reuse a state name.
Simulation applies the declared physical law to addressed base draws.

Validate each replacement through the public API against its declared quadrature and
independent economic oracle. Continuous-law identity and finite quadrature identity
remain separate facts; neither implies bitwise equality across compiler topologies.
Compare structural action ownership separately from value tolerances. Retire legacy
consumers only after their replacement's required solver, query and replay gates pass.

## EKL migration and resources

The exact reference household comes from lcm-zoo
`ad2658dae4df00aaf53934f9011adc4c60dcc959`,
`tests/test_ecksteinCareerFamilyDecisions2019_reference_household.py`. Its literal
Fraction loops are independent of production lowering. Its shared solve/simulation
decision helpers test routing agreement, not an independent derivation of household
economics.

For the full-model S5 precondition, make a per-shock table of law inputs, draw date and
every formation, divorce and counterparty projection. Action-dependent next-period
offer-set laws remain persistent/transition-local processes. They are not admitted as
current IID law callables. For every adopted shared discrete IID law, prove projection
invariance of its inputs or implement the common-uniform joint CDF refinement required
by the issue. Toy-law invariance alone is insufficient.

The initial source review uses lcm-zoo `ad2658dae4df00aaf53934f9011adc4c60dcc959`,
`src/lcm_zoo/ecksteinCareerFamilyDecisions2019/model.py`:

| Declaration                    | Actual source inputs and date                                                                                                           | Migration decision                                                                                                                                                                                                |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_job_offer_probs`             | Current hours action, current education/health and experience advanced by hours; next-period offer set (`:451`).                        | Retain a persistent offer-set state with its edge law. Do not admit this callable as current IID.                                                                                                                 |
| `marriage_offer_young_f/m`     | Current school action and an arrival-age logit (`:2120`, `:2168`).                                                                      | Preserve existing transition-local timing. A new current-IID declaration needs a genuine period-current state law; it cannot relabel the school action as a parameter.                                            |
| `_spouse_offer_probs_from_f/m` | Own education and fixed parameters (`:1389`); one partner education draw also determines partner experience (`:2717`).                  | Retain one shared innovation for counterparty and couple reads. Same-period own education is preserved by the inspected single projections (`:2234`, `:2238`); changed-law cases require the joint CDF partition. |
| `_single_projection`           | Own economic states project identically; young projections also set `married_last_period` and phase-specific schooling years (`:2323`). | Check each adopted law against changed fields. Preserving education does not establish invariance for a callable reading marital history.                                                                         |

These are source-based migration constraints, not a revalidation of the full paper
model. S5 must connect adopted declarations and projection-invariance checks through the
public API before acceptance.

For S6, use symbolic full-model dimensions, dtype bytes, query channels and all consumer
counts with the resolved per-device chunk budget. Account for source, target and
transient workspace simultaneously. Never allocate full household arrays merely to size
them. If retaining queries exceeds the budget, recompute inside the same declared chunk.
Assert retained/recomputed equality and release after the last registered consumer. A
reduced-model run cannot certify full-model memory admission.

S10 also requires the specified production regime-period reconstruction-cost measurement
through the perf-loop workflow. Preserve existing production owners and paper-ledger
holds; a correctness fixture is not that performance receipt.

## Ordered delivery

### Gate migration map

The legacy gate date is the arrival period, after the scheduled transition has produced
the landing state and before the destination action. An arrival adapter retains
`(incoming source, proposed target, source role, arrival period)` provenance; a
destination-global exit cannot stand in for an edge-specific fallback. An adapter
retains its declared Boolean comparison, rather than substituting a utility maximum.

| Legacy behavior                                           | Replacement and preserved contract                                                                                                                         | Baseline witness                                                                             |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Strict unanimous formation                                | Consent at the existing arrival date; native current-period Consent for the new household. Both values belong to the target's common participating policy. | `tests/regime_building/test_gated_edges_collective_solve.py`                                 |
| Collective participation and empty support                | Dissolution exit at arrival; realized participating-query support replaces interpolated dissolution flags. Keep zero-weight vetoes and one final action.   | `tests/simulation/test_realised_dissolution_route.py`                                        |
| Incoming-edge-specific fallback                           | Arrival adapter with incoming-leg provenance; initial and ungated occupants cannot borrow another leg's fallback.                                          | `tests/simulation/test_dissolution_route_provenance.py`                                      |
| Moving or retirement comparison                           | Arrival adapter retains the strict comparator, mapped landing and source-period utility/discounting.                                                       | `tests/solution/test_terminal_carry_kernel_edge_kwargs.py`                                   |
| Target value floor, window or mixed state/value predicate | Arrival adapter retains the entire threshold/window predicate, arrival-age grids and temporal parameter namespace.                                         | `tests/test_discrete_grid_values.py`, `tests/params/test_gate_reads_on_the_target_domain.py` |
| Reference-only or self-reference hurdle                   | Explicit typed projected reader at arrival; a value dependency is not another move.                                                                        | `tests/solution/test_gated_edge_self_referencing_ref_age_specialized_axes.py`                |
| Collective component comparison                           | Adapter over one selected joint-policy tuple; do not infer unanimous consent from a one-sided comparator.                                                  | `tests/solution/test_decoupled.py`                                                           |
| Value threshold on a self-loop                            | Adapter scoped to the edge and arrival date, carrying states once and preventing another same-period move.                                                 | `tests/regime_building/test_carried_state_through_gated_self_loop.py`                        |
| State-only or layout-only crossing                        | State-only Gate or predicate-free layout route, keeping role mapping, phase settlement and off-grid behavior.                                              | `tests/simulation/test_phased_edge_fallback.py`                                              |

Each migrated behavior retains an explicit replacement public-API test before
retirement. Apply projections to the realized landing before reading a value; apply a
predicate after reading its operands at the query point. Neither operation commutes with
interpolation. Preserve source, target, reference and edge parameter namespaces,
phase-specific settlements and exact arrival-period demand. The new realized policy
query intentionally replaces the interpolation-of-maxima residual; changed off-grid
outcomes need oracle evidence, not a claim of legacy bitwise parity.

### Arrival pricing depth

An arrival is priced exactly one level deep. The user-facing statement of this
approximation, with its error pattern and the advice to refine grids near participation
thresholds, is
[Participation checks over several periods](../user_guide/collective_regimes.md#participation-check-depth).

- **Level 1 is exact.** A source reading an arrival at date `d` rebuilds the target's
  action problem exactly at the realized landing, as above. Simulation decides each
  realized arrival the same way.
- **Below level 1 a stored surface replaces the rebuild.** The rebuilt action problem
  does not rebuild its own later arrivals: for an arrival from regime `R` into `U` at
  `d + 1` it reads the stored surface `A_{R->U}`. `A_{R->U}` holds, at every node of
  `U`'s grid at `d + 1`, the operands the level-1 decision consumes there rather than
  its verdict: each policy answer it reads (the acting target, every declared read,
  every member's fallback) through its support indicator, NaN where that answer fails
  its integrity, and each value component the decision reads, zero where the answer has
  no support. A node whose whole answer fails its integrity is NaN in every channel.
- **Off a node the decision is taken again.** Each channel is interpolated by `U`'s own
  continuation reader, as `V_U` would be. A support indicator reads as support where it
  is at least one half; a value is the support-weighted mean of the supported corners.
  The arrival decision is re-run on those answers with the landing state and its own
  date, age and parameters, and each member takes the branch it selects. On a node the
  rebuilt answers are the node's own, so the read equals the level-1 decision there.

Where each piece lives:

- `install_arrival_source_queries` (`src/_lcm/arrival_queries.py`) binds every plan
  against pricing contexts: the canonical action problems, carrying no plans of their
  own. Each surfaced target's continuation in a pricing context is replaced by
  `_ArrivalSurfaceValue` (`_surface_action_problem`), and the context names the plan
  whose surface it reads in `SolverBuildContext.arrival_surfaces`. No plan contains a
  plan, so depth one holds by construction.
- `surface_channels` fixes the channel order of one plan, from its `_ArrivalReducer`;
  `surface_queries` is the same query graph with the root answering those channels. The
  publisher and every reader derive the layout from the same plan.
- `_GridSearchArgumentBuilder.arrival_surfaces` (`src/_lcm/solution/grid_search.py`)
  substitutes the surface for the target's entry in `next_regime_to_V_arr` and supplies
  `arrival_surface_arguments`: each surfaced decision's own period, age and parameters.
- The `_arrival_surfaces` core program of `_BorrowedArrivalKernel`
  (`src/_lcm/arrival_publication.py`) tabulates the surfaces over `U`'s grid with a
  bounded `lax.map` after `R`'s period kernel. Each surface is a leaf of `R`'s dated
  `ArrivalContinuation` payload, so it is published by `R`'s period program, transported
  to `d - 1` with that payload, and counted by the same admission.
- Backward induction retains the surfaces on `SolutionResult`, simulation replays the
  same plans against them (`src/_lcm/simulation/arrival_values.py`), and
  `src/_lcm/persistence/solution.py` saves and verifies them under the archive manifest
  field `"arrival_surfaces"`, each entry tagged `"layout": "decision-branches"`; an
  archive of another layout is refused on load. A solution without them, such as a
  `dataclasses.replace` copy, refuses replay where a plan reads one.
- A decision reading anything beyond its declared reads, the landing state, `period`,
  `age` and its parameters — realized physical information — cannot be re-run between
  nodes (`decides_between_nodes`). Its surface keeps the value after the decision per
  node, integrated over that information, shaped like the plan's answer, and the pricing
  context reads it through the target's ordinary continuation reader.

Cost. A period's program prices one arrival level whatever the number of remaining
arrival dates, so the solve is linear in the horizon. Full nesting multiplies a period's
program by the target's actions times its reads and lottery nodes for each further
arrival date. On the household example with 10 wealth nodes and 5 consumption shares at
four periods (CPU), the solve went from 84.9 s and 10.9 GB peak RSS under full nesting
to 19.1 s and 2.6 GB with a stored value after the decision, and takes 37.0 s and 3.8 GB
with stored branches, compiling from an empty persistent cache (one measurement on a
shared machine). The household's period-0 query program at 3 wealth nodes and 2 shares
costs 1.57 times the one with a stored value after the decision, and is the same at four
and five periods. A surface holds `len(surface_channels(plan))` entries per node: eight
for the household's continuing couple, against two values after the decision.

Accuracy.

- Where every landing is a grid node the values are bitwise those of full nesting: a
  discrete-wealth twin of the household example at four periods, fp64 and fp32.
- Off grid, where the decision does not switch inside a cell, each member's value is a
  smooth branch read off its interpolant: second order in `h`. Where the decision
  compares values, the interpolated operands place the switch to second order, so a
  member who is overruled there carries the jump between his branches only within a band
  of order `h^2`; the member who decides has a kink and stays second order. Where the
  decision reads policy support, a Boolean, the switch can only be placed halfway across
  the cell, and every member whose value jumps there carries the jump over at most
  `h / 2`.
- References: a pure-Python two-member reference (`reference_branch.py` in the lane
  notes; she leaves rich households, he is overruled) gives a read error near the switch
  of 0.069, 0.015, 0.0036, 0.00084, 0.00026 at `h` = 4.75 to 0.30, against 0.9 to 1.6,
  not converging, for the stored value after the decision; with support deciding neither
  converges. A pylcm witness with the same structure gives a max period-0 error of
  0.048, 0.017, 0.0050, 0.0014 at `h` = 4, 2, 1, 0.5, against 2.45 to 2.48 before.
- The household example at four periods (fp64, 3 consumption shares) changes only period
  0, and its only surfaced check is the continuing couple's, on support. The couple's
  mean absolute error is 0.102, 0.116, 0.075, 0.034 and the max 0.284, 0.241, 0.693,
  0.548 at 4, 6, 10, 16 wealth nodes. Stored branches change only the nodes whose
  landings cross a support switch (zero, two, three and two nodes). At 10 and 16 nodes
  the max is bitwise the one a stored value after the decision gives (at 6 it moves from
  0.246 to 0.241, at 10 the second-largest error from 0.56 to 0.21): a decision taken on
  the smooth surface error near a threshold — the household's choice or the exact
  next-landing check — flips and moves a member's value by a jump.

There is no switch back to full nesting. `tests/solution/test_arrival_pricing_depth.py`
pins the program-cost bound, the on-grid equality and the unchanged later periods
against nested-route values in `tests/data/arrival_pricing_depth/`, the second-order
read of an overruled partner against a closed-form solution, the archive round trip and
the refusal of another archive layout.

### Stage gates

The child issues of 607 carry the native dependency graph. The implementation PR closes
the parent and children together. A child is accepted only with its own tests and all
inherited behavior, never by deleting a failing economic behavior.

S0 establishes standalone contracts and references. S1 and S2 follow S0; S3 follows S2;
S4 follows S3. S5 and S6 follow S4 and their issue-specific preconditions. S7 follows
S5/S6, S8 follows S7, and S9 follows S7/S8 with an explicit timing-preserving gate
migration map. S10 follows S2/S4 and the performance receipt. S11 follows S1/S5/S10. S12
retires legacy topology only after all prior stages are green.

Each receipt records source identity and dirty diff, exact command, environment,
precision/backend, owner, exit code and raw log/XML paths. New RNG fixtures are new
files beside frozen evidence. Completion requires the applicable CI suite and resource
gates, independent branch review with fixes, and a review-ready PR. Broad certificate
campaigns remain in CI or offline; optional certification does not block balanced
semantic closure.

### Finite S10 acceptance workloads

> Superseded: the revised gate table in the
> [speed-first amendment](#speed-first-amendment-2026-10-29) replaces this matrix as the
> completion gate.

The adopted direction is recorded in
`sciebo:pro-audits/pylcm-pr621-s10/bundle-round0/reply/pylcm-s10-direction-review-round0/`.
Its `workload-matrix.json` binds the behavior IDs below; the direction verdict does not
qualify their unfinished implementation. Compatible behaviors share fixtures and
receipts, rather than forming a Cartesian product of solver features.

| IDs    | Required behavior                                                                                                                                                                  |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1, P2 | Scalar DC resources-only publication and genuine non-resource current-shock reconstruction.                                                                                        |
| P3, P4 | Original NB resources-only bank, and NB-U with a liquid state, a passive state and genuine discrete branches.                                                                      |
| P5, P6 | Finite NEGM and NNBEGM keeper/adjuster selection, each with resources-only and genuine U variants.                                                                                 |
| C1, C2 | Positive nonlinear CE transport through eligible NB and finite NNBEGM routes, retaining typed statistics and economic directions.                                                  |
| D1–D6  | Compatible incoming GridSearch W-only, plain EGM without its own current shock, DC-to-passive-NB, original NB, NEGM inner-DC and NNBEGM inner-NB readers.                          |
| C3     | A U source reading a projected U descendant, retaining both producing dates through physical replay and archive acquisition.                                                       |
| C4, C5 | Law/weight/resource directions including a derivative-active null event; the existing signed-linear boundary and its prohibited consumers.                                         |
| C6     | Explicit balanced grid replay against default original-candidate replay, with separate policy/Q/moment diagnostics and certified-plus-grid refusal.                                |
| Q1     | A GridSearch exit source querying an off-grid scalar DC target with no exits; the mover executes that same target policy, takes no second exit and makes one scheduled transition. |
| L1     | Shared generation reuse, support specialization, single-flight cleanup, retained evicted results, finite-budget production and fresh archive authority.                            |

Q1 selects the exit adapter as the minimal generic requirement. Target Participating
equals Raw because that target has no exits. The adapter remains unfinished until its
actual EGM target read is served; current GridSearch query admission is not evidence of
that support. Add an arrival variant only for a distinct named timing or custody
consumer. Bind faithful EKL's actual solver, dates, mapped states, shared-offer
addresses, lagged-marital context and retained leaves separately; this generic row does
not assert that EKL has an EGM target.

Before accepting a row, freeze its real public declarations, parameters and support,
producer/consumer dates, required coordinates and native leaf roster, finite capacities
and owned budget, precision/backend/device/JIT, independent reference, sampled
population, common base draws and predetermined scales in its `S10-WORKLOADS.json`
record. Unbound fields and refused required reads remain open. Use the original solver
factories and existing generation/materializer; do not substitute a DC-shaped owner for
an NB or nested producer. Lazy authority roots and receiving archive copies belong to
the same admission and retained-residency contract.

The balanced budgets above apply separately to represented value, economic marginal,
associated policy and named moments. Check original nodes, required incoming
coordinates, sampled off-node shocks, served sides and relevant corners, with
invalid/capacity/read corruption controls. Assess grid replay against original-candidate
reconstruction separately from actions absent from both banks. The latter candidate-set
error may stay explicitly unknown under `report_unknown`; continuous-global-optimum or
whole-domain refinement proofs are not balanced completion gates. Required sampled
represented-channel checks and original discrete ownership/ties remain binding.

Close the existing call owner, authentic family preparations and required incoming and
dated replay/archive reads, then attach source-bound sampled receipts and an
attributable representative production reconstruction capture. Measure cold acquisition,
compilation, synchronized warm execution, required reads, physical replay, foreign
acquisition and retained memory separately. A baseline that refuses a new capability
supplies no speedup ratio. Fixture walltimes do not supply production costs. Broad
certificate qualification runs at closure or in CI, not after each small edit. Another
general direction audit requires a concrete new architectural ambiguity.

Independent S11 row-address/draw/override preparation may proceed. Integrated S11
acceptance still waits for S1/S5 and S10's generation, reads, reconstruction, ownership
and archive contracts. S12 retirement waits for every preceding stage and replacement
consumer; independent preparation does not close either gate.
