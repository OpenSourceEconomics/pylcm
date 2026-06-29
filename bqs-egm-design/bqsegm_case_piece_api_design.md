# BQSEGM Case-Piece API Design for pylcm

**Status:** Revised draft after adversarial math/code audit
**Audience:** pylcm maintainers and model authors using DAG-style model definitions
**Goal:** Let BQSEGM exploit piecewise institutional rules without adding a new object-oriented modeling layer, while preserving boundary topology, validator scope, and the exact interpolation convention needed by EGM.

---

## 1. Motivation

BQSEGM needs to solve EGM problems branchwise when institutional rules create formula changes, kinks, cliffs, or notches. In models such as ACA, rules like Medicaid eligibility, subsidy brackets, premium default, and consumption floors are often written naturally as `if` statements, lookup tables, or `jnp.where` expressions.

Those are fine for brute-force evaluation, but opaque to an EGM-style solver. BQSEGM needs to know:

- where a formula changes;
- which smooth formula applies on each side;
- which boundary surfaces deserve one-sided candidates;
- where interpolation must not cross a discontinuity;
- which candidates must be masked after EGM recovers current resources.

The design below keeps the user-facing pylcm interface close to today’s function-based DAG style. It introduces two decorators plus a small boundary metadata helper that attach metadata to ordinary functions.

---

## 2. Public concepts

Use four names consistently.

| Term | Meaning |
|---|---|
| **regime** | Existing pylcm dynamic-programming regime: separate value function, transition target, solver, and state/action structure. |
| **case boundary** | A Boolean DAG node defining a threshold surface, such as `assets < medicaid_asset_limit`, plus equality-owner and boundary-type metadata. |
| **piece** | An alternative formula for an existing DAG output under a predicate. |
| **segment** | Solver-internal smooth EGM object produced after lowering a piece combination. |

Avoid user-facing terms like “branch,” “branch schedule,” or “institutional regime.” They are too easy to confuse with pylcm regimes.

---

## 3. Minimal API

### 3.1 Declare a case boundary

A case boundary is a normal Boolean DAG function plus metadata describing the equality
surface and which predicate side owns equality. The equality owner matters for exactness: a
left-limit candidate must not be allowed to win at a boundary point that the predicate assigns
to the right side.

```python
@lcm.case_boundary(
    lcm.boundary(
        "assets",
        "medicaid_asset_limit",
        equality="otherwise",        # because the predicate uses assets < limit
        kind="jump",                 # or "continuous_kink" / "hard_constraint"
    ),
    lcm.boundary(
        "countable_income",
        "medicaid_income_limit",
        equality="otherwise",
        kind="jump",
    ),
)
def medicaid_eligible(
    assets,
    countable_income,
    medicaid_asset_limit,
    medicaid_income_limit,
):
    return (
        (assets < medicaid_asset_limit)
        & (countable_income < medicaid_income_limit)
    )
```

The function is still an ordinary DAG node returning a Boolean array. The decorator says that
the relevant equality surfaces are:

```text
assets == medicaid_asset_limit
countable_income == medicaid_income_limit
```

and that equality belongs to the `otherwise` side for both surfaces. BQSEGM uses this metadata
to insert or check one-sided candidates, prevent interpolation across discontinuities, and
apply open/closed endpoint eligibility at exact boundary queries.

For a small v1 API, `lcm.boundary(...)` can be a metadata helper rather than a decorator. A
legacy two-tuple such as `("assets", "asset_limit")` should be accepted only if the equality
side is inferred unambiguously from the predicate AST; otherwise validation should require an
explicit `equality=` argument.

### 3.2 Declare formula pieces

```python
@lcm.piece("oop", when=medicaid_eligible)
def oop_medicaid(medical_expense):
    return medicaid_oop_formula(medical_expense)


@lcm.piece("oop", otherwise=medicaid_eligible)
def oop_private(medical_expense, insurance_plan):
    return private_oop_formula(medical_expense, insurance_plan)
```

This means:

```text
if medicaid_eligible:
    oop = oop_medicaid(...)
else:
    oop = oop_private(...)
```

No separate `split=...` or `case=...` argument is needed. The predicate function is the split identity.

### 3.3 Optional brute-force fallback

The model may still define the ordinary combined function:

```python
def oop(oop_medicaid, oop_private, medicaid_eligible):
    return jnp.where(medicaid_eligible, oop_medicaid, oop_private)
```

BruteForce can use this normal DAG node. BQSEGM can ignore it and lower the decorated pieces directly.

---

## 4. Full ACA-style example

```python
@lcm.case_boundary(
    lcm.boundary(
        "assets",
        "medicaid_asset_limit",
        equality="otherwise",
        kind="jump",
    ),
    lcm.boundary(
        "countable_income",
        "medicaid_income_limit",
        equality="otherwise",
        kind="jump",
    ),
)
def medicaid_eligible(
    assets,
    countable_income,
    medicaid_asset_limit,
    medicaid_income_limit,
):
    return (
        (assets < medicaid_asset_limit)
        & (countable_income < medicaid_income_limit)
    )


@lcm.piece("premium", when=medicaid_eligible)
def premium_medicaid():
    return 0.0


@lcm.piece("premium", otherwise=medicaid_eligible)
def premium_private(gross_premium, subsidy):
    return gross_premium - subsidy


@lcm.piece("oop", when=medicaid_eligible)
def oop_medicaid(medical_expense):
    return medicaid_oop_formula(medical_expense)


@lcm.piece("oop", otherwise=medicaid_eligible)
def oop_private(medical_expense, insurance_plan):
    return private_oop_formula(medical_expense, insurance_plan)


def resources(assets, income, premium, oop):
    return assets + income - premium - oop
```

BQSEGM internally creates two smooth DAG variants:

```text
Variant 1:
    premium = premium_medicaid
    oop     = oop_medicaid
    valid where medicaid_eligible

Variant 2:
    premium = premium_private
    oop     = oop_private
    valid where ~medicaid_eligible
```

It then solves EGM under each variant, masks inconsistent endogenous candidates using the NaN-dead convention, adds side-aware boundary candidates, splits the surviving candidate chains into monotone segments, and upper-envelopes the surviving segments.

---

## 5. Supported in the first version

The first implementation should be deliberately small.

Supported:

- two-way Boolean predicates;
- explicit boundary surfaces with equality-owner metadata;
- boundary type labels: `continuous_kink`, `jump`, or `hard_constraint`;
- `@lcm.piece(output, when=predicate)`;
- `@lcm.piece(output, otherwise=predicate)`;
- multiple outputs sharing one predicate;
- predicates depending on recovered current state or resources;
- predicates used as post-EGM consistency masks;
- NaN-dead masking of invalid pre-envelope candidates;
- one-sided boundary candidates and open/closed endpoint ownership;
- ordinary fallback functions for brute force.

Examples of supported case boundaries:

- Medicaid eligible versus ineligible;
- premium default binds versus does not bind;
- consumption floor binds versus does not bind;
- borrowing rate segment, such as `savings < 0`;
- tax or subsidy bracket boundary, if modeled as a binary split.

---

## 6. Not supported in the first version

Do not support these initially:

- multiway table lowering;
- nested piece splits for the same output;
- automatic satisfiability pruning of many case combinations;
- hidden helper functions with economic case logic such as `if`, `where`, `clip`, `maximum`,
  `searchsorted`, etc.;
- unscoped primitive bans applied to solver interpolation or continuation machinery;
- dynamic Python control flow;
- arbitrary lookup tables inside smooth pieces;
- piecewise formulas hidden inside lambdas or closures with unavailable source;
- ordinary aggregate carry publication without segment topology or switch-refined grid proof;
- nonlinear or non-branchable dependence on current assets after all pieces are fixed.

A future `@lcm.piecewise_table` or `@lcm.piecewise_affine` decorator can support subsidy/tax tables more directly.

---

## 7. What must be exposed as a case boundary?

Expose a case boundary only if crossing the threshold changes an object relevant to the Bellman or Euler calculation:

- resources;
- transition;
- utility;
- feasible set;
- continuation;
- marginal return.

Do not expose:

- reporting-only flags;
- simulation-only summaries;
- debug guards;
- purely numerical clipping that has no economic interpretation.

---

## 8. Beginning-of-period versus recovered-state predicates

A predicate may be known before the EGM inversion, or it may depend on a current state recovered by EGM.

Example:

```python
def medicaid_eligible(assets, countable_income, ...):
    return assets < limit
```

In BQSEGM this is handled as:

```text
assume the Medicaid piece
solve EGM
recover assets/resources
evaluate medicaid_eligible
discard inconsistent candidates
```

The predicate does not need to be known before the solve. It only needs to be evaluable after the endogenous current state is recovered.

---

## 9. Internal lowering using DAG topology

The public API should not expose `dags.tree` to model authors. Internally, BQSEGM should use
DAG topology to lower decorated pieces while keeping solver-provided interpolation behind a
trusted boundary.

Internal lowering:

```text
1. Build the ordinary model DAG.
2. Collect decorated case boundaries, equality-owner metadata, and pieces.
3. For each output with pieces, identify its alternative producers.
4. Build specialized smooth DAG variants by replacing selected producers.
5. Use DAG reachability to identify user-authored economic nodes active in each variant.
6. Validate only those user-authored economic nodes and their user helpers.
7. Do not validate trusted solver infrastructure with the forbidden-primitive ban:
   continuation interpolation, grid location, envelope evaluation, and EGM kernel internals
   legitimately use compare/select/search-like primitives.
8. Run EGM on each variant.
9. Evaluate predicates as consistency masks and convert invalid pre-envelope candidates to
   NaN-dead triples.
10. Insert one-sided boundary candidates with open/closed endpoint metadata.
11. Split into monotone feasible subsegments and upper-envelope the resulting segments.
```

This lets `dags.tree` answer:

```text
Which functions can produce "oop"?
Which predicate guards each producer?
Which downstream nodes depend on "oop"?
What specialized DAG is created if "oop" is replaced by "oop_medicaid"?
Which user-authored nodes remain reachable after that replacement?
Which predicates must be checked after EGM recovers current assets?
```

---

## 10. Decorator metadata

The decorators should only attach metadata. They should not alter runtime behavior.

```python
from dataclasses import dataclass
from typing import Literal


BoundaryKind = Literal["continuous_kink", "jump", "hard_constraint"]
EqualityOwner = Literal["when", "otherwise"]


@dataclass(frozen=True)
class BoundarySurface:
    variable: str
    threshold: str
    equality_owner: EqualityOwner
    kind: BoundaryKind


@dataclass(frozen=True)
class CaseBoundaryMeta:
    boundaries: tuple[BoundarySurface, ...]


@dataclass(frozen=True)
class PieceMeta:
    output: str
    predicate_name: str
    side: Literal["when", "otherwise"]


def boundary(
    variable: str,
    threshold: str,
    *,
    equality: EqualityOwner,
    kind: BoundaryKind,
) -> BoundarySurface:
    return BoundarySurface(
        variable=variable,
        threshold=threshold,
        equality_owner=equality,
        kind=kind,
    )


def case_boundary(*boundaries):
    coerced = tuple(_coerce_boundary(b) for b in boundaries)

    def deco(fn):
        fn.__lcm_case_boundary__ = CaseBoundaryMeta(coerced)
        return fn

    return deco


def piece(output, *, when=None, otherwise=None):
    if (when is None) == (otherwise is None):
        raise ValueError("Use exactly one of when= or otherwise=.")

    predicate = when if when is not None else otherwise
    side = "when" if when is not None else "otherwise"

    def deco(fn):
        fn.__lcm_piece__ = PieceMeta(
            output=output,
            predicate_name=predicate.__name__,
            side=side,
        )
        return fn

    return deco
```

`_coerce_boundary` may accept a two-string tuple only as a convenience if the validator can
infer equality ownership from a simple predicate. If inference fails, the error should require
`lcm.boundary(..., equality=..., kind=...)`.

## 11. Validation goals

BQSEGM relies on formulas being smooth within each piece. Therefore, validation must guarantee:

```text
No hidden economic case logic exists in any user-authored smooth BQSEGM subgraph.
```

The scope qualifier is essential. The EGM solver itself must interpolate next-period value and
marginal objects on grids; that trusted solver infrastructure normally lowers to compare,
select, clamp, and search-like primitives. A global AST/JAXPR ban would reject every realistic
EGM model. The ban applies only to user-authored economic functions and user helpers reachable
inside a specialized smooth case piece.

This requires three checks:

1. AST validation for Python-level branching in user economic nodes.
2. JAXPR validation, including nested JAXPRs, for hidden JAX piecewise primitives in those nodes.
3. Numeric spot checks as diagnostics for node-resolution cliffs where build-time parameter
   values are available.

The ordinary fallback DAG may contain `jnp.where` or lookup logic if it is not reachable in the
BQSEGM smooth variant. Solver-provided continuation interpolation may contain search/select
primitives even when it is reachable from the mathematical Euler RHS; it is outside the
user-economic-node validation scope and is checked by separate interpolation/envelope tests.

---

## 12. AST validation

### 12.1 Smooth user economic functions

In smooth user-authored pieces and smooth user helpers, reject:

- `ast.If`;
- `ast.IfExp`;
- `ast.Match`;
- `ast.Compare`;
- `ast.BoolOp`;
- `ast.UnaryOp(Not)`;
- calls to common piecewise functions.

Forbidden function calls in smooth user economic functions:

```text
where
select
cond
switch
clip
maximum
minimum
max
min
abs
sign
heaviside
searchsorted
digitize
interp
piecewise
```

Some of these functions, such as `abs` or `clip`, can be harmless on a restricted domain. v1
should still reject them by default unless the project adds an explicit trusted-helper review
mechanism with a stated domain proof.

### 12.2 Case-boundary functions

In `@lcm.case_boundary` functions, allow vectorized comparisons and Boolean logic, such as:

```python
return (assets < limit) & (income < income_limit)
```

Still reject Python `if`, `match`, and non-JAX-traceable dynamic control flow. Also validate
that the declared equality owner matches the predicate for simple recognized comparisons, e.g.
`x < limit` implies equality belongs to `otherwise`, while `x <= limit` implies equality belongs
to `when`.

### 12.3 AST checker sketch

```python
import ast
import inspect
import textwrap


PIECEWISE_CALL_NAMES = {
    "where",
    "select",
    "cond",
    "switch",
    "clip",
    "maximum",
    "minimum",
    "max",
    "min",
    "abs",
    "sign",
    "heaviside",
    "searchsorted",
    "digitize",
    "interp",
    "piecewise",
}


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


class PiecewiseASTChecker(ast.NodeVisitor):
    def __init__(self, *, mode, fn_name):
        self.mode = mode  # "smooth_user" or "boundary"
        self.fn_name = fn_name
        self.errors = []

    def error(self, node, message):
        self.errors.append(
            {
                "function": self.fn_name,
                "line": getattr(node, "lineno", None),
                "message": message,
            }
        )

    def visit_If(self, node):
        self.error(node, "Python if is not allowed in BQSEGM-traced functions.")
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.error(node, "Conditional expression is not allowed.")
        self.generic_visit(node)

    def visit_Match(self, node):
        self.error(node, "match/case is not allowed.")
        self.generic_visit(node)

    def visit_Compare(self, node):
        if self.mode == "smooth_user":
            self.error(
                node,
                "Comparison in a smooth formula creates a hidden case boundary.",
            )
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        if self.mode == "smooth_user":
            self.error(
                node,
                "Boolean logic in a smooth formula creates a hidden case boundary.",
            )
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if self.mode == "smooth_user" and isinstance(node.op, ast.Not):
            self.error(node, "not in a smooth formula creates a hidden case boundary.")
        self.generic_visit(node)

    def visit_Call(self, node):
        name = _call_name(node.func)
        leaf = name.split(".")[-1] if name else None

        if self.mode == "smooth_user" and leaf in PIECEWISE_CALL_NAMES:
            self.error(
                node,
                f"Call to piecewise function `{name}` is not allowed "
                "inside a smooth BQSEGM piece.",
            )

        self.generic_visit(node)


def check_ast(fn, *, mode):
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        return [
            {
                "function": getattr(fn, "__name__", "<unknown>"),
                "line": None,
                "message": f"source unavailable for AST validation: {exc}",
            }
        ]
    tree = ast.parse(source)
    checker = PiecewiseASTChecker(mode=mode, fn_name=fn.__name__)
    checker.visit(tree)
    return checker.errors
```

---

## 13. JAXPR validation

AST validation does not catch piecewise logic hidden in helper functions. JAXPR validation does.
It must be applied to the same scoped set of user-authored smooth functions, not to the trusted
EGM solver or continuation interpolation.

Forbidden smooth user primitives:

```text
lt
le
gt
ge
eq
ne
select_n
cond
switch
max
min
clamp
abs
sign
sort
searchsorted
```

Checker sketch:

```python
SMOOTH_FORBIDDEN_PRIMS = {
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
    "select_n",
    "cond",
    "switch",
    "max",
    "min",
    "clamp",
    "abs",
    "sign",
    "sort",
    "searchsorted",
}


def iter_jaxpr_eqns(jaxpr_like):
    """Yield equations from a closed jaxpr and nested jaxprs in primitive params."""
    jaxpr = getattr(jaxpr_like, "jaxpr", jaxpr_like)
    for eqn in jaxpr.eqns:
        yield eqn
        for value in eqn.params.values():
            if hasattr(value, "jaxpr"):
                yield from iter_jaxpr_eqns(value)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    if hasattr(item, "jaxpr"):
                        yield from iter_jaxpr_eqns(item)


def check_jaxpr(fn, abstract_args, *, mode, trusted=False):
    if trusted:
        return []

    jaxpr = jax.make_jaxpr(fn)(*abstract_args)
    errors = []

    for eqn in iter_jaxpr_eqns(jaxpr):
        prim = eqn.primitive.name

        if mode == "smooth_user" and prim in SMOOTH_FORBIDDEN_PRIMS:
            errors.append(
                {
                    "function": fn.__name__,
                    "primitive": prim,
                    "message": (
                        f"JAX primitive `{prim}` indicates hidden piecewise logic "
                        "inside a smooth BQSEGM formula. Expose it as a case "
                        "boundary or mark a reviewed helper as trusted."
                    ),
                }
            )

    if mode == "boundary":
        # Validate that output dtype is bool or bool array.
        pass

    return errors
```

This catches hidden helper logic such as:

```python
def helper(x):
    return jnp.where(x > 0, x, 0)

@lcm.piece("resources", when=some_predicate)
def resources_piece(x):
    return helper(x)
```

The AST of `resources_piece` looks smooth, but the JAXPR contains comparisons and `select_n`.
By contrast, a BQSEGM continuation read implemented by pylcm may contain search or select
primitives and should be tested as solver infrastructure, not rejected as a hidden economic
case.

---

## 14. Reachability-aware validation

Do not validate every function in the model. Validate every user-authored function reachable in
a BQSEGM smooth variant, and explicitly stop at trusted solver-provided continuation and
envelope boundaries.

Algorithm:

```text
for each BQSEGM regime:
    build ordinary DAG
    collect piece outputs and boundary metadata

    for each feasible piece combination:
        clone producer map
        replace output producers with selected pieces
        find user-authored functions reachable from current-period economic targets
        exclude trusted solver nodes:
            continuation interpolation
            grid location
            envelope evaluation
            EGM kernel internals

        for each reachable user-authored function:
            if case_boundary:
                run boundary AST and JAXPR checks
                check Boolean output and equality-owner consistency
            elif trusted_smooth_helper:
                run the helper's declared domain checks and optional numeric tests
            else:
                run smooth-user AST and JAXPR checks
                run numeric spot checks when all required parameters are available
```

If the fallback function `oop` contains `jnp.where`, but BQSEGM replaces it with
`oop_medicaid` or `oop_private`, it is not reachable and should not fail validation.

The existing DC-EGM numerical jump checker is useful but narrower than this validator. It checks
savings-stage reads of the Euler state at node resolution and skips functions whose needed
arguments are free parameters at build time. BQSEGM may reuse or extend it, but the API should
not claim that it already detects every hidden cliff.

---

## 15. Coverage checks

For each output and predicate pair:

```text
exactly one when=predicate piece
exactly one otherwise=predicate piece
```

Valid:

```python
@lcm.piece("oop", when=medicaid_eligible)
def oop_medicaid(...): ...

@lcm.piece("oop", otherwise=medicaid_eligible)
def oop_private(...): ...
```

Invalid:

```python
@lcm.piece("oop", when=medicaid_eligible)
def oop_medicaid(...): ...
```

because the false side is missing.

Keep coverage strict in v1.

---

## 16. Boundary metadata checks

Every `case_boundary` should declare at least one boundary surface with equality ownership and
boundary type.

Valid:

```python
@lcm.case_boundary(
    lcm.boundary("assets", "asset_limit", equality="otherwise", kind="jump")
)
def asset_test(assets, asset_limit):
    return assets < asset_limit
```

Invalid:

```python
@lcm.case_boundary()
def asset_test(assets, asset_limit):
    return assets < asset_limit
```

The validator should also check:

- every boundary variable appears in the function signature or is computable in the DAG;
- every declared threshold function is reachable;
- every boundary is compatible with the state/action variables used by BQSEGM;
- the equality owner is consistent with simple recognized comparisons;
- the boundary type is one of `continuous_kink`, `jump`, or `hard_constraint`;
- discontinuous `jump` boundaries produce one-sided records or an explicit proof that the
  boundary cannot be optimal;
- open/closed endpoint flags are passed to the segment-envelope representation.

## 17. Error messages

### Hidden `where` in a smooth piece

```text
BQSEGMCaseError:
    function: oop_private
    line: 17
    reason: jnp.where found inside a smooth formula piece.
    fix:
        Define a new @lcm.case_boundary predicate and split oop_private into pieces.
```

### Hidden lookup table

```text
BQSEGMCaseError:
    function: subsidy
    primitive: searchsorted
    reason:
        lookup-table logic is hidden inside a smooth formula.
    fix:
        Use @lcm.piecewise_table(...) or expose bracket predicates.
```

### Missing otherwise piece

```text
BQSEGMCaseError:
    output: oop
    predicate: medicaid_eligible
    reason:
        missing otherwise=medicaid_eligible formula piece.
```

---

## 18. Open design questions

### 18.1 Multiway tables

Future API candidates:

```python
@lcm.piecewise_table(
    output="subsidy",
    variable="income",
    breakpoints="subsidy_breakpoints",
    side="right",
    discontinuities="subsidy_jump_mask",
)
def subsidy(...):
    ...
```

or:

```python
@lcm.piecewise_affine(
    output="tax",
    variable="income",
    breakpoints="tax_breakpoints",
    slopes="tax_slopes",
    intercepts="tax_intercepts",
)
def tax(...):
    ...
```

Do not include these in v1.

### 18.2 Binding cases

Some cases are not simple exogenous threshold predicates. They are equilibrium conditions:

- borrowing constraint binds;
- consumption floor binds;
- premium default binds.

These require one-sided KKT logic. The same metadata style can eventually represent them, but they should be implemented after simple formula pieces.

### 18.3 Satisfiability pruning

Multiple predicates create many case combinations. Initially, BQSEGM can enumerate and mask. Later it should prune impossible combinations statically when predicates logically contradict each other.

### 18.4 Trusted smooth helpers

Some low-level helpers may look piecewise but be mathematically smooth under the model domain.
A narrow trusted-helper escape hatch is acceptable only if it is explicit and reviewed:

```python
@lcm.smooth_helper(domain="x > 0; clip never binds on the solver domain")
def numerically_stable_helper(x):
    ...
```

The validator should treat this as an attestation plus a test obligation, not as a silent
bypass. Trusted helpers are for numerical smoothness devices; economic thresholds still need
`@lcm.case_boundary` metadata.

---

## 19. Implementation sequence

1. Implement metadata-only decorators: `case_boundary`, `piece`, and a `boundary(...)` metadata helper.
2. Add metadata collection in the model-building path.
3. Implement equality-owner and boundary-type validation.
4. Implement reachability-aware AST validation scoped to user economic nodes.
5. Implement reachability-aware JAXPR validation scoped to user economic nodes, including nested JAXPRs.
6. Add optional reviewed `smooth_helper` support or explicitly reject all bypasses in v1.
7. Implement two-piece DAG lowering for one output and one predicate.
8. Implement post-EGM predicate masking with the NaN-dead absent-candidate convention.
9. Implement one-sided boundary candidates and open/closed endpoint metadata.
10. Implement monotone subsegment construction with `segment_id` per subsegment.
11. Choose the continuation publication strategy: topology-preserving payload, or switch-refined aggregate grid with exactness tests.
12. Add a minimal Medicaid-style toy model.
13. Compare BQSEGM against brute force on the toy under the same interpolation/envelope convention.
14. Generalize to multiple outputs sharing the same predicate.
15. Only then consider lookup-table decorators.

## 20. Minimal test plan

### Unit tests

- decorator metadata is attached correctly;
- exactly one of `when=` or `otherwise=` is required;
- missing otherwise piece fails;
- missing boundary metadata fails;
- smooth AST rejects Python `if`;
- smooth AST rejects `jnp.where` in a user economic node;
- boundary AST permits vectorized comparisons;
- boundary metadata requires equality owner and kind;
- JAXPR catches hidden helper `jnp.where`;
- JAXPR recursively inspects nested jaxprs;
- trusted continuation interpolation does not false-positive under the scoped validator;
- fallback functions are ignored if unreachable.

### Solver tests

- one-predicate Medicaid toy;
- predicate depending on recovered current assets;
- boundary candidate at asset limit;
- invalid case candidates are masked as NaN-dead before envelope evaluation;
- open/closed endpoint ownership at exact threshold queries;
- upper envelope selects the correct side;
- folded or hole-split case gets more than one `segment_id`;
- brute force and BQSEGM agree on a dense-grid toy under the same convention;
- topology-preserving publication avoids aggregate interpolation bridges;
- BQSEGM beats asset-row mode on memory for the same toy only when the full cost inequality is favorable.

### Negative tests

- hidden `clip` inside a smooth piece;
- hidden `searchsorted` inside a smooth subsidy formula;
- incomplete true/false coverage;
- multiple pieces for the same predicate side;
- case boundary returning non-Boolean output;
- missing equality-owner metadata;
- finite `x` plus `value=-inf` used as a pre-envelope dead candidate.

---

## 21. Short summary

BQSEGM should expose institutional piecewise structure through metadata on ordinary DAG
functions:

```python
@lcm.case_boundary(
    lcm.boundary("x", "x_bar", equality="otherwise", kind="jump")
)
def below_bar(x, x_bar):
    return x < x_bar

@lcm.piece("resources", when=below_bar)
def resources_low(...):
    ...

@lcm.piece("resources", otherwise=below_bar)
def resources_high(...):
    ...
```

The user-facing interface stays close to pylcm today. Internally, pylcm uses DAG topology to
lower pieces into smooth EGM variants, validates only user-authored smooth economic nodes,
masks inconsistent endogenous candidates using the NaN-dead convention, inserts side-aware
boundary candidates, splits cases into monotone feasible segments, and upper-envelopes the
result.

The first version should support only binary predicates and formula pieces with explicit
boundary ownership. Multiway tables, binding KKT cases, and richer helper attestations can come
later. Exact backward induction also requires a topology-preserving continuation payload or a
switch-refined aggregate grid; an ordinary maxed aggregate carry is not enough in general.
