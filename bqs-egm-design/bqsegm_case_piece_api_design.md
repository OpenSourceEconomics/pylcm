# BQSEGM Case-Piece API Design for pylcm

**Status:** Draft design
**Audience:** pylcm maintainers and model authors using DAG-style model definitions
**Goal:** Let BQSEGM exploit piecewise institutional rules without adding a new object-oriented modeling layer.

---

## 1. Motivation

BQSEGM needs to solve EGM problems branchwise when institutional rules create formula changes, kinks, cliffs, or notches. In models such as ACA, rules like Medicaid eligibility, subsidy brackets, premium default, and consumption floors are often written naturally as `if` statements, lookup tables, or `jnp.where` expressions.

Those are fine for brute-force evaluation, but opaque to an EGM-style solver. BQSEGM needs to know:

- where a formula changes;
- which smooth formula applies on each side;
- which boundary surfaces deserve one-sided candidates;
- where interpolation must not cross a discontinuity;
- which candidates must be masked after EGM recovers current resources.

The design below keeps the user-facing pylcm interface close to today’s function-based DAG style. It introduces only two decorators that attach metadata to ordinary functions.

---

## 2. Public concepts

Use four names consistently.

| Term | Meaning |
|---|---|
| **regime** | Existing pylcm dynamic-programming regime: separate value function, transition target, solver, and state/action structure. |
| **case boundary** | A Boolean DAG node defining a threshold surface, such as `assets < medicaid_asset_limit`. |
| **piece** | An alternative formula for an existing DAG output under a predicate. |
| **segment** | Solver-internal smooth EGM object produced after lowering a piece combination. |

Avoid user-facing terms like “branch,” “branch schedule,” or “institutional regime.” They are too easy to confuse with pylcm regimes.

---

## 3. Minimal API

### 3.1 Declare a case boundary

```python
@lcm.case_boundary(
    ("assets", "medicaid_asset_limit"),
    ("countable_income", "medicaid_income_limit"),
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

The function is still an ordinary DAG node returning a Boolean array. The decorator says that the relevant equality surfaces are:

```text
assets == medicaid_asset_limit
countable_income == medicaid_income_limit
```

BQSEGM uses these boundaries to insert or check one-sided candidates and to avoid interpolating across discontinuities.

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
    ("assets", "medicaid_asset_limit"),
    ("countable_income", "medicaid_income_limit"),
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

It then solves EGM under each variant, masks inconsistent endogenous candidates, and upper-envelopes the surviving segments.

---

## 5. Supported in the first version

The first implementation should be deliberately small.

Supported:

- two-way Boolean predicates;
- `@lcm.piece(output, when=predicate)`;
- `@lcm.piece(output, otherwise=predicate)`;
- multiple outputs sharing one predicate;
- predicates depending on recovered current state or resources;
- predicates used as post-EGM consistency masks;
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
- hidden helper functions with `if`, `where`, `clip`, `maximum`, `searchsorted`, etc.;
- dynamic Python control flow;
- arbitrary lookup tables inside smooth pieces;
- piecewise formulas hidden inside lambdas or closures with unavailable source;
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

The public API should not expose `dags.tree` to model authors. Internally, BQSEGM should use DAG topology to lower decorated pieces.

Internal lowering:

```text
1. Build the ordinary model DAG.
2. Collect decorated case boundaries and pieces.
3. For each output with pieces, identify its alternative producers.
4. Build specialized smooth DAG variants by replacing selected producers.
5. Use DAG reachability to identify which functions are active in each variant.
6. Run EGM on each variant.
7. Evaluate predicates as consistency masks.
8. Upper-envelope the resulting segments.
```

This lets `dags.tree` answer:

```text
Which functions can produce "oop"?
Which predicate guards each producer?
Which downstream nodes depend on "oop"?
What specialized DAG is created if "oop" is replaced by "oop_medicaid"?
Which predicates must be checked after EGM recovers current assets?
```

---

## 10. Decorator metadata

The decorators should only attach metadata. They should not alter runtime behavior.

```python
@dataclass(frozen=True)
class CaseBoundaryMeta:
    boundaries: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PieceMeta:
    output: str
    predicate_name: str
    side: Literal["when", "otherwise"]


def case_boundary(*boundaries):
    def deco(fn):
        fn.__lcm_case_boundary__ = CaseBoundaryMeta(boundaries)
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

---

## 11. Validation goals

BQSEGM relies on formulas being smooth within each piece. Therefore, validation must guarantee:

```text
No hidden case logic exists in any smooth BQSEGM subgraph.
```

This requires two checks:

1. AST validation for Python-level branching.
2. JAXPR validation for hidden JAX piecewise primitives.

The ordinary fallback DAG may contain `jnp.where` or lookup logic if it is not reachable in the BQSEGM smooth variant.

---

## 12. AST validation

### 12.1 Smooth functions

In smooth pieces and smooth helpers, reject:

- `ast.If`;
- `ast.IfExp`;
- `ast.Match`;
- `ast.Compare`;
- `ast.BoolOp`;
- `ast.UnaryOp(Not)`;
- calls to common piecewise functions.

Forbidden function calls in smooth pieces:

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

### 12.2 Case-boundary functions

In `@lcm.case_boundary` functions, allow vectorized comparisons and Boolean logic, such as:

```python
return (assets < limit) & (income < income_limit)
```

Still reject Python `if`, `match`, and non-JAX-traceable dynamic control flow.

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
        self.mode = mode  # "smooth" or "boundary"
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
        if self.mode == "smooth":
            self.error(
                node,
                "Comparison in a smooth formula creates a hidden case boundary.",
            )
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        if self.mode == "smooth":
            self.error(
                node,
                "Boolean logic in a smooth formula creates a hidden case boundary.",
            )
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if self.mode == "smooth" and isinstance(node.op, ast.Not):
            self.error(node, "not in a smooth formula creates a hidden case boundary.")
        self.generic_visit(node)

    def visit_Call(self, node):
        name = _call_name(node.func)
        leaf = name.split(".")[-1] if name else None

        if self.mode == "smooth" and leaf in PIECEWISE_CALL_NAMES:
            self.error(
                node,
                f"Call to piecewise function `{name}` is not allowed "
                "inside a smooth BQSEGM piece.",
            )

        self.generic_visit(node)


def check_ast(fn, *, mode):
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    checker = PiecewiseASTChecker(mode=mode, fn_name=fn.__name__)
    checker.visit(tree)
    return checker.errors
```

---

## 13. JAXPR validation

AST validation does not catch piecewise logic hidden in helper functions. JAXPR validation does.

Forbidden smooth primitives:

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


def check_jaxpr(fn, abstract_args, *, mode):
    jaxpr = jax.make_jaxpr(fn)(*abstract_args)
    errors = []

    for eqn in jaxpr.jaxpr.eqns:
        prim = eqn.primitive.name

        if mode == "smooth" and prim in SMOOTH_FORBIDDEN_PRIMS:
            errors.append(
                {
                    "function": fn.__name__,
                    "primitive": prim,
                    "message": (
                        f"JAX primitive `{prim}` indicates hidden piecewise logic "
                        "inside a smooth BQSEGM formula."
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

---

## 14. Reachability-aware validation

Do not validate every function in the model. Validate every function reachable in a BQSEGM smooth variant.

Algorithm:

```text
for each BQSEGM regime:
    build ordinary DAG
    collect piece outputs

    for each feasible piece combination:
        clone producer map
        replace output producers with selected pieces
        find functions reachable from solver-required outputs

        for each reachable function:
            if case_boundary:
                run boundary AST and JAXPR checks
            else:
                run smooth AST and JAXPR checks
```

If the fallback function `oop` contains `jnp.where`, but BQSEGM replaces it with `oop_medicaid` or `oop_private`, it is not reachable and should not fail validation.

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

Every `case_boundary` should declare at least one boundary surface.

Valid:

```python
@lcm.case_boundary(("assets", "asset_limit"))
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
- boundary type is recorded in future versions: continuous kink, discontinuous jump, or hard constraint.

---

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

Some low-level helpers may look piecewise but be mathematically smooth under the model domain. The first version should not support bypasses. A later version could allow:

```python
@lcm.smooth_helper
def numerically_stable_log1pexp(x):
    ...
```

with explicit review.

---

## 19. Implementation sequence

1. Implement metadata-only decorators: `case_boundary` and `piece`.
2. Add metadata collection in the model-building path.
3. Implement reachability-aware AST validation.
4. Implement reachability-aware JAXPR validation.
5. Implement two-piece DAG lowering for one output and one predicate.
6. Implement post-EGM predicate masking.
7. Implement boundary metadata checks.
8. Add a minimal Medicaid-style toy model.
9. Compare BQSEGM against brute force on the toy.
10. Generalize to multiple outputs sharing the same predicate.
11. Only then consider lookup-table decorators.

---

## 20. Minimal test plan

### Unit tests

- decorator metadata is attached correctly;
- exactly one of `when=` or `otherwise=` is required;
- missing otherwise piece fails;
- missing boundary metadata fails;
- smooth AST rejects Python `if`;
- smooth AST rejects `jnp.where`;
- boundary AST permits vectorized comparisons;
- JAXPR catches hidden helper `jnp.where`;
- fallback functions are ignored if unreachable.

### Solver tests

- one-predicate Medicaid toy;
- predicate depending on recovered current assets;
- boundary candidate at asset limit;
- invalid branch candidates are masked;
- upper envelope selects correct side;
- brute force and BQSEGM agree on a dense-grid toy;
- BQSEGM beats asset-row mode on memory for the same toy.

### Negative tests

- hidden `clip` inside a smooth piece;
- hidden `searchsorted` inside a smooth subsidy formula;
- incomplete true/false coverage;
- multiple pieces for the same predicate side;
- case boundary returning non-Boolean output.

---

## 21. Short summary

BQSEGM should expose institutional piecewise structure through decorators on ordinary DAG functions:

```python
@lcm.case_boundary(("x", "x_bar"))
def below_bar(x, x_bar):
    return x < x_bar

@lcm.piece("resources", when=below_bar)
def resources_low(...):
    ...

@lcm.piece("resources", otherwise=below_bar)
def resources_high(...):
    ...
```

The user-facing interface stays close to pylcm today. Internally, pylcm uses DAG topology to lower pieces into smooth EGM segments, validates that no hidden branching remains, masks inconsistent endogenous candidates, and upper-envelopes the result.

The first version should support only binary predicates and formula pieces. Multiway tables, binding cases, and trusted smooth-helper bypasses can come later.
