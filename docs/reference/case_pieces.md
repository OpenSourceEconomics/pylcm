---
title: Case pieces
---

# Case pieces

Case pieces declare a supported binary split of one economic output into two smooth
formulas. The decorators attach metadata and return the original functions unchanged, so
the same regime remains executable under `GridSearch`.

## Boundary predicate

```python
import lcm

eligible = lcm.case_boundary(
    lcm.ref("liquid") < lcm.ref("asset_limit"),
    kind="jump",
)
```

The declaration records:

- both named operands from the `lcm.ref(...)` comparison;
- exact-boundary ownership from the comparison operator (`<` above leaves equality to
  the `otherwise` side, while `<=` gives it to `when`);
- `kind`: `"continuous_kink"`, `"jump"`, or `"hard_constraint"`.

The condition must be one `<`, `<=`, `>`, or `>=` comparison. Compound conditions and
bare callable predicates cannot declare a case-piece boundary because they do not
identify one ordered split with unambiguous ownership.

## Piece formulas

```python
@lcm.piece(output="subsidy", when=eligible)
def subsidy_when_eligible(subsidy_high):
    return subsidy_high


@lcm.piece(output="subsidy", otherwise=eligible)
def subsidy_otherwise(subsidy_low):
    return subsidy_low
```

Exactly one `when` and one `otherwise` formula define the split output. A piece should
be smooth within its declared side. `@lcm.smooth_helper` can attest a reviewed numerical
helper whose implementation contains a benign `clip`, `maximum`, or `abs` that the
automatic smoothness gate would otherwise reject.

The current NBEGM case-piece route is deliberately narrow: one supported binary split of
the additive subsidy/cash-on-hand structure on the declared liquid state, with flat
piece parameters and no taste shocks. Model construction refuses declarations outside
that route instead of silently applying a heuristic.

Use [piecewise-affine schedules](piecewise_affine.md) for several brackets, floors, or
mixed breakpoint kinds. Read
[Declared non-convex budgets](../methods/nonconvex_budgets.md) for the numerical reason
the metadata exists.
