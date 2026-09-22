## Testing

### Test-Driven Development — always

**Always write the test first, watch it fail, then implement.** No exceptions for new
behavior or bug fixes. Tests are not an afterthought, they are the spec.

The cycle:

1. **Red.** Write a failing test that asserts the desired behavior in user-facing terms.
   Run it. Confirm it fails for the *right* reason (the missing behavior — not a typo,
   not an import error).
1. **Green.** Write the smallest amount of code that makes the test pass.
1. **Refactor.** Clean up while keeping the test green.

Apply per case:

- **New feature** → red-green-refactor.
- **Bug fix** → reproduce as a failing test before writing the fix. The test then
  prevents regression.
- **Refactor (no behavior change)** → existing tests are the spec. Keep them green
  before, during, and after. No new test needed if behavior is unchanged; if you find a
  behavior gap, fill it with a new test *before* refactoring.

### Test docstrings — describe behavior, not history

Test docstrings state what *should* be true, in user-facing terms. Pretend the reader
has never seen the PR. They should not need to.

```python
# Good — behavior, in plain language
def test_simulate_with_chained_transitions_yields_expected_next_wealth():
    """`next_wealth_t = wealth_t - c_t + 0.1 * next_aime_t` holds in simulation."""


# Bad — rehearses the prior bug or implementation history
def test_solve_resolves_chain_via_dags():
    """Before the fix, `_resolve_fixed_params` raised
    `InvalidParamsError: Missing required parameter: ...` because
    `create_regime_params_template` classified ..."""
```

Rule of thumb: **would the docstring still make sense in 9 months without the PR
context?** If not, rewrite it.

### When a certificate goes red

Editing a file under `src/` can turn a `tests/test_*_certificate.py` battery red for a
reason unrelated to whether the edit is correct: the candidate certificate pins every
certified source by byte digest, and pins the callables and transport surfaces its nine
proved corridors depend on by AST digest. A moved callable stales a pin whether or not
it changed behavior.

Do not treat that as a test to relax. Re-anchor it, or find out why it moved:
[Certification and preflight](../docs/development/certification.md) gives the exit-code
decision table, the re-anchor procedure, and the rule that governs both — never refresh
a seal merely to make a changed route green.

Two related obligations: a new test file must be registered in
`tests/ci/ci-workloads.json`, and several test files are pinned by path and classname by
`cpu.yml` and the timing-report checkers, so moving one needs the checklist on that page.
See [build-and-test.md](build-and-test.md).

### Concrete-value assertions

Assert *what* the result is, not just that it didn't crash.

```python
# Good — analytical value with explicit tolerance
np.testing.assert_allclose(curr["wealth"], expected_next_wealth, atol=1e-6)

# Bad — passes whether the math is right or not
assert not jnp.any(jnp.isnan(V_arr))
assert df["wealth"].notna().all()
```

`not isnan` and `no exception raised` belong in CI smoke tests, not in the unit tests
for the feature itself.

### Precision-aware tolerances — take them from the policy, never hardcode

The suite runs at both precisions: `pytest --precision=64` (the default) and
`pytest --precision=32`. `tests/conftest.py` sets `jax_enable_x64` accordingly and
publishes the matching tolerance as `DECIMAL_PRECISION` — **12** at float64, **5** at
float32. Import it; do not write a numeric tolerance that only one precision can meet.

```python
# Good — one assertion, valid at both precisions
from numpy.testing import assert_array_almost_equal as aaae

from tests.conftest import DECIMAL_PRECISION

aaae(got, expected, decimal=DECIMAL_PRECISION)


# Bad — float32 cannot represent this at all
np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
```

A hardcoded `1e-12` is roughly five orders of magnitude below float32 machine epsilon
(`~1.19e-7`), so such a test fails at `--precision=32` for every implementation, correct
or not — it reports the format, not the code. Since `jax_enable_x64` is **False** by
default outside the suite, float32 is what users actually run, so a test that cannot
pass there leaves the default configuration uncovered.

Choose the instrument by what is being asserted:

- **A reported quantity** (a value, a policy, a moment) carries rounding, so it takes a
  tolerance — and the tolerance belongs to the precision, i.e. `DECIMAL_PRECISION`.
- **A structural predicate** — which branch owns a state, whether a constraint binds,
  which candidate is the argmax — is discrete, and no tolerance is the right instrument.
  A tie broken the wrong way moves the answer by a finite amount, not by an ULP, so a
  tolerance wide enough to absorb it is also wide enough to hide a real defect. Assert
  the decision itself, and let the arithmetic that makes it fail loudly when it cannot
  decide. See `.ai-instructions/modules/math.md`, "Floating-point decisions".
- **Absolute vs relative** matters once magnitudes vary: `assert_array_almost_equal` is
  an *absolute* check, so on large-valued arrays prefer `assert_allclose` with an `rtol`
  derived from the same policy.

A reported quantity also has an **error order**, which one number per precision cannot
express. `DECIMAL_PRECISION` is an *epsilon-order* instrument — 12 and 5 decimals are
the two machine epsilons with a couple of decades of headroom — and so is
`assert_agrees_to_ulp`. Not every continuous quantity is epsilon order, and importing
`DECIMAL_PRECISION` for one that is not produces a test that is red for no defect:

- **Epsilon** — a residual, or a closed form evaluated in the working format.
  `DECIMAL_PRECISION`, or `assert_agrees_to_ulp` where the comparison is relative.
- **Square root of epsilon** — the *abscissa* of a smooth interior maximum, and anything
  proportional to it: an implicit tangent, a marginal evaluated at the selected action,
  a finite-difference gate such a marginal drives. The objective is flat there to second
  order, so the location is determined only to `sqrt(eps)` — 1.5e-8 at float64, 3.4e-4
  at float32. `DECIMAL_PRECISION` is four decades too tight for it at float64 and ~1.4
  at float32, so **this is not a float32-only gap**.
- **Cube root of epsilon** — the *step* of a central finite difference, balancing `h**2`
  truncation against `eps/h` cancellation. A fixed `1e-6` is the float64 optimum and
  sits three decades below the float32 one, where cancellation makes the reference
  difference quotient itself wrong by ~4e-2. Derive the step, not only the tolerance.

The order is worth measuring rather than inferring: in
`collapse_continuous_candidate_bank` the collapsed value is epsilon order (1 ULP at
float64, ~13 at float32) while its own marginal is square-root order (1.1e-8 and 1.3e-3)
— one call, two orders, so one shared `atol` cannot be right for both.

Two checks before rescaling any existing constant. A loose float64 constant is **not**
an epsilon-calibrated bound: `rtol=1e-9` is ~4.5e6 eps, and multiplying it by the
epsilon ratio yields `rtol=0.5`. Write `max(<old constant>, <epsilon term>)` so the
float64 leg is provably unchanged, and flag in the source any site where it is not. And
some constants are not precision bounds at all — the `< 1e-4` envelope-consistency gate
in `tests/egm/test_outer_carry.py` is a *truncation* bound, `dm**2/6` at `dm = 0.01`,
with rounding contributing ~1e-14. Establish what a constant *is* before rescaling it.

Which invariances a structural predicate has to satisfy is not uniform, so assert only
the ones that carry numerical content:

- **Across batch size — required, but not bit for bit.** A `batch_size` or block-size
  knob partitions a computation whose *result* does not depend on the partition: no
  operation and no operand order changes. What does change is the vmap width each block
  is compiled for, and XLA emits a differently vectorized kernel per width — so two
  partitions can land on adjacent representable neighbours of the same real number.
  (Turning the backend optimizer off collapses every width onto one bit pattern, which
  is what identifies the effect as code generation rather than arithmetic.) Assert
  accordingly: structural properties — which discrete choice is taken, which nodes carry
  a feasible action — exactly; published values with
  `tests.conftest.assert_agrees_to_ulp`, which bounds the gap in units of the working
  format's spacing and so says the same thing at either precision. The defect this
  guards against is a partition-dependent *reduction* — a sum or max evaluated over a
  block instead of the whole axis, or padding cells left in it — which moves a value by
  orders of magnitude more than a few ULP.
- **Across precision — not required.** Two candidates separated at float64 can be
  indistinguishable at float32, and the honest float32 answer is then a *deterministic*
  tie-break, not agreement with float64. Requiring the two to match would forbid the
  ordinary working pattern of a coarse float32 first pass followed by a float64 polish.
  What float32 owes is that repeated runs agree with each other, and that the value it
  publishes is within its own resolution of the optimum — not that it picks the same
  leg.
- **Across device — out of scope.** Reduction order and library kernels vary, and a
  whole program has many places for that to surface. Do not write cross-device equality
  tests.

### Mechanics

- Use plain pytest functions, never test classes (`class TestFoo`)
- Use `@pytest.mark.parametrize` for test variations
