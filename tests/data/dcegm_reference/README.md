# DC-EGM cross-check fixtures

`ijrs_taste_shocks_reference.csv` holds the solution of the Iskhakov et al. (2017)
consumption-retirement model with EV1 taste shocks (scale 0.2) and deterministic income,
computed by the independent [`dcegm`](https://github.com/OpenSourceEconomics/dcegm)
package. It pins pylcm's logsum / probability-weighted-marginal-utility pipeline against
an implementation that shares no code with pylcm.

Consumed by `tests/test_dcegm_fixture_crosscheck.py`, which builds the pylcm twin of the
model (`tests/test_models/dcegm_paper_twin.py`) and compares value functions at the
fixture's wealth points.

## Regeneration

`dcegm` is intentionally not a pylcm dependency. To regenerate, use a throwaway venv:

```bash
uv venv /tmp/dcegm-fixtures-venv --python 3.12
uv pip install --python /tmp/dcegm-fixtures-venv/bin/python \
    "git+https://github.com/OpenSourceEconomics/dcegm.git@7d7af991f0ca" pandas
cd tests/data/dcegm_reference
/tmp/dcegm-fixtures-venv/bin/python generate_fixtures.py
```

Generated with `dcegm` at upstream commit `7d7af991f0ca` (pre-release; the interfaces
used are `setup_model`, `model.solve`, and the
`choice_{values,policies,probabilities}_for_states` accessors). If upstream has moved
and the interfaces changed, pin the commit above rather than chasing the API.

The model specification (functional forms, parameter values, choice coding — including
the upstream docstring/code discrepancy on which choice code means "work") is documented
in `generate_fixtures.py`'s module docstring.

## Excluded rows: low-wealth resolution of the uniform savings grid

The retiree rows at the lowest wealth node (`lagged_choice = 1`, `wealth = 1.0`; 9 rows,
one per period) are excluded from the cross-check. The run's exogenous savings grid is
uniform on `[0, 50]`, which under-resolves the sharply curved (CRRA, `rho = 1.95`)
retiree value function near the borrowing limit: the published value is interpolated
linearly between endogenous grid points whose spacing the savings grid dictates, and the
resulting error compounds backward across periods. Both implementations inherit the
problem from the shared grid, but the error is implementation-specific in size — at
period 0 the fixture stores about −66.0 and pylcm's DC-EGM about −76, while an
independent fine-grid value-iteration recursion of the model (consistent with the
fixture's own `policy_retire` column, which implies about −54) and pylcm's brute-force
twin both put the value near −54. A pylcm DC-EGM run with the same number of savings
nodes clustered toward the borrowing limit also reproduces the recursion at those nodes
(locked in by `test_clustered_savings_grid_resolves_excluded_low_wealth_nodes`), so the
exclusion reflects the fixture run's grid choice, not a kernel defect.

On the remaining rows, agreement between the two DC-EGM implementations certifies that
the implementations match on a shared discretization; the brute-force comparison and the
recursion anchor accuracy.
