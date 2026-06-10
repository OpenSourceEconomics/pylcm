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

## Known upstream artifact

The retiree rows at the lowest wealth node (`lagged_choice = 1`, `wealth = 1.0`) carry
an upstream numerical artifact: the stored `value_retire` contradicts the value implied
by the fixture's own `policy_retire` column (period 0: stored value −66.0, while
consuming the stored policy `0.124 · wealth` per period is worth about −54), and an
independent fine-grid value-iteration recursion of the same model confirms the
policy-implied value. The cross-check test therefore excludes those rows; all other rows
agree with the recursion at the test's tolerances.
