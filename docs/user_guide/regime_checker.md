---
title: Check a regime graph
---

# Check a regime graph

The regime checker looks for ways to simplify the age- or period-indexed regime graph of
a constructed `Model`. It uses the model's declarations and original grid sizes to
propose changes such as removing unused state variables or combining compatible regimes.
Model construction is sufficient. The checker does not solve or simulate the model.

## Try the bundled example

From the root of a PyLCM checkout, run the following commands on Linux with Pixi.

```sh
cd tools/regime-finder
pixi run -e default python -m regime_finder.pipeline \
    examples/declarations/tiny-huge-grid.json --output regime-check-results
```

Open `regime-check-results/next-actions.html` for the recommendation and
`regime-check-results/report.html` for the graph, alternatives and supporting checks.

## Check your model

Construct your `Model` using your usual setup. Add `tools/regime-finder` from the
checkout to `PYTHONPATH` if your setup runs outside that directory. Run the following
code after construction, in the same Pixi environment as your model.

```python
from pathlib import Path

from regime_finder.pytask_api import export_model

export_model(
    model,
    Path("model-snapshot.json"),
    production_metadata={"recipe": "my production model setup"},
)
```

From `tools/regime-finder`, rerun the example command with the path to your snapshot and
a fresh output directory. Keep the snapshot and its companion files together. Review
them before sharing because they can contain source and model inputs.

For an in-memory Python check, capture once with `read_model(model)` from
`regime_finder.model_snapshot`. Pass the same snapshot to `find_regimes(snapshot)` and
`verify_production_result(snapshot, report)` from `regime_finder`.

The default search independently checks its proposed changes against the captured
declarations. It can leave a proposal unresolved if it cannot establish a required
dependency or transition property. The search does not establish a global optimum. Its
work and storage estimates describe the graph and grid dimensions.
