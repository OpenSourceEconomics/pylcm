---
title: Check a regime graph
---

# Check a regime graph

The regime checker looks for ways to simplify the age- or period-indexed regime graph of
a constructed `Model`. It uses the model's declarations and original grid sizes to
propose changes such as removing unused state variables or combining compatible regimes.
Model construction is sufficient. The checker does not solve or simulate the model.

## Check your model

After installing PyLCM, construct your `Model` using your usual setup. Call
`find_regimes` in the same Python environment.

```python
from pathlib import Path

from lcm.graphs import find_regimes

report = find_regimes(model, output=Path("regime-check-results"))
print(report["recommendations"])
```

Open `regime-check-results/report.html` for the graph, alternatives and supporting
checks. Use a new output directory, or omit `output` to get the report in memory. Review
saved reports before sharing them because they can contain source and model inputs.

The default search independently checks its proposed changes against the captured
declarations. It can leave a proposal unresolved if it cannot establish a required
dependency or transition property. The search does not establish a global optimum. Its
work and storage estimates describe the graph and grid dimensions.

## Replay the bundled snapshot

From the root of a PyLCM checkout on Linux, run the following with Pixi.

```sh
pixi run -e default python -m _lcm_regime_finder.pipeline \
    tools/regime-finder/examples/declarations/tiny-huge-grid.json \
    --output regime-snapshot-results
```

Open `regime-snapshot-results/next-actions.html` for the recommendation.
