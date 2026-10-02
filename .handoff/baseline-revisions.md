# Starting revisions (plan §0)

| Repository | Revision | Role |
|---|---|---|
| pylcm | 2e2392eb1eeca117cb09f808d6ff575f5db77a88 | #482 head the plan inspected |
| pylcm | 4674c0386e311c16c75c3ed66bc10ca082334b35 | #482 successor head at branch time (2026-10-01); branch base |
| pylcm | 52407b79f5c25f93f6b80c33f60f7c0e2dd45bc8 | #482 base (main); secondary historical control |
| aca-model | ad38653696ec366e318ac61b9a81b597a4ecb700 | ACA revision pinned by #482's benchmark environment |

Intervening change 2e2392eb..4674c038, reviewed: one commit, "Select the async allocator through XLA_PYTHON_CLIENT_ALLOCATOR". It touches `_fail_if_the_pool_grows_in_regions` and `visible_device_pool_limits` in `execution_plan.py` (allocator environment check; a pool limit of 0 counts as no limit), plus tests and docs. None of the plan's surfaces are touched (processing/fixed components, core_program, value_transfer, placement, backward_induction, grid_search/max_Q_over_a, stores, simulation). No baseline evidence existed yet to regenerate.
