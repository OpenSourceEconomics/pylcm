# Example model specification

## Example Stats

| Example name                              | Description                                                                                                                                                                         | Runtime                      |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| [`Mahler_Yum_2024`](./Mahler_Yum_2024.py) | Replication of the lifecycle model from the paper 'Lifestyle Behaviors and Wealth-Health Gaps in Germany' by Lukas Mahler and Minchul Yum (2024, https://doi.org/10.3982/ECTA20603) | CPU: not working, GPU: \<40s |

## Running the example

First set up the pixi environment. Make sure you load the `test-gpu` environment, as the
model will have trouble running on the CPU. Then run `python run-model.py`.
