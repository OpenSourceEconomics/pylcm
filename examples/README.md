# Example model specifications

## Example Model Stats

| Example name                              | Description                                                                                                                                                                         | Runtime                      |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| [`long_running`](./long_running.py)       | Simple consumption-savings model with health and leisure                                                                                                                            | CPU: 45s, GPU: 4s            |
| [`Mahler_Yum_2024`](./Mahler_Yum_2024.py) | Replication of the lifecycle model from the paper 'Lifestyle Behaviors and Wealth-Health Gaps in Germany' by Lukas Mahler and Minchul Yum (2024, https://doi.org/10.3982/ECTA20603) | CPU: not working, GPU: \<40s |

## Running an example model

If you want to solve and simulate the example locally. First, clone this repository,
[install pixi if required](https://pixi.sh/latest/#installation), move into the examples
folder and type:

```console
$ git clone https://github.com/opensourceeconomics/pylcm.git
$ cd lcm/example/#Name of Example#
$ pixi run python simulate_model.py
```

If you want to run a model on the GPU you need to use the pixi 'cuda' environment. Run
`pixi run -e cuda python simulate_model.py` instead.
