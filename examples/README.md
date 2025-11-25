# Example model specifications

## Example Model Stats

| Example name                                    | Description                                                                                                                                                    | Runtime                 |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| [`consumption_saving`](./consumption_saving.py) | Simple illustrative consumption-savings model with health and leisure                                                                                          | CPU: ~20s, GPU: \<5s    |
| [`mahler_yum_2024`](./mahler_yum_2024.py)       | Replication of the lifecycle model from the paper "Lifestyle Behaviors and Wealth-Health Gaps in Germany" by Lukas Mahler and Minchul Yum (Econometrica, 2024) | CPU: ~20min, GPU: \<40s |

> [!NOTE]
> Runtime refers to a single simulation call for 1,000 individuals. CPU times come from
> a MacBook Pro with M4 Pro chip with 24GB RAM.

## Running an example model

If you want to solve and simulate an example locally,
[install pixi](https://pixi.sh/latest/#installation) and follow these steps:

```console
$ git clone https://github.com/opensourceeconomics/pylcm.git
$ cd pylcm/examples/#example name#
$ pixi run python model.py
```

If you want to run a model on the GPU you need to use the pixi 'cuda' environment, run

```console
$ pixi run -e cuda python model.py
```
