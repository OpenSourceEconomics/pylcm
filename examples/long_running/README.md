# Example model specification

## Example Stats

| Example name                        | Description                                       | Runtime       |
| ----------------------------------- | ------------------------------------------------- | ------------- |
| [`long_running`](./long_running.py) | Consumption-savings model with health and leisure | a few minutes |

## Running the example

If you want to solve the `long_running` example locally. First, clone this repository,
[install pixi if required](https://pixi.sh/latest/#installation), move into the examples
folder, and open the interactive Python shell. In a console, type:

```console
$ git clone https://github.com/opensourceeconomics/pylcm.git
$ cd lcm/examples
$ pixi run ipython
```

In that shell, run the following code:

```python
from lcm.entry_point import get_lcm_function

from long_running import LONG_RUNNING_MODEL, PARAMS


V_arr_list = LONG_RUNNING_MODEL.solve(params=PARAMS)
```
