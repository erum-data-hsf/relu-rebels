---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
--- 



# Quadratic Function Example

Usually, Optuna is used to optimize hyperparameters, but as an example, let’s optimize a simple quadratic function $(x-2)^2$

First of all, import `optuna`.

```{code-cell}
import optuna
```

In optuna, conventionally functions to be optimized are named objective.

```{code-cell}
def objective(trial):
    x = trial.suggest_float("x", 0, 4)
    return (x - 2) ** 2
```

This function returns the value of $(x-2)^2$. Our goal is to find the value of `x` that minimizes the output of the `objective` function. 
This is the “optimization.” During the optimization, Optuna repeatedly calls and evaluates the objective function with different values of `x`.

The `suggest` APIs (for example, `suggest_float()`) are called inside the objective function to obtain parameters for a trial. `suggest_float()`
selects parameters uniformly within the range provided. In our example, from $0$ to $4$.

To start the optimization, we create a study object and pass the objective function to method `optimize()` as follows.

```{code-cell}
study = optuna.create_study()
study.optimize(objective, n_trials=3)
```

You can get the best parameter as follows.

```{code-cell}
best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))
```


We can see that the `x` value found by Optuna is close to the optimal value of `2`.


