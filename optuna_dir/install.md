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
Optuna is a great framework for hyperparameter optimization---it's
efficient, flexible, and integrates well with deep learning libraries
like PyTorch, TensorFlow, and Keras, as well as scikit-learn.

------------------------------------------------------------------------

## 🔹 What is Optuna?
Optuna is an **automatic hyperparameter optimization framework**. It allows you to define a search space for hyperparameters and then uses efficient algorithms to find the best-performing set of parameters.

Optuna works by asking two important questions during the optimization process:\
1. **Search Space Definition:** 
What range or type of hyperparameters should we try?\
- Example: learning rate (float between 1e-5 and 1e-1), number of layers
(integer 1--5), optimizer type (categorical: Adam, SGD, RMSprop).\
2. **Optimization Target:** 
What metric do we want to maximize or minimize?\
- Example: accuracy (maximize), loss (minimize), F1 score (maximize).

# Installation of Optuna 
Optuna supports Python 3.8 or newer.

We recommend to install Optuna via pip:

```
$ pip install optuna
```
