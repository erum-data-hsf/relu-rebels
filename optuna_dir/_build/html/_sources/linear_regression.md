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

# Linear Regression with scikit-learn

This chapter teaches linear regression using `scikit-learn` with a dataset from 
[Deep Learning from HEP](https://hsf-training.github.io/deep-learning-intro-for-hep/).

Download the dataset from [here](https://github.com/hsf-training/deep-learning-intro-for-hep/blob/main/deep-learning-intro-for-hep/data/penguins.csv) 
and place it in a `data/` folder.

```{admonition} Objectives
- Have fun with penguins!
```

The dataset contains the basic measurements on 3 species of penguins!   

![A penguin](https://hsf-training.github.io/deep-learning-intro-for-hep/_images/culmen_depth.png)

```{warning}
Penguins can be very cute.
```


```{code-cell}
print(2 + 2)
```

```{code-cell} ipython3
import pandas as pd
penguins_df = pd.read_csv("data/penguins.csv")
penguins_df
```