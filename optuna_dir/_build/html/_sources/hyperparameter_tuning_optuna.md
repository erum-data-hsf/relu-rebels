# Hyperparameter Tuning with Optuna

Optuna is a great framework for hyperparameter optimization---it's
efficient, flexible, and integrates well with deep learning libraries
like PyTorch, TensorFlow, and Keras, as well as scikit-learn.

------------------------------------------------------------------------

## 🔹 1. Install Optuna

``` bash
pip install optuna
```

------------------------------------------------------------------------

## 🔹 2. Define an Objective Function

The **objective function** tells Optuna what hyperparameters to sample
and how to evaluate them.

Example with a simple scikit-learn classifier:

``` python
import optuna
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    # Define hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    
    # Model with suggested params
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc
```
<!-- 
------------------------------------------------------------------------

## 🔹 3. Run the Optimization

``` python
study = optuna.create_study(direction="maximize")  # "minimize" for loss
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print("  Value:", trial.value)
print("  Params:", trial.params)
```

------------------------------------------------------------------------

## 🔹 4. Example with Deep Learning (PyTorch)

You can use Optuna to tune learning rate, hidden layers, dropout, etc.

``` python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def objective(trial):
    # Suggest hyperparams
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    n_hidden = trial.suggest_int("n_hidden", 16, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Define simple model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(4, n_hidden)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(n_hidden, 3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)
    
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training loop
    for epoch in range(10):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32))
        acc = (preds.argmax(dim=1).numpy() == y_test).mean()
    return acc
```

------------------------------------------------------------------------

## 🔹 5. Visualize Results

Optuna has built-in visualization:

``` python
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

------------------------------------------------------------------------

✅ **Key Tips:** - Use `suggest_int`, `suggest_float`,
`suggest_categorical`, `suggest_loguniform` depending on the parameter
type. - For deep learning, you can prune unpromising trials early using
`optuna.integration.PyTorchLightningPruningCallback` or similar. - Set a
proper number of trials (start small like `n_trials=20`, scale up
later). - Combine with distributed/parallel execution for faster tuning. -->
