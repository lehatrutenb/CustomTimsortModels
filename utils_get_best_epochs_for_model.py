import operator
import numpy as np
import functools as ft
from model_keeper import ModelKeeper
from sklearn.model_selection import GridSearchCV


def get_best_epochs(model, data, parameters, kol):
    """Train model and return best by err"""
    left, right = parameters.epochs[0], parameters.epochs[1]

    param_grid = {'epochs': list(np.linspace(left, right, num=kol, dtype=int))}
    grid_search = GridSearchCV(ModelKeeper(model), param_grid)
    grid_search.fit(data.train, data.train_labels)
    grid_search.score(list(ft.reduce(operator.iconcat, data.test, [])),
                      list(ft.reduce(operator.iconcat, data.test_labels, [])))

    all_results = grid_search.cv_results_
    print(all_results)

    epochs = [act["epochs"] for act in all_results["params"]]
    results = all_results["mean_test_score"]
    best_epochs_and_score = sorted(
        [(results[i], epochs[i]) for i in range(len(results))])

    return best_epochs_and_score[0]

