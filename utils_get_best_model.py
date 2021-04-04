import pandas as pd
import numpy as np
import functools as ft
import operator
import itertools
from collections import namedtuple
from itertools import product
from model_keeper import ModelKeeper
from sklearn.model_selection import GridSearchCV

act_params = namedtuple("act", ["neirons", "regularizer", "dropout", "activation"])
d_act_params = namedtuple("d_act", ["relu", "sigmoid", "softmax", "dropout", "regularizer"])

def named_product(**items):
    return [act_params._make(list(prod)) for prod in product(*items.values())]


def get_act_by_layer(layer, d_act):
    drops, regularizer = [None], [None]
    if "dropout" in layer:
        drops = list(np.linspace(d_act.dropout[0],
                                  d_act.dropout[1],
                                  num=d_act.dropout[2]))
    if "regularizer" in layer:
        regularizer = list(np.linspace(d_act.regularizer[0],
                                         d_act.regularizer[1],
                                         num=d_act.regularizer[2]))
    neirons = list(np.linspace(getattr(d_act, layer[0])[0],
                               getattr(d_act, layer[0])[1],
                               num=getattr(d_act, layer[0])[2]))

    return named_product(neirons=neirons, regularizer=regularizer, dropout=drops, activation=[layer[0]])


def get_d_act(parameters):
    d_act = d_act_params
    d_act.relu = [parameters.neirons[0], parameters.neirons[1],
                  parameters.neirons[2]]
    d_act.sigmoid, d_act.softmax = d_act.relu, d_act.relu
    d_act.dropout = [parameters.dropout[0], parameters.dropout[1],
                     parameters.dropout[2]]
    d_act.regularizer = [parameters.reg_params.l2[0],
                         parameters.reg_params.l2[1],
                         parameters.reg_params.l2[2]]
    return d_act


def get_network(data, parameters, num_layers, layers=None, save=None):
    """Return best generated network by parameters"""
    d_act = get_d_act(parameters)

    if layers is None:
        layers = [["relu", "dropout"], ["relu", "regularizer"], ["relu"], ["sigmoid"]]
    # If some layers haven't None or regularizer
    layers = [layer if len(layer) > 1 else layer + [None] for layer in layers]

    best_networks = []
    for layer in layers:
        acts = get_act_by_layer(layer, d_act)
        if not isinstance(acts, list):
            acts = [acts]
        for act in acts:
            param_grid = {'act': [act]}
            grid_search = GridSearchCV(ModelKeeper(), param_grid)
            grid_search.fit(data.train, data.train_labels, epochs=200)
            flatten = itertools.chain.from_iterable
            grid_search.score(list(ft.reduce(operator.iconcat, data.test, [])), list(ft.reduce(operator.iconcat, data.test_labels, [])))

            all_results = grid_search.cv_results_

            acts = [([act["act"]], layer) for act in all_results["params"]]
            results = all_results["mean_test_score"]

            best_networks += sorted([(results[i], acts[i]) for i in range(len(results))])

    best_networks_by_iterations = []
    best_networks_by_iterations += best_networks
    submission = pd.DataFrame({'best_networks_by_iterations': best_networks_by_iterations})
    submission.to_csv('best_networks_by_iterations.csv', index=True)
    # Look over best_networks num_layers - 1 times
    for _ in range(num_layers - 1):
        k = len(best_networks)
        for i in range(k):
            network = best_networks[i]
            act_second = get_act_by_layer(network[1][1], d_act)

            param_grid = {'act': act_second}
            grid_search = GridSearchCV(ModelKeeper(network[1][0]), param_grid)
            grid_search.fit(data.train, data.train_labels, epochs=200)
            grid_search.score(list(ft.reduce(operator.iconcat, data.test, [])), list(ft.reduce(operator.iconcat, data.test_labels, [])))

            all_results = grid_search.cv_results_

            acts = [(network[1][0] + [act["act"]], network[1][1]) for act in all_results["params"]]
            results = all_results["mean_test_score"]

            best_networks += sorted([(results[i], acts[i]) for i in range(len(results))], reverse=True)[:2]

        best_networks = sorted(best_networks, reverse=True)[:len(best_networks) // 3 + 1]
        best_networks_by_iterations += best_networks

        if save:
            # Save result by iteration
            best_networks_by_iterations += best_networks
            submission = pd.DataFrame({'best_networks_by_iterations': best_networks_by_iterations})
            submission.to_csv('best_networks_by_iterations.csv', index=True)

    return best_networks[:3], best_networks_by_iterations

