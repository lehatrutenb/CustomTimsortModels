"""Module that gets best networks using model_keeper and utils"""
import os
import copy
import time
import keras
import random
import logging
import json as js
import pandas as pd
import matplotlib.pyplot as plt
from colour import Color

from model_keeper import *
from customtimsort import timsort
from utils_get_best_model import get_network
from utils_get_best_model import act_params as act
from Timsort_lib_num import TimSort, get_minrun
from utils_get_best_epochs_for_model import get_best_epochs
from utils_get_and_parse_data import normalize_data, renormalize_data
from utils_get_and_parse_data import get_parsed_data, config_parameters

predict_params = namedtuple("predict", ["size", "minrun", "time"])
plots_params = namedtuple("plots", ["time", "minrun"])
data_params = namedtuple("data", ["std", "mean"])

logging.basicConfig(filename="logs.log", level=logging.INFO)


def Get_best_minrun_by_arr(arr, k=3, minrun_step=1):
    """Funtion that culculates minruns for given array.

    Args:
        arr ([..]): Array of smth that we want to sort.
        k (int): number of resorting for one minrun value. (the more the better)
        minrun_step (int): step of checking minrun

    Returns:
        NamedTuple(Minrun, Time): Namedtuple of minruns and their working time.

    """
    minrun_params = namedtuple("minrun", ["minrun", "time"])
    minrun_and_time = []
    for minrun in range(1, len(arr) + 1, minrun_step):
        time_sr = 0
        for p in range(k):
            now_s = time.time()
            timsort(minrun, copy.deepcopy(arr))
            now2_s = time.time()
            time_r = abs(now2_s - now_s)
            time_sr += time_r / k
        minrun_and_time += [[time_sr, minrun]]

    minrun_and_time.sort()
    result = minrun_params(minrun=list(map(lambda x: x[1], minrun_and_time)),
                           time=list(map(lambda x: x[0], minrun_and_time)))
    return result


class SortingBenchmarkFramework:
    """Framework to find, save and compare models

    Attributes::
        algo: alg of training neural networks
        like fast, secondary, hard, specific
        model: trained model
        save: true, if during training program should print info about modules
        otherwise - false
        epochs: num of epochs for training
        model_params: SortingBenchmarkFramework model params
        data NamedTuple("train", "train_labels", "test",
                          "test_labels", "exam", "exam_labels"): just data
        parameters: parameters from config file for generating neural network
        data_params (NamedTuple data_params): info about data like std, mean

    """
    def __init__(self, algo, model=None, save=None,
                 epochs=None, model_params=None):
        self.save = save
        self.model = model
        self.epochs = epochs
        self.algo = algo
        self.model_params = model_params
        self.data = None
        self.parameters = None
        self.data_params = None

    def CreateAlgo(self, specific_params=None):
        if self.data is None or self.parameters is None:
            raise ValueError("model or data or parameters is None")

        num_epochs = self.parameters.epochs[1] - self.parameters.epochs[0]
        if self.algo == "fast":
            layers = [["relu"]]
            num_layers = 2
            num_cheks = num_epochs // 75
            self.parameters.neirons[2] = 3
        elif self.algo == "secondary":
            layers = [["relu"], ["sigmoid"]]
            num_layers = 3
            num_cheks = num_epochs // 50
            self.parameters.neirons[2] = 4
        elif self.algo == "hard":
            layers = [["relu"], ["sigmoid"], ["relu", "regularizer"],
                      ["sigmoid", "regularizer"]]
            num_layers = 4
            num_cheks = num_epochs // 25
            self.parameters.neirons[2] = 5
        elif self.algo == "hard_with_dropout":
            layers = [["relu"], ["sigmoid"], ["relu", "regularizer"],
                      ["sigmoid", "regularizer"], ["relu", "dropout"],
                      ["sigmoid", "dropout"]]
            num_layers = 5
            num_cheks = num_epochs // 25
            self.parameters.neirons[2] = 5
        elif self.algo == "specific":
            layers, num_layers = specific_params.layer, \
                                 specific_params.num_layer
            num_cheks = num_epochs // specific_params.cheks_num
            self.parameters.neirons[2] = specific_params.neirons
        else:
            raise ValueError("self.algo can't be {}".format(self.algo))

        models, _ = get_network(self.data, self.parameters, num_layers,
                                layers=layers, save=self.save)
        model = models[0][1][0]
        _, self.epochs = get_best_epochs(model, self.data, self.parameters,
                                         num_cheks)
        self.model_params = model
        self.model = ModelKeeper(model)
        self.model.epochs = self.epochs

    def SaveModel(self, model_name_json=None, model_name=None):
        if "Models" not in os.listdir(os.curdir):
            os.mkdir("Models")
        if model_name is not None:
            self.model_fitted.save("{}/Models/{}".format(
                os.path.abspath(os.getcwd()), model_name))
        if model_name_json is not None:
            f = open("{}/Models/{}".format(os.path.abspath(os.getcwd()),
                                           model_name_json), 'w')
            f.write(str({"model": self.model_params, "epochs": self.epochs,
                         "algo": self.algo}))
            f.close()

    def LoadModel(self, model_name):
        f = open("{}/Models/{}".format(os.path.abspath(os.getcwd()),
                                       model_name), 'r')
        saved_data = eval(f.read())
        self.model = ModelKeeper(saved_data["model"])
        self.epochs = saved_data["epochs"]
        self.algo = saved_data["algo"]
        self.model.epochs = self.epochs
        f.close()

    def LoadConfig(self, config_name):
        self.parameters = config_parameters(config_name)

    def LoadData(self, path_to_data):
        self.data = get_parsed_data(path_to_data)

    def LoadDataConfig(self, path_to_data):
        with open(path_to_data) as f:
            data = js.load(f)
            self.data_params = data_params(float(data["std"]),
                                           float(data["mean"]))

    def Train(self):
        """Train with given data"""
        if self.model is None or self.data is None:
            raise ValueError("model or data is None")
        self.model_fitted = self.model.fit(self.data.train,
                                           self.data.train_labels)

    def Validate(self):
        """Uses the standard dataset to get benchmark"""
        if self.model is None or self.data is None:
            raise ValueError("model or data is None")
        return self.model.score(self.data.exam, self.data.exam_labels)

    def PredictMinrunBySize(self, size):
        norm_len = normalize_data(size,
                                  self.data_params.mean,
                                  self.data_params.std)
        minrun = self.model.predict([norm_len])[0][0]
        return min(size, max(1, abs(int(minrun))))

    def Predict(self, test, test_labels=None):
        """Make predictions for arrays"""
        predictions = []
        logging.info("Make predictions for {}".format(self.algo))
        if self.algo == "standard algorithm":
            predictions = [get_minrun(len(arr)) for arr in test] # not true
        elif self.algo == "best algorithm":
            predictions = [int(label) for label in test_labels]
        else:
            if self.model is None:
                raise ValueError("model is None")
            for arr in test:
                predictions += [self.PredictMinrunBySize(len(arr))]

        result = []
        for i in range(len(test)):
            if i % 100 == 0:
                logging.info("Completed {} from {}".format(i, len(test)))
            t_first = time.time()
            if self.algo == "standard algorithm":
                sorted(test[i])
            else:
                timsort(max(1, abs(predictions[i])), test[i])
            t_second = time.time()
            result += [predict_params(size=len(test[i]),
                                      minrun=predictions[i],
                                      time=(t_second - t_first) * 1000)]

        return result

    def Plots(self, models, name_of_result, dpi=100):
        """
        Draw some plots
        models -> [SortingBenchmarkFramework, ...]
        """
        if self.data is None or name_of_result is None:
            raise ValueError("data or name_of_result is None")
        if name_of_result is not None and name_of_result[-4:] != ".png":
            raise ValueError("picture_name has to end .png")

        arr = [[random.randint(1, 10000) for _ in range(int(renormalize_data(
            self.data.exam[i], self.data_params.mean, self.data_params.std)))]
               for i in range(len(self.data.exam))]

        plots = [model.Predict(arr) if model.algo != "best algorithm"
                 else model.Predict(arr, self.data.exam_labels)
                 for model in models]
        subplots = plots_params
        fig, (subplots.minrun, subplots.time) = plt.subplots(
            nrows=2, ncols=1)

        subplots.minrun.set_xlabel('Model', fontsize=12)
        subplots.minrun.set_ylabel('Minrun', fontsize=12)
        subplots.minrun.set_title('Minruns from all models', fontsize=12)
        subplots.minrun.grid(True)

        subplots.time.set_xlabel('Model', fontsize=12)
        subplots.time.set_ylabel('Time (s * 1000)', fontsize=12)
        subplots.time.set_title('Time from all models', fontsize=12)
        subplots.time.grid(True)

        gradient_pairs = [[Color("red"), Color("blue")],
                          [Color("green"), Color("yellow")]]
        colors = [list(color[0].range_to(Color(color[1]), (len(plots) + 2) // 2))
                  for color in gradient_pairs]

        for i in range(len(plots)):
            plot = plots[i]
            color = colors[i % 2][(i - i // 2) % len(colors[0])].rgb
            color = [max(random.uniform(0.0, 0.2), color[i] - 0.3) for i in range(3)]
            X = [j for j in range(len(plot))]
            y = [plot[j].minrun for j in range(len(plot))]
            subplots.minrun.scatter(X, y, c=color, s=20, label=models[i].algo)

            y = [plot[j].time for j in range(len(plot))]
            time_label = models[i].algo + " summary time: " + str(sum(y))
            subplots.time.scatter(X, y, c=color, s=20, label=time_label)

        subplots.minrun.legend(loc="upper right", fontsize="xx-large")
        subplots.time.legend(loc="upper right", fontsize="xx-large")
        plt.gcf().set_size_inches((65, 25))
        plt.tight_layout()
        if name_of_result is not None:
            plt.savefig(name_of_result, dpi=dpi, bbox_inches="tight")
        else:
            plt.tight_layout()

    def MakeMinrunFile(filename, left, right, step=1):
        MinrunData = str(left) + "\n" + str(step) + "\n"
        for size in range(left, right, step):
            minrun = str(self.PredictMinrunBySize(size))
            MinrunData += minrun + "\n"

        with open(filename, 'w') as f:
            f.write(MinrunData)

def CreateMinrunData(data, name_of_csv=None, k=3, minrun_step=1):
    minrun_data = []
    for arr in data:
        minrun_data += [(len(arr), Get_best_minrun_by_arr(arr, k, minrun_step).minrun[0])]

    if name_of_csv is not None:
        namedtuple("data", ["std", "mean"])
        csv_data = pd.DataFrame({"DataPairs": minrun_data})
        csv_data.to_csv(name_of_csv, index=True)
    return minrun_data

def DivideArr(arr):
    return [arr[i:i + i] for i in range(1, len(arr) // 2 - 1)]

