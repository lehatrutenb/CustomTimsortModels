"""Keep and work with neural models"""
from collections import namedtuple
from keras import models, regularizers, layers

model_settings = namedtuple("model_params",
                            ["activation", "neirons",
                             "dropout", "regularizer"])


class ModelKeeper:
    """Class that can keep, fit, train neural models"""
    def __init__(self, model_params=None, epochs=None):
        """
        model_params ->
        [[(activation, "relu" or "sigmoid" or "softmax" or "dropout"),
        (regularizer, float), (neirons, int or float(drop_percent)],
        [layer2], [layer3], ...]
        """
        self.epochs = epochs
        self.model_params = model_params

        """self.model_params = [model_settings(
            neirons=None, dropout=None,
            regularizer=None, activation=None)] * 5"""

        if self.model_params is not None:
            self.model = self._build_model()

    def _build_model(self):
        """Build model layer by layer from model_params"""
        model = models.Sequential()
        model.add(layers.Dense(1, input_dim=1))
        for i in range(len(self.model_params)):
            if self.model_params[i].regularizer is not None:
                model.add(layers.Dense(
                    int(self.model_params[i].neirons),
                    kernel_regularizer=regularizers.l2(
                        self.model_params[i].regularizer),
                    activation=self.model_params[0].activation))
            elif self.model_params[i].neirons is not None:
                model.add(layers.Dense(int(
                    self.model_params[i].neirons),
                    activation=self.model_params[0].activation))

            if self.model_params[i].dropout is not None:
                model.add(layers.Dropout(self.model_params[i].dropout))

        model.add(layers.Dense(1))
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mse"])
        return model

    def set_params(self, act=None, epochs=None):
        if act is not None:
            layer = 0
            if self.model_params == None:
                self.model_params = [model_settings(
                                        neirons=None, dropout=None,
                                        regularizer=None, activation=None)]

            neirons = self.model_params[layer].neirons
            while neirons is not None:
                if layer + 1 == len(self.model_params):
                    self.model_params += [model_settings(
                                            neirons=None, dropout=None,
                                            regularizer=None, activation=None)]

                layer += 1
                neirons = self.model_params[layer].neirons
            
            self.model_params[layer] = act
            self.model = self._build_model()
        if epochs is not None:
            self.epochs = epochs
        return self

    def get_params(self, deep):
        import copy
        if not deep:
            params = {"epochs": self.epochs}
            params["model_params"] = self.model_params
            return params

    def score(self, test, test_labels):
        """Evaluate self model"""
        err = self.model.evaluate(test, test_labels)[0]
        return err

    def fit(self, train, train_labels, epochs=None, verbose=None):
        if verbose is None:
            verbose = 0
        if self.epochs is None or epochs is not None:
            self.epochs = epochs

        self.model.fit(train, train_labels,
                       epochs=self.epochs, verbose=verbose)

        return self.model

    def predict(self, arr_for_predict):
        return self.model.predict(arr_for_predict)

