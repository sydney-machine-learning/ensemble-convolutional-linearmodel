import numpy as np
import math
import random


class EnsembleUtils:

    @staticmethod
    def draw_bagging_indices(n_features, n_estimators):
        bagging_features_indices = list()
        for i in range(n_estimators):
            bagging_features_indices.append(random.sample(range(n_features), math.ceil(math.sqrt(n_features))))
        return bagging_features_indices

    @staticmethod
    def draw_bootstrap(X_train, y_train):
        bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace=True))
        oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        X_oob = X_train[oob_indices]
        y_oob = y_train[oob_indices]
        return X_bootstrap, y_bootstrap, X_oob, y_oob
