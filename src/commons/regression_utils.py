import numpy as np


class RegressionUtils:
    @staticmethod
    def initialize_weights(X):
        weights = np.zeros((np.shape(X)[1] + 1, 1))
        X = np.c_[np.ones((np.shape(X)[0], 1)), X]
        return weights, X

    @staticmethod
    def standardize(X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)