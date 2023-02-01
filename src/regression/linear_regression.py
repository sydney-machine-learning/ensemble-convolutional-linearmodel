import numpy as np
from src.commons.regression_utils import RegressionUtils


class LinearRegression:
    def __init__(self):
        self.weights = None

    @staticmethod
    def linear_function(X, theta):
        return np.matmul(X, theta)

    def cost(self, X, y, theta):
        return ((self.linear_function(X, theta)-y).T@(self.linear_function(X, theta)-y))/(2*y.shape[0])

    def fit(self, X, y, alpha=0.001, epochs=100):
        params, X = RegressionUtils.initialize_weights(X)
        cost_list = np.zeros(epochs, )
        for i in range(epochs):
            gradient = np.dot(X.T, self.linear_function(X, params) - np.reshape(y, (len(y), 1)))
            params = params - alpha * gradient
            cost_list[i] = self.cost(X, y, params)
        self.weights = params
        return cost_list

    def predict(self, X):
        z = np.dot(RegressionUtils.initialize_weights(X)[1], self.weights)
        lis = []
        for i in self.linear_function(z):
            if i > 0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis


