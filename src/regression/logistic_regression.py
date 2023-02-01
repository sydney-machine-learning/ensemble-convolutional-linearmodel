import numpy as np
from src.commons.regression_utils import RegressionUtils


class LogisticRegression:
    def __init__(self):
        self.weights = None
        print(1)

    @staticmethod
    def sigmoid(z):
        sig = 1 / (1 + np.exp(-z))
        return sig

    def cost(self, X, y, theta):
        z = np.dot(X, theta)
        cost0 = y.T.dot(np.log(self.sigmoid(z)))
        cost1 = (1 - y).T.dot(np.log(1 - self.sigmoid(z)))
        cost = -(cost1 + cost0) / len(y)
        return cost

    def fit(self, X, y, alpha=0.001, epochs=100):
        params, X = RegressionUtils.initialize_weights(X)
        cost_list = np.zeros(epochs, )
        for i in range(epochs):
            params = params - alpha * np.dot(X.T, self.sigmoid(np.dot(X, params)) - np.reshape(y, (len(y), 1)))
            cost_list[i] = self.cost(X, y, params)
        self.weights = params
        return cost_list

    def predict(self, X):
        z = np.dot(RegressionUtils.initialize_weights(X)[1], self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i > 0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis
