from src.regression.linear_regression import LinearRegression
from src.regression.logistic_regression import LogisticRegression


class Regression:
    def __init__(self, type: str):
        if type == "logistic":
            self.algorithm = LogisticRegression()
        elif type == "linear":
            self.algorithm = LinearRegression()
        else:
            raise ValueError("Please specify a valid type: logistic or linear")

    def fit(self, X, y, lr, epochs):
        self.algorithm.fit(X, y, lr, epochs)

    def predict(self, X):
        predictions = self.algorithm.predict(X)
        print(predictions)
        return predictions


