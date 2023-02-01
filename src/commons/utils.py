import random
import math
import numpy as np


class Utils:
    @staticmethod
    def error_score(prediction, ground_truth):
        mis_label = 0
        for i in range(len(prediction)):
            if prediction[i] != ground_truth[i]:
                mis_label += 1
        return mis_label / len(prediction)

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

    @staticmethod
    def hard_voting_prediction(test_sample, forest, bagging_indices=None):
        predictions_list = list()
        for i in range(len(test_sample)):
            predictions = dict()
            predictions[0] = 0
            predictions[1] = 0
            predictions_list.append(predictions)
        for i in range(len(forest)):
            prediction = list()
            if bagging_indices is not None:
                test_sample_temp = test_sample[:, bagging_indices[i]]
                prediction = forest[i].predict(test_sample_temp)
            else:
                prediction = forest[i].predict(test_sample)

            for i in range(len(prediction)):
                predictions_list[i][prediction[i]] += 1

        final_pred = list()
        for predictions in predictions_list:
            final_pred.append(max(predictions, key=lambda x: predictions[x]))

        return final_pred
