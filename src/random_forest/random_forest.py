from src.commons.utils import Utils
from sklearn import tree


class RandomForest:
    def __init__(self, type: str = None) -> None:
        self.n_estimators = [1, 10, 50, 100, 250, 500, 750, 1000]
        self.errors = []
        if type == "bootstrapping":
            self.algorithm = self.random_forest_bootstrapping
        elif type == "bagging":
            self.algorithm = self.random_forest_bagging
        else:
            raise ValueError("Invalid Algorithm Type")

    def predict(self):
        for est in self.n_estimators:
            forest = self.algorithm(X_train, y_train, est)
            predictions = Utils.hard_voting_prediction(X_test, forest)
            self.errors.append(Utils.error_score(predictions, y_test))
        return self.errors, self.n_estimators

    @staticmethod
    def random_forest_bootstrapping(X_train, y_train, n_estimators):
        forest = list()
        oob_ls = list()
        for i in range(n_estimators):
            X_bootstrap, y_bootstrap, X_oob, y_oob = Utils.draw_bootstrap(X_train, y_train)
            decision_tree = tree.DecisionTreeClassifier()
            decision_tree.fit(X_bootstrap, y_bootstrap)
            forest.append(decision_tree)
            oob_pred = decision_tree.predict(X_oob)
            oob_error = Utils.error_score(oob_pred, y_oob)
            oob_ls.append(oob_error)
        print("OOB error estimate: {:.2f}".format(np.mean(oob_ls)))
        return forest

    @staticmethod
    def random_forest_bagging(X_train, y_train, n_estimators):
        forest = list()
        bagging_indices = Utils.draw_bagging_indices(X_train.shape[1], n_estimators)
        for i in range(n_estimators):
            X_bag, y_bag = X_train[:, bagging_indices[i]], y_train
            decision_tree = tree.DecisionTreeClassifier()
            decision_tree.fit(X_bag, y_bag)
            forest.append(decision_tree)
        return forest, bagging_indices
