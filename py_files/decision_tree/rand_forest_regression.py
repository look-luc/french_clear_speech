from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class RandomForest_nasal:
    def __init__(self, x_train, x_test, y_train, y_test, num_estimators, depth):
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        self.num_estimators = num_estimators
        self.depth = depth

    def modeling(self):
        self.model = RandomForestClassifier(
            n_estimators=self.num_estimators,
            max_depth=self.depth,
            random_state=42,
            oob_score=True,
            n_jobs=-1
        )
        return self.model.fit(self.x_train, self.y_train)

    def prediction(self):
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")

        predictions = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        oob_score = self.model.oob_score_

        return accuracy, report, oob_score

    def getEstimator(self):
        return self.model.estimators_

    def importance(self):
        if self.model is None:
            return None
        return self.model.feature_importances_
