from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class RandomForest_nasal:
    def __init__(self, x_train, x_test, y_train, y_test, num_estimators):
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        self.num_estimators = num_estimators

    def modeling(self):
        self.model = RandomForestClassifier(n_estimators=self.num_estimators, random_state=0, oob_score=True)
        return self.model.fit(self.x_train, self.y_train)

    def prediction(self):
        self.done_modeling = self.modeling()
        self.x_prediction = self.done_modeling.predict(self.x_test)
        self.accuracy = accuracy_score(self.y_test, self.prediction())
        self.classification = classification_report(self.y_test, self.prediction())
        return self.accuracy, self.classification
