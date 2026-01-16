from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class RandomForest_nasal:
    def __init__(self, dataset, label, num_estimators):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            dataset,
            label,
            test_size=0.2,
            random_state=42
        )
        self.num_estimators = num_estimators

    def modeling(self):
        self.model = RandomForestClassifier(n_estimators=self.num_estimators, random_state=42)
        self.model.fit(self.X_train, self.Y_train)

    def prediction(self):
        self.y_prediction = self.model.predict(self.X_test)

    def report(self):
        self.accuracy = accuracy_score(self.Y_test, self.prediction())
        self.classification = classification_report(self.Y_test, self.prediction())
        return self.accuracy, self.classification
