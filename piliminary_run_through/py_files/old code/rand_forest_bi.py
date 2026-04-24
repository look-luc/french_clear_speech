from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

class RandomForest_multi:
    def __init__(self, x_train, x_test, y_train, y_test, num_estimators, depth):
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        self.num_estimators = num_estimators
        self.depth = depth

    def modeling(self):
        self.model = RandomForestClassifier(
            n_estimators=self.num_estimators,
            max_depth=self.depth,
            random_state=42,
            oob_score=True)
        return self.model.fit(self.x_train, self.y_train)

    def prediction(self):
        self.done_modeling = self.modeling()
        y_pred = self.done_modeling.predict(self.x_test)
        classification = classification_report(self.y_test, y_pred)
        subset_acc = accuracy_score(self.y_test, y_pred)
        h_loss = hamming_loss(self.y_test, y_pred)

        return classification, subset_acc, h_loss