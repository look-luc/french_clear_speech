import warnings
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

warnings.filterwarnings('ignore')

class rand_tree():
    def __init__(self, x_train, x_test, y_train, y_test, num_estimators, depth):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
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

def main():
    df = pd.read_csv("../../data/ML Copy - FRENCH RESULTS - 2025 - Matching Datasets (VLOOKUP).csv")
    x = df.drop(columns=['clarity','timepoint','Label'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    classification_model = rand_tree(
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_estimators=1200,
        depth=10,
    )

    accuracy, classification, h_loss = classification_model.prediction()
    print("-------------------Accuracy-------------------")
    print(accuracy)
    print("-------------Classification Report------------")
    print(classification)
    print("------------------hamming loss-----------------")
    print(h_loss)
    return 0

if __name__ == '__main__':
    main()