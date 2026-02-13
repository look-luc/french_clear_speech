import warnings

import pandas as pd
from show_tree import importance_trees, permutation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class rand_tree:
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
            oob_score=True,
        )
        return self.model.fit(self.x_train, self.y_train)

    def prediction(self):
        self.done_modeling = self.modeling()
        y_pred = self.done_modeling.predict(self.x_test)
        classification = classification_report(self.y_test, y_pred)
        subset_acc = accuracy_score(self.y_test, y_pred)
        h_loss = hamming_loss(self.y_test, y_pred)

        return classification, subset_acc, h_loss

    def getEstimator(self):
        return self.model.estimators_

    def importance(self):
        return self.model.feature_importances_


def main():
    df = pd.read_csv(
        "../../data/ML Copy - FRENCH RESULTS - 2025 - Matching Datasets (VLOOKUP).csv"
    )
    # x = df.drop(
    #     columns=[
    #         "clarity",
    #         "timepoint",
    #         "freq_f1",
    #         "amp_f1",
    #         "freq_f2",
    #         "amp_f2",
    #         "width_f2",
    #         "freq_f3",
    #         "amp_f3",
    #         "width_f3",
    #         "amp_p0",
    #         "freq_p0",
    #         "p0prominence",
    #         "vwl_amp_rms",
    #         "vwl_duration",
    #         "appendix_dur",
    #         "Label",
    #     ]
    # )

    x = df.drop(columns=["clarity", "timepoint", "Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    classification_model = rand_tree(
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_estimators=1200,
        depth=10,
    )

    classification, accuracy, h_loss = classification_model.prediction()
    print("-------------------Accuracy-------------------")
    print(accuracy)
    print("-------------Classification Report------------")
    print(classification)
    print("------------------hamming loss-----------------")
    print(h_loss)

    print(len(classification_model.getEstimator()))

    importance_trees(
        classification_model,
        "importance features",
        "importance",
        "features",
        "Mean decrease in impurity",
        x,
    )

    permutation(
        classification_model.model,
        "Importance features via permutation on full model",
        "permutation",
        "features",
        "Mean accuracy decrease",
        x,
        X_test,
        y_test,
    )
    return 0


if __name__ == "__main__":
    main()
