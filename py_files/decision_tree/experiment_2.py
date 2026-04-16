import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

import rand_forest_regression
from show_tree import importance_trees, permutation

warnings.filterwarnings('ignore')

def main():
    full_data = pd.read_csv("../../data/tache_lecture.csv").drop("clarity", axis=1)
    full_data = full_data[full_data["timepoint"] == 3]

    nasal_appendix = full_data.drop(
        ['timepoint', 'appendix', 'appendix_dur', 'nasal_app', 'creak_app', 'clarity', 'target'],
        axis=1
    )
    nasal_target = full_data['nasal_app']

    creak_appendix = full_data.drop(
        ['timepoint', 'appendix', 'appendix_dur', 'nasal_app', 'creak_app', 'clarity', 'target'],
        axis=1
    )
    creak_target = full_data['creak_app']

    data = [nasal_appendix, creak_appendix]
    targets = [nasal_target, creak_target]
    exp = ["nasal_only", "creak_only"]

    for x, y, experiment in zip(data, targets, exp):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        regression_model = rand_forest_regression.RandomForest_nasal(
            x_train=X_train,
            x_test=X_test,
            y_train=y_train,
            y_test=y_test,
            num_estimators=1000,
            depth=30
        )

        regression_model.modeling()
        accuracy, report, oob = regression_model.prediction()

        with open(f"{experiment}_scores.txt", "w") as file:
            file.write(f"Validation Accuracy: {accuracy:.4f}\n")
            file.write(f"OOB Score:           {oob:.4f}\n")
            file.write(f"\nClassification Report:\n{report}")

        importance_trees(
            regression_model,
            "MDI Feature Importance",
            f"importance_{experiment}",
            "features",
            "Mean decrease in impurity",
            x
        )

        permutation(
            regression_model.model,
            "Importance features via permutation on full model",
            f"permutation_{experiment}",
            "features",
            "Mean accuracy decrease",
            x,
            X_test,
            y_test,
        )

    return 0


if __name__ == '__main__':
    main()
