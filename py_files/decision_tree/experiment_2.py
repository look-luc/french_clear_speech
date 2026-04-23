import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

import rand_forest_regression
from show_tree import importance_trees, permutation

warnings.filterwarnings('ignore')

def main():
    controll_data = pd.read_csv("../../data/append_dur.csv").drop("clarity", axis=1)

    full_data = controll_data[controll_data["timepoint"] == 3]

    data = full_data.drop(columns=["timepoint", "appendix", "creak_app", "target"])
    data_target = full_data["target"]

    X_train, X_test, y_train, y_test = train_test_split(data, data_target, test_size=0.2, random_state=42)

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

    with open(f"sep_dur_scores.txt", "w") as file:
        file.write(f"Validation Accuracy: {accuracy:.4f}\n")
        file.write(f"OOB Score:           {oob:.4f}\n")
        file.write(f"\nClassification Report:\n{report}")

    importance_trees(
        regression_model,
        "MDI Feature Importance",
        f"importance_sep_dur",
        "features",
        "Mean decrease in impurity",
        data
    )

    permutation(
        regression_model.model,
        "Importance features via permutation on full model",
        f"permutation_sep_dur",
        "features",
        "Mean accuracy decrease",
        data,
        X_test,
        y_test,
    )

    return 0


if __name__ == '__main__':
    main()