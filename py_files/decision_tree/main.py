import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

import rand_forest_regression
from show_tree import importance_trees, permutation

warnings.filterwarnings('ignore')

#TODO:
# \box nasal to predict nasal
# \box nasal to predict neighborhood density
# \box nasal to predict clarity

def main():
    df = pd.read_csv("../../data/tache_lecture.csv")
    df = df[df["timepoint"] != 3]
    x = df.drop(
        columns=[
            'Search Query', 'speaker', 'task', 'clarity', 'ND', 'timepoint', 'word', 'vowel', 'target'
        ]
    )
    y = df['target']

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

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"OOB Score:           {oob:.4f}")
    print(f"\nClassification Report:\n{report}")

    # Feature Importance Visualizations
    importance_trees(
        regression_model,
        "MDI Feature Importance",
        "importance_with_tache",
        "features",
        "Mean decrease in impurity",
        x
    )

    permutation(
        regression_model.model,
        "Importance features via permutation on full model",
        "permutation_with_tache",
        "features",
        "Mean accuracy decrease",
        x,
        X_test,
        y_test,
    )
    return 0

if __name__ == '__main__':
    main()
