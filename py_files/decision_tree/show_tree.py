import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def importance_trees(
    model, title: str, filename: str, x_label: str, y_label: str, X: pd.DataFrame
):
    importances = model.importance()
    std = np.std([tree.feature_importances_ for tree in model.getEstimator()], axis=0)

    forest_importances = pd.Series(importances, index=X.columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    plt.savefig(f"{filename}.png")
    print(f"graph made and saved under {filename}.png")


def permutation(
    model,
    title: str,
    filename: str,
    x_label: str,
    y_label: str,
    X: pd.DataFrame,
    X_test,
    y_test,
):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    importance = pd.Series(result.importances_mean, index=X.columns)

    fig, ax = plt.subplots()
    importance.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    plt.savefig(f"{filename}.png")

    print(f"graph made and saved under {filename}.png")
