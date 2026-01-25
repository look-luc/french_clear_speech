import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def show_tree(model, title: str, filename: str, df: pd.DataFrame):
    model_estimator = model.estimators_[0]

    plt.figure()
    plt.title(title)
    plot_tree(
        model_estimator,
        feature_names= df.columns.tolist(),
        show_leaf_values=True,
        show_shapes=True,
        show_edges=True,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.savefig(f'{filename}.png')