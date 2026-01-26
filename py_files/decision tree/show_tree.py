import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def show_tree(model, title: str, filename: str, x: pd.DataFrame, y: pd.DataFrame, num_features: int):
    for i in range(num_features):
        plt.figure(figsize=(20, 10))
        plt.title(title)
        plot_tree(
            model.estimators_[i],
            feature_names= x.columns.tolist(),
            class_names= y.columns.tolist(),
            show_leaf_values=True,
            show_shapes=True,
            show_edges=True,
            filled=True,
            rounded=True,
            fontsize=10,
        )
        plt.savefig(f'{filename}_{i}.png')