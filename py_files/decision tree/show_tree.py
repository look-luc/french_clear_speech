import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def importance_trees(model, title:str, filename:str, x_label:str, y_label:str, X:pd.DataFrame):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=X.columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    plt.savefig(f"{filename}.png")