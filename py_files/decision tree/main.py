import warnings
import rand_forest_regression
import rand_forest_bi
import pandas as pd
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def main():
    df = pd.read_csv("../../data/vowel_data_all_LabPhon.csv")  # getting the data

    # starting to work on random forest
    x = df.drop(columns=['vowelSAMPA', 'Target'])
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    regression_model = rand_forest_regression.RandomForest_nasal(
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_estimators=75,
        depth=200,
    )

    accuracy, classification = regression_model.prediction()
    print(f'accuracy:\n{accuracy}')
    print(f'classification:\n{classification}')
    return 0


if __name__ == '__main__':
    main()
