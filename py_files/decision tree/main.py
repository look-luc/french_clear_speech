import warnings

import pandas as pd

warnings.filterwarnings('ignore')

def main():
    df = pd.read_csv("../../data/vowel_data_all_LabPhon.csv")  # getting the data
    df = df.dropna(subset=['Target'])
    # starting to work on random forest
    x = df.drop(columns=['vowelSAMPA', 'Target'])
    y = df['Target']
    return 0


if __name__ == '__main__':
    main()
