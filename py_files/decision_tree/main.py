import warnings

import pandas as pd

warnings.filterwarnings('ignore')

#TODO:
# \box nasal to predict nasal
# \box nasal to predict neighborhood density
# \box nasal to predict clarity

def main():
    ds = pd.read_csv("./data/tache_lecture.csv")
    print(ds.head())
    return 0


if __name__ == '__main__':
    main()
