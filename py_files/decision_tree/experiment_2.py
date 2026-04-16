import warnings

import pandas as pd

warnings.filterwarnings('ignore')

def main():
    full_data = pd.read_csv("../../data/tache_lecture.csv")
    full_data = full_data[full_data["timepoint"] == 3]

    nasal_appendix = full_data[
        full_data["appendix"] == 'n' or full_data["appendix"] == 'n+c'
    ].drop([
        'timepoint', 'appendix', 'appendix_dur'
    ], axis=1)

    creak_appendix = full_data[
        full_data["appendix"] == 'c' or full_data["appendix"] == 'n+c'
        ].drop([
        'timepoint', 'appendix', 'appendix_dur'
    ], axis=1)


    return 0


if __name__ == '__main__':
    main()
