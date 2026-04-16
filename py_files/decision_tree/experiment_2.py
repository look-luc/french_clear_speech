import warnings

import pandas as pd

warnings.filterwarnings('ignore')

def main():
    full_data = pd.read_csv("../../data/tache_lecture.csv")
    full_data = full_data[full_data["timepoint"] == 3]

    nasal_appendix = full_data.drop(
        ['timepoint', 'appendix', 'appendix_dur', 'creak_app'],
        axis=1
    )

    creak_appendix = full_data.drop(
        ['timepoint', 'appendix', 'appendix_dur', 'nasal_app'],
        axis=1
    )

    nasal_appendix_dur = full_data.drop(
        ['timepoint', 'appendix', 'nasal_app', 'creak_app'],
        axis=1
    )

    return 0


if __name__ == '__main__':
    main()
