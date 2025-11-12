import pandas as pd
from first_pass import first_pass
from sec_pass import sec_pass
from third_pass import appendix

def getdata(filepath: str) -> list[dict]:
    if(filepath == ""):
        return []
    else:
        df = pd.read_csv(filepath, encoding='latin-1')
        values = df.to_dict(orient='records')
        return values

def main()->int:
    # data_french_results_1_pass = getdata(
    #     "/Users/lucdenardi/Desktop/python/french_clear_speach/first_pass_data/Copy of FRENCH RESULTS - 2025.xlsx - Data.csv"
    # )
    # first_pass(data_french_results_1_pass)

    # data_french_results_2_pass = getdata(
    #     "/Users/lucdenardi/Desktop/python/french_clear_speach/second_pass_data/vowel_data_all_LabPhon.xlsx - dataLabPhon_all.csv"
    # )
    # sec_pass(data_french_results_2_pass)
    data = getdata(
        "/Users/lucdenardi/Desktop/python/french_clear_speach/appendix/ML Copy - FRENCH RESULTS - 2025 - Nasal Data Integrated.csv"
    )
    appendix(data)
    return 0

if __name__ == '__main__':
    main()
