import numpy as np
import pandas as pd
from scipy import stats

def mean_sd(data,vowels,storage,type):
    for key,value in vowels.items():
        temp = [d.get("appendix_dur") for d in data if d["vowel"]==value]
        storage[f"{type}_{key}_mean"] = sum(temp)/len(temp)
        storage[f"{type}_{key}_sd"] = np.std(temp)
    return storage

def t_tests(data_1,data_2,vowels,storage,type):
    for key,value in vowels.items():
        temp1 = [d.get("appendix_dur") for d in data_1 if d["vowel"]==value]
        temp2 = [d.get("appendix_dur") for d in data_2 if d["vowel"]==value]

        storage[f"{type}_{key}_t"], storage[f"{type}_{key}_p"] = stats.ttest_ind(temp1,temp2)
    return storage

def appendix(data):
    vowels = {"ɑ̃":"É\x91Ì\x83","ɛ̃":"É\x9bÌ\x83","ɔ̃":"É\x94Ì\x83"}
    stat = {}
    lo_normal = [d for d in data if d["ND"] == "lo" and d["clarity"]=="normal" and d["appendix"]!="none"]
    hi_normal = [d for d in data if d["ND"] == "hi" and d["clarity"]=="normal" and d["appendix"]!="none"]
    lo_clear = [d for d in data if d["ND"] == "lo" and d["clarity"]=="clear" and d["appendix"]!="none"]
    hi_clear = [d for d in data if d["ND"] == "hi" and d["clarity"]=="clear" and d["appendix"]!="none"]

    stat = mean_sd(lo_normal,vowels,stat,"lo_normal")
    stat = mean_sd(hi_normal,vowels,stat,"hi_normal")
    stat = mean_sd(lo_clear,vowels,stat,"lo_clear")
    stat = mean_sd(hi_clear,vowels,stat,"hi_clear")

    lo_nor_n = [d for d in lo_normal if d["appendix"]=="n"]
    lo_nor_c = [d for d in lo_normal if d["appendix"]=="c"]

    lo_clr_n = [d for d in lo_clear if d["appendix"]=="n"]
    lo_clr_c = [d for d in lo_clear if d["appendix"]=="c"]

    hi_nor_n = [d for d in hi_normal if d["appendix"]=="n"]
    hi_nor_c = [d for d in hi_normal if d["appendix"]=="c"]

    hi_clr_n = [d for d in hi_clear if d["appendix"]=="n"]
    hi_clr_c = [d for d in hi_clear if d["appendix"]=="c"]

    stat = mean_sd(lo_nor_n,vowels,stat,"lo_nor_n")
    stat = mean_sd(lo_nor_c,vowels,stat,"lo_nor_c")

    stat = mean_sd(lo_clr_n,vowels,stat,"lo_clr_n")
    stat = mean_sd(lo_clr_c,vowels,stat,"lo_clr_c")

    stat = mean_sd(hi_nor_n,vowels,stat,"hi_nor_n")
    stat = mean_sd(hi_nor_c,vowels,stat,"hi_nor_c")

    stat = mean_sd(hi_clr_n,vowels,stat,"hi_clr_n")
    stat = mean_sd(hi_clr_c,vowels,stat,"hi_clr_c")

    stat = t_tests(lo_normal,lo_clear,vowels,stat,"lo_nor_lo_clr")
    stat = t_tests(hi_normal,hi_clear,vowels,stat,"hi_nor_hi_clr")

    stat = t_tests(lo_nor_n,lo_nor_c,vowels,stat,"lo_nor_n_c")
    stat = t_tests(lo_clr_n,lo_clr_c,vowels,stat,"lo_clr_n_c")
    stat = t_tests(hi_nor_n,hi_nor_c,vowels,stat,"hi_nor_n_c")
    stat = t_tests(hi_clr_n,hi_clr_c,vowels,stat,"hi_clr_n_c")

    with open("appendix.txt","a") as file:
        for key,value in stat.items():
            file.write(f"{key}: \n\t{value}\n")
    return 0