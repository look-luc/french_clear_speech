import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def compare_len(data1, data2):
    if len(data1) < len(data2):
        data1.extend([np.nan] * (len(data2) - len(data1)))
    if len(data1) > len(data2):
        data2.extend([np.nan] * (len(data1) - len(data2)))
    return data1,data2

def plot(data_1, data_2, title:str, data_1_key:str, data_2_key:str):
    data_1_listF1,data_2_listF1 = compare_len([d["F1_Hz"] for d in data_1],[d["F1_Hz"] for d in data_2])
    data_1_listF2,data_2_listF2 = compare_len([d["F2_Hz"] for d in data_1],[d["F2_Hz"] for d in data_2])
    
    plt.figure(figsize=(6, 5))
    plt.scatter(data_1_listF1, data_1_listF2,c="blue",marker="+", label=data_1_key)
    plt.scatter(data_2_listF1,data_2_listF2,c="red",marker="o", label=data_2_key)
    plt.xlabel("F1 frequency")
    plt.ylabel("F2 frequency")
    plt.legend()
    plt.savefig(title)

def mean_sd(data1, name1:str, store_dict:dict):
    store_dict[name1+"_F1_mean"] = sum([d.get("F1_Hz") for d in data1])/len([d.get("F1_Hz") for d in data1])
    store_dict[name1+"_F1_sd"] = np.std([d.get("F1_Hz") for d in data1])
    store_dict[name1+"_F2_mean"] = sum([d.get("F2_Hz") for d in data1])/len([d.get("F2_Hz") for d in data1])
    store_dict[name1+"_F2_sd"] = np.std([d.get("F2_Hz") for d in data1])
    return store_dict

def t_tests(data1,data2,name_ttest,namep,store_data):
    ttest,pval = stats.ttest_ind([d.get("F1_Hz") for d in data1],[d.get("F1_Hz") for d in data2])
    store_data[name_ttest+"F1"]=ttest
    store_data[namep+"F1"]=pval
    ttest,pval = stats.ttest_ind([d.get("F2_Hz") for d in data1],[d.get("F2_Hz") for d in data2])
    store_data[name_ttest+"F2"]=ttest
    store_data[namep+"F2"]=pval
    return store_data

def sec_pass(data):
    a_lo_nasal = [d for d in data if d["ND"]=="lo" and d["vowelSAMPA"]=="a~"]
    a_lo_unnasal = [d for d in data if d["ND"]=="lo" and d["vowelSAMPA"]=="a"]
    o_lo_nasal = [d for d in data if d["ND"]=="lo" and d["vowelSAMPA"]=="O~"]
    o_lo_unnasal = [d for d in data if d["ND"]=="lo" and d["vowelSAMPA"]=="o"]
    e_lo_nasal = [d for d in data if d["ND"]=="lo" and d["vowelSAMPA"]=="3"]
    e_lo_unnasal = [d for d in data if d["ND"]=="lo" and d["vowelSAMPA"]=="E~"]

    plot(a_lo_nasal,a_lo_unnasal,"a_nasal_un_lo.png","a~","a")
    plot(o_lo_nasal,o_lo_unnasal,"o_nasal_un_lo.png","o~","o")
    plot(e_lo_nasal,e_lo_unnasal,"e_nasal_un_lo.png","e~","e")

    a_hi_nasal = [d for d in data if d["ND"] == "hi" and d["vowelSAMPA"] == "a~"]
    a_hi_unnasal = [d for d in data if d["ND"] == "hi" and d["vowelSAMPA"] == "a"]
    o_hi_nasal = [d for d in data if d["ND"] == "hi" and d["vowelSAMPA"] == "O~"]
    o_hi_unnasal = [d for d in data if d["ND"] == "hi" and d["vowelSAMPA"] == "o"]
    e_hi_nasal = [d for d in data if d["ND"] == "hi" and d["vowelSAMPA"] == "3"]
    e_hi_unnasal = [d for d in data if d["ND"] == "hi" and d["vowelSAMPA"] == "E~"]

    plot(a_hi_nasal, a_hi_unnasal, "a_nasal_un_hi.png", "a~", "a")
    plot(o_hi_nasal, o_hi_unnasal, "o_nasal_un_hi.png", "O~", "o")
    plot(e_hi_nasal, e_hi_unnasal, "e_nasal_un_hi.png", "e~", "e")

    lo_V={"a~_lo":a_lo_nasal,"a_lo":a_lo_unnasal,"o~_lo":o_lo_nasal,"o_lo":o_lo_unnasal,"e~_lo":e_lo_nasal,"e_lo":e_lo_unnasal}
    hi_V={"a~_hi":a_hi_nasal,"a_hi":a_hi_unnasal,"o~_hi":o_hi_nasal,"o_hi":o_hi_unnasal,"e~_hi":e_hi_nasal,"e_hi":e_hi_unnasal}

    stat_store = {}
    for key,val in lo_V.items():
        stat_store = mean_sd(val,key,stat_store)
    for key_2,val_2 in hi_V.items():
        stat_store = mean_sd(val_2,key_2,stat_store)

    for (key1, value1), (key2, value2) in zip(lo_V.items(), hi_V.items()):
        stat_store = t_tests(value1,value2,key1+" "+key2+" t",key1+" "+key2+" p",stat_store)

    for i,j in stat_store.items():
        with open("sec_pass_measure.txt","a") as f:
            f.write(f"{i}: {j}\n\n")
    return 0