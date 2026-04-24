import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def get_safe(d, key):
    return d.get(key, None)

def first_pass(data):
    lo_nd_a = [d for d in data if get_safe(d, "ND") == "lo" and get_safe(d, "vowel") == "É\x91Ì\x83"]
    hi_nd_a = [d for d in data if get_safe(d, "ND") == "hi" and get_safe(d, "vowel") == "É\x91Ì\x83"]

    lo_nd_e = [d for d in data if get_safe(d, "ND") == "lo" and get_safe(d, "vowel") == "É\x9bÌ\x83"]
    hi_nd_e = [d for d in data if get_safe(d, "ND") == "hi" and get_safe(d, "vowel") == "É\x9bÌ\x83"]

    lo_nd_o = [d for d in data if get_safe(d, "ND") == "lo" and get_safe(d, "vowel") == "É\x94Ì\x83"]
    hi_nd_o = [d for d in data if get_safe(d, "ND") == "hi" and get_safe(d, "vowel") == "É\x94Ì\x83"]

    f1_hi_a, f1_lo_a = [],[]
    for lo_f1 in lo_nd_a:
        f1_lo_a.append(lo_f1["freq_f1"])

    for hi_f1 in hi_nd_a:
        f1_hi_a.append(hi_f1["freq_f1"])

    f1_hi_e, f1_lo_e = [], []
    for lo_f1, hi_f1 in zip(lo_nd_e, hi_nd_e):
        f1_hi_e.append(hi_f1["freq_f1"])
        f1_lo_e.append(lo_f1["freq_f1"])

    f1_hi_o, f1_lo_o = [], []
    for lo_f1, hi_f1 in zip(lo_nd_o, hi_nd_o):
        f1_hi_o.append(hi_f1["freq_f1"])
        f1_lo_o.append(lo_f1["freq_f1"])

    f2_hi_a, f2_lo_a = [], []
    for lo_f2 in lo_nd_a:
        f2_lo_a.append(lo_f2["freq_f2"])
    for hi_f2 in hi_nd_a:
        f2_hi_a.append(hi_f2["freq_f2"])


    f2_hi_e, f2_lo_e = [], []
    for lo_f2, hi_f2 in zip(lo_nd_e, hi_nd_e):
        f2_hi_e.append(hi_f2["freq_f2"])
        f2_lo_e.append(lo_f2["freq_f2"])

    f2_hi_o, f2_lo_o = [], []
    for lo_f2, hi_f2 in zip(lo_nd_o, hi_nd_o):
        f2_hi_o.append(hi_f2["freq_f2"])
        f2_lo_o.append(lo_f2["freq_f2"])
    
    plt.figure(figsize=(6, 5))
    plt.scatter(f1_lo_a, f2_lo_a, color="blue", marker="+", label="lo nd a")
    plt.scatter(f1_hi_a, f2_hi_a, color="red", marker="o", label="hi nd a")
    plt.xlabel("F1 frequency")
    plt.ylabel("F2 frequency")
    plt.legend()
    plt.savefig("a.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(f1_lo_e, f2_lo_e, color="blue", marker="+", label="lo nd e")
    plt.scatter(f1_hi_e, f2_hi_e, color="red", marker="o", label="hi nd e")
    plt.xlabel("F1 frequency")
    plt.ylabel("F2 frequency")
    plt.legend()
    plt.savefig("e.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(f1_lo_o, f2_lo_o, color="blue", marker="+", label="lo nd o")
    plt.scatter(f1_hi_o, f2_hi_o, color="red", marker="o", label="hi nd o")
    plt.xlabel("F1 frequency")
    plt.ylabel("F2 frequency")
    plt.legend()
    plt.savefig("o.png")

    f1_lo_a_mean = sum(f1_lo_a) / len(f1_lo_a)
    f1_lo_a_sd = np.std(f1_lo_a)

    f1_hi_a_mean = sum(f1_hi_a) / len(f1_hi_a)
    f1_hi_a_sd = np.std(f1_hi_a)

    f1_lo_e_mean = sum(f1_lo_e) / len(f1_lo_e)
    f1_lo_e_sd = np.std(f1_lo_e)

    f1_hi_e_mean = sum(f1_hi_e) / len(f1_hi_e)
    f1_hi_e_sd = np.std(f1_hi_e)

    f1_lo_o_mean = sum(f1_lo_o) / len(f1_lo_o)
    f1_lo_o_sd = np.std(f1_lo_o)

    f1_hi_o_mean = sum(f1_hi_o) / len(f1_hi_o)
    f1_hi_o_std = np.std(f1_hi_o)

    with open("mean_sd.txt", "w") as f:
        f.write("F1:\n")
        f.write("ɑ̃ low ND:\n")
        f.write(str(f1_lo_a_mean))
        f.write("\n")
        f.write(str(f1_lo_a_sd))
        f.write("\n")
        f.write("ɑ̃ high ND:\n")
        f.write(str(f1_hi_a_mean))
        f.write("\n")
        f.write(str(f1_hi_a_sd))
        f.write("\n")
        f.write("ɛ̃ low ND:\n")
        f.write(str(f1_lo_e_mean))
        f.write("\n")
        f.write(str(f1_lo_e_sd))
        f.write("\n")
        f.write("ɛ̃  high ND:\n")
        f.write(str(f1_hi_e_mean))
        f.write("\n")
        f.write(str(f1_hi_e_sd))
        f.write("\n")
        f.write("ɔ̃ low ND:\n")
        f.write(str(f1_lo_o_mean))
        f.write("\n")
        f.write(str(f1_lo_o_sd))
        f.write("\n")
        f.write("ɔ̃ high ND:\n")
        f.write(str(f1_hi_o_mean))
        f.write("\n")
        f.write(str(f1_hi_o_std))
        f.write("\n\n")
        f.close()

    f2_lo_a_mean = sum(f2_lo_a) / len(f2_lo_a)
    f2_lo_a_sd = np.std(f2_lo_a)

    f2_hi_a_mean = sum(f2_hi_a) / len(f2_hi_a)
    f2_hi_a_sd = np.std(f2_hi_a)

    f2_lo_e_mean = sum(f2_lo_e) / len(f2_lo_e)
    f2_lo_e_sd = np.std(f2_lo_e)

    f2_hi_e_mean = sum(f2_hi_e) / len(f2_hi_e)
    f2_hi_e_sd = np.std(f2_hi_e)

    f2_lo_o_mean = sum(f2_lo_o) / len(f2_lo_o)
    f2_lo_o_sd = np.std(f2_lo_o)

    f2_hi_o_mean = sum(f2_hi_o) / len(f2_hi_o)
    f2_hi_o_sd = np.std(f2_hi_o)

    with open("mean_sd.txt", "a") as f:
        f.write("F2:\n")
        f.write("ɑ̃ low ND:\n")
        f.write(str(f2_lo_a_mean))
        f.write("\n")
        f.write(str(f2_lo_a_sd))
        f.write("\n")
        f.write("ɑ̃ high ND:\n")
        f.write(str(f2_hi_a_mean))
        f.write("\n")
        f.write(str(f2_hi_a_sd))
        f.write("\n")
        f.write("ɛ̃ low ND:\n")
        f.write(str(f2_lo_e_mean))
        f.write("\n")
        f.write(str(f2_lo_e_sd))
        f.write("\n")
        f.write("ɛ̃  high ND:\n")
        f.write(str(f2_hi_e_mean))
        f.write("\n")
        f.write(str(f2_hi_e_sd))
        f.write("\n")
        f.write("ɔ̃ low ND:\n")
        f.write(str(f2_lo_o_mean))
        f.write("\n")
        f.write(str(f2_lo_o_sd))
        f.write("\n")
        f.write("ɔ̃ high ND:\n")
        f.write(str(f2_hi_o_mean))
        f.write("\n")
        f.write(str(f2_hi_o_sd))
        f.write("\n\n")
        f.close()

    h1_hi_a, h1_lo_a = [], []
    for lo_h1 in lo_nd_a:
        h1_lo_a.append(lo_h1["freq_h1"])
    for hi_h1 in hi_nd_a:
        h1_hi_a.append(hi_h1["freq_h1"])

    h1_hi_e, h1_lo_e = [], []
    for hi_h1 in hi_nd_e:
        h1_hi_e.append(hi_h1["freq_h1"])
    for lo_h1 in lo_nd_e:
        h1_lo_e.append(lo_h1["freq_h1"])

    h1_hi_o, h1_lo_o = [], []
    for hi_h1 in hi_nd_o:
        h1_hi_o.append(hi_h1["freq_h1"])
    for lo_h1 in lo_nd_o:
        h1_lo_o.append(lo_h1["freq_h1"])

    p0_hi_a, p0_lo_a = [], []
    for hi_p0 in hi_nd_a:
        p0_hi_a.append(hi_p0["freq_p0"])
    for lo_p0 in lo_nd_a:
        p0_lo_a.append(lo_p0["freq_p0"])

    p0_hi_e, p0_lo_e = [], []
    for hi_p0 in hi_nd_e:
        p0_hi_e.append(hi_p0["freq_p0"])
    for lo_p0 in lo_nd_e:
        p0_lo_e.append(lo_p0["freq_p0"])

    p0_hi_o, p0_lo_o = [], []
    for hi_p0 in hi_nd_o:
        p0_hi_o.append(hi_p0["freq_p0"])
    for lo_p0 in lo_nd_o:
        p0_lo_o.append(lo_p0["freq_p0"])

    plt.figure(figsize=(6, 5))
    plt.scatter(h1_lo_a, p0_lo_a, c='blue', marker='+', label='low nb a')
    plt.scatter(h1_hi_a, p0_hi_a, c='red', marker='+', label='high nb a')
    plt.xlabel('H1')
    plt.ylabel('P0')
    plt.legend()
    plt.savefig("h1_p0_a.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(h1_lo_e, p0_lo_e, c='blue', marker='o', label='low nb e')
    plt.scatter(h1_hi_e, p0_hi_e, c='red', marker='o', label='high nb e')
    plt.xlabel('H1')
    plt.ylabel('P0')
    plt.legend()
    plt.savefig("h1_p0_e.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(h1_lo_o, p0_lo_o, c='blue', marker='x', label='low nb o')
    plt.scatter(h1_hi_o, p0_hi_o, c='red', marker='x', label='high nb o')
    plt.xlabel('H1')
    plt.ylabel('P0')
    plt.legend()
    plt.savefig("h1_p0_o.png")

    h1_lo_a_mean = sum(h1_lo_a) / len(h1_lo_a)
    h1_lo_a_sd = np.std(h1_lo_a)

    h1_hi_a_mean = sum(h1_hi_a) / len(h1_hi_a)
    h1_hi_a_sd = np.std(h1_hi_a)

    h1_lo_e_mean = sum(h1_lo_e) / len(h1_lo_e)
    h1_lo_e_sd = np.std(h1_lo_e)

    h1_hi_e_mean = sum(h1_hi_e) / len(h1_hi_e)
    h1_hi_e_sd = np.std(h1_hi_e)

    h1_lo_o_mean = sum(h1_lo_o) / len(h1_lo_o)
    h1_lo_o_sd = np.std(h1_lo_o)

    h1_hi_o_mean = sum(h1_hi_o) / len(h1_hi_o)
    h1_hi_o_sd = np.std(h1_hi_o)

    with open("mean_sd.txt", "a") as f:
        f.write("H1:\n")
        f.write("ɑ̃  low ND:\n")
        f.write(str(h1_lo_a_mean))
        f.write("\n")
        f.write(str(h1_lo_a_sd))
        f.write("\n")
        f.write("ɑ̃  high ND:\n")
        f.write(str(h1_hi_a_mean))
        f.write("\n")
        f.write(str(h1_hi_a_sd))
        f.write("\n")
        f.write("ɛ̃ low ND:\n")
        f.write(str(h1_lo_e_mean))
        f.write("\n")
        f.write(str(h1_lo_e_sd))
        f.write("\n")
        f.write("ɛ̃ high ND:\n")
        f.write(str(h1_hi_e_mean))
        f.write("\n")
        f.write(str(h1_hi_e_sd))
        f.write("ɔ̃ low ND:\n")
        f.write(str(h1_lo_o_mean))
        f.write("\n")
        f.write(str(h1_lo_o_sd))
        f.write("\n")
        f.write("ɔ̃ high ND:\n")
        f.write(str(h1_hi_o_mean))
        f.write("\n")
        f.write(str(h1_hi_o_sd))
        f.write("\n\n")
        f.close()

    t_test_af1, p_val_af1 = stats.ttest_ind(f1_lo_a, f1_hi_a)
    t_test_ef1, p_val_ef1 = stats.ttest_ind(f1_lo_e, f1_hi_e)
    t_test_of1, p_val_of1 = stats.ttest_ind(f1_lo_o, f1_hi_o)

    t_test_af2, p_val_af2 = stats.ttest_ind(f2_lo_a, f2_hi_a)
    t_test_ef2, p_val_ef2 = stats.ttest_ind(f2_lo_e, f2_hi_e)
    t_test_of2, p_val_of2 = stats.ttest_ind(f2_lo_o, f2_hi_o)

    t_test_ah1, p_val_ah1 = stats.ttest_ind(h1_lo_a, h1_hi_a)
    t_test_eh1, p_val_eh1 = stats.ttest_ind(h1_lo_e, h1_hi_e)
    t_test_oh1, p_val_oh1 = stats.ttest_ind(h1_lo_o, h1_hi_o)

    t_tests = {
        "F1": {
            "a":{
                "t_test":t_test_af1,
                "p_value":p_val_af1
            },
            "e":{
                "t_test":t_test_ef1,
                "p_value":p_val_ef1
            },
            "o":{
                "t_test":t_test_of1,
                "p_value":p_val_of1
            }
        },
        "F2":{
            "a":{
                "t_test":t_test_af2,
                "p_value":p_val_af2
            },
            "e":{
                "t_test":t_test_ef2,
                "p_value":p_val_ef2
            },
            "o":{
                "t_test":t_test_of2,
                "p_value":p_val_of2
            }
        },
        "H1":{
            "a":{
                "t_test":t_test_ah1,
                "p_value":p_val_ah1
            },
            "e":{
                "t_test":t_test_eh1,
                "p_value":p_val_eh1
            },
            "o":{
                "t_test":t_test_oh1,
                "p_value":p_val_oh1
            }
        }
    }

    with open("t-test.txt", "w") as f:
        for outer_key, middle_dict in t_tests.items():
            f.write(f"{outer_key}\n")
            for middle_key, inner_dict in middle_dict.items():
                f.write(f"  {middle_key}\n")
                for inner_key, value in inner_dict.items():
                    f.write(f"    {inner_key}: {value}\n")
        f.close()
    return 0
