import pandas as pd

def getdata(filepath: str) -> list[dict]:
    if(filepath == ""):
        return []
    else:
        df = pd.read_csv(filepath, encoding='latin-1')
        values = df.to_dict(orient='records')
        return values

def un_nasal(nasal_conver: dict, data):
    nasals = ["5", "1", "@", "¤"]
    conv = {}
    for word in data:
        if(len(word)>1):
            for nasal in nasals:
                if nasal in word["2_phon"]:
                    conv[word["2_phon"]] = word["2_phon"][0:word["2_phon"].index(nasal)]+nasal_conver[nasal]+word["2_phon"][word["2_phon"].index(nasal)+1:]
        else:
            for nasal in nasals:
                if nasal in word["2_phon"]:
                    conv[word["2_phon"]] = nasal_conver[nasal]
    return conv

#from ChatGPT
def get_min_pair(data,filter):
    return_list = []
    seen = set()

    #re-organize data so that the key is the ipa and the value is the normal dict easy lookup
    phon_to_word = {item["2_phon"]: item for item in data}

    #nasal_phon is key and oral_phon is the converted, .items() gives key and value as different values
    for nasal_phon, oral_phon in filter.items(): 
        if nasal_phon in phon_to_word and oral_phon in phon_to_word:
            #since phone_to_word has the key as the ipa, to be able to add it to return_list, need get original dict
            nasal_word = phon_to_word[nasal_phon]
            oral_word = phon_to_word[oral_phon]

            #prevents reduplication, didn't do this previously, that's why it was 200 in length then adds both to the seen and 
            #return_list
            if nasal_phon not in seen:
                return_list.append(nasal_word)
                seen.add(nasal_phon)
            if oral_phon not in seen:
                return_list.append(oral_word)
                seen.add(oral_phon)

    return return_list

def lexique(data):
    more_than_10 = list(filter(
            lambda d: int(int(d["7_freqlemfilms2"])+int(d["8_freqlemlivres"])+int(d["9_freqfilms2"])+int(d["10_freqlivres"]))>30, data))

    nasals = ["5", "1", "@", "¤"]

    lax_counterpart = un_nasal({"5":"E", "1":"9", "@":"¡", "¤":"O"},more_than_10)
    tense_counterpart = un_nasal({"5":"e", "1":"2", "@":"A", "¤":"o"}, more_than_10)

    nasalV_laxV_min_pair = get_min_pair(more_than_10, lax_counterpart)
    
    nasalV_tenseV_min_pair = get_min_pair(more_than_10, tense_counterpart)
    
    nasalV_laxV_min_pair_csv = pd.DataFrame(nasalV_laxV_min_pair)
    nasalV_laxV_min_pair_csv.to_csv("nasalV_laxV_min_pair.csv")

    nasalV_tenseV_min_pair_csv = pd.DataFrame(nasalV_tenseV_min_pair)
    nasalV_tenseV_min_pair_csv.to_csv("nasalV_tenseV_min_pair.csv")

def main()->int:
    data_lexique = getdata("/Users/lucdenardi/Desktop/python/french_clear_speach/Lexique383/Lexique383.csv")
    lexique(data_lexique)
    return 0

if __name__ == '__main__':
    main()