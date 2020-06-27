import os
import pandas as pd
import glob
import spacy
from conditionExtrection.run_Lexico_pattern import *
from filterSentenceByVerb.get_verb_entities import  *
def getDataList(verb_data):
    if verb_data == "nan":
        return []
    data_list = set()
    lines = str(verb_data).split("\n")
    for line in lines:
        if len(line) == 0 or line == "nan":
            continue
        data_raw = line.split("--->")[-1].strip()
        data_list.add(data_raw)
    return data_list


def getVerbList(verb_data):
    if verb_data == "nan":
        return []
    verb_list = set()
    lines = str(verb_data).split("\n")
    for line in lines:
        if len(line) == 0 or line == "nan":
            continue
        verb_raw = line.split("--->")[0]
        verb_list.add(verb_raw)
    return verb_list




def propagates(data_token,doc,total_data):
    if data_token.dep_ == "conj" and  policy_verb_entity.valiateData(data_token.head):
        total_data.add(data_token.head)
        propagates(data_token.head, doc,total_data)
    else:
        return


nlp = spacy.load("en_core_web_sm")
def expend_pair(filename):
    data = {}
    new_sentence_list = []
    new_data_list = []
    new_condition_list = []
    new_description_list=[]

    sheet = pd.read_excel(filename)
    sentence_list = sheet["sentence_list"].tolist()

    for index, sentence in enumerate(sentence_list):

        #### 1.filter out sentence unrelated to data policies
        condition = sheet.loc[index,"all_matched_condition_list"]
        score = sheet.loc[index,"score_list"]
        if pd.isnull(sheet.loc[index, "all_matched_condition_list"]):
            continue
        if score >= 4:
            continue

        ### determine if the verb is "combine","associate","connect"
        verb_nmod_data = set()
        verb_data =  sheet.loc[index,"verb_entity_list_with_filter"]
        nmod_data =  sheet.loc[index,"nmod_entity_list"]
        verb_data_list = getDataList(verb_data)
        nmod_data_list = getDataList(nmod_data)
        if len(verb_data_list) != 0:
            verb_nmod_data = verb_data_list
        else:
            verb_nmod_data = nmod_data_list

        verb_list = getVerbList(verb_data)
        verb_filter = ["combin","associat","connect"]
        flag = False
        for v1 in verb_filter:
            if flag:
                break
            for v2 in verb_list:
                if v1 in v2 :
                    flag = True
                    break
        if flag:
            note = "and"
            new_sentence_list.append(sentence)
            new_data_list.append(str(verb_data))
            new_condition_list.append(condition)
            new_description_list.append(note)
            continue



        #### 3.expand the data object:
        total_data = set()
        doc = nlp(sentence)
        for phrase in list(doc.noun_chunks):
            phrase.merge(phrase.root.tag_, phrase.root.lemma_, phrase.root.ent_type_)
        ### 3. find all conjuntive data item
        for d in verb_nmod_data:
            for token in doc:
                if token.text == d:
                    total_data.add(token)
                    propagates(token, doc, total_data)



        ### 4. find all lexico patterns
        expand_lexico_data = set()
        key_data = [d.text for d in total_data]
        lp = LexicoPattern(sentence,key_data)
        if len(lp.lexicoPattern) != 0:
            for d in lp.lexicoPattern[2:]:
                if d not in key_data:
                    token = findToken(doc, d)
                    if policy_verb_entity.valiateData(token):
                        expand_lexico_data.add(token.text)



        ### 5. condtruct new sheet
        note = ""
        for d in total_data:
            if d.text in verb_nmod_data:
                note = "org"
            else:
                note = "conjuntive"

            new_sentence_list.append(sentence)
            new_data_list.append(d)
            new_condition_list.append(condition)

            new_description_list.append(note)


        for d in expand_lexico_data:
            note = "lexico"

            new_sentence_list.append(sentence)
            new_data_list.append(d)
            new_condition_list.append(condition)

            new_description_list.append(note)


    data["sentence_list"] =new_sentence_list
    data["data"] =new_data_list
    data["condition"] =new_condition_list
    data["description"] =new_description_list

    df = pd.DataFrame(data)
    to_filemame = "./data/condition_data_pair.xlsx"
    df.to_excel(to_filemame, index=False, encoding="utf8",
                header=["sentence_list", "data", "condition", "description"])



def findToken(doc, data):
    for t in doc:
        if t.text == data:
            return t
    return None

def main():
    filename="./data/outputSentence_condition.xlsx"
    expend_pair(filename)







if __name__ == "__main__":
    main()
