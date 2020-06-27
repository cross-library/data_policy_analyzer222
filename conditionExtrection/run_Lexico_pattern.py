import spacy
import networkx as nx
from conditionExtrection.prunTree import *
import matplotlib.pyplot as plt
import pandas as pd
from allennlp.predictors.predictor import Predictor

import datetime


nlp = spacy.load("en_core_web_sm")
class LexicoPattern:


    def __init__(self,sentence,phrase_set):
        self.sentence = sentence
        self.doc = nlp(sentence)
        for phrase in list(self.doc.noun_chunks):
            phrase.merge(phrase.root.tag_, phrase.root.lemma_, phrase.root.ent_type_)
        self.phrase_set = phrase_set
        self.is_flag = False
        self.lexicoPattern = ""
        self.match_pattern_1__2()
        self.match_pattern_3()
        self.match_pattern_4()
        self.match_pattern_6_7()
        self.match_pattern_8()
        self.match_pattern_9()


    def find_conj(self,parent,res):
        dep_list = ["conj","dobj","compound","appos"]
        for t in self.doc:
            if t.head == parent and t.dep_ in dep_list:
                res.append(t.text)
                self.find_conj(t, res)


    # 25
    # X, such as Y1, Y2, Y3. . , Yn
    # such X as Y1, Y2, ...Yn
    def match_pattern_1__2(self):
        doc = self.doc
        phrase = self.phrase_set
        for index,token  in enumerate(doc):

            #print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index == 0:
                continue
            if token.text == "as" and token.head.text in phrase and "such" in doc[index-1].text.lower():
                self.is_flag = True
                match_list = []
                match_list.append(token.head.text)
                match_list.append("-->")
                next_word_list = [t for t in doc if t.head.text == "as" and t.dep_ == "pobj"]
                if len(next_word_list) == 0:
                    continue
                next_word = next_word_list[0]
                res = []
                res.append(next_word.text)
                self.find_conj(next_word, res)
                match_list.extend(list(set(res)))
                self.lexicoPattern = match_list



    # 101
    # "X [and|or]other Y1, Y2, . . .Yn"
    def match_pattern_3(self):
        doc = self.doc
        phrase = self.phrase_set
        for index,token  in enumerate(doc):
            #print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index == 0:
                continue
            if "other" in token.text and token.head.text in phrase and (doc[index-1].text.lower() == "and" or doc[index-1].text.lower() == "or"):
                self.is_flag = True
                match_list = []
                match_list.append(token.head.text)
                match_list.append("-->")
                next_word_list = [t for t in doc if t.head.text == token.head.text and t.dep_ == "conj"]
                if len(next_word_list) == 0:
                    continue
                next_word = next_word_list[0]
                res = []
                res.append(next_word.text)
                self.find_conj(next_word, res)
                match_list.extend(list(set(res)))
                self.lexicoPattern = match_list


    # 93
    # X, including Y1, Y2, . . .Yn
    def match_pattern_4(self):
        doc = self.doc
        phrase = self.phrase_set
        for index,token  in enumerate(doc):
            #print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index == 0:
                continue
            if "including" in token.text and (token.head.text in phrase or token.head.head.text in phrase):
                self.is_flag = True
                match_list = []
                match_list.append(token.head.text)
                match_list.append("-->")
                next_word_list = [t for t in doc if t.head == token and (t.dep_ == "pobj" or t.dep_ =="conj")]
                if len(next_word_list) == 0:
                    continue
                next_word = next_word_list[0]
                res = []
                res.append(next_word.text)
                self.find_conj(next_word, res)
                match_list.extend(list(set(res)))
                self.lexicoPattern = match_list

    # 0
    # X, especially Y1, Y2, . . .Yn
    def match_pattern_5(self):
        doc = self.doc
        phrase = self.phrase_set
        for index,token  in enumerate(doc):
            #print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index == 0:
                continue
            if "especially" in token.text and  (token.head.text in phrase or token.head.head.text in phrase):
                self.is_flag = True
                match_list = []
                match_list.append(token.head.text)
                match_list.append("-->")
                next_word_list = [t for t in doc if t.head == token and t.dep_ == "conj"]
                if len(next_word_list) == 0:
                    continue
                next_word = next_word_list[0]
                res = []
                res.append(next_word.text)
                self.find_conj(next_word, res)
                match_list.extend(list(set(res)))
                self.lexicoPattern = match_list


    # X, [eg|ie] Y1, Y2, . . .Yn
    # X, ([eg|ie]) Y1, Y2, . . .Yn
    # 16

    def determine_target_data(self,index, phrase):
        if self.doc[index - 1].text in phrase:
            return True
        if self.doc[index - 2].text in phrase:
            return True
        if index - 3 >= 0 and self.doc[index - 3].text in phrase:
            return True
        if index - 4 >= 0 and self.doc[index - 4].text in phrase:
            return True
        return False


    def match_pattern_6_7(self):
        doc = self.doc
        phrase = self.phrase_set
        for index,token  in enumerate(doc):
            #print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index == 0:
                continue
            if ("e.g." in token.text or "i.e." in token.text) and self.determine_target_data(index, phrase):
                content = ""
                match_list = []
                match_list.append(token.head.text)
                match_list.append("-->")
                if "e.g." in token.text:
                    content = "(e.g." + self.sentence.split("e.g.")[1].split(")")[0] + ")"
                elif "i.e." in token.text:
                    content = "(i.e." + self.sentence.split("i.e.")[1].split(")")[0] + ")"

                while index < len(doc) - 1 and not doc[index + 1].text.isalpha():
                    index += 1
                next_word = doc[index+1]
                res = []
                res.append(next_word.text)
                self.find_conj(next_word, res)
                match_list.extend(list(set(res)))
                self.lexicoPattern = match_list



    # X, for example Y1, Y2, . . .Yn
    #
    def match_pattern_8(self):
        doc = self.doc
        phrase = self.phrase_set
        #print(phrase)
        for index,token  in enumerate(doc):
            #print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index <= 2:
                continue
            is_target = False
            for p in phrase:
                if token.head.text.lower() in p or token.head.head.text.lower() in p or token.head.head.head.text.lower() in p:
                    is_target = True
            #if "example" in token.text and doc[index-3].text.lower() in phrase :

            if "example" in token.text and is_target:
                self.is_flag = True
                match_list = []
                match_list.append(token.head.text)
                match_list.append("-->")

                while index < len(doc) - 1 and not doc[index + 1].text.isalpha():
                    index += 1
                next_word = doc[index + 1]
                res = []
                res.append(next_word.text)
                self.find_conj(next_word, res)
                match_list.extend(list(set(res)))
                self.lexicoPattern = match_list



    # X, which may include Y1, Y2, . . .Yn
    def match_pattern_9(self):
        doc = self.doc
        phrase = self.phrase_set
        for index,token  in enumerate(doc):
            #print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index == 0:
                continue
            if "which" in token.text and doc[index-2].text in phrase and doc[index + 1].text == "may" and doc[index + 2].text == "include":
                self.is_flag = True
                match_list = []
                match_list.append(token.head.text)
                match_list.append("-->")
                next_word_list = [t for t in doc if t.head == doc[index + 2] and  t.dep_ =="dobj"]
                if len(next_word_list) == 0:
                    continue
                next_word = next_word_list[0]
                res = []
                res.append(next_word.text)
                self.find_conj(next_word, res)
                match_list.extend(list(set(res)))
                self.lexicoPattern = match_list


# read raw data
def read_data(filename):
    sheet = pd.read_excel(filename)
    return sheet

def getDataSet(raw):
    data_set = []
    lines = str(raw).split("\n")
    for line in lines:
        item = line.split("--->")[-1]
        data_set.append(item.strip())
    #print(data_set)
    return data_set


def extract_lexico_pattern(filename):
    sheet = read_data(filename)
    sentence_list = sheet["sentence_list"]
    verb_entity_list_with_filter = sheet["verb_entity_list_with_filter"].tolist()
    lexico_pattern_list = []
    for index, sentence in enumerate(sentence_list):
        raw = verb_entity_list_with_filter[index]
        phase_set = getDataSet(raw)
        lp = LexicoPattern(sentence,phase_set)
        lexico_pattern_list.append(lp.lexicoPattern)

    new_sheet = sheet.copy()
    new_sheet["lexico_pattern_list"] = lexico_pattern_list


    to_filemame = "clean_v3.xlsx"
    new_sheet.to_excel(to_filemame, index=False, encoding="utf8",
                       header=["sentence_list",
                               "verb_entity_list_with_filter", "nmod_entity_list", "all_matched_condition_list",
                               "pattern_condition_list","score_list", "count_data_object","lexico_pattern_list"])





def main():

    sentence = "The advertising identifier must not be connected to personally-identifiable information or associated with any persistent device identifier (for example: SSAID, MAC address, IMEI, etc.) without explicit consent of the user."
    lp = LexicoPattern(sentence, ["any persistent device identifier"])

    #sentence = "Don’t combine multiple end-advertisers or their Facebook connections (i.e. Pages) in the same ad account."
    #lp = LexicoPattern(sentence, ["their Facebook connections"])
    print(lp.lexicoPattern)



if __name__ == '__main__':

    main()