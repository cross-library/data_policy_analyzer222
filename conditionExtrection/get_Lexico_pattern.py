import spacy
import networkx as nx
from conditionExtrection.prunTree import *
import matplotlib.pyplot as plt
import pandas as pd
from allennlp.predictors.predictor import Predictor
from filterSentenceByVerb.get_entities import *

from conditionExtrection.get_clauses import *
from conditionExtrection.pattern import *
import datetime



class LexicoPattern:


    def __init__(self,sentence,policy):
        self.sentence = sentence
        self.policy = policy
        self.is_flag = False
        self.lexicoPattern = ""
        #self.match_pattern_1__2()
        #self.match_pattern_3()
        #self.match_pattern_4()
        #self.match_pattern_6_7()
        #self.match_pattern_8()
        self.match_pattern_9()


    def find_conj(self,parent,res):
        dep_list = ["conj","dobj","prep","pobj","compound"]
        for t in self.policy.doc:
            if t.head == parent and t.dep_ in dep_list:
                res.append(t.text)
                self.find_conj(t, res)


    # 25
    # X, such as Y1, Y2, Y3. . , Yn
    # such X as Y1, Y2, ...Yn
    def match_pattern_1__2(self):
        doc = self.policy.doc
        phrase = self.policy.phrase_set
        for index,token  in enumerate(doc):
            print(token.text + "->" + token.dep_ + "->" +token.head.text )
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
        doc = self.policy.doc
        phrase = self.policy.phrase_set
        for index,token  in enumerate(doc):
            print(token.text + "->" + token.dep_ + "->" +token.head.text )
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
        doc = self.policy.doc
        phrase = self.policy.phrase_set
        for index,token  in enumerate(doc):
            print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index == 0:
                continue
            if "including" in token.text and token.head.text in phrase:
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
        doc = self.policy.doc
        phrase = self.policy.phrase_set
        for index,token  in enumerate(doc):
            print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index == 0:
                continue
            if "especially" in token.text and token.head.text in phrase:
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
    def match_pattern_6_7(self):
        doc = self.policy.doc
        phrase = self.policy.phrase_set
        for index,token  in enumerate(doc):
            print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index == 0:
                continue
            if ("e.g." in token.text or "i.e." in token.text) and doc[index -1].text in phrase:
                content = ""
                match_list = []
                if "e.g." in token.text:
                    content = "(e.g." + self.sentence.split("e.g.")[1].split(")")[0] + ")"
                elif "i.e." in token.text:
                    content = "(i.e." + self.sentence.split("i.e.")[1].split(")")[0] + ")"

                match_list.append(content)
                self.lexicoPattern = match_list



    # X, for example Y1, Y2, . . .Yn
    #
    def match_pattern_8(self):
        doc = self.policy.doc
        phrase = self.policy.phrase_set
        for index,token  in enumerate(doc):
            print(token.text + "->" + token.dep_ + "->" +token.head.text )
            if index <= 2:
                continue
            if "example" in token.text and doc[index-3].text.lower() in phrase :
                self.is_flag = True
                match_list = []
                content = ""

                content = "(for " + self.sentence.split("example")[1].split(")")[0] + ")"

                match_list.append(content)
                self.lexicoPattern = match_list



    # X, which may include Y1, Y2, . . .Yn
    def match_pattern_9(self):
        doc = self.policy.doc
        phrase = self.policy.phrase_set
        for index,token  in enumerate(doc):
            print(token.text + "->" + token.dep_ + "->" +token.head.text )
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


def extract_lexico_pattern(filename):
    sheet = read_data(filename)
    sentence_list = sheet["sentence_list"]
    align_paragraph = sheet["align_paragraph"]
    lexico_pattern_list = []
    for index, sentence in enumerate(sentence_list):
        paragraph = align_paragraph[index]
        policy = policy_verb_entity.predict(sentence, paragraph)
        policy.extractEntity()
        lp = LexicoPattern(sentence,policy)
        lexico_pattern_list.append(lp.lexicoPattern)

    new_sheet = sheet.copy()
    new_sheet["lexico_pattern_list"] = lexico_pattern_list


    to_filemame = "../raw data/40_post_processed_data/apply_Lexicosyntatic_patterns/" + filename.split("v3/")[1]
    new_sheet.to_excel(to_filemame, index=False, encoding="utf8",
                       header=["sentence_list", "align_paragraph", "subject_co_reference", "co_reference_list",
                               "verb_subject_list", "score", "predict_label", "verb_entity_list_without_filter",
                               "verb_entity_list_with_filter", "nmod_entity_list", "all_matched_condition_list",
                               "pattern_condition_list","lexico_pattern_list"])


def get_raw_file():
    xlsx_list = glob.glob('../raw data/40_post_processed_data/assign_condition/v3/*.xlsx')  # get all the filenames of the prepossed 40 toses
    print(u'have found %s xlsx files' % len(xlsx_list))
    print(u'正在处理............')
    return xlsx_list


def main():
    start = datetime.datetime.now()
    xlsx_list = get_raw_file()
    print(xlsx_list)
    for filename in xlsx_list:
        to_filemame = "../raw data/40_post_processed_data/apply_Lexicosyntatic_patterns/" + filename.split("v3/")[1]
        if filename.split("v3/")[1] != "test.xlsx":
            continue
        if os.path.exists(to_filemame):
            continue
        print("===============================" + str(to_filemame))
        extract_lexico_pattern(filename)
    end = datetime.datetime.now()
    print (end-start)


if __name__ == '__main__':

    main()