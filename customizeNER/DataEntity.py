import pandas as pd
import os
from customizeNER.generate_features import *
import spacy
import sys
import getopt

class DataEntity:
    def __init__(self, sentence):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence = sentence
        self.consituency_preditor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
        self.doc = self.nlp(sentence)
        self.extractEntity()

    # get sensitive data items and their phrase
    # input: Vungle and its Demand Partners use Tracking Technologies in order to collect certain Ad Data.
    # output: {'certain Ad Data'}, ['Ad', 'Data']
    def extractEntity(self):
        pharse_list = self.generate_testFile()
        testFile = "tmp.tsv"

        write(testFile, self.consituency_preditor)
        to_filemame = "tmp_feature_v1.tsv"
        self.generate_results(to_filemame)
        sensitive_data = self.getLabeledWord()
        phrase_set = set()
        for data in sensitive_data:
            for phrase in pharse_list:
                p_list = phrase.split(" ")
                if data in p_list:
                    phrase_set.add(phrase)
        # print(phrase_set)
        self.deleteTmpFile()
        self.phrase_set = phrase_set
        self.sensitive_data = sensitive_data
        return phrase_set, sensitive_data

    def generate_testFile(self):
        sentence_list = [token.text for token in self.doc]
        # assign O to each word
        label_list = ["O" for i in range(0, len(sentence_list))]
        data = {}
        data["sentence_list"] = sentence_list
        data["label_list"] = label_list
        df = pd.DataFrame(data)
        df.to_csv("tmp.tsv", sep='\t', index=False, header=False, encoding="utf8", mode='w')

        # Merge the noun phrases
        for phrase in list(self.doc.noun_chunks):
            phrase.merge(phrase.root.tag_, phrase.root.lemma_, phrase.root.ent_type_)
        pharse_list = [phrase.text for phrase in self.doc]
        return pharse_list

    # invoke trained model and get the results
    # parameter: testFile
    def generate_results(self, to_filemame):
        curr = os.getcwd()
        ner_path = "./model"
        os.chdir(ner_path)
        cmd = "java -jar ner.jar " + curr + "/" + to_filemame
        os.system(cmd)
        del_file1 = "rm -rf features-true.txt"
        del_file2 = "rm -rf true"
        os.system(del_file1)
        os.system(del_file2)
        os.chdir(curr)

    # read the result.txt file and return word whose label is SEC
    def getLabeledWord(self):
        # filename = "/Users/huthvincent/Desktop/paper_works/filter_Sentence_Based_Verb/ner_model/result.txt"
        filename = "./model/result.txt"
        f = open(filename)
        content = f.readlines()
        f.close()
        sensitive_data = []
        for row in content:
            item_list = row.split("\t")
            # print("==============" + str(len(item_list)))
            # print(item_list)
            # print("\n")
            if len(item_list) < 3:
                continue
            if "SEC" in item_list[2]:
                sensitive_data.append(item_list[0])
        return sensitive_data

    def deleteTmpFile(self):
        del_file1 = "rm -rf tmp.tsv"
        del_file2 = "rm -rf tmp_feature_v1.tsv"
        # del_file3 = "rm -rf /Users/huthvincent/Desktop/paper_works/filter_Sentence_Based_Verb/ner_model/result.txt"
        del_file3 = "rm -rf ./model/result.txt"
        os.system(del_file1)
        os.system(del_file2)
        os.system(del_file3)




def main():
    inputSentence = "Vungle and its Demand Partners use Tracking Technologies in order to collect certain Ad Data."
    extractData = DataEntity(inputSentence)
    print(extractData.sensitive_data)



if __name__ == "__main__":
    main()