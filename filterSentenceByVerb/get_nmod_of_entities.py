# -*- coding: utf-8 -*-
import glob
import os
import json
import csv
import spacy
import os
from allennlp.predictors.predictor import Predictor
import numpy as np
import nltk
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree import ParentedTree
from nltk.tree import *
import stanza


def main():
    '''
    nlp = stanza.Pipeline()
    doc = nlp("Your use of Google user data must be limited to the practices explicitly disclosed in your private policy.")
    doc.sentences[0].print_dependencies()
    for sentence in doc.sentences:
        for word in sentence.words:
            print()
    '''
    nlp = stanza.Pipeline()
    sentence = "Your use of Google user data must be limited to the practices explicitly disclosed in your private policy."
    sensitive_word = "data"
    print(get_word_head(sentence, sensitive_word, nlp))


# return the head word, head.deprel
def get_word_head(sentence, sensitive_word, nlp):
    # print("get================")
    # print(sensitive_word)
    doc = nlp(sentence)
    for sent in doc.sentences:
        for word in sent.words:
            if word.text == sensitive_word and word.deprel == "nmod":
                return sent.words[word.head - 1].text

    # print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')


if __name__ == "__main__":
    main()