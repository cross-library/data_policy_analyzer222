# -*- coding: utf-8 -*-

from allennlp.predictors.predictor import Predictor
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
import pandas as pd

from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tree import *


def main():
    filename = "../data/result_v5.tsv"
    predictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
    write(filename, predictor)


def write(filename, predictor):
    sentence = read_sentence(filename)
    for s in sentence:
        sentence_list, label_list = process_sentence(s)
        sen = mergeWords(sentence_list)
        # print(sen)

        #####assign pos#############################################3
        pos_list = []
        # truple = tree2conlltags(ne_chunk(pos_tag(word_tokenize(sen))))
        truple = tree2conlltags(ne_chunk(pos_tag(sentence_list)))
        # the truple contains word, pos, ner-label
        for item in truple:
            pos_list.append(item[1])

        ################get words lemma and stem######################
        wordnet_lemmatizer = WordNetLemmatizer()
        lemma_list = []
        for word in sentence_list:
            lemma_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

        stem_list = []
        lancaster = LancasterStemmer()
        for word in sentence_list:
            stem_list.append(lancaster.stem(word))
        # print(stem_list)

        #####assign consituency parent pos############################
        pos_parent_list, right_sublings_list, chunk_position, left_sublings_list = parse_consituency_tree(sentence_list,
                                                                                                          predictor)
        # print("=========pos===")
        # print(len(sentence_list))
        # print(len(chunk_position))

        #####add space to each sentence############################
        sentence_list.append(" ")
        label_list.append(" ")
        pos_list.append(" ")
        pos_parent_list.append(" ")
        right_sublings_list.append(" ")
        chunk_position.append(" ")
        lemma_list.append(" ")
        stem_list.append(" ")
        left_sublings_list.append(" ")

        data = {}
        data["word"] = sentence_list
        data["label"] = label_list
        data["pos"] = pos_list
        data["chunk"] = pos_list
        data["pos_parent"] = pos_parent_list
        data["right_sublings_list"] = right_sublings_list
        data["chunk_position"] = chunk_position
        data["lemma_list"] = lemma_list
        data["stem_list"] = stem_list
        data["left_sublings_list"] = left_sublings_list
        df = pd.DataFrame(data)

        #####write features to the file############################
        to_file = filename.split(".tsv")[0]
        to_file1 = to_file + "_feature_v1" + ".tsv"
        df.to_csv(to_file1, sep='\t', index=False, header=False, encoding="utf8", mode='a')
        # print(sen)


def predict(sent, predictor):
    results = predictor.predict(sentence=sent)
    return results['trees']


def parse_consituency_tree(sentence_list, predictor):
    pos_parent = []
    right_sublings_list = []
    left_sublings_list = []
    chunk_position = []
    sen = mergeWords(sentence_list)

    '''
    parser = CoreNLPParser(url = "http://localhost:9000")
    parse, = parser.raw_parse(sen)
    '''

    tree = predict(sen, predictor)
    parse = Tree.fromstring(tree)
    # parse.pretty_print()
    newtree = ParentedTree.convert(parse)
    leaf_values = newtree.leaves()
    for i, word in enumerate(sentence_list):
        index = find_closest_words(i, word, leaf_values)
        if index >= 0 and index < len(leaf_values):
            tree_location = newtree.leaf_treeposition(index)
            parent = newtree[tree_location[:-2]].label()
            pos_parent.append(parent)

            #####################find right_sibling###########################
            right_sibling = newtree[tree_location[:-1]].right_sibling()
            # print(right_sibling)
            # print("=================")
            # print("\n")
            if (parent == "NP" or parent == "NX") and right_sibling is not None and calcuate_nodes(right_sibling) == 1:
                count = calcuate_nodes((right_sibling))
                # print(count)
                right_sublings_list.append(right_sibling.leaves()[0])
            else:
                right_sublings_list.append(" ")

            ########################find left_sibling#########################
            left_sibling = newtree[tree_location[:-1]].left_sibling()
            # count = calcuate_nodes((right_sibling))
            if (parent == "NP" or parent == "NX") and left_sibling is not None and calcuate_nodes(left_sibling) == 1:
                count = calcuate_nodes((left_sibling))
                # print(count)
                left_sublings_list.append(left_sibling.leaves()[0])
            else:
                left_sublings_list.append(" ")

            ###########################find chunk item position##########################
            height = newtree[tree_location[:-2]].height()
            chunk_item_list = newtree[tree_location[:-2]].leaves()
            # 只处理最底层的NP tree_height == 3
            if (parent == "NP" or parent == "NX") and height == 3 and len(chunk_item_list) > 1:

                for i, item in enumerate(chunk_item_list):
                    if item == leaf_values[index]:
                        # chunk_position.append(i+1)
                        chunk_position.append(1)
                        break

            else:
                chunk_position.append(0)


        else:
            pos_parent.append("null")
            right_sublings_list.append("null")
            left_sublings_list.append("null")
            chunk_position.append(0)
    return pos_parent, right_sublings_list, chunk_position, left_sublings_list


# determine if word1 is the same with word2
def same_word(word1, word2):
    if word1.strip() == word2.strip() or word1.strip() in word2.strip() or word2.strip() in word1.strip():
        return True
    else:
        return False


# find the index of the leavs. in order to solve the duplicate problems [the twitter ...use twitter api]. there are 2 twitter
def find_closest_words(index, word, leaf_values):
    if leaf_values[index] == word:
        return index

    left_index = index
    right_index = index

    while left_index >= 0 and not same_word(leaf_values[left_index], word):
        left_index -= 1

    while right_index < len(leaf_values) and not same_word(leaf_values[right_index], word):
        right_index += 1

    if (index - left_index) < (right_index - index):
        return left_index
    else:
        return right_index


'''

#change list ["1","2","3"] ->1-2-3
def obtain_dash_list(list):
    str =""
    for item  in list:
        str += item
        str+="-"
    print("====================="+str[:-1])
    return str[:-1]
'''


def calcuate_nodes(tree):
    if not isinstance(tree, Tree):
        return 0

    count = 0
    for t in tree.subtrees():
        count += 1
    return count


def mergeWords(sentence_list):
    sen = ""
    for word in sentence_list:
        sen += word
        sen += " "
    return sen.strip()


def read_sentence(filename):
    # filename="../data/data/result.tsv"
    f = open(filename, encoding="utf8")
    content = f.readlines()
    f.close()
    i = 0
    # print(len(content))

    sentence = []
    i = 0
    while i < len(content):
        # print(i)
        start = i
        end = i
        while end < len(content) and not content[end].isspace():
            end += 1
        sentence.append(content[start:end])
        i = end + 1
        # print(len(sentence))
    return sentence


def process_sentence(sent):
    sentence_list = []
    label_list = []
    for row in sent:
        # print(row.strip().split('\t')[0])
        # print("--------------------")
        # clean data
        word = row.strip().split('\t')[0].strip("\"")
        word = word.replace("(", "")
        if "," not in word:
            word = word.replace(")", ",")
        else:
            word = word.replace(")", "")
        if len(word) != 0 and not word.isspace():
            # print("================")
            # print(word)
            # print("\n")
            try:
                label_list.append(row.strip().split('\t')[1])
                sentence_list.append(word)
            except:
                continue
    return sentence_list, label_list


if __name__ == "__main__":
    main()
