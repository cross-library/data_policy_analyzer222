import matplotlib.pyplot as plt
from conditionExtrection.get_clauses import *
from conditionExtrection.pattern import *
import datetime
from conditionExtrection.get_sentiment import *
from filterSentenceByVerb.extract_policy_statement import *
from conditionExtrection.get_data_condition_pair import *


class PolicyStatement:

    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
    co_refence_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
    nlp = spacy.load("en_core_web_sm")
    standford_nlp = stanza.Pipeline()

    def __init__(self, sentence,paragraph):
        self.sentence = sentence
        self.paragraph = paragraph
        self.graph = None
        self.convert_spacy_tree_to_graph()




    #covert the tree genereated by spacy to graph
    def convert_spacy_tree_to_graph(self):
        doc = PolicyStatement.nlp(self.sentence)
        for phrase in list(doc.noun_chunks):
            phrase.merge(phrase.root.tag_, phrase.root.lemma_, phrase.root.ent_type_)
        G = nx.DiGraph()
        root = None
        for token in doc:
            #print(token.text + " --> " + token.dep_ + "-->" + token.head.text)
            if token == token.head:
                root = token
            G.add_edge(token, token.head, dep=token.dep_)
            ##check if there is condition role
            if self.is_condition(token):
                G.nodes[token]['role'] = "condition"
            else:
                G.nodes[token]['role'] = "other"
        G.remove_edge(root, root)
        self.graph = G
        return G


    def is_condition(self,token):
        conditionList = ["notify","notice","privacy practices","subject","consent","permission","approval", "privacy policy","privacy notice","only","comply","complies","compliance","according","accord","accordance", "if","when","solely"]
        for c in conditionList:
            if c in token.text.lower():
                return True
        return False

    @classmethod
    def is_sentence_condition_pattern_1(self,sentence):
        conditionList = ["consent","permission","approval"]
        for c in conditionList:
            if c in sentence.lower():
                return True
        return False

    @classmethod
    def is_sentence_condition_pattern_2(self,sentence):

        conditionList = ["privacy policy","privacy notice","privacy practices"]
        for c in conditionList:
            if c in sentence.lower():
                return True
        return False

    @classmethod
    def is_sentence_condition_pattern_3(self,sentence):

        conditionList = ["only","solely"]
        for c in conditionList:
            if c in sentence.lower():
                return True
        return False

    @classmethod
    def is_sentence_condition_pattern_4(self,sentence):
        conditionList = ["comply","complies","compliance","subject to"]
        for c in conditionList:
            if c in sentence.lower():
                return True
        return False

    @classmethod
    def is_sentence_condition_pattern_5(self,sentence):
        conditionList = ["according","accord","accordance"]
        for c in conditionList:
            if c in sentence.lower():
                return True
        return False

    @classmethod
    def is_sentence_condition_pattern_6(self,sentence):
        conditionList = ["if ", "when"]
        for c in conditionList:
            if c in sentence.lower():
                return True
        return False

    @classmethod
    def is_sentence_condition_pattern_7(self,sentence):
        conditionList = ["except", "unless","other than", "as long as","agree to"]
        for c in conditionList:
            if c in sentence.lower() and "exceptionally" not in sentence.lower():
                return True
        return False

    @classmethod
    def pattern_7_key_word(self, sentence):
        keyword = ""
        conditionList = ["except", "unless", "other than", "as long as", "agree to","agrees to"]
        for c in conditionList:
            if c in sentence.lower():
                keyword = c
                return keyword
        return keyword

    @classmethod
    def is_sentence_condition_pattern_8(self,sentence):
        conditionList = ["notice","notify"]
        for c in conditionList:
            if c in sentence.lower():
                return True
        return False


    # find condition node
    def getCondition(self):
        for node in self.graph.nodes():
            if self.graph.nodes[node]['role'] == "condition":
                return node
        return None

    #based on data anchor
    def getDataAnchor(self,dataText):
        curr = None
        node_list = self.graph.nodes()
        for index, n in enumerate(node_list) :
            if n.text in dataText or dataText in n.text:
                #print("yueyueyue===" + str(index))
                self.graph.nodes[n]['role'] ="data"
                curr = n
                return curr
        return curr


    # find the verb or the noun modifier of the data anchor
    def getAction(self, dataText):
        verb = getDirectVerb(dataText, self.sentence, self.nlp)
        items = dataText.split()
        nmod=""
        for item in items:
            n = get_word_head(self.sentence, item, self.standford_nlp)
            if n is not None:
                nmod = n

        action = ""
        if verb:
            action = verb
        elif nmod:
            action = nmod

        node_list = self.graph.nodes()
        for index, n in enumerate(node_list) :
            if action in n.text.split():
                if self.graph.nodes[n]['role'] != "condition":
                    self.graph.nodes[n]['role'] ="action"
        return action


    #nodesList is a list of node, this method return subgraph contains those nodes
    def getSubGraph(self,nodesList):
        subGraphList = []
        for nodes in nodesList:
            sub = self.graph.subgraph(nodes).copy()
            subGraphList.append(sub)
        return subGraphList


    def get_most_similar_graph(self, category,maxDepth,pattern):

        if category == "consent" or category == "privacy_policy" or category == "comply" or category == "if" or category == "only" or category == "accord" or category == "notice":
            # find condition node
            condition_node = self.getCondition()
            #print("=====current condition anchor is ===========" + condition_node.text)
            curr = condition_node

        elif  category == "other":
            object = policy_verb_entity.predict(self.sentence, self.paragraph)
            object.extractEntity()
            phrase_set = object.phrase_set
            sensitive_data = object.sensitive_data

            if len(sensitive_data) != 0:
                dataText = sensitive_data[0]
                data_node = self.getDataAnchor(dataText)
                print("=====current data node is ===========" + data_node.text)

                # find action and assign role
                action = self.getAction(data_node.text)
                print("the verb or the modifier of the data is:  " +action)
            else:
                data_node = None


            if data_node is not None:
                curr = data_node
            else:
                curr = self.getCondition()


        # find all sub_graph contains anchor
        res = []

        # if the raw sentences has condition node, thus we need to find sub_graph contains such node
        condition_node = self.getCondition()
        if condition_node is not None:
            require_node = condition_node
        else:
            require_node = curr
        self.search([], [curr], [curr], res, 0, maxDepth, require_node)
        sub_graph_list = self.getSubGraph(res)
        print("===the total subgraph is " + str(len(sub_graph_list)))
        score = []
        sub = []
        match_pattern = []
        count = 0
        flag = False
        for G1 in sub_graph_list:
            # if we have found the smallest subgraph, no need to find any more
            if flag:
                break

            ## if test_graph contains 里面有标condition, but subgraph does not contain, hence continue
            if not self.contains_condition_anchor(G1):
                continue
            count += 1
            print("=================the nodes=======" + str(count))
            print(G1.nodes())
            for G2 in pattern:
                match_pattern.append(G2)
                # score.append(nx.graph_edit_distance(G1, G2, edge_match=PolicyStatement.ematch))
                s =  nx.graph_edit_distance(G1, G2, node_match=PolicyStatement.nmatch, edge_match=PolicyStatement.ematch)
                score.append(s)
                # score.append(nx.graph_edit_distance(G1, G2))
                sub.append(G1)
                if s == 0:
                    flag = True
                    break
        print(score)
        print("-----------------------------")
        #no result, return
        if len(score) == 0:
            return [],[],-1

        min_score = min(score)
        print(min(score))
        ########################stat get all min socre#####################
        min_score_list = []
        for index, s in enumerate(score):
            if s == min_score:
                min_score_list.append(index)
        ########################end get all min socre######################
        print(min_score_list)

        all_matched_condition = []

        for index in min_score_list:
            condition_nodes = self.plot_min_sub(index, sub)
            all_matched_condition.append(condition_nodes)
        p = match_pattern[score.index(min(score))]
        self.plot_match_pattern(min_score_list[0], match_pattern)
        get_match_node = [n for n in p.nodes()]
        return all_matched_condition, get_match_node,min(score)



    def search(self, cur_subgraph, visited, to_visit, res, depth,maxDepth,require_node):
        res_text = []
        for s in res:
            for token in s:
                res_text.append(token.text.lower())

        if require_node.text.lower() in res_text and depth > maxDepth:
            return
        if (cur_subgraph and cur_subgraph not in res):
            res.append(cur_subgraph.copy())
        if (to_visit):
            vertex = to_visit.pop(0)
            # do not select vertex
            self.search(cur_subgraph.copy(), visited.copy(), to_visit.copy(), res,depth+1,maxDepth,require_node)
            # select vertex
            for node in  self.get_neighbour(self.graph,vertex):
                if node not in visited:
                    to_visit.append(node)
                    visited.append(node)
            cur_subgraph.append(vertex)
            self.search(cur_subgraph.copy(), visited.copy(), to_visit.copy(), res,depth+1,maxDepth,require_node)



    #the node will only have one neighbour each time(data --> collect)
    #because tree structure
    def get_neighbour(self, g, currNode):
        neighnour_list =[]
        for n, nbrs in g.adj.items():
            if n == currNode:
                pre_node = g.predecessors(currNode)
                for p in pre_node:
                    neighnour_list.append(p)
                for nbr, eattr in nbrs.items():
                    neighnour_list.append(nbr)
        return list(set(neighnour_list))



    @classmethod
    def ematch(self, e1,e2):
        return e1['dep'] == e2['dep']

    @classmethod
    def nmatch(self, n1,n2):

        if n1['role'] == n2['role']:
            return True
        else:
            return False


    def constructEdgeLabels(self,graph):
        lables = {}
        for n, nbrs in graph.adj.items():
            if n.dep_ == "ROOT":
                continue
            for nbr, eattr in nbrs.items():
                pair = []
                print(n.text +" : " + str(nbr)+ " : "+ str(eattr) )
                pair.append(n)
                pair.append(nbr)
                key = tuple(pair)
                value = eattr['dep']
                lables[key] = value
        return lables


    @classmethod
    def customize_draw(self,graph,nodelist):
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, arrows=True, edge_color='black', width=1, linewidths=1, node_size=500, node_color='pink',alpha=0.9, labels={node: node for node in graph.nodes()})
        edge_label_map = self.constructEdgeLabels(graph)
        nx.draw_networkx_nodes(graph, pos, arrows=True, nodelist=nodelist)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_label_map, font_color='red')
        plt.show()


    def plot_min_sub(self, min_sub_index,sub):

        min_sub = sub[min_sub_index]
        print(min_sub.nodes())
        print(min_sub.edges())
        for n, nbrs in min_sub.adj.items():
            for nbr, eattr in nbrs.items():
                print(n.text + " : " + str(nbr) + " : " + str(eattr))

        node_list = []
        for n in self.graph.nodes():
            for sub_n in min_sub.nodes():
                if n.text == sub_n.text and n.dep_ == sub_n.dep_ and n.head == sub_n.head:
                    node_list.append(n)
        #PolicyStatement.customize_draw(test_graph, node_list)
        return min_sub.nodes()



    def plot_match_pattern(self, min_sub_index, match_pattern):
        p = match_pattern[min_sub_index]
        print("\n")
        print(p.nodes())
        print(p.edges())
        for n, nbrs in p.adj.items():
            for nbr, eattr in nbrs.items():
                print(n.text +" : " + str(nbr)+ " : "+ str(eattr) )
        print("-----------------------------")
        #PolicyStatement.customize_draw(p, p.nodes())
        return p.nodes()


    def contains_condition_anchor(self,G1):
        parent_has_condition = False
        child_has_condition = False
        for n in self.graph.nodes():
            if self.graph.nodes[n]['role'] == "condition":
                parent_has_condition = True
                break
        if parent_has_condition:
            for n in G1.nodes():
                if G1.nodes[n]['role'] == "condition":
                    child_has_condition = True
                    break
        return child_has_condition



# read raw data
def read_data(filename):
    sheet = pd.read_excel(filename)
    return sheet


# to better present the results
def liststoString(all_matched_condition):
    s= ""
    for c in all_matched_condition:
        s +=  str(c)+ "\n"
    return s


def max_length(all_matched_condition):
    max_len = 0
    max_index = -1
    for index, c in enumerate(all_matched_condition):
        if len(c) >= max_len:
            max_len = len(c)
            max_index = index

    candidate = []
    for c in all_matched_condition:
        if len(c) == max_len:
            candidate.append(c)

    string_len_max = 0
    string_len_max_index = -1
    for index , c in enumerate(candidate):
        if len(str(c)) >= string_len_max:
            string_len_max = len(str(c))
            string_len_max_index = index

    return candidate[string_len_max_index]



neg_marks = ["not allow","prohibited","not allowed","neither","have no right","agree not","protect","never do","may not do","will not","protecting","shall not","protected","you may not","in no event"]
neg_pair = [["keep","private"], ["keep","secure"],["keep","secret"],["prevent","unauthorized"],["keep","within your control"],["keep","confidential"]]


#### determine if the v in sentence is negtive
def is_negative(sentence,verb_data):
    ## if sentence has neg marks:
    for neg in neg_marks:
        if neg in sentence.lower():
            return True
    for pair in neg_pair:
        if pair[0] in sentence.lower() and pair[1] in sentence.lower():
            return True
    ## if sentence begin with don't or do not
    s = sentence.lower()
    if sentence.lower().startswith("don't") or sentence.lower().startswith("do not"):
        return True

    verb_list =set()
    lines = str(verb_data).split("\n")
    for line in lines:
        if len(line) == 0:
            continue
        verb_raw = line.split("--->")[0]
        verb_list.add(verb_raw)
    for v in verb_list:
        senti_object = sentiment(sentence,v)
        flag = senti_object.verb_sentiment()
        # 如果有一个 neg 返回
        if not flag :
            return True
    return False



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


def getDataList(verb_data):
    if verb_data == "nan":
        return []
    data_list = set()
    lines = str(verb_data).split("\n")
    for line in lines:
        if len(line) == 0 or line == "nan":
            continue
        data_raw = line.split("--->")[1].strip()
        data_list.add(data_raw)
    return data_list



def extract_statement(filename,pattern_map):

    sheet = read_data(filename)

    sentence_list = sheet["sentence_list"]
    all_matched_condition_list = []
    pattern_condition_list = []
    score_list=[]

    for index, sentence in enumerate(sentence_list):
        ##############################################obtain_condition##################################################
        ## 1. deteminie if predict label  is 1, if not, continue
        pre_label = sheet.loc[index,"predict_label"]
        if pre_label == 0:
            all_matched_condition_list.append("")
            pattern_condition_list.append("")
            score_list.append("")
            continue

        ## 2. if sentence is a policy statement, we need to extract condition
        paragraph = sheet.loc[index, "sentence_list"]
        verb_data = sheet.loc[index, "verb_entity_list_with_filter"]
        all_matched_condition, pattern_condition,score = get_condition(sentence,paragraph,pattern_map,verb_data)

        ### 3. when condition is none, determine the neg verbs
        if len(all_matched_condition) == 0:
            if is_negative(sentence,verb_data):
                all_matched_condition_list.append("neg")
                pattern_condition_list.append("neg")
                score_list.append(0)
            else:
                all_matched_condition_list.append("")
                pattern_condition_list.append("")
                score_list.append("")
        else:
            all_matched_condition_list.append(str(max_length(all_matched_condition)))
            pattern_condition_list.append(str(pattern_condition))
            score_list.append(score)



    new_sheet = sheet.copy()
    new_sheet["all_matched_condition_list"] = all_matched_condition_list
    new_sheet["pattern_condition_list"] = pattern_condition_list
    new_sheet["score_list"] = score_list
    to_filemame = "./data/outputSentence_condition.xlsx"
    new_sheet.to_excel(to_filemame, index=False, encoding="utf8",
                header=["sentence_list",  "subject_co_reference", "co_reference_list",
                        "verb_subject_list", "predict_label", "verb_entity_list_without_filter",
                        "verb_entity_list_with_filter", "nmod_entity_list","all_matched_condition_list","pattern_condition_list","score_list"])

    expend_pair(to_filemame)


def clean_except_condition(clause):
    p =[]
    for t in PolicyStatement.nlp(clause):
        if len(p) >= 18:
            break
        if t.text == ":" or t.text == "," or t.text == ";":
            break
        if t.text == " ":
            continue
        p.append(t.text)
    return p


def cut_if_condition(new_condition_token):
    p =[]
    for t in new_condition_token:
        if len(p) >= 18:
            break
        if t == ":" or t == "," or t == ";":
            break
        if t == " ":
            continue
        p.append(t)
    return p



def get_condition(sentence,paragraph,pattern_map,verb_data):
    all_matched_condition = []
    pattern_condition = []
    score = None
    verb_list = getVerbList(verb_data)
    data_list = getDataList(verb_data)

    ### if sentence match pattern 1
    if PolicyStatement.is_sentence_condition_pattern_1(sentence):
        pattern = pattern_map["consent"]
        clauses = split_clauses(PolicyStatement.predictor, sentence, 1)
        try:
            target_clause = [c for c in clauses if PolicyStatement.is_sentence_condition_pattern_1(c)][0]
            print("the target clause is :  " + target_clause)
            policy_state_object = PolicyStatement(target_clause,paragraph)
        except:
            policy_state_object = PolicyStatement(sentence, paragraph)
        all_matched_condition, pattern_condition ,score= policy_state_object.get_most_similar_graph("consent",3,pattern)
        print("all matched condition is " +str(all_matched_condition))
        print("pattern_condition " + str(pattern_condition))


    elif PolicyStatement.is_sentence_condition_pattern_7(sentence):
        clauses = split_clauses(PolicyStatement.predictor, sentence, 1)
        try:

            target_clause = [c for c in clauses if PolicyStatement.is_sentence_condition_pattern_7(c)][0]
            keyword = PolicyStatement.pattern_7_key_word(target_clause)
            index = clauses.index(target_clause)
            sub = target_clause.lower().split(keyword)[1]
            if len(sub) != 0:
                key_clause =keyword+ " " + target_clause.split(keyword)[1]
                p = clean_except_condition(key_clause)
                all_matched_condition.append(p)
            else:
                next_clause = clauses[index+1]
                p=[]
                key_clause = keyword + " " + next_clause
                p = clean_except_condition(key_clause)
                #p = [t.text for t in PolicyStatement.nlp(key_clause)]
                all_matched_condition.append(p)

        except:
            keyword = PolicyStatement.pattern_7_key_word(sentence)
            key_sentence = keyword + " "+sentence.lower().split(keyword)[1]
            ### clean the condition to be concise
            p = clean_except_condition(key_sentence)
            all_matched_condition.append(p)
        pattern_condition = ["Except", "to",  "you","have","a separate agreement"]
        score = -1


    ### if sentence match pattern 2
    elif PolicyStatement.is_sentence_condition_pattern_2(sentence):
        pattern = pattern_map["privacy_policy"]
        clauses = split_clauses(PolicyStatement.predictor, sentence, 1)
        try:
            target_clause = [c for c in clauses if PolicyStatement.is_sentence_condition_pattern_2(c)][0]
            print("the target clause is :  " + target_clause)
            policy_state_object = PolicyStatement(target_clause, paragraph)
        except:
            policy_state_object = PolicyStatement(sentence, paragraph)
        all_matched_condition, pattern_condition,score = policy_state_object.get_most_similar_graph("privacy_policy",5,pattern)
        print("all matched condition is " + str(all_matched_condition))
        print("pattern_condition " + str(pattern_condition))


    ### if sentence match pattern 3
    elif PolicyStatement.is_sentence_condition_pattern_3(sentence):
        pattern = pattern_map["only"]
        clauses = split_clauses(PolicyStatement.predictor, sentence, 0)
        try:
            target_clause = [c for c in clauses if PolicyStatement.is_sentence_condition_pattern_3(c)][0]
            print("the target clause is :  " + target_clause)
            policy_state_object = PolicyStatement(target_clause, paragraph)
        except:
            policy_state_object = PolicyStatement(sentence, paragraph)
        all_matched_condition_0, pattern_condition,score= policy_state_object.get_most_similar_graph( "only", 8,pattern)
        all_matched_condition = clean_all_matched_condition(all_matched_condition_0, verb_list, data_list)
        print("all matched condition is " + str(all_matched_condition))
        print("pattern_condition " + str(pattern_condition))

    ### if sentence match pattern 4
    elif PolicyStatement.is_sentence_condition_pattern_4(sentence):
        pattern = pattern_map["comply"]
        clauses = split_clauses(PolicyStatement.predictor, sentence, 1)
        try:
            target_clause = [c for c in clauses if PolicyStatement.is_sentence_condition_pattern_4(c)][0]
            print("the target clause is :  " + target_clause)
            policy_state_object = PolicyStatement(target_clause, paragraph)
        except:
            policy_state_object = PolicyStatement(sentence, paragraph)
        all_matched_condition, pattern_condition,score= policy_state_object.get_most_similar_graph("comply", 7, pattern)
        print("all matched condition is " + str(all_matched_condition))
        print("pattern_condition " + str(pattern_condition))


    ### if sentence match pattern 5
    elif PolicyStatement.is_sentence_condition_pattern_5(sentence):
        pattern = pattern_map["accord"]
        clauses = split_clauses(PolicyStatement.predictor, sentence, 0)
        try:
            target_clause = [c for c in clauses if PolicyStatement.is_sentence_condition_pattern_5(c)][0]
            print("the target clause is :  " + target_clause)
            policy_state_object = PolicyStatement(target_clause, paragraph)
        except:
            policy_state_object = PolicyStatement(sentence, paragraph)
        all_matched_condition_0, pattern_condition,score = policy_state_object.get_most_similar_graph( "accord",7,pattern)
        ## remove all the verb and data type node

        all_matched_condition = clean_all_matched_condition(all_matched_condition_0,verb_list,data_list)
        print("all matched condition is " + str(all_matched_condition))
        print("pattern_condition " + str(pattern_condition))

    ### if sentence match pattern 6
    elif PolicyStatement.is_sentence_condition_pattern_6(sentence):
        pattern = pattern_map["if"]
        clauses = split_clauses(PolicyStatement.predictor, sentence, 0)
        try:
            target_clause = [c for c in clauses if PolicyStatement.is_sentence_condition_pattern_6(c)][0]
            print("the target clause is :  " + target_clause)
            policy_state_object = PolicyStatement(target_clause, paragraph)
        except:
            policy_state_object = PolicyStatement(sentence, paragraph)
        all_matched_condition_0, pattern_condition ,score= policy_state_object.get_most_similar_graph( "if", 4,pattern)
        #### 补全if 条件句
        all_matched_condition = []
        new_condition = clean_if_Condition(sentence, PolicyStatement.predictor)
        new_condition_token = new_condition.split(" ")
        all_matched_condition.append(cut_if_condition(new_condition_token))
        print("all matched condition is " + str(all_matched_condition))
        print("pattern_condition " + str(pattern_condition))

    ### if sentence match pattern 8
    elif PolicyStatement.is_sentence_condition_pattern_8(sentence):
        pattern = pattern_map["notice"]
        clauses = split_clauses(PolicyStatement.predictor, sentence, 0)
        try:
            target_clause = [c for c in clauses if PolicyStatement.is_sentence_condition_pattern_8(c)][0]
            print("the target clause is :  " + target_clause)
            policy_state_object = PolicyStatement(target_clause, paragraph)
        except:
            policy_state_object = PolicyStatement(sentence, paragraph)
        all_matched_condition, pattern_condition,score= policy_state_object.get_most_similar_graph( "notice", 5,pattern)
        print("all matched condition is " + str(all_matched_condition))
        print("pattern_condition " + str(pattern_condition))


    return all_matched_condition, pattern_condition,score




### 补全if条件句
def clean_if_Condition(sentence,predictor):
    clauses = split_clauses(predictor, sentence,1)
    for index, c in enumerate(clauses):

        if ("when" in c.lower() or "if" in c.lower()) and len(c) >8 and (c.lower().startswith("when") or c.lower().startswith("if")):
            print(c)
            return c
        elif ("when" in c.lower() or "if" in c.lower()) :

            try:
                print(c + " " + clauses[index + 1])
                return c + " " + clauses[index+1]
            except:
                return ""
    return ""


## delete data and action nodes
def clean_all_matched_condition(all_matched_condition,verb_list,data_list):
    all_matched_condition_new =[]
    for condition_list in all_matched_condition:
        new_condition_list = [node for node in condition_list if node.text not in verb_list and node.text not in data_list]
    all_matched_condition_new.append(new_condition_list)
    return all_matched_condition_new


def main():
    pattern = Pattern()
    pattern_map = pattern.map
    filename = "./data/outputSentence_policy.xlsx"
    extract_statement(filename,pattern_map)



if __name__ == '__main__':
    main()
