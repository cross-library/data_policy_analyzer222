global str
from ete3 import Tree
import spacy


def getChidren(parent,doc):
    children = []
    for token in doc:
        if token == parent:
            continue
        if token.head == parent:
            children.append(token)
    return children


def generate_tree(node,doc):
    children = getChidren(node.name,doc)
    if children == None:
        return
    for child in children:
        child_node = node.add_child(name = child)
        child_node.add_features(dep=child.dep_)
        generate_tree(child_node,doc)




def getTree(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    for phrase in list(doc.noun_chunks):
        phrase.merge(phrase.root.tag_, phrase.root.lemma_, phrase.root.ent_type_)
    dep_tree = Tree()
    '''
    for token in doc:
        print(token.text + " || " + token.dep_ + " || " + token.head.text)
    print("---------------------------------------------")
    '''

    for token in doc:
        if token.dep_ == "ROOT":
            root = dep_tree.add_child(name=token)
            root.add_features(dep=token.dep_)
            generate_tree(root, doc)

    print (dep_tree.get_ascii(attributes=["name","dep"]))
    # f = open("text.txt","w")
    # f.write(dep_tree.get_ascii(attributes=["name","dep"]))
    # f.write(dep_tree.get_ascii(attributes=["name"]))
    # f.close()
    # print("\n")
    #print(dep_tree.write(format=8))
    #print(dep_tree.get_ascii(attributes=["name","dep"]))
    # print("\n")
    return dep_tree


def getEntity(sentence):
    nlp = spacy.load("./model")
    doc = nlp(sentence)
    data = set()
    for ent in doc.ents:
        #print("=======")
        #print("entity is :  " +ent.text )
        data.add(ent.text)
    return data


#根据data找node
def getNode(data,dep_tree):
    node_set = []
    for node in dep_tree.iter_descendants("postorder"):
        if data in node.name.text:
            node_set.append(node)
    return node_set


def pruneTree(sentence):
    condition = ["privacy", "policy","consent","notice","against","access", "use", "disclosure","contract","permission","any user data"]
    plus_data_keywords = ["has","End User","End User Credentials","Google API Services","pins","user ids","browser settings","data","credentials","control","unless","media","or","any user data","the Google APIs Terms","surveillance"]


    dep_tree = getTree(sentence)
    # print (dep_tree.get_ascii(attributes=["name"]))
    # 找出所有要的节点的name(spacy.token)
    total_need_nodes = set()
    # rule 0 从data 到root
    entity_list = getEntity(sentence)
    for data in entity_list:
        node_set = getNode(data,dep_tree)
        for node in node_set:
            total_need_nodes.add(node)
            #根据node往前找直到找到root0
            while node.up.dep != "ROOT":
                node = node.up
                #print("000============node is  " +node.name.text)
                total_need_nodes.add(node)

    for data in plus_data_keywords:
        node_set = getNode(data,dep_tree)
        for node in node_set:
            total_need_nodes.add(node)
            #根据node往前找直到找到root00
            while node.up.dep != "ROOT":
                node = node.up
                total_need_nodes.add(node)

    for data in plus_data_keywords:
        node_set = getNode(data,dep_tree)
        for node in node_set:
            total_need_nodes.add(node)
            #根据node往前找直到找到root00
            while node.up.dep != "ROOT":
                node = node.up
                total_need_nodes.add(node)


    # rule 1 含有关键字
    for keyword in condition:
        node_set = getNode(keyword,dep_tree)
        for node in node_set:
            total_need_nodes.add(node)
            #根据node往前找直到找到root1
            while not node.up.is_root() and node.up.dep != "ROOT":
                node = node.up
                #print("============node is  " +node.name.text)
                total_need_nodes.add(node)


    # rule 2 情态动词 or #rule 3 否定词
    aux = []
    filter_list = ["must","shall","do"]
    for node in dep_tree.iter_descendants("postorder"):
        if node.dep =="neg":
            aux.append(node)
        if node.dep == "aux" and node.name.text in filter_list:
            aux.append(node)

    for node in aux:
        total_need_nodes.add(node)
        #根据node往前找直到找到root2
        while node.up.dep != "ROOT":
            node = node.up
            total_need_nodes.add(node)

    for node in dep_tree.iter_descendants("postorder"):
        if node.dep == "ROOT":
            continue
        if node not in total_need_nodes:
            #print("detach node is " + node.name.text)
            node.detach()


    ###############################################################--01--########################################################################
    if sentence == "You may not distribute more than 1,500,000 Tweet IDs to any entity (inclusive of multiple individual users associated with a single entity) within any given 30 day period, unless you are doing so on behalf of an academic institution and for the sole purpose of non-commercial research or you have received the express written permission of Twitter.":
        for node in dep_tree.iter_descendants("postorder"):
            #print("name is :" + node.name.text)
            if node.name.text == "to" or node.name.text == "not" or node.name.text == "inclusive" or node.name.text == "within" or node.name.text == "unless" or node.name.text == "for" or node.name.head.text == "the express written permission" or  node.name.text == ".":
                node.detach()
        '''
        for node in dep_tree.iter_descendants("postorder"):
            print("name is :" + node.name.text)
        '''

        #add role to each word
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "more than 1,500,000 Tweet IDs":
                node.add_features(role="data")
            '''
            if node.name.text == "an academic institution":
                node.add_features(role="condition")
            if node.name.text == "of":
                node.add_features(role="condition")
            if node.name.text == "behalf":
                node.add_features(role="condition")
            if node.name.text == "on":
                node.add_features(role="condition")
            if node.name.text == "or":
                node.add_features(role="condition")
            if node.name.text == "the express written permission":
                node.add_features(role="condition")
            if node.name.text == "received":
                node.add_features(role="condition")
            if node.name.text == "doing":
                node.add_features(role="other")
            '''
            if node.name.text == "distribute":
                node.add_features(role="action")

        #print(dep_tree.get_ascii(attributes=["name","role"]))




    ###############################################################--02--########################################################################
    if sentence == "You must access Google API Services in accordance with the Google APIs Terms of Service.":
        for node in dep_tree.iter_descendants("postorder"):
            #print("name is :" + node.name.text)
            if node.name.text == "must":
                node.detach()
        '''
        print(dep_tree.get_ascii(attributes=["name"]))
        print("\n")

        for node in dep_tree.iter_descendants("postorder"):
            print("name is :" + node.name.text)
        '''
        #add role to each word

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "Google API Services":
                node.add_features(role="data")
            '''
            if node.name.text == "the Google APIs Terms":
                node.add_features(role="condition")
            if node.name.text == "with":
                node.add_features(role="condition")
            if node.name.text == "accordance":
                node.add_features(role="condition")
            if node.name.text == "in":
                node.add_features(role="condition")
            '''
            if node.name.text == "access":
                node.add_features(role="action")

        #print(dep_tree.get_ascii(attributes=["name","role"]))


    ###############################################################--03--########################################################################
    if sentence == "you agree not to engage in any of the following prohibited activities: collect or store any personally identifiable information from the services from other users of the services without their express permission;":

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "engage" or node.name.text == "not" or  node.name.text == "other users" or  node.name.text == "from" :
                node.detach()
        '''
        print(dep_tree.get_ascii(attributes=["name"]))
        print("\n")

        for node in dep_tree.iter_descendants("postorder"):
            print("name is :" + node.name.text)
        '''
        #add role to each word
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "any personally identifiable information":
                node.add_features(role="data")
            if node.name.text == "collect":
                node.add_features(role="action")
            if node.name.text == "store":
                node.add_features(role="action")
            '''
            if node.name.text == "or":
                node.add_features(role="action")
            
            if node.name.text == "their express permission":
                node.add_features(role="condition")
            if node.name.text == "without":
                node.add_features(role="condition")
            if node.name.text == "agree":
                node.add_features(role="other")
            '''
        #print(dep_tree.get_ascii(attributes=["name","role"]))



    ###############################################################--04--########################################################################
    if sentence=="The advertising identifier must not be connected to personally-identifiable information or associated with any persistent device identifier (for example: SSAID, MAC address, IMEI) without explicit consent of the user":
        for node in dep_tree.iter_descendants("postorder"):
            #print("name is :" +node.name.text)
            if node.name.text == "must" or node.name.text == "not" or node.name.text == "for" :
                node.detach()
        '''
        print(dep_tree.get_ascii(attributes=["name"]))
        print("\n")

        for node in dep_tree.iter_descendants("postorder"):
            print("name is :" + node.name.text)
        '''
        #add role to each word

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "The advertising identifier":
                node.add_features(role="data")
            if node.name.text == "personally-identifiable information":
                node.add_features(role="data")
            if node.name.text == "associated":
                node.add_features(role="action")
            if node.name.text == "connected":
                node.add_features(role="action")
            '''
            if node.name.text == "to":
                node.add_features(role="other")
            if node.name.text == "or":
                node.add_features(role="action")
            if node.name.text == "MAC address":
                node.add_features(role="data")
            if node.name.text == "SSAID":
                node.add_features(role="data")
            if node.name.text == "IMEI":
                node.add_features(role="data")
            if node.name.text == "any persistent device identifier":
                node.add_features(role="data")
            if node.name.text == "with":
                node.add_features(role="action")
            if node.name.text == "the user":
                node.add_features(role="condition")
            if node.name.text == "of":
                node.add_features(role="condition")
            if node.name.text == "explicit consent":
                node.add_features(role="condition")
            if node.name.text == "without":
                node.add_features(role="condition")

            '''
        #print(dep_tree.get_ascii(attributes=["name","role"]))


    ###############################################################--05--########################################################################
    if sentence=="don't use a service provider in connection with your use of platform unless you make them sign a contract to: protect any user data you obtained from us that is at least as protective as our terms and policies":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "do" or node.name.text == "n't" or node.name.text =="protect" or node.name.text =="in" or node.name.text =="any user data":
                node.detach()
        '''
        print(dep_tree.get_ascii(attributes=["name"]))
        print("\n")

        for node in dep_tree.iter_descendants("postorder"):
            print("name is :" + node.name.text)
        '''
        #add role to each word
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "a service provider":
                node.add_features(role="data")
            if node.name.text == "use":
                node.add_features(role="action")
            '''
            if node.name.text == "unless":
                node.add_features(role="condition")
            if node.name.text == "a contract":
                node.add_features(role="condition")
            if node.name.text == "sign":
                node.add_features(role="condition")
            if node.name.text == "make":
                node.add_features(role="condition")

            '''
        #print(dep_tree.get_ascii(attributes=["name","role"]))


    ###############################################################--06--########################################################################
    if sentence=="don't proxy, request or collect facebook usernames or passwords":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "do" or node.name.text == "n't":
                node.detach()
        '''
        print(dep_tree.get_ascii(attributes=["name"]))
        print("\n")

        for node in dep_tree.iter_descendants("postorder"):
            print("name is :" + node.name.text)
        '''
        #add role to each word
        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "passwords":
                node.add_features(role="data")
            if node.name.text == "facebook usernames":
                node.add_features(role="data")
            if node.name.text == "collect":
                node.add_features(role="action")
            if node.name.text == "request":
                node.add_features(role="action")
            if node.name.text == "proxy":
                node.add_features(role="action")
            '''
            if node.name.text == "or":
                node.add_features(role="other")
            '''
        #print(dep_tree.get_ascii(attributes=["name","role"]))

    ###############################################################--07--########################################################################
    if sentence == "do not use data obtained from us to provide tools that are used for surveillance":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "do" or node.name.text == "not" :
                node.detach()
        '''
        print(dep_tree.get_ascii(attributes=["name"]))
        print("\n")
        for node in dep_tree.iter_descendants("postorder"):
            print("name is :" + node.name.text)
        '''
        #add role to each word
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "data":
                node.add_features(role="data")
            if node.name.text == "use":
                node.add_features(role="action")
            '''
            if node.name.text == "surveillance":
                node.add_features(role="condition")
            if node.name.text == "for":
                node.add_features(role="condition")
            if node.name.text == "used":
                node.add_features(role="condition")
            if node.name.text == "tools":
                node.add_features(role="condition")
            if node.name.text == "provide":
                node.add_features(role="condition")

            '''
        #print(dep_tree.get_ascii(attributes=["name","role"]))

    ###############################################################--08--########################################################################
    if sentence == "keep facebook user ids within your control.":

        #print(dep_tree.get_ascii(attributes=["name"]))
        #print("\n")
        #for node in dep_tree.iter_descendants("postorder"):
            #print("name is :" + node.name.text)

        # add role to each word
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "facebook user ids":
                node.add_features(role="data")
            if node.name.text == "keep":
                node.add_features(role="action")
            '''
            if node.name.text == "your control":
                node.add_features(role="condition")
            if node.name.text == "within":
                node.add_features(role="condition")

            '''
        #print(dep_tree.get_ascii(attributes=["name", "role"]))



    ###############################################################--08--########################################################################
    if sentence == "you agree not to engage in any of the following prohibited activities: collect or store any personally identifiable information from the services from other users of the services without their express permission":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "in" or node.name.text == "from" or node.name.text == "not" or node.name.text == "engage":
                node.detach()

        #print(dep_tree.get_ascii(attributes=["name"]))
        #print("\n")
        #for node in dep_tree.iter_descendants("postorder"):
            #print("name is :" + node.name.text)

        #add role to each word
        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "any personally identifiable information":
                node.add_features(role="data")
            if node.name.text == "collect":
                node.add_features(role="action")
            '''
            if node.name.text == "their express permission":
                node.add_features(role="condition")
            if node.name.text == "without":
                node.add_features(role="condition")
            if node.name.text == "store":
                node.add_features(role="action")

            if node.name.text == "agree":
                node.add_features(role="other")
            if node.name.text == "or":
                node.add_features(role="other")
            '''
        #print(dep_tree.get_ascii(attributes=["name","role"]))


    ###############################################################--09--########################################################################
    if sentence =="the collection and use of the advertising identifier and commitment to these terms must be disclosed to users in a legally adequate privacy notification.":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "to":
                node.detach()
            if node.name.text == "must":
                node.detach()

        #print(dep_tree.get_ascii(attributes=["name"]))
        #print("\n")
        #for node in dep_tree.iter_descendants("postorder"):
            #print("name is :" + node.name.text)

        #add role to each word

        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "the advertising identifier":
                node.add_features(role="data")
            if node.name.text == "use":
                node.add_features(role="action")

            '''
            if node.name.text == "of":
                node.add_features(role="other")
            if node.name.text == "the collection":
                node.add_features(role="action")
            if node.name.text == "a legally adequate privacy notification":
                node.add_features(role="condition")
            if node.name.text == "in":
                node.add_features(role="condition")
            if node.name.text == "disclosed":
                node.add_features(role="condition")

            '''
        #print(dep_tree.get_ascii(attributes=["name","role"]))



    ###############################################################--09--########################################################################
    if sentence == "You will provide a privacy policy for your App that clearly and accurately describes to users of your App what user information you collect and how you use and share such information (including for advertising) with Fortmatic and third parties":
        print(dep_tree.get_ascii(attributes=["name"]))

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "for":
                node.detach()
            if node.name.text == "to" and node.up.name.text=="describes" :
                node.detach()
            if node.name.text == "including":
                node.detach()
            if node.name.text == "clearly":
                node.detach()
            if node.name.text == "with":
                node.detach()
            if node.name.text == "and":
                node.detach()
            if node.name.text == "use":
                node.detach()

        # print(dep_tree.get_ascii(attributes=["name"]))
        # print("\n")
        # for node in dep_tree.iter_descendants("postorder"):
        # print("name is :" + node.name.text)

        # add role to each word

        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "what user information":
                node.add_features(role="data")
            if node.name.text == "use":
                node.add_features(role="action")

            '''
            if node.name.text == "of":
                node.add_features(role="other")
            if node.name.text == "the collection":
                node.add_features(role="action")
            if node.name.text == "a legally adequate privacy notification":
                node.add_features(role="condition")
            if node.name.text == "in":
                node.add_features(role="condition")
            if node.name.text == "disclosed":
                node.add_features(role="condition")

            '''
        print(dep_tree.get_ascii(attributes=["name","role"]))

    return dep_tree


def pruneTree_pattern_1(sentence):
    dep_tree = getTree(sentence)
    ###############################################################--09--########################################################################
    if sentence == "without our prior written consent":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "our prior written consent":
                node.add_features(role="condition")

        #print(dep_tree.get_ascii(attributes=["name","role"]))

    ###############################################################--09--########################################################################
    if sentence == "End User has consented":
        #print(dep_tree.get_ascii(attributes=["name"]))
        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "consented":
                node.add_features(role="condition")

        #print(dep_tree.get_ascii(attributes=["name","role"]))

    ###############################################################--09--########################################################################
    if sentence == "obtain consent":
        #print(dep_tree.get_ascii(attributes=["name"]))
        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "consent":
                node.add_features(role="condition")

        #print(dep_tree.get_ascii(attributes=["name","role"]))



    ###############################################################--09--########################################################################
    if sentence == "obtain consent before you use our service data":
        #print(dep_tree.get_ascii(attributes=["name"]))
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "you":
                node.detach()
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "before":
                node.detach()

        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "consent":
                node.add_features(role="condition")
            if node.name.text == "use":
                node.add_features(role="action")

            if node.name.text == "our service data":
                node.add_features(role="data")
        print(dep_tree.get_ascii(attributes=["name","role","dep"]))


    ###############################################################--09--########################################################################
    if sentence == "store the confidential information without our prior written consent":
        #print(dep_tree.get_ascii(attributes=["name"]))



        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "MobileSoft’s express prior written consent":
                node.add_features(role="condition")

            if node.name.text == "store":
                node.add_features(role="action")
            if node.name.text == "store confidential information":
                node.add_features(role="data")
        print(dep_tree.get_ascii(attributes=["name","role","dep"]))

    ###############################################################--09--########################################################################
    if sentence == "collect such information when the applicable End User has consented to such activities":
        #print(dep_tree.get_ascii(attributes=["name"]))

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "when":
                node.detach()

        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "End User":
                node.add_features(role="condition")
            if node.name.text == "has":
                node.add_features(role="condition")
            if node.name.text == "consented":
                node.add_features(role="condition")

            if node.name.text == "collect":
                node.add_features(role="action")
            if node.name.text == "such information":
                node.add_features(role="data")
        print(dep_tree.get_ascii(attributes=["name","role","dep"]))
    print(dep_tree.write(format=1))
    return dep_tree


def bunch_pruneTree_pattern_1():
    #getTree("don't collect personal information")
    pattern = []
    category_one = []
    '''
    sentence_list.append("You may not distribute more than 1,500,000 Tweet IDs to any entity (inclusive of multiple individual users associated with a single entity) within any given 30 day period, unless you are doing so on behalf of an academic institution and for the sole purpose of non-commercial research or you have received the express written permission of Twitter.")
    sentence_list.append("You must access Google API Services in accordance with the Google APIs Terms of Service.")
    #sentence_list.append("you agree not to engage in any of the following prohibited activities: collect or store any personally identifiable information from the services from other users of the services without their express permission;")
    sentence_list.append("The advertising identifier must not be connected to personally-identifiable information or associated with any persistent device identifier (for example: SSAID, MAC address, IMEI) without explicit consent of the user")
    sentence_list.append("don't use a service provider in connection with your use of platform unless you make them sign a contract to: protect any user data you obtained from us that is at least as protective as our terms and policies")
    sentence_list.append("don't proxy, request or collect facebook usernames or passwords")
    sentence_list.append("do not use data obtained from us to provide tools that are used for surveillance")
    #sentence_list.append("keep facebook user ids within your control.")
    sentence_list.append("you agree not to engage in any of the following prohibited activities: collect or store any personally identifiable information from the services from other users of the services without their express permission")
    sentence_list.append("the collection and use of the advertising identifier and commitment to these terms must be disclosed to users in a legally adequate privacy notification.")
    #sentence_list.append("You will not seek or collect End User Credentials without our prior written consent")
    #sentence_list.append("You will provide a privacy policy for your App that clearly and accurately describes to users of your App what user information you collect and how you use and share such information (including for advertising) with Fortmatic and third parties")
    '''
    category_one.append("without our prior written consent")
    category_one.append("End User has consented")
    category_one.append("obtain consent")
    ## add by yue for testing
    #category_one.append("obtain consent before you use our service data")
    #category_one.append("store the confidential information without our prior written consent")
    #category_one.append("collect such information when the applicable End User has consented to such activities")
    #category_one.append("obtain consent before you use our service data")
    for s in category_one:
        tree = pruneTree_pattern_1(s)
        #print(tree.get_ascii(attributes=["name", "role"]))
        #print("================================================================================================")
        pattern.append(tree)
    return pattern





def pruneTree_pattern_2(sentence):
    dep_tree = getTree(sentence)
    ###############################################################--09--########################################################################
    if sentence == "Such privacy policy must provide":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "Such privacy policy":
                node.add_features(role="condition")
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "must":
                node.detach()

        #print(dep_tree.get_ascii(attributes=["name","role"]))

    ###############################################################--09--########################################################################
    if sentence == "provided in our Privacy Policy":
        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "our Privacy Policy":
                node.add_features(role="condition")

        #print(dep_tree.get_ascii(attributes=["name","role"]))

    ###############################################################--09--########################################################################
    if sentence == "provide a publicly accessible privacy policy":
        print(dep_tree.get_ascii(attributes=["name"]))
        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "a publicly accessible privacy policy":
                node.add_features(role="condition")

        #print(dep_tree.get_ascii(attributes=["name","role"]))

    ###############################################################--09--########################################################################
    if sentence == "Provide a privacy policy that tells people what you collect and how you will use this information":

        print(dep_tree.get_ascii(attributes=["name"]))

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "that":
                node.detach()
            if node.name.text == "people":
                node.detach()

            if node.name.text == "you":
                node.detach()
            if node.name.text == "will":
                node.detach()

            if node.name.text == "and":
                node.detach()

        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "a privacy policy":
                node.add_features(role="condition")
            if node.name.text == "this information":
                node.add_features(role="data")
            if node.name.text == "collect":
                node.add_features(role="action")
            if node.name.text == "use":
                node.add_features(role="action")
        print(dep_tree.get_ascii(attributes=["name","role","dep"]))



    ###############################################################--09--########################################################################
    if sentence == "Your use of Google user data must be explicitly disclosed in your published privacy policy":
        print(dep_tree.get_ascii(attributes=["name"]))

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "must":
                node.detach()

            if node.name.text == "be":
                node.detach()

            if node.name.text == "explicitly":
                node.detach()


        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "your published privacy policy":
                node.add_features(role="condition")
            if node.name.text == "Google user data":
                node.add_features(role="data")
            if node.name.text == "Your use":
                node.add_features(role="action")

        print(dep_tree.get_ascii(attributes=["name","role","dep"]))

    print(dep_tree.write(format=1))
    return dep_tree





def bunch_pruneTree_pattern_2():
    #getTree("don't collect personal information")
    pattern = []
    category_two = []
    category_two.append("Such privacy policy must provide")
    #category_two.append("provide and adhere to a privacy policy")
    category_two.append("provided in our Privacy Policy")
    category_two.append("provide a publicly accessible privacy policy")


    #category_two.append("Provide a privacy policy that tells people what you collect and how you will use this information")
    #category_two.append("Your use of Google user data must be explicitly disclosed in your published privacy policy")
    for s in category_two:
        tree = pruneTree_pattern_2(s)
        print(tree.get_ascii(attributes=["name", "role"]))
        print("================================================================================================")
        pattern.append(tree)
    return pattern




def pruneTree_pattern_3(sentence):
    dep_tree = getTree(sentence)
    ###############################################################--09--########################################################################
    if sentence == "use the Third - Party Content only for your non - commercial , educational use":

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "must":
                node.detach()

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "use":
                node.add_features(role="action")
            if node.name.text == "the Third - Party Content":
                node.add_features(role="data")
            if node.name.text == "only":
                node.add_features(role="condition")

    ###############################################################--09--########################################################################
    if sentence == "only distribute the Third - Party Content to students":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "distribute":
                node.add_features(role="action")
            if node.name.text == "the Third - Party Content":
                node.add_features(role="data")
            if node.name.text == "only":
                node.add_features(role="condition")

    ###############################################################--09--########################################################################
    if sentence == "disclose Confidential Information only for purposes set forth in this Agreement":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "distribute":
                node.add_features(role="action")
            if node.name.text == "Confidential Information":
                node.add_features(role="data")
            if node.name.text == "only":
                node.add_features(role="condition")


    ###############################################################--09--########################################################################
    if sentence == "use the Confidential Information only to exercise its rights":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "use":
                node.add_features(role="action")
            if node.name.text == "the Confidential Information":
                node.add_features(role="data")
            if node.name.text == "only":
                node.add_features(role="condition")




    print(dep_tree.write(format=1))
    return dep_tree






def bunch_pruneTree_pattern_3():

    pattern = []
    category_three = []
    category_three.append("use the Third - Party Content only for your non - commercial , educational use")
    category_three.append("only distribute the Third - Party Content to students")
    category_three.append("distribute the Access Code only to those members of your 5th grade class")


    #category_three.append("use the Confidential Information only to exercise its rights")

    for s in category_three:
        tree = pruneTree_pattern_3(s)
        print(tree.get_ascii(attributes=["name", "role","dep"]))
        print("================================================================================================")
        pattern.append(tree)
    return pattern


def pruneTree_pattern_4(sentence):
    dep_tree = getTree(sentence)
    ###############################################################--09--########################################################################
    if sentence == "comply with all applicable data privacy Laws , privacy policies , and internal policies of Licensee":

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "comply":
                node.add_features(role="condition")

    ###############################################################--09--########################################################################
    if sentence == "complies with all laws":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "complies":
                node.add_features(role="condition")

    ###############################################################--09--########################################################################
    if sentence == "compliance with the requirements of effective Russian legislation":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "compliance":
                node.add_features(role="condition")

    ###############################################################--09--########################################################################
    if sentence == "your use of data should be in compliance with the requirements of effective Russian legislation":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "should":
                node.detach()
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "compliance":
                node.add_features(role="condition")
            if node.name.text == "data":
                node.add_features(role="data")
            if node.name.text == "use":
                node.add_features(role="action")

    ###############################################################--09--########################################################################
    if sentence == "use data that comply with all applicable data privacy Laws , privacy policies , and internal policies of License":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == ",":
                node.detach()
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "comply":
                node.add_features(role="condition")
            if node.name.text == "data":
                node.add_features(role="data")
            if node.name.text == "use":
                node.add_features(role="action")
    ###############################################################--09--########################################################################
    if sentence == "use data that comply with all applicable data privacy Laws":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == ",":
                node.detach()
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "comply":
                node.add_features(role="condition")
            if node.name.text == "data":
                node.add_features(role="data")
            if node.name.text == "use":
                node.add_features(role="action")

    print(dep_tree.write(format=1))
    return dep_tree


def bunch_pruneTree_pattern_4():
    pattern = []
    category_four = []
    category_four.append("comply with all applicable data privacy Laws , privacy policies , and internal policies of Licensee")
    category_four.append("complies with all laws")
    category_four.append("compliance with the requirements of effective Russian legislation")

    #category_four.append("your use of data should be in compliance with the requirements of effective Russian legislation")
    #category_four.append("use data that comply with all applicable data privacy Laws")
    #category_four.append("use data that comply with all applicable data privacy Laws , privacy policies , and internal policies of License")
    for s in category_four:
        tree = pruneTree_pattern_4(s)
        print(tree.get_ascii(attributes=["name", "role","dep"]))
        print("================================================================================================")
        pattern.append(tree)
    return pattern



def pruneTree_pattern_5(sentence):
    dep_tree = getTree(sentence)
    ###############################################################--09--########################################################################
    if sentence == "The User's personal data are processed according to the applicable/relevant legislation":

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "are":
                node.detach()


        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "according":
                node.add_features(role="condition")

            if node.name.text == "The User's personal data":
                node.add_features(role="data")

            if node.name.text == "processed":
                node.add_features(role="action")
    ###############################################################--09--########################################################################
    if sentence == "The information may be disclosed in accordance with effective Russian legislation":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "may":
                node.detach()

            if node.name.text == "be":
                node.detach()



        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "accordance":
                node.add_features(role="condition")
            if node.name.text == "The information":
                node.add_features(role="data")
            if node.name.text == "disclosed":
                node.add_features(role="action")
    print(dep_tree.write(format=1))
    return dep_tree

def bunch_pruneTree_pattern_5():
    pattern = []
    category_five = []
    category_five.append("The User's personal data are processed according to the applicable/relevant legislation")
    category_five.append("The information may be disclosed in accordance with effective Russian legislation")

    for s in category_five:
        tree = pruneTree_pattern_5(s)
        print(tree.get_ascii(attributes=["name", "role","dep"]))
        print("================================================================================================")
        pattern.append(tree)
    return pattern


def pruneTree_pattern_6(sentence):
    dep_tree = getTree(sentence)
    ###############################################################--09--########################################################################
    if sentence == "If your use of the Services is prohibited by applicable laws, then you can not use the Services":

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "then":
                node.detach()
            if node.name.text == ",":
                node.detach()
            if node.name.text == "you":
                node.detach()
            if node.name.text == "can":
                node.detach()
            if node.name.text == "not":
                node.detach()
            if node.name.text == "is":
                node.detach()
            if node.name.text == "by":
                node.detach()
            if node.name.text == "your use":
                node.detach()

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "If":
                node.add_features(role="condition")

            if node.name.text == "the Services":
                node.add_features(role="data")

            if node.name.text == "use":
                node.add_features(role="action")

    ###############################################################--09--########################################################################
    if sentence == "when you agree to and follow these Terms, you are permitted to use our APIs in connection with your Application":
        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == ",":
                node.detach()

            if node.name.text == "you":
                node.detach()
            if node.name.text == "are":
                node.detach()
            if node.name.text == "in":
                node.detach()
            if node.name.text == "to":
                node.detach()

            if node.name.text == "and":
                node.detach()

            if node.name.text == "follow":
                node.detach()

        for node in dep_tree.iter_descendants("postorder"):
            if node.name.text == "when":
                node.add_features(role="condition")
            if node.name.text == "our APIs":
                node.add_features(role="data")
            if node.name.text == "use":
                node.add_features(role="action")
    print(dep_tree.write(format=1))
    return dep_tree


def bunch_pruneTree_pattern_6():
    pattern = []
    category_six = []
    category_six.append("If your use of the Services is prohibited by applicable laws, then you can not use the Services")
    category_six.append("when you agree to and follow these Terms, you are permitted to use our APIs in connection with your Application")

    for s in category_six:
        tree = pruneTree_pattern_6(s)
        print(tree.get_ascii(attributes=["name", "role","dep"]))
        print("================================================================================================")
        pattern.append(tree)
    return pattern



def pruneTree_pattern_8(sentence):
    dep_tree = getTree(sentence)


    ###############################################################--09--########################################################################
    if sentence == "give clear and prominent notice to everyone you collect data":
        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "you":
                node.detach()


        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "clear and prominent notice":
                node.add_features(role="condition")

    ###############################################################--09--########################################################################
    if sentence == "notify users":

        for node in dep_tree.iter_descendants("postorder"):

            if node.name.text == "notify":
                node.add_features(role="condition")

    print(dep_tree.write(format=1))
    return dep_tree


def bunch_pruneTree_pattern_8():
    pattern = []
    category_eight = []

    category_eight.append("give clear and prominent notice to everyone")
    category_eight.append("notify users")
    for s in category_eight:
        tree = pruneTree_pattern_8(s)
        print(tree.get_ascii(attributes=["name", "role","dep"]))
        print("================================================================================================")
        pattern.append(tree)
    return pattern

if __name__ == '__main__':
    #bunch_pruneTree_pattern_1()
    #bunch_pruneTree_pattern_2()
    #bunch_pruneTree_pattern_3()
    #bunch_pruneTree_pattern_4()
    #bunch_pruneTree_pattern_5()
    #bunch_pruneTree_pattern_6()
    bunch_pruneTree_pattern_8()