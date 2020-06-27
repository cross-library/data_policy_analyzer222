# -*- coding: utf-8 -*-
import pandas as pd
from filterSentenceByVerb.assign_features import write
from filterSentenceByVerb.get_nmod_of_entities import *
from filterSentenceByVerb.convert_word_format import *
from filterSentenceByVerb.request_co_reference import *
import re

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

class policy_verb_entity:
    ### config verb related to data collect and share here
    collection = ["provide", "disclosure", "protect", "process", "utilize", "prohibit", "disclose", "distribute",
                  "give", "rent", "sell", "send", "share", "trade", "transfer", "transmit",
                  "connect", "associate",  "combine", "lease",  "afford", "resell",
                  "deliver", "disseminate", "transport", "access", "collect", "gather", "obtain", "receive",
                  "save",
                  "store", "use", "Keep", "proxy", "request", "track", "aggregate", "augment", "cache",
                  "republish", "get", "seek", "possess", "accumulate", "keep", "convert"]


    ### config first party here
    filter_first_subject = ["tencent", "travelport", "mobileSoft", "we", "us", "twitter", "our", "facebook",
                            "google",
                            "alipay", "edmodo", "pollfish", "adobe", "appnext", "here", "spotify", "uber", "vimeo",
                            "amplitude", "ionicframework", "mixpanel", "seattleclouds", "swmansion", "tiktok",
                            "dropbox", "kakao", "wechat", "line", "linkedin", "pinterest", "vk", "snapchat",
                            "instagram", "zendesk", "squareup", "airmap", "zoom", "gotenna", "vimeo", "fortmatic",
                            "slack", "matterport", "trello", "mindbodyonline", "spotify", "snap", "unity", "fabric",
                            "onesignal", "flurry", "startapp", "applovin", "chartboost", "firebase", "amazon",
                            "paypal", "appsflyer", "airbnb", "mopub", "adcolony", "vungle","square","tapjoy","appbrain","ironsource","aws"]


    nlp = spacy.load("en_core_web_sm")
    consituency_preditor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
    co_refence_predictor =Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
    standford_nlp = stanza.Pipeline()

    def __init__(self, sentence,paragraph, consituency_preditor, co_refence_predictor,nlp,standford_nlp):
        self.sentence = sentence
        self.paragraph = paragraph
        self.consituency_preditor = consituency_preditor
        self.co_refence_predictor = co_refence_predictor
        self.doc = nlp(sentence)
        self.standford_nlp = standford_nlp
        self.sensitive_data = None
        self.phrase_set = None
        self.co_reference = None
        self.verb_entitiy_string_without_filter = None
        self.verb_entitiy_string_with_filter = None
        self.verb_subjects = None
        self.subject_co_reference = None
        self.nmod_entitiy_string = None




    # Alternate constructor
    @classmethod
    def predict(cls,sentence,paragraph):
        return cls(sentence,paragraph,cls.consituency_preditor, cls.co_refence_predictor,cls.nlp,cls.standford_nlp)



    ### check the noun phrase: the use of the google user data
    def assign_nmod_of_sensitive_word(self):
        if self.sentence.strip().endswith("?"):
            self.nmod_entitiy_string = ""
            return ""
        string = ""
        for sensitive_word in self.sensitive_data:
            nmod = get_word_head(self.sentence, sensitive_word, self.standford_nlp)
            if nmod is None:
                continue
            # transfer the nmod from noun to the verb format: disclosure --> disclose
            verb_format = convert(nmod, 'n', 'v')
            verb_list = []
            for v in verb_format:
                if v in policy_verb_entity.collection:
                    string = string + nmod + "-->" + v + "--->" + sensitive_word + "\n"
        self.nmod_entitiy_string = string
        return string



    # if subject has co_reference, print "it --> [your app, application] "
    def get_subject_co_reference(self, subject_list):
        string = ""
        for subject in subject_list:
            for pair in self.co_reference:
                if subject in pair:
                    string = string + subject + "-->" + str(pair) + "\n"
        self.subject_co_reference = string
        return string


    def is_captial_sentence(self):
        s = re.sub(r'[^\w\s]', '', self.sentence)
        if s.isupper():
            return True
        else:
            return False

    ### considering prevent us/authorizes us and so on
    def is_special_verb_first_party(self):
        for item in self.filter_first_subject:
            keyword1 = "to" + item
            keyword2 = "authorizes" + item
            keyword3 = "prevent" + item
            keyword4 = item + "shall"
            keyword5 = item + "may"
            key_list = []
            key_list.append(keyword1)
            key_list.append(keyword2)
            key_list.append(keyword3)
            key_list.append(keyword4)
            key_list.append(keyword5)
            for key in key_list:
                if key in self.sentence.lower():
                    return True
        return False


    # get the verb of the sensitve data with filter
    # filter 1: filter sentence which is not a statement or all captial sentence
    # filter 5: filter sentence which first verb is "means"(in order to remove definitions)
    # filter 6: filter sentence about Disclaimers(upper cases) which are always unrelated to policeis
    # filter 2: filter sentence which the subject of the verb is the first party ("we","us")("twitter","google")
    # filter 3: filter sentence which the verb is not in the target collection
    # filter 4: filter sentence which contains "grant us", "grant google the rights"...
    # filter 8: filter sentence contains authorizes + first party: for example you authorizes us to collect your data
    def assign_verb_and_entity_with_filter(self):
        ###condition 1: if the sentence is end with "?", return null string, and True which means it will not process nmod
        if self.sentence.strip().endswith("?") or self.is_captial_sentence():
            self.verb_entitiy_string_with_filter = ""
            self.verb_subjects = []
            return "", True, []

        ### condition 5: move "means". some sentences are descriting the defintion of the sensitive data which is not in our scope.
        for item in self.doc:
            if item.pos_ == "VERB":
                if "mean" in item.text:
                    self.verb_entitiy_string_with_filter = ""
                    self.verb_subjects = []
                    return "", True, []
                break

        ### condition 6: if the sentence is all uppercase, which are always the Disclaimers that are unrelated to policeis
        if self.is_captial_sentence():
            return "", True, []

        ###condition 4: if the sentences contains give us, grant us, we return null string, and True which means it will not process nmod
        ###this condition can not be covered in condition2 because "give us the rights" the rights is not the sensitive data
        ###condition 4+: if the sentences contains give + sdk target(give Amplitude), grant us, we return null string, and True which means it will not process nmod
        special_verb = ["grant", "grants", "granted", "give", "gave", "provide", "provided", "providing"]
        for word in special_verb:
            if word in self.sentence.strip().lower():
                subject_of_special_verb = self.get_subject(word)

                print("=============subject")
                print(subject_of_special_verb)
                for s in subject_of_special_verb:
                    if self.is_first_party(s):
                        self.verb_entitiy_string_with_filter = ""
                        self.verb_subjects = []
                        return "", True, []

        ### if the sentence contains "authorizes + first party" : for example you authorizes us to collect your data, filter it out
        if self.is_special_verb_first_party():
            return "", True, []


        total_verb_list = []
        total_subject_list = []
        string = ""
        for phrase_item in self.phrase_set:
            verb_list = self.getVerbBasedOnEntity(phrase_item)
            total_verb_list.extend(verb_list)
            for verb in verb_list:
                subject_list = self.get_subject(verb.text)
                total_subject_list.extend(subject_list)

                ### condition 3: filter the verb which not in the collection
                if verb.lemma_.lower() not in policy_verb_entity.collection:
                    continue
                else:
                    string = string + verb.text + "--->" + phrase_item + "\n"

        #### condition2: check if the the subject of verb is "we", "us"
        subject_candidate = []
        for s in total_subject_list:
            subject_candidate.append(s)
            for synoym in self.co_reference:
                if s in synoym:
                    subject_candidate.extend(synoym)

        flag = False
        for s in list(set(subject_candidate)):
            if self.is_first_party(s):
                flag = True
        self.verb_entitiy_string_with_filter = string
        self. verb_subjects = list(set(total_subject_list))
        if flag:
            return "", flag, list(set(total_subject_list))
        else:
            return string, flag, list(set(total_subject_list))





    # give the verb, return its heads
    def get_subject(self, verb):
        subject_list = []
        ####obtain subject
        for i in range(0, len(self.doc)):
            # print(doc[i].text + "--->"  + doc[i].dep_  + "---->" + doc[i].head.text)
            if self.doc[i].head.text == verb:
                subject_list.append(self.doc[i].text)
        return subject_list



    # check given the subject of verb
    def is_first_party(self, subject):
        if subject.lower() in policy_verb_entity.filter_first_subject:
            return True
        else:
            return False



    # get the verb of the sensitve data without any filter
    # input: By downloading or using the Vungle SDK, you and any company, entity, or
    # output: downloading - -->theVungle SDK, using - -->the Vungle SDK
    def assign_verb_and_entity_without_filter(self):
        string = ""
        for phrase_item in self.phrase_set:
            verb_list = self.getVerbBasedOnEntity(phrase_item)
            for verb in verb_list:
                string = string + verb.text + "--->" + phrase_item + "\n"
        self.verb_entitiy_string_without_filter = string
        return string


    # give a sensitive data, get verb of it.
    def getVerbBasedOnEntity(self,phrase_item):
        # Merge the noun phrases
        for phrase in list(self.doc.noun_chunks):
            phrase.merge(phrase.root.tag_, phrase.root.lemma_, phrase.root.ent_type_)

        verb_list = []
        for token in self.doc:
            if token.text == phrase_item:
                curr = token
                while curr.dep_ != "ROOT" and curr.pos_ != "VERB":
                    curr = curr.head
                if curr.pos_ == "VERB":
                    verb_list.append(curr)

        ###extend conjuction verb
        total_verb = set()
        for verb in verb_list:
            total_verb.add(verb)
            self.find_conj_verb(verb, total_verb)
        return total_verb


    ## condition 3 find conjuction verb
    ## for example, Do not modify, translate or delete a portion of the Twitter Content.
    ## the verb should be modify, translate, delete --> Twitter content
    def find_conj_verb(self, verb, total_verb):
        if verb.dep_ == "conj":
            total_verb.add(verb.head)
            self.find_conj_verb(verb.head, total_verb)
        else:
            return



    # get all co_reference relationship in the paragraph
    # input: Vungle and its Demand Partners use Tracking Technologies in order to collect certain Ad Data.
    # output: [['Vungle', 'its']]
    def get_co_reference(self):
        try:
            results = json.loads(request_co_reference(self.paragraph))
        except:
            self.co_reference = [[]]
            return [[]]
        clusters = results["clusters"]
        relationship = []
        for cluster in clusters:
            pair = []
            for item in cluster:
                if item[0] == item[1]:
                    # print(results["document"][item[0]])
                    pair.append(results["document"][item[0]])
                else:
                    # print(' '.join(results["document"][item[0]:item[1]]))
                    pair.append(' '.join(results["document"][item[0]:item[1]]))
            relationship.append(pair)
        self.co_reference = relationship
        return relationship


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
                p_list = phrase.text.split(" ")
                if data in p_list and self.valiateData(phrase):
                    phrase_set.add(phrase.text)
        # print(phrase_set)
        self.deleteTmpFile()
        self.phrase_set = self.deduplicate(phrase_set)
        self.sensitive_data = sensitive_data
        return self.phrase_set, self.sensitive_data



    ##deduplicate phrase
    ## fix {customer, customer data} --> {customer data}
    def deduplicate(self,phrase_set):
        phrase_list = list(phrase_set)
        depulicated_index = []
        new_set = set()
        for i in range(0,len(phrase_list)):
            for j in range(0,len(phrase_list)):
                if i == j:
                    continue
                if phrase_list[i] in phrase_list[j]:
                    depulicated_index.append(i)
        for i in range(0,len(phrase_list)):
            if i in depulicated_index:
                self.write("duplicate_pharse :: " + phrase_list[i] + "\n")
                continue
            new_set.add(phrase_list[i])
        return new_set



    @classmethod
    def valiateData(self, phrase):
        data_pos = phrase.pos_
        valid_phrase_list = ["PROPN","NOUN"]
        if data_pos not in valid_phrase_list:
            #policy_verb_entity.write("verb_pos_ :: " + phrase.text+ "\n" + str(data_pos) + "\n")
            return False
        ### fix spacy parser error(filter verb in phrase)
        mistaken_verb_pharse = ["permit","disclosure","share","rent","lease","sublicense","loan","license","store","market","underwriting","order","defense","transfer","transmit","defect","proxy","return","us","supply","purchase","display","translate","monitor","operates","copy","record","analysis","summary","alter","access","revise","process","syndicate","request","reformat","charge","move","pledge","assign","support","post","redistribute","inquiry","redirect","conduct"]
        if lemmatizer.lemmatize(phrase.text.lower()) in mistaken_verb_pharse:
            #policy_verb_entity.write("mistaken_verb_pharse :: " + phrase.text + "\n")
            return False
        mistaken_nonu_pharse = ["broker","copyright","term","obligation","terrorism","worm","virus","purpose","law","loss","consent","regulation","employee","permission","officer","employee","attorne","auditor","advisor","director","contractor","bombs","agent","robot","software","fee","policy"]
        for nonu in mistaken_nonu_pharse:
            if nonu in phrase.text.lower():
                #policy_verb_entity.write("mistaken_nonu_pharse :: " + phrase.text + "\n")
                return False
        mistaken_non_sensi_pharse = ["services","racism","violence","investor","lender","acquirer","hosts","software","spider","end users","customer","criterion","adware","labor","energy","war","riot","scraper","bot","gdpr","reproduction","service","our services","all services","the services","the service","display","distribute","reformat","transmit","modify","grant","resell",'redistribute',"assign","warrant","save","keep","use","policies"]
        if lemmatizer.lemmatize(phrase.text.lower()) in mistaken_non_sensi_pharse or phrase.text.lower() in mistaken_non_sensi_pharse :
            #policy_verb_entity.write("mistaken_non_sensi_pharse :: " + phrase.text + "\n")
            return False
        return True



    @classmethod
    def write(self,content):
        f=open("log.txt","a+")
        f.write("==============="+"\n")
        f.write(content)
        f.write("===============")
        f.write("\n")

        f.close()


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
        pharse_list = [phrase for phrase in self.doc]
        # print("====================================================================")
        # print(pharse_list)
        return pharse_list


    # invoke trained model and get the results
    # parameter: testFile
    def generate_results(self, to_filemame):
        curr = os.getcwd()
        ner_path = "../customizeNER/model"
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
        filename = "../customizeNER/model/result.txt"
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
        del_file3 = "rm -rf ../customizeNER/model/result.txt"
        os.system(del_file1)
        os.system(del_file2)
        os.system(del_file3)




def main():
    #sentence = "By downloading or using the Vungle SDK, you and any company, entity, or organization on behalf of which you are accepting this Agreement (“Developer”) hereby agrees to be bound by all terms and conditions of this Agreement, and you represent and warrant that you are an authorized representative of Developer with the authority to bind Developer to this Agreement"
    #paragraph = "This Vungle SDK License and Publisher Terms (“Agreement”) is made available by Vungle, Inc. (“Vungle”). By downloading or using the Vungle SDK, you and any company, entity, or organization on behalf of which you are accepting this Agreement (“Developer”) hereby agrees to be bound by all terms and conditions of this Agreement, and you represent and warrant that you are an authorized representative of Developer with the authority to bind Developer to this Agreement.  IF YOU DO NOT AGREE TO ALL TERMS AND CONDITIONS OF THIS AGREEMENT, DO NOT DOWNLOAD OR USE THE VUNGLE SDK."

    sentence = "       “Developer Apps” means the mobile applications owned and/or controlled by Developer, including all content, images, music and text contained therein, that Developer wishes to use with the Vungle SDK and Vungle Platform."
    paragraph = "       “Developer Apps” means the mobile applications owned and/or controlled by Developer, including all content, images, music and text contained therein, that Developer wishes to use with the Vungle SDK and Vungle Platform."
    object = policy_verb_entity.predict(sentence,paragraph)
    object.extractEntity()
    object.get_co_reference()
    object.assign_verb_and_entity_without_filter()
    verb_entitiy_string, flag, subject_list = object.assign_verb_and_entity_with_filter()
    object.get_subject_co_reference(subject_list)
    object.assign_nmod_of_sensitive_word()
    print(object.phrase_set)
    print(object.sensitive_data)
    print(object.co_reference)
    print(object.verb_entitiy_string_without_filter)
    print(object.verb_entitiy_string_with_filter)
    print("----")
    print(object.subject_co_reference)
    print("----")
    print(object.nmod_entitiy_string)
    print("----")
    print(object.verb_subjects)




if __name__ == "__main__":
    main()
