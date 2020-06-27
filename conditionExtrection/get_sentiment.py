import re
import spacy
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}


class sentiment:
    def __init__(self, sentence,verb):
        self.contraction_dict = contraction_dict
        self.sentence = self.cleanSentence(sentence.lower())
        self.verb = verb.lower()
        self.nlp = spacy.load("en_core_web_sm")


    def cleanSentence(self,text):
        return self.replace_contractions(text)

    def _get_contractions(self):
        contraction_re = re.compile('(%s)' % '|'.join(self.contraction_dict.keys()))
        return contraction_dict, contraction_re


    def replace_contractions(self, text):
        contractions, contractions_re = self._get_contractions()
        def replace(match):
            return contractions[match.group(0)]
        return contractions_re.sub(replace, text)



    def propagates(self,verb_token,doc,total_verb):
        if verb_token.dep_ == "conj" or verb_token.dep_ == "xcomp":
            total_verb.add(verb_token.head)
            self.propagates(verb_token.head, doc,total_verb)
        else:
            return

    def propa_xcomp(self, token, doc, total_verb):
        if token.dep_ == "xcomp":
            total_verb.add(token.head)


    def propa_advcl(self, token, doc, total_verb):
        if token.dep_ == "advcl" or "ccomp":
            total_verb.add(token.head)


    def determine_verb_neg(self,verb,doc):
        for d in doc:
            if d.dep_ == "neg" and d.head == verb:
                return True
        return False



    def verb_sentiment(self):
        doc = self.nlp(self.sentence)
        for phrase in list(doc.noun_chunks):
            phrase.merge(phrase.root.tag_, phrase.root.lemma_, phrase.root.ent_type_)
        total_verb = set()
        for token in doc:
            #print(token.text + " :: " + token.dep_ + "-->" + token.head.text)

            if token.text == self.verb:

                total_verb.add(token)
                ### case1: "We do not sell,rent, or trade your personal information,” means “not sell,” “not rent,” and “not trade."
                self.propagates(token,doc,total_verb)
                ### case2: "We do not require you to disclose any personal information,” initially has “require” marked with negative sentiment"
                self.propa_xcomp(token, doc, total_verb)
                ### case3: We do not collect your information to share with advertisers,”
                self.propa_advcl(token, doc, total_verb)
        print(total_verb)
        for v in total_verb:
            if self.determine_verb_neg(v,doc):
                return False
        return True





def main():
    sent_object = sentiment("Don't sell, license, or purchase any data obtained from us or our services.", "sell")
    flag = sent_object.verb_sentiment()
    print(flag)





if __name__ == '__main__':
    main()