# -*- coding: utf-8 -*-
from allennlp.predictors.predictor import Predictor
from nltk.tree import *

from string import punctuation

predictor = Predictor.from_path(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
def main():

    #sent = "In jurisdictions that require informed consent for the storing and accessing of cookies or other information on an end user’s device (such as the European Union), ensure, in a verifiable manner, that an end user provides the necessary consent before you use Facebook technologies that enable us to store and access cookies or other information on the end user’s device. "
    #sent = "You represent and warrant that you are authorized to process your data and make such data available to MINDBODY or its customers for uses as set out in this Agreement and the MINDBODY Privacy Policy , including through appropriate notice, disclosures, consent and by your referring individuals to our Privacy Policy (notwithstanding MINDBODY' ability and right, to which you agree, to request consent, and provide notice and its Privacy Policy  separately to individuals)."
    #sent = "You represent and warrant that the Subscriber consents to the disclosure and processing of Personal Data relating to it or its representatives and End Users (as defined in the MINDBODY Privacy Policy )."
    sent = "Without Matterport' prior written consent, You shall not provide to Matterport any personally identifiable information or personal data, as defined under applicable law. "
    #sent ="You may not assign or otherwise transfer any of Your rights hereunder without Matterport' prior written consent"
    #sent = "In the event of a Claim in respect of which a Matterport Indemnitee seeks indemnification from You under this Section, the Matterport Indemnitee will promptly notify You in writing of the Claim, cooperate with You in defending or settling the Claim at Your expense, and allow You to control the defense and settlement of the Claim, including the selection of attorneys; provided, however, that You shall not settle any Claim unless such settlement completely and forever releases the Matterport Indemnitee from all liability with respect to such Claim or unless the Matterport Indemnitee consents to such settlement in writing."
    #sent = "Where such data includes non-public content relating to a user, such content must not be exposed to other users or to third parties without proper consent from that user."
    #sent = "If your use of the Uber API Services or access to Uber Data requires or will likely result in the provision of personal information directly to Uber, you agree to adequately inform and obtain all necessary consents and authorizations from the applicable users to provide such personal information to Uber and retain written records of such consents."
    #sent = "The advertising identifier must not be connected to personally-identifiable information or associated with any persistent device identifier (for example: SSAID, MAC address, IMEI, etc.) without explicit consent of the user."
    #sent = "If reset, a new advertising identifier must not be connected to a previous advertising identifier or data derived from a previous advertising identifier without the explicit consent of the user. "
    #sent = "Get the user's express consent before you do any of the following: Take any actions on a user's behalf, including posting Twitter Content, following/unfollowing other users, modifying profile information, starting a Periscope Broadcast or adding hashtags or other data to the user's Tweets."
    #sent = "No Application may disable, override, or otherwise interfere with any system alerts, warnings, display panels, consent panels and the like implemented or required by AirMap, including those that are intended to notify the End User that the End User‘s location data is being collected, transmitted, maintained, processed, or used, or intended to obtain consent for such use. "
    #sent ="Without limiting the generality of the foregoing, Licensee will notify and obtain consent from each End User prior to the collection, transmission, maintenance, processing, or other use of End User‘s location data and if such consent is denied or withdrawn, Licensee will not, whether via an Application or otherwise, collect, transmit, maintain, process or use the applicable End User‘s location data or Personal Data or perform any other actions for which such End User‘s consent has been denied or withdrawn."
    #sent = "Each party agrees that it will use the Confidential Information of the other party solely in accordance with the provisions of this Agreement and it will not disclose, or permit to be disclosed, the same directly or indirectly, to any third party without the other party’s prior written consent, except as otherwise permitted hereunder."
    #sent = "Licensee will ensure that each Application does not use, incorporate or make available any airspace data (whether owned, controlled or licensed by Licensee or any third party) in or through such Application, other than the AirMap content made available by AirMap to Licensee via the Service under this Agreement; "
    #sent = "Licensee will ensure that each Application contains protections that are adequate to keep secure and prevent the interception of any information transmitted to, from or through the Service or SDK. "
    #sent = "If any Application accesses, uses, associates, or collects from End Users any information that constitutes Personal Data, Licensee must comply, and cause all Applications to comply, with all applicable data privacy Laws, privacy policies, and internal policies of Licensee; and"
    #sent = "Licensee agrees to permit access to Confidential Information only to those employees, contractors and agents of Licensee who need to know such information, and who have agreed to keep such information confidential under confidentiality obligations at least as protective of AirMap as those set forth in this Agreement; "
    #sent = "Licensee agrees to not disclose, publish or communicate Confidential Information to any third party (or authorize its employees, agents or anyone else to do so) except as expressly permitted under this Agreement"
    #sent = "Licensee agrees to use and disclose Confidential Information only for purposes set forth in this Agreement."
    #sent = "You must agree and accept all of the Terms, or you don’t have the right to use the Services."
    #sent = "Your using the Services in any way means that you agree to all of these Terms, and these Terms will remain in effect while you use the Services."
    #sent = "For example, if you wished to establish a closed group amongst yourself, as a teacher, and your 5th grade class, you would distribute the Access Code only to those members of your 5th grade class that you wanted to view what was going on within that group."
    sent = "Student accounts are currently not permitted or enabled to access Third Party Content unless the Publisher authorizes student accounts to access such Third-Party Content."
    sent = "In addition to any restrictions imposed by the Publisher, Edmodo requires you to: (i) use the Third-Party Content only for your non-commercial, educational use; (ii) only distribute the Third-Party Content to students who are currently enrolled in courses you teach; (iii) not transfer the"
    sent = "The right to use the brand name, trademarks, domain names and other distinctive marks of the Site Administration may be granted only as agreed in writing with the Site Administration."
    sent = "The User understands the necessity of receiving from the Site Administration some technical information in connection with the use of the Site by means of services of the Site, electronic mails, sms and confirms his/her approval for such possible technical notifications. "
    sent = "The right to use the Site non-activated data and commands is granted to the User providing that the User meets the provisions of the License Agreement mentioned above (see 1.2.of these Terms). "
    sent = "The Site Administration takes all necessary measures to protect the User's personal data from unauthorized access, modification, disclosure or destruction. "
    sent = "The information provided by the User, including personal data, may be used by the Site Administration to ensure compliance with the requirements of effective Russian legislation, other applicable/relevant legislation, as well as to transfer it to the third-parties in order to protect right and interests of the Users, the Site Administration and third-parties (including in order to prevent, control/investigate and/or suppress illegal actions). "
    sent = "The information provided by the User may be disclosed only in accordance with effective Russian legislation on the demand of the court, law-enforcement bodies as well as in other cases as provided by Russian legislation."
    sent = "In case of unauthorized access to the User's login and password and/or personal page or dissemination of the User's login and password, the User is obliged to notify the Site Administration immediately according to the established procedure."
    sent = "As the owner of the information posted on his/her own personal page, the User understands that, except as provided by these Terms and effective Russian legislation, the Site Administration does not participate in forming and using the content and controlling other users' access to the User's personal page. "
    sent = "In case of the violation of third parties' legitimate rights and interests provided by Russian legislation or these Terms in the Community, the Site Administration has the right to take the following measures at its opinion:   transfer the Community administration rights and the use of sub-domain to the legal right holder that has confirmed their right for the Community Content according to the established procedure, including the right to the objects of copyright and related rights, the right to means of individualization, confusingly similar to those used in the Community sub-domain names; "
    sent= "The User acquires the right to use non-activated data and commands of the Site in accordance with the License Agreement available at: http://vk.com/licence , and which is concluded by the Site Administration with the User. "
    sent = "API applications created by the Users should only use API methods published on the Site, as well as the ID, secure key and service token specified in these applications' settings. "
    sent = "By posting his/her Content in any part of the Site, the User automatically grants the Site Administration free of charge the non-exclusive right to use it by copying, public performance, reproduction, processing, translation and distribution for or in connection with the purposes of the Site, including for the purpose of increasing its popularity. "
    sent = "The users bear liability for their own actions in connection with the creation and posting of information on their own personal page on the Site as well as in connection with the posting of information on the personal pages of other users and other sections of the Site in accordance with effective Russian legislation. "
    sent = "If you have executed another agreement with us related to your use of our APIs, then those terms would control your use of our APIs."
    sent = "You may use the self-serve APIs if your Application follows these Terms, the Developer Documentation, and is designed to help Members be more productive and successful."
    sent = "These Terms shall apply to any use of our APIs unless you have executed a separate signed partnership agreement, in which case that agreement shall apply (Partner Program)"
    sent = "LinkedIn also offers a variety of plugins you can integrate into your website to allow visitors to access and view Content from LinkedIn websites if you have agreed to be bound, in addition to these Terms, to the LinkedIn Plugins License Agreement."
    sent = "Hiring: Applications to find a job or fill a job, such as ones used to advertise, post, apply for, search for, recommend, refer, suggest, enhance listings for, or fill permanent jobs, contract positions, or volunteer opportunities, or for anything with similar functionality (you can, however, use the APIs to develop an Application to let people apply for a job at your company on your company’s career site, as long as it abides by the specifications set forth here)."
    sent = "Only use an entity's data on behalf of the entity (i.e., only to provide services to that entity and not for your own business purposes or another entity's purposes)."
    sent = "Only use Facebook Login, social plugins, and publishing channels. "
    sent = "All other data may not be transferred outside your app, except to your service provider (per, Section 3.7) who needs that information to provide services to your app. "
    sent = "Delete all of a person’s data you have received from us (including friend data) if that person asks you to, unless you are required to keep it by law, regulation, or separate agreement with us. "
    sent = "Don't use a service provider in connection with your use of Platform unless you make them sign a contract to: (a) protect any user data you obtained from us (that is at least as protective as our terms and policies)."
    sent = "Don't use a service provider in connection with your use of Platform unless you make them sign a contract to: (b) limit their use of that user data solely to using it on your behalf to provide services to your app (and not for their own purposes or any other purposes)."
    sent ="Marketplace Lead Data: Don't use or transfer to a third party (except to your merchant providing the listing) any data you access via the Marketplace Platform lead form features or Lead Generation API or Business Manager (or any successor APIs or technology) except to contact the applicable user about the specific listing for which the data was provided. "
    clauses = split_clauses(predictor, sent,1)
    for c in clauses:
        print(c)
        print("------")


## based on granularity to separate sentence
## granularity == 0: "SBAR"
## granularity == 1: s and "SBAR"
def get_subtrees(granularity,t):
    subtexts = []
    if granularity == 0:
        for subtree in t.subtrees():
            if subtree.label() == "SBAR":
                subtexts.append(' '.join(subtree.leaves()))

    elif granularity == 1:
        for subtree in t.subtrees():
            if subtree.label() == "S" or subtree.label() == "SBAR":
                subtexts.append(' '.join(subtree.leaves()))
    return subtexts


def split_clauses1(predictor, sent,granularity):
    # get the systax parser
    t = testAllenNlp(sent, predictor)
    t.pretty_print()
    # in order to solve ",","." space , keep consistence
    restructure_sent = ' '.join(t.leaves())

    # find all clauses which one can be nested in another, we need to process those.
    subtexts = get_subtrees(granularity, t)

    for i in reversed(range(len(subtexts) - 1)):
        try:
            subtexts[i] = subtexts[i][0:subtexts[i].index(subtexts[i + 1])]
        except:
            continue

    split_clauses = []
    for substring in subtexts:
        # print(substring)
        start = restructure_sent.index(substring)
        end = start + len(substring)
        split_clauses.append(start)
        split_clauses.append(end)
    # append the total length of the sentence to the list to get a complete list
    total_len = len(restructure_sent)
    split_clauses.append(total_len)
    split_clauses.append(0)
    new_list = list(set(split_clauses))
    new_list.sort()

    clauses = []
    for i in range(0, len(new_list) - 1):
        # print(restructure_sent[new_list[i]:new_list[i+1]])
        clauses.append(restructure_sent[new_list[i]:new_list[i + 1]])
    new_clauses = []
    for c in clauses:
        item  = c.strip(punctuation).strip(" , ").strip()
        if len(item) != 0:
            new_clauses.append(item)

    return new_clauses



def split_clauses(predictor, sent,granularity):
    try:
        new_clauses = split_clauses1(predictor, sent, granularity)
        return new_clauses
    except:
        return [sent]


def predict(sent, predictor):
    results = predictor.predict(sentence=sent)
    return results['trees']


def testAllenNlp(sent, predictor):
    tree = predict(sent, predictor)
    constituency_tree = Tree.fromstring(tree)

    return constituency_tree


if __name__ == "__main__":
    main()