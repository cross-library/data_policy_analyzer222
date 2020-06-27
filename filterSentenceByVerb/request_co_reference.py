import requests
import json
from filterSentenceByVerb.request_co_reference import *
from fake_useragent import UserAgent

def main():
    document = "We are looking for a region of central Italy bordering the Adriatic Sea. The area is mostly mountainous and includes Mt. Corno, the highest peak of the mountain range. It also includes many sheep and an Italian entrepreneur has an idea about how to make a little money of them."
    request_co_reference(document)

### AllenNLP: the provided mdoel is not newest which is not consistent with the results query directly by services
### hence we construt a request to query its services
### TODO: it will be blocked by services as too many request
def request_co_reference(document):
    headers = {
        'authority': 'demo.allennlp.org',
        'accept': 'application/json',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36',
        'content-type': 'application/json',
        'origin': 'https://demo.allennlp.org',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': 'https://demo.allennlp.org/coreference-resolution',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'cookie': '_ga=GA1.2.1200486564.1590360406; _gid=GA1.2.1388054061.1592845943',
    }

    ua = UserAgent()
    headers['User-Agent'] = ua.random

    dic = {}

    dic["document"] = document
    dumpJson = json.dumps(dic)

    response = requests.post('https://demo.allennlp.org/predict/coreference-resolution', headers=headers, data=dumpJson)

    return response.text


if __name__ == "__main__":
    main()