
## Data_policy_analyzer
The goal of Data Policy Analyzer(DPA) is to extract third-party  data  sharing  policies  from  an  SDK  ToS

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1.Create a clean virtual python environment. It's suggested to use pyenv to build virtualenv(you can also use other tools).
``` 
pyenv virtualenv nlp-3.7.3
``` 

2. We levarage existing NLP techniques to build our tool. Hence we need to install commom NLP tools. (eg., nltk, allennlp, spacy,stanza) and some useful util tool to process tree and graph structure. (eg., ete3,networkx). Please install the requirements before deploying it.
``` 
pip install -r requirements.txt
``` 

## Usage

Here we will explain how to run the each module for this system.

### Part One: web crawler 

The crawler will extract all the text from website and split them into sentence. Put urls you want to crawl in the file "domainList_developer.txt" and specify which folder you want to store the crawl results. Each url will generate a csv file that contains sentences of the webpage.

```
python3 execute.py -i domainList_developer.txt -o /Users/huthvincent/Desktop/

```

### Part Two: customized ner model
The ner model is a sensitive data extracter. You can input a sentence as a paramater, and it will show you the sentensive data entity in the sentence. 

```
python3 extract_sensitive_data.py -i "Do not store Twitter passwords."
```
The output is {'Twitter passwords'}

### Part Three: policy statement discovery
Put the pre-processed Tos docs under the folder "raw data/40_pre_processed_data", then run extract_policy_statement.py. The results will be in the folder "raw data/40_post_processed_data/policy_statement_discovery". The column "predict_label" will shown whether the sentence is related to third party data sharing.

```
python3 extract_policy_statement.py
```
### Part Four: condition extraction
After we found the sentences related to the data sharing and collection. We want to extract the condition of such usage. We run "conditionExtrection/condition_extractor.py" to extract such condition. 

```
python3 condition_extractor.py
```

## Authors

* **Yue Xiao** - *Initial work* - [PurpleBooth](https://github.com/xiaoyue10131748)

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

