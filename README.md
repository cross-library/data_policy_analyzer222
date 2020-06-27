
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

### Part one: customized ner model
The ner model is a sensitive data extracter. Put the sentences in to ./data/inputSentence.txt, then run extract_sensitive_data.py. For example, "Do not store Twitter passwords." the output will be {'Twitter passwords'}
```
python3 extract_sensitive_data.py -i "Do not store Twitter passwords."
```
### Part two: policy statement discovery
Put the pre-processed Tos docs under the folder "raw data/40_pre_processed_data", then run extract_policy_statement.py. The results will be in the folder "raw data/40_post_processed_data/policy_statement_discovery". The column "predict_label" will shown whether the sentence is related to third party data sharing.

```
python3 extract_policy_statement.py
```
### Part three: condition extraction
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

