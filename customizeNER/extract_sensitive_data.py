from customizeNER.DataEntity import *


def readData():
    f = open("data/inputSentence.txt")
    content = f.readlines()
    f.close()
    return content




def main():
    sentences = readData()
    for s in sentences:
        dataObject = DataEntity(s)
        print(dataObject.sensitive_data)


if __name__ == "__main__":
    main()