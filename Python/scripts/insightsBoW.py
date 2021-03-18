from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_validate
from metricUtils import tn, fp, tp, fn, precision, recall, fpr, tpr, tnr, f1, auc, mcc
from sklearn.metrics import make_scorer
from tqdm import tqdm
import sys
import os
import json 
import pandas as pd
from pprint import pprint

def main():
    # Checks
    checkUsage()

    # Load Data
    datasetPath = sys.argv[1]
    data = pd.read_json(datasetPath)
    flaky = data[data["Label"] == True]
    nonFlaky = data[data["Label"] == False]

    body = data['Body'].values
    bodyFlaky = flaky['Body'].values
    bodyNonFlaky = nonFlaky['Body'].values
    bodyAndCut = data['Body'].values + data['CUT_1'].values + data['CUT_2'].values + data['CUT_3'].values + data['CUT_4'].values

    # Build Bag of Words
    tokenizer = Tokenizer(filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(body)

    tokenizerFlaky = Tokenizer(filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizerFlaky.fit_on_texts(bodyFlaky)

    tokenizerNonFlaky = Tokenizer(filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizerNonFlaky.fit_on_texts(bodyNonFlaky)

    tokenizerCut = Tokenizer(filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizerCut.fit_on_texts(bodyAndCut)

    # Information 
    print("\nProject: ", data.iloc[0]["projectName"])
    print("Data length: ", len(data))
    print("Length of vocabulary: ", len(tokenizer.word_counts))
    print("Length of vocabulary with CUT: ", len(tokenizerCut.word_counts))
    print("\nNumber of Flaky: ", len(flaky))
    print("Length of vocabulary: ", len(tokenizerFlaky.word_counts))
    print("\nNumber of Non Flaky: ", len(nonFlaky))
    print("Length of vocabulary: ", len(tokenizerNonFlaky.word_counts))

    # Create and fit classifier, check most important words 
    fitModelAndCheckWords(tokenizer, tokenizerFlaky, tokenizerNonFlaky, body, data)

    # Same but for Test + CUT
    fitModelAndCheckWords(tokenizerCut, tokenizerFlaky, tokenizerNonFlaky, body, data)

        
def fitModelAndCheckWords(tokenizer, tokenizerFlaky, tokenizerNonFlaky, body, data):
    # Model, to get information on most important features
    classifierKFold = RandomForestClassifier(n_estimators = 100, random_state = 0) 
    X = tokenizer.texts_to_matrix(body, mode="count")
    y = data['Label'].values
    classifierKFold.fit(X, y)

    importantWords = featuresUnderstanding(tokenizer, classifierKFold, 10)

    # Further details
    for word in importantWords:
        print(word)
        print("Number of occurence in Flaky Tests", tokenizerFlaky.word_counts.get(word))
        print("Number of occurence in Non Flaky Tests", tokenizerNonFlaky.word_counts.get(word))

def featuresUnderstanding(tokenizer, classifier, num):
    featureImportances = classifier.feature_importances_
    featureImportancesSorted = sorted(range(len(featureImportances)), key=lambda k: featureImportances[k], reverse=True)
    mostImportantFeatures = featureImportancesSorted[:num]
    # mostImportantFeatureIndex = np.argmax(featureImportances)
    # mostImportantFeatureValue = featureImportances[np.argmax(featureImportances)]

    tokenList = list(tokenizer.word_index.keys())

    MostImportantWords = []
    # For 25 Most Important Features
    for i in mostImportantFeatures:
        # Print the corresponding token
        MostImportantWords.append(tokenList[i])

    # print("Features importances: ", featureImportances)
    # print("Features importances sorted: ", featureImportancesSorted)
    # print("Most 25 important features: ", mostImportantFeatures)
    # print("Index of most important feature: ", mostImportantFeatureIndex)
    # print("Value of most important feature: ", mostImportantFeatureValue)
    # print("Corresponding token for Most Important Feature: ", tokenList[mostImportantFeatureIndex])
    print("\nMost Important Words: ", MostImportantWords, "\n")
    return MostImportantWords

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 insightsBoW.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()


    # Dictionaries
    # print("\nword_counts: A dictionary of words and their counts.")
    # print("\nGlobal")
    # print(tokenizer.word_counts)
    # print("\nFlaky")
    # print(tokenizerFlaky.word_counts)
    # print("\nNon Flaky")
    # print(tokenizerNonFlaky.word_counts)
    # print("\nword_docs: A dictionary of words and how many documents each appeared in.")
    # print(tokenizer.document_count)
    # print("\nword_index: A dictionary of words and their uniquely assigned integers.")
    # print(tokenizer.word_index)
    # print("\ndocument_count: An integer count of the total number of documents that were used to fit the Tokenizer.")
    # print(tokenizer.word_docs)