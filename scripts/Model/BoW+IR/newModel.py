from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score, matthews_corrcoef, recall_score, roc_auc_score, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
import json 
import pandas as pd
import pickle
import numpy as np
from pprint import pprint
from metricUtils import tn, fp, tp, fn, precision, recall, fpr, tpr, tnr, f1, auc, mcc
import pickle
from sklearn import tree
from matplotlib import pyplot as plt
import seaborn as sns

def main():
    # Checks
    checkUsage()

    # Parameters
    nbTrees = 50

    # Load Data
    datasetPath = sys.argv[1]
    data = pd.read_json(datasetPath)
    
    # Shuffle Data
    # data = shuffle(data)

    # Work only on Oozie project
    # data = data[data["ProjectName"] == "oozie"]
    data_train, data_test = train_test_split(data, test_size=0.1, stratify=data['Label'])

    # Get Bodies, split Train and Test
    dataBody = getTestAndCUT(data)
    dataBody_train = getTestAndCUT(data_train)
    dataBody_test = getTestAndCUT(data_test)

    print("\n#### Vocabulary size: \n")

    # Building Tokenizer, fit on whole data (Train + Test)
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(dataBody)
    print("Bag Of Words: ", len(tokenizer.word_index))

    # TF-IDF Approach
    vectorizer = TfidfVectorizer()
    vectorizer.fit(dataBody)
    print("TF-IDF: ", len(vectorizer.get_feature_names()))
    
    # Random Forest Model
    bow_classifier = RandomForestClassifier(n_estimators = nbTrees, random_state = 0, verbose=0) 
    tfidf_classifier = RandomForestClassifier(n_estimators = nbTrees, random_state = 0, verbose=0) 
    
    # Tokenize data
    # BoW
    X_train_bow = tokenizer.texts_to_matrix(dataBody_train, mode='count')
    X_test_bow = tokenizer.texts_to_matrix(dataBody_test, mode='count')
    # TF-IDF
    X_train_tfidf = vectorizer.transform(dataBody_train)
    X_test_tfidf = vectorizer.transform(dataBody_test)

    y_train = data_train['Label'].values
    y_test = data_test['Label'].values
    
    # Fit model
    bow_classifier.fit(X_train_bow, y_train)
    tfidf_classifier.fit(X_train_tfidf, y_train)

    # Prediction
    prediction_bow = bow_classifier.predict(X_test_bow)
    prediction_tfidf = tfidf_classifier.predict(X_test_tfidf)

    # Understand TN, TP, FN, FP
    print("\n#### Prediction Matrix")
    print("\nBag Of Words model:")
    fp, tn, tp, fn = predictionMatrix(data_test, y_test, prediction_bow)
    # print("\n Bag Of Words model:")
    # predictionMatrix(data_test, y_test, prediction_tfidf)

    # Get Metrics
    print("\n#### Metrics:")
    print("\nBag Of Words model:")
    metrics(y_test, prediction_bow)
    print("\nTF-IDF model:")
    metrics(y_test, prediction_tfidf)

    # Understanding the features
    print("\n#### Features understanding:\n")
    MostImportantWordsInBoW = featuresUnderstanding(tokenizer, bow_classifier, "tokenizer")
    MostImportantWordsInTFIDF = featuresUnderstanding(vectorizer, tfidf_classifier, "vectorizer")

    # tpAnalysis(MostImportantWordsInBoW, tp)
    # sandbox(data, datasetPath, MostImportantWordsInBoW)

    # visualizeTree()

    # Save the model to disk
    # saveModel(classifier)

    # Load Model
    # classifier = loadModel()

def getTest(data):
    return data['Body'].values

def getTestAndCUT(data):
    return data['Body'].values + data['CUT_1'].values + data['CUT_2'].values + data['CUT_3'].values + data['CUT_4'].values + data['CUT_5'].values

def getCUT(data):
    return data['CUT_1'].values + data['CUT_2'].values + data['CUT_3'].values + data['CUT_4'].values + data['CUT_5'].values

def loadModel():
    return pickle.load(open(sys.argv[2], 'rb'))  

def saveModel(classifier):
    filename = 'RFClassifierBoW+CUT.sav'
    print("\nSaving model to: ", filename)
    pickle.dump(classifier, open(filename, 'wb'))

def featuresUnderstanding(tokenizer, classifier, tokenType):
    featureImportances = classifier.feature_importances_
    featureImportancesSorted = sorted(range(len(featureImportances)), key=lambda k: featureImportances[k], reverse=True)
    mostImportantFeatures = featureImportancesSorted[:25]
    # mostImportantFeatureIndex = np.argmax(featureImportances)
    # mostImportantFeatureValue = featureImportances[np.argmax(featureImportances)]

    tokenList = []
    if tokenType == "vectorizer": 
        print("TF-IDF model:")
        tokenList = list(tokenizer.vocabulary_.keys())
    else:
        print("Bag Of Words model:")
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
    print("Most Important Words: ", MostImportantWords, "\n")
    return MostImportantWords

def metrics(y_test, prediction):
    print("Precision: ", precision_score(y_test, prediction))
    print("Recall: ", recall_score(y_test, prediction))
    print("MCC: ", matthews_corrcoef(y_test, prediction))
    print("AUC: ", roc_auc_score(y_test, prediction))
    print("Test set size: ", len(prediction))

def predictionMatrix(data_test, y_test, prediction):
    # This line only gives the numbers
    # tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
    
    FP = []
    TN = []
    TP = []
    FN = []

    for i in range(len(prediction)): 
        if prediction[i] == 1 and y_test[i] != prediction[i]:
           FP.append(data_test.iloc[i,:])
        if y_test[i] == prediction[i] == 0:
           TN.append(data_test.iloc[i,:])
        if y_test[i] == prediction[i] == 1:
           TP.append(data_test.iloc[i,:])
        if prediction[i] == 0 and y_test[i] != prediction[i]:
           FN.append(data_test.iloc[i,:])

    print("FP: ", len(FP))
    print("TN: ", len(TN))
    print("TP: ", len(TP))
    print("FN: ", len(FN))
    return FP, TN, TP, FN

def tpAnalysis(MostImportantWords, tp):
    print("\nOne sample TP:\n")
    print(tp[0]["ClassName"] + "." + tp[0]["MethodName"])
    print(tp[0]["Body"])
    
    c = 0
    for word in MostImportantWords:
        if word in tp[0]["Body"]:
            c += 1
            print(word)
    print("Number of Important words present in test body: ", c)
    return

def sandbox(data, datasetPath, MostImportantWords):
    print("#### Sandbox:\n")

    FT = []
    NFT = []
    dicFT = {}
    dicNFT = {}
    totalLenFT = 0
    totalLenNFT = 0

    with open(datasetPath, 'r') as json_file:
        data = json.load(json_file)

        for test in data:
            # Add info about presence of common words
            for word in MostImportantWords:
                body = test["Body"]
                test[word] = body.count(word) / len(body)
            # Add test to FT or NFT set
            if test["ProjectName"] == "oozie":
                if test["Label"] == 1:
                    FT.append(test)
                if test["Label"] == 0:
                    NFT.append(test)
    print(FT[0].keys())
    # Init dictionary
    # for word in MostImportantWords:
    #     dicFT[word] = 0
    #     dicNFT[word] = 0

    # for test in FT:
    #     totalLenFT += len(test["Body"])
    #     for comWordPres in test["CommonWordsPresence"]:
    #         dicFT[comWordPres] += test["CommonWordsPresence"][comWordPres]
    # averageLenFT = totalLenFT / len(FT)

    # for test in NFT:
    #     totalLenNFT += len(test["Body"])
    #     for comWordPres in test["CommonWordsPresence"]:
    #         dicNFT[comWordPres] += test["CommonWordsPresence"][comWordPres]
    # averageLenNFT = totalLenNFT / len(NFT)

    # print("Flaky Tests:\n")
    # pprint(dicFT)
    # print("Number of tests: ", len(FT))
    # print("Average Length Body: ", averageLenFT)


    # print("\nNon Flaky Tests:\n")
    # pprint(dicNFT)
    # print("Number of tests: ", len(NFT))
    # print("Average Length Body: ", averageLenNFT)

    # Seaborn here
    TESTS = FT + NFT


    df = pd.DataFrame(TESTS)

    ax = sns.boxplot(x="Label", y="new", data=df, linewidth=1)
    plt.show()

    # print("\nWord \"",MostImportantWords[0], "\" present in", counterFT, "out of", len(FT), "flaky tests.", round(100 * counterFT / len(FT), 2) ,"%")
    # print("Word \"",MostImportantWords[0], "\" present in", counterNFT, "out of", len(NFT), "non flaky tests.", round(100 * counterNFT / len(NFT), 2) ,"%")

    return

def visualizeTree(bow_classifier, tokenizer):
    print("\n")
    print(bow_classifier.estimators_[0])

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (32,32), dpi=300)
    cn = ['Non Flaky', 'Flaky']
    fn = list(tokenizer.word_index.keys())

    tree.plot_tree(bow_classifier.estimators_[0], 
    class_names=cn, feature_names=fn)
    fig.savefig('visTree.png')

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 model.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()