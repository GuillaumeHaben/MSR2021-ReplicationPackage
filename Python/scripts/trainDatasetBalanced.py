
import sys
import os
import json 
import pandas as pd
import pickle
import numpy as np
import pickle
from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score, matthews_corrcoef, recall_score, roc_auc_score, f1_score, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from pprint import pprint
from metricUtils import tn, fp, tp, fn, precision, recall, fpr, tpr, tnr, f1, auc, mcc
from sklearn import tree
from matplotlib import pyplot as plt
from collections import Counter

# Parameters
nbTrees = 100
nbCUT = 0
featureType = "Body" # "Body" or "CodeMetric"
vectorType = "BagOfWords" # "BagOfWords" or "TF-IDF"
trainSplit = 0.8
smote = False


def main():
    # Checks
    checkUsage()

    infoParam()

    # Load Data
    datasetPath = sys.argv[1]
    data = pd.read_json(datasetPath)

    results = {
        "precision": [],
        "recall": [],
        "f1": [],
        "mcc": [],
        "auc": []
    }
    
    for i in range(0, 9):

        # Build sets
        data_train, data_test = buildTrainTestSets(data)
        
        # [TEST] Shuffle sets (no time constraint)
        # data_train, data_test = train_test_split(data, test_size=testSplit, shuffle=True)

        # if featureType == "CodeMetric":
        #     X_train = getFeatures(data_train)
        #     X_test = getFeatures(data_test)
        #     y_train = data_train['Label'].values
        #     y_test = data_test['Label'].values
        # elif featureType == "Body":
            # Tokenize, Vectorize and Build sets
        X_train, X_test, y_train, y_test, tokenizer = tokenizeAndBuildSets(data, data_train, data_test)
        # else:
        #     sys.exit(0)

        
        # SMOTE

        # Before over sampling
        counter = Counter(y_train)
        counterTest = Counter(y_test)
        # print(counter)
        # print(counterTest)
        # if smote == True:
        #     oversample = SMOTE()
        #     X_train, y_train = oversample.fit_resample(X_train, y_train)
        #     # After over sampling
        #     counter = Counter(y_train)
        #     print(counter)


        # Classifier
        classifier = RandomForestClassifier(n_estimators = nbTrees, random_state = 0, verbose=0) 
    
        # Fit model
        classifier = classifier.fit(X_train, y_train)

        # Prediction
        prediction = classifier.predict(X_test)

        # Get Metrics
        prec, rec, f1, mcc, auc = metrics(y_test, prediction)
        results["precision"].append(prec)
        results["recall"].append(rec)
        results["f1"].append(f1)
        results["mcc"].append(mcc)
        results["auc"].append(auc)
    
    displayScores(results)


    # Understanding the features
    # if featureType == "CodeMetric" and nbCUT == 0:
    #     mostImportantMetrics(classifier)
    # elif featureType == "Body":
    mostImportantWords(tokenizer, classifier)

    fp, tn, tp, fn = predictionMatrix(data_test, y_test, prediction)

    pprint(fp)
    # else:
    #     sys.exit(0)

    # [OPTION] Load Model
    # classifier = loadModel("classifier.sav")

    # [OPTION] Save the model to disk
    # saveModel(classifier)

def infoParam():
    print("\n[STEP] Info Params")
    print("Number of Trees:", nbTrees)
    print("Training set size:", trainSplit)
    print("Type of Features:", featureType)
    print("Number of similar methods added to the test:", nbCUT)
    print("SMOTE enable:", smote)
    if featureType == "Body":
        print("Type of Text Representation:", vectorType)
    return

def buildTrainTestSets(data):
    """
    Build Train and Test sets

    Parameters
    ----------
    data: The initial panda frame, from JSON built by buildDataset.py

    Returns
    -------
    data_train: FT and NFT for Train set
    data_test: FT and NFT for Test set
    """
    print("\n[STEP] Build Train and Test sets")
    
    # RQ1 WITH TIME VALIDATION
    # # Get FT and NFT
    # FT = data[data["Label"] == 1]
    # NFT = data[data["Label"] == 0]

    # # Separate FT and NFT in train and test
    # FT_train, FT_test = train_test_split(FT, test_size=1-trainSplit, shuffle=False)
    # if getCommitNFT("train") == getCommitNFT("test"):
    #     NFT_train, NFT_test = train_test_split(NFT, test_size=1-trainSplit, shuffle=False)
    # else:
    #     NFT_train = NFT[NFT["Commit"] == getCommitNFT("train")]
    #     NFT_test = NFT[NFT["Commit"] == getCommitNFT("test")]

    # print("len(NFT_train):", len(NFT_train))
    # print("len(NFT_test):", len(NFT_test))
    # print("len(FT_train):", len(FT_train))
    # print("len(FT_test):", len(FT_test))

    # # Regroup FT and NFT together
    # data_train = FT_train.append(NFT_train)
    # data_test = FT_test.append(NFT_test)

    # RQ1 WITHOUT TIME VALIDATION
    # Get FT and NFT
    FT = data[data["Label"] == 1]
    NFT = data[data["Label"] == 0]
    
    FT_train, FT_test = train_test_split(FT, test_size=1-trainSplit, shuffle=True)
    NFT_train, NFT_test = train_test_split(NFT, test_size=1-trainSplit, shuffle=True)

    print("len(NFT_train):", len(NFT_train))
    print("len(NFT_test):", len(NFT_test))
    print("len(FT_train):", len(FT_train))
    print("len(FT_test):", len(FT_test))

    # Regroup FT and NFT together
    data_train = FT_train.append(NFT_train)
    data_test = FT_test.append(NFT_test)

    # Info
    print("Number of Flaky Tests:", len(FT))
    print("Number of Non Flaky Tests:", len(NFT))
    return data_train, data_test

def tokenizeAndBuildSets(data, data_train, data_test):
    """
    Tokenize Code as Bag of Words or TF-IDF

    Parameters
    ----------
    data: All Tests
    data_train: Tests in Train set
    data_test: Tests in Test set

    Returns
    -------
    X_train, X_test: Vector for Train and Test set
    y_train, y_test: Info about the class Flaky (1) or Non Flaky (0)
    tokenizer: Tokenizer object to use for Feature Understanding
    """
    print("\n[STEP] Tokenize and Build Sets")
    # Get Bodies
    allBody = getFeatures(data)
    trainBody = getFeatures(data_train)
    testBody = getFeatures(data_test)

    if vectorType == "BagOfWords":
        # Building Tokenizer, fit on whole data (Train + Test)
        tokenizer = Tokenizer(lower=True)
        tokenizer.fit_on_texts(allBody)
        X_train = tokenizer.texts_to_matrix(trainBody, mode='count')
        X_test = tokenizer.texts_to_matrix(testBody, mode='count')
        y_train = data_train['Label'].values
        y_test = data_test['Label'].values
        # Info
        print("Vocabulary size:", len(tokenizer.word_index))
        print("X_train size:", len(X_train))
        print("X_test size:", len(X_test))
        return X_train, X_test, y_train, y_test, tokenizer
    elif vectorType == "TF-IDF":
        # Building Tokenizer, fit on whole data (Train + Test)
        tokenizer = TfidfVectorizer()
        tokenizer.fit(allBody)
        X_train = tokenizer.transform(trainBody)
        X_test = tokenizer.transform(testBody)
        y_train = data_train['Label'].values
        y_test = data_test['Label'].values
        # Info
        print("Vocabulary size:", len(tokenizer.get_feature_names()))
        return X_train, X_test, y_train, y_test, tokenizer
    else: sys.exit(0)

def getFeatures(data):
    """
    Get Features from the Test and possibly from the Code Under Test

    Parameters
    ----------
    data: Data Frame

    Returns
    -------
    Body of Tests and possibly their CUT
    """
    # Extend data with new Columns
    # data = extendData(data)
    if nbCUT == 0:
        if featureType == "Body":
            data = data["Body"].values
        return data
    elif nbCUT == 1:
        if featureType == "Body":
            data = data["Body"].values + data["CUT_1"].values
        return data
    elif nbCUT == 2:
        if featureType == "Body":
            data = data["Body"].values + data["CUT_1"].values + data["CUT_2"].values
        return data
    elif nbCUT == 3:
        if featureType == "Body":
            data = data["Body"].values + data["CUT_1"].values + data["CUT_2"].values + data["CUT_3"].values
        return data
    elif nbCUT == 4:
        if featureType == "Body":
            data = data["Body"].values + data["CUT_1"].values + data["CUT_2"].values + data["CUT_3"].values + data["CUT_4"].values
        return data
    elif nbCUT == 5:
        if featureType == "Body":
            data = data["Body"].values + data["CUT_1"].values + data["CUT_2"].values + data["CUT_3"].values + data["CUT_4"].values + data["CUT_5"].values
        return data
    else:
        sys.exit(0)
    
# def extendData(data):
#     data['CUT_1_Body'] = [val["Body"] for val in data["CUT_1"]]
#     data['CUT_1_CyclomaticComplexity'] = [val["CyclomaticComplexity"] for val in data["CUT_1"]]
#     data['CUT_1_HasTimeoutInAnnotations'] = [val["HasTimeoutInAnnotations"] for val in data["CUT_1"]]
#     data['CUT_1_NumberOfAsserts'] = [val["NumberOfAsserts"] for val in data["CUT_1"]]
#     data['CUT_1_NumberOfAsynchronousWaits'] = [val["NumberOfAsynchronousWaits"] for val in data["CUT_1"]]
#     data['CUT_1_NumberOfDates'] = [val["NumberOfDates"] for val in data["CUT_1"]]
#     data['CUT_1_NumberOfFiles'] = [val["NumberOfFiles"] for val in data["CUT_1"]]
#     data['CUT_1_NumberOfLines'] = [val["NumberOfLines"] for val in data["CUT_1"]]
#     data['CUT_1_NumberOfRandoms'] = [val["NumberOfRandoms"] for val in data["CUT_1"]]
#     data['CUT_1_NumberOfThreads'] = [val["NumberOfThreads"] for val in data["CUT_1"]]
#     data['CUT_2_Body'] = [val["Body"] for val in data["CUT_2"]]
#     data['CUT_2_HasTimeoutInAnnotations'] = [val["HasTimeoutInAnnotations"] for val in data["CUT_2"]]
#     data['CUT_2_CyclomaticComplexity'] = [val["CyclomaticComplexity"] for val in data["CUT_2"]]
#     data['CUT_2_NumberOfAsserts'] = [val["NumberOfAsserts"] for val in data["CUT_2"]]
#     data['CUT_2_NumberOfAsynchronousWaits'] = [val["NumberOfAsynchronousWaits"] for val in data["CUT_2"]]
#     data['CUT_2_NumberOfDates'] = [val["NumberOfDates"] for val in data["CUT_2"]]
#     data['CUT_2_NumberOfFiles'] = [val["NumberOfFiles"] for val in data["CUT_2"]]
#     data['CUT_2_NumberOfLines'] = [val["NumberOfLines"] for val in data["CUT_2"]]
#     data['CUT_2_NumberOfRandoms'] = [val["NumberOfRandoms"] for val in data["CUT_2"]]
#     data['CUT_2_NumberOfThreads'] = [val["NumberOfThreads"] for val in data["CUT_2"]]
#     data['CUT_3_Body'] = [val["Body"] for val in data["CUT_3"]]
#     data['CUT_3_CyclomaticComplexity'] = [val["CyclomaticComplexity"] for val in data["CUT_3"]]
#     data['CUT_3_HasTimeoutInAnnotations'] = [val["HasTimeoutInAnnotations"] for val in data["CUT_3"]]
#     data['CUT_3_NumberOfAsserts'] = [val["NumberOfAsserts"] for val in data["CUT_3"]]
#     data['CUT_3_NumberOfAsynchronousWaits'] = [val["NumberOfAsynchronousWaits"] for val in data["CUT_3"]]
#     data['CUT_3_NumberOfDates'] = [val["NumberOfDates"] for val in data["CUT_3"]]
#     data['CUT_3_NumberOfFiles'] = [val["NumberOfFiles"] for val in data["CUT_3"]]
#     data['CUT_3_NumberOfLines'] = [val["NumberOfLines"] for val in data["CUT_3"]]
#     data['CUT_3_NumberOfRandoms'] = [val["NumberOfRandoms"] for val in data["CUT_3"]]
#     data['CUT_3_NumberOfThreads'] = [val["NumberOfThreads"] for val in data["CUT_3"]]
#     data['CUT_4_Body'] = [val["Body"] for val in data["CUT_4"]]
#     data['CUT_4_CyclomaticComplexity'] = [val["CyclomaticComplexity"] for val in data["CUT_4"]]
#     data['CUT_4_HasTimeoutInAnnotations'] = [val["HasTimeoutInAnnotations"] for val in data["CUT_4"]]
#     data['CUT_4_NumberOfAsserts'] = [val["NumberOfAsserts"] for val in data["CUT_4"]]
#     data['CUT_4_NumberOfAsynchronousWaits'] = [val["NumberOfAsynchronousWaits"] for val in data["CUT_4"]]
#     data['CUT_4_NumberOfDates'] = [val["NumberOfDates"] for val in data["CUT_4"]]
#     data['CUT_4_NumberOfFiles'] = [val["NumberOfFiles"] for val in data["CUT_4"]]
#     data['CUT_4_NumberOfLines'] = [val["NumberOfLines"] for val in data["CUT_4"]]
#     data['CUT_4_NumberOfRandoms'] = [val["NumberOfRandoms"] for val in data["CUT_4"]]
#     data['CUT_4_NumberOfThreads'] = [val["NumberOfThreads"] for val in data["CUT_4"]]
#     data['CUT_5_Body'] = [val["Body"] for val in data["CUT_5"]]
#     data['CUT_5_CyclomaticComplexity'] = [val["CyclomaticComplexity"] for val in data["CUT_5"]]
#     data['CUT_5_HasTimeoutInAnnotations'] = [val["HasTimeoutInAnnotations"] for val in data["CUT_5"]]
#     data['CUT_5_NumberOfAsserts'] = [val["NumberOfAsserts"] for val in data["CUT_5"]]
#     data['CUT_5_NumberOfAsynchronousWaits'] = [val["NumberOfAsynchronousWaits"] for val in data["CUT_5"]]
#     data['CUT_5_NumberOfDates'] = [val["NumberOfDates"] for val in data["CUT_5"]]
#     data['CUT_5_NumberOfFiles'] = [val["NumberOfFiles"] for val in data["CUT_5"]]
#     data['CUT_5_NumberOfLines'] = [val["NumberOfLines"] for val in data["CUT_5"]]
#     data['CUT_5_NumberOfRandoms'] = [val["NumberOfRandoms"] for val in data["CUT_5"]]
#     data['CUT_5_NumberOfThreads'] = [val["NumberOfThreads"] for val in data["CUT_5"]]
#     return data

# def getCommitNFT(tset):
#     projectName = sys.argv[1].split("/")[-1].split(".")[-2]
#     if tset == "train":
#         return infoCommitNFT[projectName][trainSplit]
#     elif tset == "test":
#         return infoCommitNFT[projectName][1.0]
#     else:
#         sys.exit(0)

# def mostImportantMetrics(classifier):
#     """
#     Find Most Important Metrics for the Classifier 

#     Parameters
#     ----------
#     classifier: Random Forest Classifier object

#     Returns
#     -------
#     """
#     print("\n[STEP] Most Important Metrics")
#     featureImportances = classifier.feature_importances_
#     featureImportancesSorted = sorted(range(len(featureImportances)), key=lambda k: featureImportances[k], reverse=True)
#     print("featureImportancesSorted", featureImportancesSorted)
#     for i in featureImportancesSorted:
#         print(featuresMeaning[featuresPositionAll[i]])
#     return

def mostImportantWords(tokenizer, classifier):
    """
    Find Most Important Words for the Classifier given the Tokenizer

    Parameters
    ----------
    classifier: Random Forest Classifier object
    tokenizer: Tokenizer object built in tokenizeAndBuildSets

    Returns
    -------
    MostImportantWords: 25 Most important words
    """
    print("\n[STEP] Most Important Words")
    featureImportances = classifier.feature_importances_
    featureImportancesSorted = sorted(range(len(featureImportances)), key=lambda k: featureImportances[k], reverse=True)
    mostImportantFeatures = featureImportancesSorted[:15]

    # Different functions for different tokenizer types
    tokenList = []
    if vectorType == "TF-IDF": 
        print("TF-IDF model:")
        tokenList = list(tokenizer.vocabulary_.keys())
    elif vectorType == "BagOfWords":
        print("Bag Of Words model:")
        tokenList = list(tokenizer.word_index.keys())
    else:
        sys.exit(0)
        
    MostImportantWords = []
    for i in mostImportantFeatures:
        MostImportantWords.append(tokenList[i])

    print("Most Important Words: ", MostImportantWords, "\n")
    return

# def loadModel(fileName):
#     print("\n[STEP] Load model...")
#     return pickle.load(open(fileName, 'rb'))  

# def saveModel(classifier):
#     print("\n[STEP] Saving model...")
#     filename = 'classifier.sav'
#     pickle.dump(classifier, open(filename, 'wb'))
#     print("\nModel saved to: ", filename)
#     return

def metrics(y_test, prediction):
    print("\n[STEP] Metrics")
    print("Precision: ", precision_score(y_test, prediction))
    print("Recall: ", recall_score(y_test, prediction))
    print("F1: ", f1_score(y_test, prediction))
    print("MCC: ", matthews_corrcoef(y_test, prediction))
    print("AUC: ", roc_auc_score(y_test, prediction))
    prec = precision_score(y_test, prediction)
    rec = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    mcc = matthews_corrcoef(y_test, prediction)
    auc = roc_auc_score(y_test, prediction)
    return prec, rec, f1, mcc, auc



def displayScores(scores):
    print("\n[STEP] Overall Results")
    print("Precision: %0.2f (+/- %0.2f)" % (np.nanmean(scores["precision"]), np.nanstd(scores["precision"]) * 2))
    print("Recall: %0.2f (+/- %0.2f)" % (np.nanmean(scores["recall"]), np.nanstd(scores["recall"]) * 2))
    print("F1: %0.2f (+/- %0.2f)" % (np.nanmean(scores["f1"]), np.nanstd(scores["f1"]) * 2))
    print("MCC: %0.2f (+/- %0.2f)" % (np.nanmean(scores["mcc"]), np.nanstd(scores["mcc"]) * 2))
    print("AUC: %0.2f (+/- %0.2f)" % (np.nanmean(scores["auc"]), np.nanstd(scores["auc"]) * 2))

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 model.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()