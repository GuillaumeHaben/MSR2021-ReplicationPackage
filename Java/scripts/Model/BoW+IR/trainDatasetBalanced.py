
import sys
import os
import json 
import pandas as pd
import pickle
import numpy as np
import pickle
import seaborn as sns
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

# Global var
# Code Metric Features without CUT
featuresPosition = [8,10,13,14,15,16,17,18,19]
# Code Metric Features with 5 CUT
codeMetricPositionCUT1 = [23, 24, 25, 26, 27, 28, 29, 30, 31]
codeMetricPositionCUT2 = [33, 34, 35, 36, 37, 38, 39, 40, 41]
codeMetricPositionCUT3 = [43, 44, 45, 46, 47, 48, 49, 50, 51]
codeMetricPositionCUT4 = [53, 54, 55, 56, 57, 58, 59, 60, 61]
codeMetricPositionCUT5 = [63, 64, 65, 66, 67, 68, 69, 70, 71]
featuresPositionAll = featuresPosition + codeMetricPositionCUT1 + codeMetricPositionCUT2 + codeMetricPositionCUT3 + codeMetricPositionCUT4 + codeMetricPositionCUT5
featuresMeaning = {
    8: "CyclomaticComplexity",
    10: "HasTimeoutInAnnotations",
    13: "NumberOfAsserts",
    14: "NumberOfAsynchronousWaits",
    15: "NumberOfDates",
    16: "NumberOfFiles",
    17: "NumberOfLines",
    18: "NumberOfRandoms",
    19: "NumberOfThreads",
    22: "CUT_1_Body",
    23: "CUT_1_CyclomaticComplexity",
    24: "CUT_1_HasTimeoutInAnnotations",
    25: "CUT_1_NumberOfAsserts",
    26: "CUT_1_NumberOfAsynchronousWaits",
    27: "CUT_1_NumberOfDates",
    28: "CUT_1_NumberOfFiles",
    29: "CUT_1_NumberOfLines",
    30: "CUT_1_NumberOfRandoms",
    31: "CUT_1_NumberOfThreads",
    32: "CUT_2_Body",
    33: "CUT_2_HasTimeoutInAnnotations",
    34: "CUT_2_CyclomaticComplexity",
    35: "CUT_2_NumberOfAsserts",
    36: "CUT_2_NumberOfAsynchronousWaits",
    37: "CUT_2_NumberOfDates",
    38: "CUT_2_NumberOfFiles",
    39: "CUT_2_NumberOfLines",
    40: "CUT_2_NumberOfRandoms",
    41: "CUT_2_NumberOfThreads",
    42: "CUT_3_Body",
    43: "CUT_3_CyclomaticComplexity",
    44: "CUT_3_HasTimeoutInAnnotations",
    45: "CUT_3_NumberOfAsserts",
    46: "CUT_3_NumberOfAsynchronousWaits",
    47: "CUT_3_NumberOfDates",
    48: "CUT_3_NumberOfFiles",
    49: "CUT_3_NumberOfLines",
    50: "CUT_3_NumberOfRandoms",
    51: "CUT_3_NumberOfThreads",
    52: "CUT_4_Body",
    53: "CUT_4_CyclomaticComplexity",
    54: "CUT_4_HasTimeoutInAnnotations",
    55: "CUT_4_NumberOfAsserts",
    56: "CUT_4_NumberOfAsynchronousWaits",
    57: "CUT_4_NumberOfDates",
    58: "CUT_4_NumberOfFiles",
    59: "CUT_4_NumberOfLines",
    60: "CUT_4_NumberOfRandoms",
    61: "CUT_4_NumberOfThreads",
    62: "CUT_5_Body",
    63: "CUT_5_CyclomaticComplexity",
    64: "CUT_5_HasTimeoutInAnnotations",
    65: "CUT_5_NumberOfAsserts",
    66: "CUT_5_NumberOfAsynchronousWaits",
    67: "CUT_5_NumberOfDates",
    68: "CUT_5_NumberOfFiles",
    69: "CUT_5_NumberOfLines",
    70: "CUT_5_NumberOfRandoms",
    71: "CUT_5_NumberOfThreads"
}
infoCommitNFT = {
    "achilles": {0.2: "4466d8d396438b08b3291346a0db2554623db367", 0.6: "4466d8d396438b08b3291346a0db2554623db367", 0.7: "4466d8d396438b08b3291346a0db2554623db367", 0.8: "4466d8d396438b08b3291346a0db2554623db367", 0.9: "a10c468cad70a42ca3e9acdca1f0ad0b6bbfe365", 1.0: "a10c468cad70a42ca3e9acdca1f0ad0b6bbfe365"},
    "hbase": {0.2: "a309d26e8c6636b968f6beb8dab6510a6972f76c", 0.6: "851b6a9cfc98e4e0283f5babea156b8b5298fde2", 0.7: "0b590b5703f22d8a4f831dc20108f136bb8448f0", 0.8: "0b590b5703f22d8a4f831dc20108f136bb8448f0", 0.9: "d380a628bfac05b158a059306edf68baa2b33abd", 1.0: "d59d054fc9abeab776d90709f64bb5bb59d1b673"},
    "logback": {0.2: "147ed05b09fc60430d87aa1da984d7a24092aa6a", 0.6: "16a26fdc6e144de18ee8805724ec7341d3bf06f3", 0.7: "16a26fdc6e144de18ee8805724ec7341d3bf06f3", 0.8: "16a26fdc6e144de18ee8805724ec7341d3bf06f3", 0.9: "16a26fdc6e144de18ee8805724ec7341d3bf06f3", 1.0: "16a26fdc6e144de18ee8805724ec7341d3bf06f3"},
    "okhttp": {0.2: "f3cc7930a5d1e95b707eeb5ddc182746177d570e", 0.6: "f5111c28356431c6c5d1e71dd1bbced9a8016cd8", 0.7: "d614f056145d07cfa77140f03027f385afe57c85", 0.8: "7bb06e78bac05e0e24c6ea81b34aa11f498ad61f", 0.9: "eec5b4a8c942d6468d1f6ba389185116228de63d", 1.0: "58f6cf5130a06e95e9f0ef078abea082417340f7"},
    "oozie":{0.2: "d37ac455bfcfb0c1a13a5235c9ab0705dd546833", 0.6: "88d28bed27bc63df2cfe8f20f0b78c0637f82122", 0.7: "88d28bed27bc63df2cfe8f20f0b78c0637f82122", 0.8: "88d28bed27bc63df2cfe8f20f0b78c0637f82122", 0.9: "264ec14b7e1551ef0147f6072ec068d860cbf2c5", 1.0: "4412cbc22e77dd712e8b5d4d2930c7b278623786"},
    "oryx": {0.2: "7ab42c4fedc1e71eb93891ab572dda34b44ee325", 0.6: "7ab42c4fedc1e71eb93891ab572dda34b44ee325", 0.7: "7ab42c4fedc1e71eb93891ab572dda34b44ee325", 0.8: "7ab42c4fedc1e71eb93891ab572dda34b44ee325", 0.9: "7ab42c4fedc1e71eb93891ab572dda34b44ee325", 1.0: "7ab42c4fedc1e71eb93891ab572dda34b44ee325"},
    "togglz": {0.2: "84cbb08a9eaf2cb3a7688db02a06da62af0b8269", 0.6: "3ad266a54dd2dbe0bdb198fd14d8159fccb4cb26", 0.7: "3ad266a54dd2dbe0bdb198fd14d8159fccb4cb26", 0.8: "3ad266a54dd2dbe0bdb198fd14d8159fccb4cb26", 0.9: "e400470fa98bb8fa832d379af8c62160cbf8814f", 1.0: "e7c7fa01edbb5e52003d7722eaf024e19a5d2a1f"}
}

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
    
    # for i in range(0, 9):

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
    mostImportantWords(tokenizer, classifier)
    words = ['window', 'enqueue', 'store', 'takerequest', 'timeunit', 'httpurlconnection', 'hpackwriter', 'goaway']
    
    fp, tn, tp, fn = predictionMatrix(data_test, y_test, prediction)

    for test in fp:
        for word in words:
            if (word in test["CUT_1"]["Body"] or word in test["CUT_2"]["Body"] or word in test["CUT_3"]["Body"]) and word not in test["Body"]:
                print("\n",word)
                pprint(test["MethodName"])
                pprint(test["ClassName"])
                pprint(test["Commit"])
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
    
    # Get FT and NFT
    FT = data[data["Label"] == 1]
    NFT = data[data["Label"] == 0]

    # RQ1 WITH TIME VALIDATION

    # Separate FT and NFT in train and test
    FT_train, FT_test = train_test_split(FT, test_size=1-trainSplit, shuffle=False)
    if getCommitNFT("train") == getCommitNFT("test"):
        NFT_train, NFT_test = train_test_split(NFT, test_size=1-trainSplit, shuffle=True)
    else:
        NFT_train = NFT[NFT["Commit"] == getCommitNFT("train")]
        NFT_test = NFT[NFT["Commit"] == getCommitNFT("test")]

    # RQ1 WITHOUT TIME VALIDATION

    # Achilles
    # NFT = NFT[NFT["Commit"] == "a10c468cad70a42ca3e9acdca1f0ad0b6bbfe365"]
    # Hbase
    # NFT = NFT[NFT["Commit"] == "d59d054fc9abeab776d90709f64bb5bb59d1b673"]
    # # Okhttp
    # NFT = NFT[NFT["Commit"] == "58f6cf5130a06e95e9f0ef078abea082417340f7"]
    # # Oozie
    # NFT = NFT[NFT["Commit"] == "4412cbc22e77dd712e8b5d4d2930c7b278623786"]
    # # Oryx
    # NFT = NFT[NFT["Commit"] == "7ab42c4fedc1e71eb93891ab572dda34b44ee325"]
    # # Togglz
    # NFT = NFT[NFT["Commit"] == "e7c7fa01edbb5e52003d7722eaf024e19a5d2a1f"]
    # FT_train, FT_test = train_test_split(FT, test_size=1-trainSplit, shuffle=True)
    # NFT_train, NFT_test = train_test_split(NFT, test_size=1-trainSplit, shuffle=True)


    print("len(NFT_train):", len(NFT_train))
    print("len(NFT_test):", len(NFT_test))
    print("len(FT_train):", len(FT_train))
    print("len(FT_test):", len(FT_test))

    # # Regroup FT and NFT together
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
    data = extendData(data)
    if nbCUT == 0:
        if featureType == "Body":
            data = data["Body"].values
        if featureType == "CodeMetric":
            data = data.iloc[:, featuresPosition].values
        return data
    elif nbCUT == 1:
        if featureType == "Body":
            data = data["Body"].values + data["CUT_1_Body"].values
        if featureType == "CodeMetric":
            featuresPositionExtended = featuresPosition + codeMetricPositionCUT1
            data = data.iloc[:, featuresPositionExtended].values
        return data
    elif nbCUT == 2:
        if featureType == "Body":
            data = data["Body"].values + data["CUT_1_Body"].values + data["CUT_2_Body"].values
        if featureType == "CodeMetric":
            featuresPositionExtended = featuresPosition + codeMetricPositionCUT1 + codeMetricPositionCUT2
            data = data.iloc[:, featuresPositionExtended].values
        return data
    elif nbCUT == 3:
        if featureType == "Body":
            data = data["Body"].values + data["CUT_1_Body"].values + data["CUT_2_Body"].values + data["CUT_3_Body"].values
        if featureType == "CodeMetric":
            featuresPositionExtended = featuresPosition + codeMetricPositionCUT1 + codeMetricPositionCUT2 + codeMetricPositionCUT3
            data = data.iloc[:, featuresPositionExtended].values
        return data
    elif nbCUT == 4:
        if featureType == "Body":
            data = data["Body"].values + data["CUT_1_Body"].values + data["CUT_2_Body"].values + data["CUT_3_Body"].values + data["CUT_4_Body"].values
        if featureType == "CodeMetric":
            featuresPositionExtended = featuresPosition + codeMetricPositionCUT1 + codeMetricPositionCUT2 + codeMetricPositionCUT3 + codeMetricPositionCUT4
            data = data.iloc[:, featuresPositionExtended].values
        return data
    elif nbCUT == 5:
        if featureType == "Body":
            data = data["Body"].values + data["CUT_1_Body"].values + data["CUT_2_Body"].values + data["CUT_3_Body"].values + data["CUT_4_Body"].values + data["CUT_5_Body"].values
        if featureType == "CodeMetric":
            featuresPositionExtended = featuresPosition + codeMetricPositionCUT1 + codeMetricPositionCUT2 + codeMetricPositionCUT3 + codeMetricPositionCUT4 + codeMetricPositionCUT5
            data = data.iloc[:, featuresPositionExtended].values
        return data
    else:
        sys.exit(0)
    
def extendData(data):
    data['CUT_1_Body'] = [val["Body"] for val in data["CUT_1"]]
    data['CUT_1_CyclomaticComplexity'] = [val["CyclomaticComplexity"] for val in data["CUT_1"]]
    data['CUT_1_HasTimeoutInAnnotations'] = [val["HasTimeoutInAnnotations"] for val in data["CUT_1"]]
    data['CUT_1_NumberOfAsserts'] = [val["NumberOfAsserts"] for val in data["CUT_1"]]
    data['CUT_1_NumberOfAsynchronousWaits'] = [val["NumberOfAsynchronousWaits"] for val in data["CUT_1"]]
    data['CUT_1_NumberOfDates'] = [val["NumberOfDates"] for val in data["CUT_1"]]
    data['CUT_1_NumberOfFiles'] = [val["NumberOfFiles"] for val in data["CUT_1"]]
    data['CUT_1_NumberOfLines'] = [val["NumberOfLines"] for val in data["CUT_1"]]
    data['CUT_1_NumberOfRandoms'] = [val["NumberOfRandoms"] for val in data["CUT_1"]]
    data['CUT_1_NumberOfThreads'] = [val["NumberOfThreads"] for val in data["CUT_1"]]
    data['CUT_2_Body'] = [val["Body"] for val in data["CUT_2"]]
    data['CUT_2_HasTimeoutInAnnotations'] = [val["HasTimeoutInAnnotations"] for val in data["CUT_2"]]
    data['CUT_2_CyclomaticComplexity'] = [val["CyclomaticComplexity"] for val in data["CUT_2"]]
    data['CUT_2_NumberOfAsserts'] = [val["NumberOfAsserts"] for val in data["CUT_2"]]
    data['CUT_2_NumberOfAsynchronousWaits'] = [val["NumberOfAsynchronousWaits"] for val in data["CUT_2"]]
    data['CUT_2_NumberOfDates'] = [val["NumberOfDates"] for val in data["CUT_2"]]
    data['CUT_2_NumberOfFiles'] = [val["NumberOfFiles"] for val in data["CUT_2"]]
    data['CUT_2_NumberOfLines'] = [val["NumberOfLines"] for val in data["CUT_2"]]
    data['CUT_2_NumberOfRandoms'] = [val["NumberOfRandoms"] for val in data["CUT_2"]]
    data['CUT_2_NumberOfThreads'] = [val["NumberOfThreads"] for val in data["CUT_2"]]
    data['CUT_3_Body'] = [val["Body"] for val in data["CUT_3"]]
    data['CUT_3_CyclomaticComplexity'] = [val["CyclomaticComplexity"] for val in data["CUT_3"]]
    data['CUT_3_HasTimeoutInAnnotations'] = [val["HasTimeoutInAnnotations"] for val in data["CUT_3"]]
    data['CUT_3_NumberOfAsserts'] = [val["NumberOfAsserts"] for val in data["CUT_3"]]
    data['CUT_3_NumberOfAsynchronousWaits'] = [val["NumberOfAsynchronousWaits"] for val in data["CUT_3"]]
    data['CUT_3_NumberOfDates'] = [val["NumberOfDates"] for val in data["CUT_3"]]
    data['CUT_3_NumberOfFiles'] = [val["NumberOfFiles"] for val in data["CUT_3"]]
    data['CUT_3_NumberOfLines'] = [val["NumberOfLines"] for val in data["CUT_3"]]
    data['CUT_3_NumberOfRandoms'] = [val["NumberOfRandoms"] for val in data["CUT_3"]]
    data['CUT_3_NumberOfThreads'] = [val["NumberOfThreads"] for val in data["CUT_3"]]
    data['CUT_4_Body'] = [val["Body"] for val in data["CUT_4"]]
    data['CUT_4_CyclomaticComplexity'] = [val["CyclomaticComplexity"] for val in data["CUT_4"]]
    data['CUT_4_HasTimeoutInAnnotations'] = [val["HasTimeoutInAnnotations"] for val in data["CUT_4"]]
    data['CUT_4_NumberOfAsserts'] = [val["NumberOfAsserts"] for val in data["CUT_4"]]
    data['CUT_4_NumberOfAsynchronousWaits'] = [val["NumberOfAsynchronousWaits"] for val in data["CUT_4"]]
    data['CUT_4_NumberOfDates'] = [val["NumberOfDates"] for val in data["CUT_4"]]
    data['CUT_4_NumberOfFiles'] = [val["NumberOfFiles"] for val in data["CUT_4"]]
    data['CUT_4_NumberOfLines'] = [val["NumberOfLines"] for val in data["CUT_4"]]
    data['CUT_4_NumberOfRandoms'] = [val["NumberOfRandoms"] for val in data["CUT_4"]]
    data['CUT_4_NumberOfThreads'] = [val["NumberOfThreads"] for val in data["CUT_4"]]
    data['CUT_5_Body'] = [val["Body"] for val in data["CUT_5"]]
    data['CUT_5_CyclomaticComplexity'] = [val["CyclomaticComplexity"] for val in data["CUT_5"]]
    data['CUT_5_HasTimeoutInAnnotations'] = [val["HasTimeoutInAnnotations"] for val in data["CUT_5"]]
    data['CUT_5_NumberOfAsserts'] = [val["NumberOfAsserts"] for val in data["CUT_5"]]
    data['CUT_5_NumberOfAsynchronousWaits'] = [val["NumberOfAsynchronousWaits"] for val in data["CUT_5"]]
    data['CUT_5_NumberOfDates'] = [val["NumberOfDates"] for val in data["CUT_5"]]
    data['CUT_5_NumberOfFiles'] = [val["NumberOfFiles"] for val in data["CUT_5"]]
    data['CUT_5_NumberOfLines'] = [val["NumberOfLines"] for val in data["CUT_5"]]
    data['CUT_5_NumberOfRandoms'] = [val["NumberOfRandoms"] for val in data["CUT_5"]]
    data['CUT_5_NumberOfThreads'] = [val["NumberOfThreads"] for val in data["CUT_5"]]
    return data

def getCommitNFT(tset):
    projectName = sys.argv[1].split("/")[-1].split(".")[-2]
    if tset == "train":
        return infoCommitNFT[projectName][trainSplit]
    elif tset == "test":
        return infoCommitNFT[projectName][1.0]
    else:
        sys.exit(0)

def mostImportantMetrics(classifier):
    """
    Find Most Important Metrics for the Classifier 

    Parameters
    ----------
    classifier: Random Forest Classifier object

    Returns
    -------
    """
    print("\n[STEP] Most Important Metrics")
    featureImportances = classifier.feature_importances_
    featureImportancesSorted = sorted(range(len(featureImportances)), key=lambda k: featureImportances[k], reverse=True)
    print("featureImportancesSorted", featureImportancesSorted)
    for i in featureImportancesSorted:
        print(featuresMeaning[featuresPositionAll[i]])
    return

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
    mostImportantFeatures = featureImportancesSorted[:20]

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
    return MostImportantWords

def loadModel(fileName):
    print("\n[STEP] Load model...")
    return pickle.load(open(fileName, 'rb'))  

def saveModel(classifier):
    print("\n[STEP] Saving model...")
    filename = 'classifier.sav'
    pickle.dump(classifier, open(filename, 'wb'))
    print("\nModel saved to: ", filename)
    return

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