from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
import json 
import pandas as pd
import pickle
import numpy as np
from pprint import pprint
from metricUtils import tn, fp, tp, fn, precision, recall, fpr, tpr, tnr, f1, auc, mcc


# Info
# Metric inspired by https://towardsdatascience.com/metrics-for-imbalanced-classification-41c71549bbb5
# Change parameters below (K and nbTree)

def main():
    # Checks
    checkUsage()

    # Parameters
    k = 10
    nbTrees = 50

    # Load Data
    datasetPath = sys.argv[1]
    data = pd.read_json(datasetPath)

    data = data[data["ProjectName"] == "hbase"]
    # data = data[data["Label"] == 0]
    # When test for a false positive:
    # data = data[data["MethodName"] == "testPrepareAdded"]
    print("Data length: ", len(data))
    
    # Shuffle Data
    data = shuffle(data)
    # dataBodyAndCUT = data['Body'].values
    # dataBodyAndCUT = data['Body'].values + data['CUT_1'].values + data['CUT_2'].values + data['CUT_3'].values + data['CUT_4'].values + data['CUT_5'].values
    dataBodyAndCUT = data['CUT_1'].values + data['CUT_2'].values + data['CUT_3'].values + data['CUT_4'].values + data['CUT_5'].values
    # saveResults(dataBodyAndCUT, "F_BODY")
    
    # TF-IDF Approach
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(dataBodyAndCUT)
    # indices = np.argsort(vectorizer.idf_)[::-1]
    # features = vectorizer.get_feature_names()
    # top_n = 25
    # top_features = [features[i] for i in indices[:top_n]]
    
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(X_tfidf.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    finalList = []
    for item in sorted_scores[:25]:
        finalList.append(item[0])

    print("\n[TFIDF]")
    # print("Vocabulary: ", vectorizer.get_feature_names())
    print("Vocabulary size: ", len(vectorizer.get_feature_names()))
    print(finalList)
    
    
    # Building Tokenizer and Vocabulary
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(dataBodyAndCUT)
    print("\n[Keras]")
    print("Vocabulary size: ", len(tokenizer.word_index) + 1)
    print(list(tokenizer.word_index.keys())[:25])

    # Random Forest Model
    classifierKFold = RandomForestClassifier(n_estimators = nbTrees, random_state = 0, verbose=0) 
    # X_keras = tokenizer.texts_to_matrix(dataBodyAndCUT, mode='count')
    y = data['Label'].values

    # Cross validation, K = 10, using stratified folds
    scoring = {
        'precision': make_scorer(precision), 
        'recall': make_scorer(recall), 
        'f1': make_scorer(f1), 
        'auc': make_scorer(auc), 
        'fpr': make_scorer(fpr), 
        'tnr': make_scorer(tnr), 
        'mcc': make_scorer(mcc)
    }

    scores = cross_validate(classifierKFold, X_tfidf, y, cv=k, scoring=scoring, verbose=4, n_jobs=10)
    
    print("\nMetrics with K-fold:",k,", nbTree:",nbTrees)
    displayScoresInline(scores)
    # displayScores(scores['test_precision'], "Precision")
    # displayScores(scores['test_recall'], "Recall")
    # displayScores(scores['test_f1'], "F1")
    # # displayScores(scores['test_tnr'], "TNR")
    # # displayScores(scores['test_fpr'], "FPR")
    # displayScores(scores['test_auc'], "AUC")
    # displayScores(scores['test_mcc'], "MCC")
    

def displayScores(scores, title):
    print("\nMetric: ", title)
    print("Scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.nanmean(scores), np.nanstd(scores) * 2))

def displayScoresInline(scores):
    print(
        round(np.nanmean(scores['test_precision']), 2), ", ", 
        round(np.nanmean(scores['test_recall']), 2), ", ", 
        round(np.nanmean(scores['test_f1']), 2), ", ", 
        round(np.nanmean(scores['test_auc']), 2), ", ", 
        round(np.nanmean(scores['test_mcc']), 2), 
    sep='')
    print(
        round(np.nanstd(scores['test_precision'] * 2), 2), ", ", 
        round(np.nanstd(scores['test_recall'] * 2), 2), ", ", 
        round(np.nanstd(scores['test_f1'] * 2), 2), ", ", 
        round(np.nanstd(scores['test_auc'] * 2), 2), ", ", 
        round(np.nanstd(scores['test_mcc'] * 2), 2), 
    sep='')

def saveResults(dataset, name):
    """
    Save an array to a file.

    Parameters
    ----------
    dataset: the array to save

    Returns
    -------
    Nothing
    """
    filename = "./data/dataset_" + str(name) + ".json"
    with open(filename, 'w') as json_file:
        json.dump(dataset, json_file, cls=NumpyEncoder)
    print("File saved to: ", filename)

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 crossValidateModel.py [path/to/dataset.json]")
        sys.exit(1)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main()