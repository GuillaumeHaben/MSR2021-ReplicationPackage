from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle
import sys
import os
import json 
import pandas as pd
import pickle
import numpy as np

# Info
# Metric inspired by https://towardsdatascience.com/metrics-for-imbalanced-classification-41c71549bbb5
# Change parameters below (K and nbTree)

def main():
    # Checks
    checkUsage()

    # Parameters
    k = 10
    nbTrees = 10

    # Load Data
    datasetPath = sys.argv[1]
    data = pd.read_json(datasetPath)
    
    print("Length of data: ", len(data))
    project = "okhttp"
    data = data[(data['ProjectName'] == "/Users/guillaume.haben/Desktop/results-NFT-bodyText/" + project) | (data['ProjectName'] == "/Users/guillaume.haben/Desktop/results-FT-bodyText/" + project)]
    print("Length of new data: ", len(data))

    # Shuffle Data
    data = shuffle(data)

    # Building Tokenizer and Vocabulary
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(data['Body'].values)
    print("\nVocabulary size: ", len(tokenizer.word_index) + 1)

    # Random Forest Model
    classifierKFold = RandomForestClassifier(n_estimators = nbTrees, random_state = 0, verbose=0) 

    X = tokenizer.texts_to_matrix(data['Body'].values, mode='count')
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

    scores = cross_validate(classifierKFold, X, y, cv=k, scoring=scoring, verbose=4, n_jobs=10)
    
    print("\nMetrics with K-fold:",k,", nbTree:",nbTrees)
    
    displayScores(scores['test_precision'], "Precision")
    displayScores(scores['test_recall'], "Recall")
    displayScores(scores['test_f1'], "F1")
    displayScores(scores['test_tnr'], "TNR")
    displayScores(scores['test_fpr'], "FPR")
    displayScores(scores['test_auc'], "AUC")
    displayScores(scores['test_mcc'], "MCC")
    

def displayScores(scores, title):
    print("\nMetric: ", title)
    print("Scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.nanmean(scores), np.nanstd(scores) * 2))

# Scoring definitions

# TN
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]

# FP
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

# FN
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]

#TP
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

# Recall
def recall(y_true, y_pred): return tp(y_true, y_pred) / (tp(y_true, y_pred) + fn(y_true, y_pred))

# FPR
def fpr(y_true, y_pred): return fp(y_true, y_pred) / (fp(y_true, y_pred) + tn(y_true, y_pred))

# TPR
def tpr(y_true, y_pred): return tp(y_true, y_pred) / (tp(y_true, y_pred) + fn(y_true, y_pred))

# TNR
def tnr(y_true, y_pred): return 1 - fpr(y_true, y_pred)

# Precision
def precision(y_true, y_pred): return tp(y_true, y_pred) / (tp(y_true, y_pred) + fp(y_true, y_pred))

# F1
def f1(y_true, y_pred): return 2*tp(y_true, y_pred) / (2*tp(y_true, y_pred) + fp(y_true, y_pred) + fn(y_true, y_pred))

# AUC
def auc(y_true, y_pred): return (tpr(y_true, y_pred) + tnr(y_true, y_pred)) / 2

# MCC
def mcc(y_true, y_pred): return (tp(y_true, y_pred) * tn(y_true, y_pred) - fp(y_true, y_pred) * fn(y_true, y_pred)) / np.sqrt((tp(y_true, y_pred) + fp(y_true, y_pred)) * (tp(y_true, y_pred) + fn(y_true, y_pred)) * (tn(y_true, y_pred) + fp(y_true, y_pred)) * (tn(y_true, y_pred) + fn(y_true, y_pred)))


def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 bow.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()