from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle
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
    k = 2
    nbTrees = 10

    # Load Data
    datasetPath = sys.argv[1]
    data = pd.read_json(datasetPath)
    
    # Shuffle Data
    data = shuffle(data)
    dataBodyAndCUT = data['Body'].values + data['CUT_1'].values + data['CUT_2'].values + data['CUT_3'].values + data['CUT_4'].values + data['CUT_5'].values

    # Building Tokenizer and Vocabulary
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(dataBodyAndCUT)
    print("\nVocabulary size: ", len(tokenizer.word_index) + 1)

    # Random Forest Model
    classifierKFold = RandomForestClassifier(n_estimators = nbTrees, random_state = 0, verbose=0) 
    X = tokenizer.texts_to_matrix(dataBodyAndCUT, mode='count')
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



def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 crossValidateModel.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()