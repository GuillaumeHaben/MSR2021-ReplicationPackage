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
import pickle

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
    classifier = RandomForestClassifier(n_estimators = nbTrees, random_state = 0, verbose=0) 
    X = tokenizer.texts_to_matrix(dataBodyAndCUT, mode='count')
    y = data['Label'].values

    # Fit model
    classifier.fit(X, y)   

    # Save the model to disk
    filename = 'RFClassifierBoW+CUT.sav'
    print("\nSaving model to: ", filename)
    pickle.dump(classifier, open(filename, 'wb'))

    # Load Model
    # classifier = pickle.load(open(sys.argv[2], 'rb'))


def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 crossValidateModel.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()