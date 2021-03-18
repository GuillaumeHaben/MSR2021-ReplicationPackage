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
from tqdm import tqdm
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

finalScores = []

def main():
    # Checks
    checkUsage()

    # Parameters
    k = 10
    nbTrees = 100
    modes = ["count"] # ["binary", "count", "tfidf", "freq"]
    numWords = [100, 2000]
    lowerStates = [True] # [True, False]
    cuts = [0] # [0, 4]

    # Load Data
    datasetPath = sys.argv[1]
    data = pd.read_json(datasetPath)

    print("Data length: ", len(data))
    
    for i in tqdm(range(0, len(modes))):
        mode = modes[i]
        for numWord in numWords:
            for lowerState in lowerStates:
                for cut in cuts:

                    # Shuffle Data
                    data = shuffle(data)
                    if cut == 0:
                        body = data['Body'].values
                    if cut == 2:
                        body = data['Body'].values + data['CUT_1'].values + data['CUT_2'].values 
                    if cut == 4:
                        body = data['Body'].values + data['CUT_1'].values + data['CUT_2'].values + data['CUT_3'].values + data['CUT_4'].values
                    
                    # Building Tokenizer and Vocabulary
                    tokenizer = Tokenizer(lower=lowerState, num_words=numWord, filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
                    tokenizer.fit_on_texts(body)
                    # print("Vocabulary size: ", len(tokenizer.word_index) + 1)
                    # print("Most important words:")
                    # print(list(tokenizer.word_index.keys())[:10], "\n")

                    # Random Forest Model
                    classifierKFold = RandomForestClassifier(n_estimators = nbTrees, random_state = 0) 
                    X = tokenizer.texts_to_matrix(body, mode=mode)
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

                    scores = cross_validate(classifierKFold, X, y, cv=k, scoring=scoring, verbose=0, n_jobs=10)
                    
                    # Display results
                    # print("\nMetrics")
                    # displayScores(scores['test_precision'], "Precision")
                    # displayScores(scores['test_recall'], "Recall")
                    # displayScores(scores['test_f1'], "F1")
                    # displayScores(scores['test_auc'], "AUC")
                    # displayScores(scores['test_mcc'], "MCC")

                    # Save score
                    o = {
                        "variables": {
                            "mode": mode,
                            "numWord": numWord,
                            "lowerState": lowerState,
                            "cut": cut,
                        },
                        "Precision": round(np.nanmean(scores['test_precision']), 2),
                        "Recall": round(np.nanmean(scores['test_recall']), 2),
                        "F1": round(np.nanmean(scores['test_f1']), 2),
                        "AUC": round(np.nanmean(scores['test_auc']), 2),
                        "MCC": round(np.nanmean(scores['test_mcc']), 2)
                    }
                    finalScores.append(o)

    # Display final scores
    sortedScores = sorted(finalScores, key=lambda x: x["F1"], reverse=True)

    # Best / Worst config
    print("\nBest configuration:")
    pprint(sortedScores[:1])
    print("\nWorst configuration:")
    pprint(sortedScores[-1:])

    # All config
    # pprint(sortedScores)

def displayScores(scores, title):
    print("\n",title,":", sep="")
    #print("Scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.nanmean(scores), np.nanstd(scores) * 2))


def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 model.py [path/to/dataset.json]")
        sys.exit(1)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main()