from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import sys
import os
import json 
import pandas as pd
import pickle
import joblib


def main():
    # Checks
    checkUsage()

    # Load Data
    datasetPath = sys.argv[1]
    data = pd.read_json(datasetPath)


    # Build training and test set, carefully stratifying the sets to keep a correct distribution of each class
    data_train, data_test = train_test_split(data, test_size=0.3, stratify=data['Label'])

    # Building Tokenizer and Vocabulary
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(data['Body'].values)
    print("\nVocabulary size: ", len(tokenizer.word_index) + 1)

    # Building X_train, y_train, X_test, y_test
    X_train = tokenizer.texts_to_matrix(data_train['Body'].values, mode='count')
    y_train = data_train['Label'].values
    
    X_test = tokenizer.texts_to_matrix(data_test['Body'].values, mode='count')
    y_test = data_test['Label'].values

    # Load Model
    classifier = pickle.load(open(sys.argv[2], 'rb'))

    # Prediction
    prediction_train = classifier.predict(X_train)
    prediction_test = classifier.predict(X_test)


    print("\nTraining set metrics: ")
    print("Precision: ", precision_score(y_train, prediction_train))
    print("Recall: ", recall_score(y_train, prediction_train))
    print("MCC: ", matthews_corrcoef(y_train, prediction_train))

    print("\nTest set metrics: ")
    print("Precision: ", precision_score(y_test, prediction_test))
    print("Recall: ", recall_score(y_test, prediction_test))
    print("MCC: ", matthews_corrcoef(y_test, prediction_test))



def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 3 or not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
        print("Usage: python3 existingBow.py [path/to/dataset.json] [path/to/RFClassifierBoW.sav]")
        sys.exit(1)

if __name__ == "__main__":
    main()