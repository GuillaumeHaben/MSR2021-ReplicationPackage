from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score 
from sklearn.metrics import matthews_corrcoef, recall_score, roc_auc_score 
from sklearn.model_selection import train_test_split
import sys
import os
import json 
import pandas as pd
import pickle


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

    # Random Forest Model
    classifier = RandomForestClassifier(n_estimators = 10, random_state = 0, verbose=2) 
  
    # Fit the regressor with X_train and y_train 
    classifier.fit(X_train, y_train)   

    # Prediction
    prediction_train = classifier.predict(X_train)
    prediction_test = classifier.predict(X_test)


    print("\nTraining set metrics: ")
    print("Precision: ", precision_score(y_train, prediction_train))
    print("Recall: ", recall_score(y_train, prediction_train))
    print("MCC: ", matthews_corrcoef(y_train, prediction_train))
    print("AUC: ", roc_auc_score(y_train, prediction_train))

    print("\nTest set metrics: ")
    print("Precision: ", precision_score(y_test, prediction_test))
    print("Recall: ", recall_score(y_test, prediction_test))
    print("MCC: ", matthews_corrcoef(y_test, prediction_test))
    print("AUC: ", roc_auc_score(y_test, prediction_test))

    # Save the model to disk
    filename = 'RFClassifierBoW.sav'
    print("\nSaving model to: ", filename)
    pickle.dump(classifier, open(filename, 'wb'))


def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 bow.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()