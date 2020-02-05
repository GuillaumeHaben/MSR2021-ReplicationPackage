import sys
import os
import json
import glob
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.model_selection import train_test_split


def main():
    # Checks
    checkUsage()

    dataSetPath = sys.argv[1]

    # Load Data
    data = pd.read_json(dataSetPath)
    print("Data: ")
    print(data) 
    data_train, data_test = train_test_split(data, test_size=0.3)
    print("\n")


    # Build X_train
    x_train = data_train.iloc[:, [1,3,6,7,8,9,10,11,12]].values
    print("X train: ")
    print(x_train)
    print("\n")

    # Build Y_train
    y_train = data_train.iloc[:, 4].values
    print("Y train: ")
    print(y_train)
    print("\n")

    # Build X_test
    x_test = data_test.iloc[:, [1,3,6,7,8,9,10,11,12]].values
    print("X test: ")
    print(x_test)
    print("\n")

    # Build Y_test
    y_test = data_test.iloc[:, 4].values
    print("Y test: ")
    print(y_test)
    print("\n")

    # Create classifier object 
    classifier = RandomForestClassifier(n_estimators = 100, random_state = 0) 
  
    # Fit the regressor with x and y data 
    classifier.fit(x_train, y_train)   


    # Print Features importances
    print("\nFeatures array: [CC, DoI, HasTO, Assert, ,Async Waits, Date, File, Lines, Random, Thread]\n")
    print("Features importances: ", classifier.feature_importances_)

    prediction_test = classifier.predict(x_test)
    prediction_train = classifier.predict(x_train)


    print("\nTraining set metrics: ")
    print("Precision: ", precision_score(y_train, prediction_train))
    print("Recall: ", recall_score(y_train, prediction_train))

    print("\nTest set metrics: ")
    print("Precision: ", precision_score(y_test, prediction_test))
    print("Recall: ", recall_score(y_test, prediction_test))



def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 main.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()