import sys
import os
import json
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 


def main():
    # Checks
    checkUsage()

    dataSetPath = sys.argv[1]
    testMethodsPath = sys.argv[2]

    dataSet = pd.read_json(dataSetPath)
    testMethods = pd.read_json(testMethodsPath)


    # Build X_train
    x_train = dataSet.iloc[:, [1,3,6,7,8,9,10,11,12]].values
    print("X train: ")
    print(x_train)
    print("\n")

    # Build Y_train
    y_train = dataSet.iloc[:, 4].values
    print("Y train: ")
    print(y_train)
    print("\n")

    # Build X_test
    x_test = testMethods.iloc[:, [1,3,6,7,8,9,10,11,12]].values

    # Create classifier object 
    classifier = RandomForestClassifier(n_estimators = 100, random_state = 0) 
  
    # Fit the regressor with x and y data 
    classifier.fit(x_train, y_train)   
    
    
    print("\nTest Methods predictions:")
    prediction = classifier.predict_proba(x_test)
    #print(prediction)
    
    # Global Analysis
    flakyTestsFound = []
    nbTests = len(prediction)
    for i in range(0, nbTests):
        el = prediction[i]

        # Count number of Flaky Tests
        if el[0] < 0.5:
            flakyTestsFound.append(i)
        
    print("Number of Tests: ", str(nbTests))
    print("Number of Flaky Tests: ", str(len(flakyTestsFound)))
    print("Percentage: ", str(int(100 * len(flakyTestsFound) / nbTests)),"%")


    # Flaky Tests Found Analysis
    # for flakyTestFoundIndex in flakyTestsFound:
    #     print("\n")
    #     print(testMethods.iloc[flakyTestFoundIndex])

    print(classifier.feature_importances_)


def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 3 or not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
        print("Usage: python3 testModel.py [path/to/dataset.json] [path/to/testMethods.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()