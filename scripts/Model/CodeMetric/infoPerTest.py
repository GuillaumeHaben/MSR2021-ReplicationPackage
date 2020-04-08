import sys
import os
import json
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Checks
    checkUsage()

    dataSetPath = sys.argv[1]
    dataSet = []

    with open(dataSetPath, 'r') as json_file:
        dataSet = json.load(json_file)
    
    nbFlaky = 0
    nbNonFlaky = 0
    countFlaky = 0
    countNonFlaky = 0

    for test in dataSet:
        if test["Label"] == 1:
            nbFlaky += 1
            if test["CyclomaticComplexity"] > 0  or test["HasTimeoutInAnnotations"] > 0 or test["NumberOfAsserts"] or test["NumberOfAsynchronousWaits"] or test["NumberOfDates"] or test["NumberOfFiles"] or test["NumberOfRandoms"] or test["NumberOfThreads"] :
                countFlaky += 1
        else:
            nbNonFlaky += 1
            if test["CyclomaticComplexity"] > 0  or test["HasTimeoutInAnnotations"] > 0 or test["NumberOfAsserts"] or test["NumberOfAsynchronousWaits"] or test["NumberOfDates"] or test["NumberOfFiles"] or test["NumberOfRandoms"] or test["NumberOfThreads"] :
                countNonFlaky += 1
    print("Number of FLAKY tests having at least one of the characteristic: ", countFlaky, "/", nbFlaky)
    print("Percentage: ", 100 * countFlaky / nbFlaky)
    print("Number of NON FLAKY tests having at least one of the characteristic: ", countNonFlaky, "/", nbNonFlaky)
    print("Percentage: ", 100 * countNonFlaky / nbNonFlaky)


def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 infoPerTest.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()