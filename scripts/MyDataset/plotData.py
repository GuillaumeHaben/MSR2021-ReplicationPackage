import sys
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Checks
    checkUsage()

    flakyTestsPath = sys.argv[1]
    

    # Get dictionary of Flaky Tests
    with open(flakyTestsPath, 'r') as f:
        dicFlaky = json.load(f)
    
    CyclomaticComplexity = [i['CyclomaticComplexity'] for i in dicFlaky]
    DepthOfInheritance = [i['DepthOfInheritance'] for i in dicFlaky]
    HasTimeoutInAnnotations = [i['HasTimeoutInAnnotations'] for i in dicFlaky]
    NumberOfAsserts = [i['NumberOfAsserts'] for i in dicFlaky]
    NumberOfAsynchronousWaits = [i['NumberOfAsynchronousWaits'] for i in dicFlaky]
    NumberOfLines = [i['NumberOfLines'] for i in dicFlaky]
    NumberOfThreads = [i['NumberOfThreads'] for i in dicFlaky]
    NumberOfDates = [i['NumberOfDates'] for i in dicFlaky]
    NumberOfRandoms = [i['NumberOfRandoms'] for i in dicFlaky]
    NumberOfFiles = [i['NumberOfFiles'] for i in dicFlaky]

    printInfo(NumberOfLines, "Number of Lines")
    printInfo(NumberOfAsynchronousWaits, "Number of Asynchronous waits")
    printInfo(NumberOfAsserts, "Number of Asserts")
    printInfo(HasTimeoutInAnnotations, "Has Timeout Annotations")
    printInfo(DepthOfInheritance, "Depth of Inheritance")
    printInfo(CyclomaticComplexity, "Cyclomatic Complexity")
    printInfo(NumberOfThreads, "Number of Threads")
    printInfo(NumberOfDates, "Number of Dates")
    printInfo(NumberOfRandoms, "Number of Randoms")
    printInfo(NumberOfFiles, "Number of Files")



def printInfo(entity, title):
    dicOcc = {}
    for el in entity:
        if el not in dicOcc:
            dicOcc[el] = 1
        else:
            dicOcc[el] += 1

    plt.bar(dicOcc.keys(), dicOcc.values(), 1, color='b')
    plt.title(title)
    plt.show()


def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 plotData.py [path/to/flakyTests.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()