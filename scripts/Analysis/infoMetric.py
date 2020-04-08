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
    
    # List of metrics
    CyclomaticComplexity = [i['CyclomaticComplexity'] for i in dicFlaky]
    DepthOfInheritance = [i['DepthOfInheritance'] for i in dicFlaky]
    HasTimeoutInAnnotations = [i['HasTimeoutInAnnotations'] for i in dicFlaky]
    NumberOfAsserts = [i['NumberOfAsserts'] for i in dicFlaky]
    NumberOfAsynchronousWaits = [i['NumberOfAsynchronousWaits'] for i in dicFlaky]
    NumberOfLines = [i['NumberOfLines'] for i in dicFlaky]
    
    # Sort lists
    CyclomaticComplexity.sort()
    DepthOfInheritance.sort()
    HasTimeoutInAnnotations.sort()
    NumberOfAsserts.sort()
    NumberOfAsynchronousWaits.sort()
    NumberOfLines.sort()

    # Threshold
    Threshold = 0.6
    CyclomaticComplexityThreshold = 2
    DepthOfInheritanceThreshold = DepthOfInheritance[int(len(DepthOfInheritance)*Threshold)]
    HasTimeoutInAnnotationsThreshold = 1
    NumberOfAssertsThreshold = NumberOfAsserts[int(len(NumberOfAsserts)*Threshold)]
    NumberOfAsynchronousWaitsThreshold = 1
    NumberOfLinesThreshold = NumberOfLines[int(len(NumberOfLines)*Threshold)]

    print(CyclomaticComplexityThreshold)
    print(DepthOfInheritanceThreshold)
    print(HasTimeoutInAnnotationsThreshold)
    print(NumberOfAssertsThreshold)
    print(NumberOfAsynchronousWaitsThreshold)
    print(NumberOfLinesThreshold)

    finalFlakyTests = []

    for flakyTest in dicFlaky:

        CyclomaticComplexity = flakyTest['CyclomaticComplexity']
        DepthOfInheritance = flakyTest['DepthOfInheritance']
        HasTimeoutInAnnotations = flakyTest['HasTimeoutInAnnotations']
        NumberOfAsserts = flakyTest['NumberOfAsserts']
        NumberOfAsynchronousWaits = flakyTest['NumberOfAsynchronousWaits']
        NumberOfLines = flakyTest['NumberOfLines']

        if CyclomaticComplexity >= CyclomaticComplexityThreshold or DepthOfInheritance >= DepthOfInheritanceThreshold or HasTimeoutInAnnotations >= HasTimeoutInAnnotationsThreshold or NumberOfAsserts >= NumberOfAssertsThreshold or NumberOfAsynchronousWaits >= NumberOfAsynchronousWaitsThreshold or NumberOfLines >= NumberOfLinesThreshold:
            finalFlakyTests.append(flakyTest)
    
    print("Length of Initial Flaky Tests", len(dicFlaky))
    print("Length of Final Flaky Tests: ", len(finalFlakyTests))
    print("Percentage: ", int(100 * len(finalFlakyTests) / len(dicFlaky)), "%")




def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 infoMetric.py [path/to/flakyTests.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()