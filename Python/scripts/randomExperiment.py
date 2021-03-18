import sys
import os
import glob
import json
import pandas as pd
import random


def main():
    # Checks
    checkUsage()

    datasetPath = sys.argv[1]

    # Read dataset.json
    with open(datasetPath, 'r') as json_file:
        data = json.load(json_file)


        # Options
        percentage = 4
        numberOfFlakyToCreate = int( len(data) * percentage / 100 )


        # Creating list of random integers
        randomlist = []
        while len(randomlist) != numberOfFlakyToCreate:
            n = random.randint(0, len(data)-1)
            if n not in randomlist:
                randomlist.append(n)

        # We put all Label to 0
        for test in data:
            test["Label"] = 0

        # We randomly affect test to Label 1
        for i in range(0, len(data)):
            if i in randomlist:
                data[i]["Label"] = 1
        
        # Check up
        c = 0
        for test in data:
            if test["Label"] == 1:
                c += 1


        print("Data length: ", len(data))
        print("Percentage of Flaky Tests wished: ", percentage)
        print("Number of Flaky Tests to create: ", numberOfFlakyToCreate)
        print(c, "Flaky Tests created.")
    saveResults(data, "experiment")


def saveResults(dic, name):
    """Save results to file"""
    fileName = "./" + name + ".json"
    with open(fileName, 'w') as json_file:
        json.dump(dic, json_file, indent=4) 

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 randomExperiment.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()