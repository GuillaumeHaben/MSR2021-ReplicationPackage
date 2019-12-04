import sys
import os
import pprint
import json
import subprocess

# Global Variables
dicFlaky = {}
resultsDirectory = "../results"

def main():
    # Checks
    checkUsage()

    # Setup env
    if not os.path.exists(resultsDirectory):
        os.mkdir(resultsDirectory)

    # Variables
    jsonPath = sys.argv[1]

    # Get dictionary of Flaky Tests
    with open(jsonPath, 'r') as f:
        dicFlaky = json.load(f)

    # Generate a file containing all flaky tests for each commit for each project
    for project in dicFlaky:
        projectFolder = resultsDirectory + "/" + project
        if not os.path.exists(projectFolder):
            os.mkdir(projectFolder)
        for commit in dicFlaky[project]:
            fileName = projectFolder + "/" + commit + ".txt"
            f = open(fileName, "w")
            for flakyTest in dicFlaky[project][commit]:
                f.write(flakyTest + "\n")
            f.close()

def saveResults(dicFlaky):
    with open('flakyTests.json', 'w') as json_file:
        json.dump(dicFlaky, json_file)

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 generateListPaths.py [path/to/flakyTests.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()