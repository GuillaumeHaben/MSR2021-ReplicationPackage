import sys
import os
import pprint
import json

# Global Variables
dicFlaky = {}

def main():
    # Checks
    checkUsage()

    # Variables
    path = sys.argv[1]
    csvFile = open(path, "r", encoding='utf-8-sig')

    # Logic
    for line in csvFile:
        # Remove new line character and create an array
        line = line.rstrip("\n\r").split(",")

        project = line[0]
        sha = line[1]
        flakyTest = line[2] + "." + line[3]

        dictionaryOfFlakyTests(project, sha, flakyTest)
    
    #Results
    printResults()
    saveResults()

def dictionaryOfFlakyTests(project, sha, flakyTest):
    if flakyTest not in dicFlaky:
        dicFlaky[flakyTest] = [project, sha]

def printResults():
    # Dictionary
    pprint.pprint(dicFlaky)
    print("Number of Flaky Tests: " + str(len(dicFlaky)))

def saveResults():
    with open('flakyTests.json', 'w') as json_file:
        json.dump(dicFlaky, json_file)

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 generateFlakyDictionary.py [path/to/file]")
        sys.exit(1)

if __name__ == "__main__":
    main()