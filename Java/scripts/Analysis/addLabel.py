import sys
import os
import json
import glob

def main():
    # Checks
    checkUsage()

    nonFlakyTestsPath = sys.argv[1]
    dicNonFlaky = []

    with open(nonFlakyTestsPath, 'r') as json_file:
        nonFlakyTest = json.load(json_file)
        dicNonFlaky = nonFlakyTest
    
    for test in dicNonFlaky:
        test["Label"] = 0

    saveResults(dicNonFlaky)

def saveResults(dicFlaky):
    with open('nonFlakyTest+Label.json', 'w') as json_file:
        json.dump(dicFlaky, json_file)    

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 addLabel.py [path/to/nonFlakyTests.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()