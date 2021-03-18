import sys
import os
import json
import glob

def main():
    # Checks
    checkUsage()

    datasetPath = sys.argv[1]
    dic = []
    dicWdoi = []

    # Put current dataset in dic
    with open(datasetPath, 'r') as json_file:
        dic = json.load(json_file)
    
    # Go through dataset
    for test in dic:
        if (test["DepthOfInheritance"] == 0):
            dicWdoi.append(test)
            
    # Save new dic 
    saveResults(dicWdoi)

def saveResults(dic):
    with open('dataset.wDoI.json', 'w') as json_file:
        json.dump(dic, json_file)    

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 getOnlyDoi0.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()