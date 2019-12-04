import sys
import os
import json
import glob

def main():
    # Checks
    checkUsage()

    path = sys.argv[1]
    dicFlaky = []

    os.chdir(path)
    for jsonFile in glob.glob("*.json"):
        print(jsonFile)
        with open(jsonFile, 'r') as json_file:
            flakyTest = json.load(json_file)
            dicFlaky.append(flakyTest)
    
    os.chdir("../")
    saveResults(dicFlaky)



def saveResults(dicFlaky):
    with open('testMethods.json', 'w') as json_file:
        json.dump(dicFlaky, json_file)

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print("Usage: python3 allInOne.py [path/to/folder]")
        sys.exit(1)

if __name__ == "__main__":
    main()