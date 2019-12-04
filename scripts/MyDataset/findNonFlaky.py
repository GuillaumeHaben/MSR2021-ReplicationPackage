import sys
import os
import json
import glob

def main():
    # Checks
    checkUsage()

    flakyListPath = sys.argv[1]
    allMethodsPath = sys.argv[2]

    dicFlaky = []
    allMethods = []

    dicNonFlaky = []


    with open(flakyListPath, 'r') as flakyListFile:
        flakyTests = json.load(flakyListFile)
        dicFlaky = flakyTests

    with open(allMethodsPath, 'r') as allMethodsFile:
        testMethods = json.load(allMethodsFile)
        allMethods = testMethods
    
    count = 0

    for project in allMethods:
        for classes in project["Classes"]:
            for currentClass in classes:
                for currentMethod in classes[currentClass]:
                    for flakyTest in dicFlaky:
                        if project["ProjectName"] == flakyTest["ProjectName"]:
                            if currentMethod == flakyTest["MethodName"] and currentClass == flakyTest["ClassName"]:
                                count += 1
                                print(currentMethod)
        #print("Project: ", project["ProjectName"])
        #print("Flaky Tests found: ", count)

                
        
    



    saveResults(dicNonFlaky)

def saveResults(dicFlaky):
    with open('nonFlakyTests.json', 'w') as json_file:
        json.dump(dicFlaky, json_file)

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 3 or not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
        print("Usage: python3 findNonFlaky.py [path/to/flakyList.json] [path/to/allMethods.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()