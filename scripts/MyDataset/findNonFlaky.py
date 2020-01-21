import sys
import os
import json
import glob
import pprint

def main():
    # Checks
    checkUsage()

    flakyListPath = sys.argv[1]
    allMethodsPath = sys.argv[2]

    dicFlaky = []
    dicNonFlaky = []
    dicAllMethods = []

    with open(flakyListPath, 'r') as flakyListFile:
        flakyTests = json.load(flakyListFile)


    # Build Array of All methods.
    # Format: [project.class.method, ...]
    for file in os.listdir(allMethodsPath):
        if file.endswith(".txt"):
            fileName = os.path.join(allMethodsPath, file)

            with open(fileName, 'r') as currentProjectFile:
                count = 0
                for line in currentProjectFile:
                    if count == 0:
                        projectName = line.rstrip()
                        # print(line.rstrip())
                    else:
                        dicAllMethods.append(projectName + "." + line.rstrip())
                    count +=1 

    # Build Array of Flaky Tests.
    # Format: [project.class.method, ...]
    for flakyTest in flakyTests:
        projectName = flakyTest["ProjectName"].split("/")[-1]
        className = flakyTest["ClassName"]
        methodName = flakyTest["MethodName"]
        dicFlaky.append(projectName + "." + className + "." + methodName)
   
    # For el in All Methods, check if it is in Flaky.
    for el in dicAllMethods:
        if el not in dicFlaky:
            dicNonFlaky.append(el)
    
    saveResults(dicNonFlaky)

def saveResults(dic):
    for el in dic:
        projectName = el.split(".")[0]
        className = el.split(".")[1]
        methodName = el.split(".")[2]
        print(projectName)
        print(className)
        print(methodName)
        if not os.path.exists("./NF4ME/" + projectName):
            os.makedirs("./NF4ME/" + projectName)
        else:
            f = open("./NF4ME/" + projectName + "/master.txt", "a+")
            f.write(className + "." + methodName + "\n")
            f.close()


    #with open('nonFlakyTests.json', 'w') as json_file:
    #    json.dump(dic, json_file)

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 3 or not os.path.isfile(sys.argv[1]) or not os.path.isdir(sys.argv[2]):
        print("Usage: python3 findNonFlaky.py [path/to/flakyList.json] [path/to/listTestMethods/]")
        sys.exit(1)

if __name__ == "__main__":
    main()