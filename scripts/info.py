import sys
import os
import pprint

# Global Variables
flakyTests = []
projects = []
dicFlaky = {}
dicShaToDate = {}
finalDic = {}

def main():
    # Checks
    checkUsage()

    # Variables
    path = sys.argv[1]
    csvFile = open(path, "r")

    # Logic
    for line in csvFile:
        # Remove new line character and create an array
        line = line.rstrip("\n\r").split(",")

        project = line[0]
        sha = line[1]
        flakyTest = line[2] + "." + line[3]

        numberOfProjects(project)
        numberOfFlakyTests(flakyTest)
        dictionaryOfFlakyTests(project, sha, flakyTest)
    checkDate()

    
    #Results
    printResults()
  
def numberOfProjects(project):
    if project not in projects:
        projects.append(project)

def numberOfFlakyTests(flakyTest):
    if flakyTest not in flakyTests:
        flakyTests.append(flakyTest)

def dictionaryOfFlakyTests(project, sha, flakyTest):
    if flakyTest not in dicFlaky:
        dicFlaky[flakyTest] = [sha]
    else:
        dicFlaky[flakyTest].append(sha)

def checkDate():
    csvFileWithDate = open("../data/historical_project_versions.csv", "r")
    for line in csvFileWithDate:
        # Remove new line character and create an array
        line = line.rstrip("\n\r").split(",")

        project = line[0]
        shaFile = line[1]
        date = line[2]

        if shaFile not in dicShaToDate:
            dicShaToDate[shaFile] = date
        else:
            print("ERROR, sha already in dicShaToDate")
            sys.exit(1)

    # pprint.pprint(dicShaToDate)

    for test in dicFlaky:
        for sha in dicFlaky[test]:
            print(test)
            print(sha)
            print(dicShaToDate[sha])
            # Adding the date of sha to dicFlaky
            # print(dicFlaky[test])
            if test not in finalDic:
                finalDic[test] = [[sha, dicShaToDate[sha]]]
            else:
                finalDic[test].append([sha, dicShaToDate[sha]])


def printResults():
    # Projects
    # print(projects)
    # print(len(projects))

    # Flaky Tests
    # print(flakyTests)
    # print(len(flakyTests))
    
    # Dictionary
    pprint.pprint(finalDic)
    print(len(finalDic))

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 info.py [path/to/file]")
        sys.exit(1)

if __name__ == "__main__":
    main()