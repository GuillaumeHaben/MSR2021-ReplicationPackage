import sys
import os
import pprint
import json
import subprocess

# Global Variables
projects = {}
finalResults = {}
rootProjectsDirectory = "../projects/"

def main():
    # Checks
    checkUsage()

    # Setup env
    if not os.path.exists(rootProjectsDirectory):
        os.mkdir(rootProjectsDirectory)

    # Variables
    jsonPath = sys.argv[1]
    csvPath = sys.argv[2]
    csvFile = open(csvPath, "r", encoding='utf-8-sig')

    # Get dictionary of Flaky Tests
    with open(jsonPath, 'r') as f:
        dicFlaky = json.load(f)
        listOfSha(dicFlaky)
    
    # Build dictionary of projects and addresses
    for line in csvFile:
        # Get data
        line = line.rstrip("\n\r").split(",")
        project = line[0]
        address = line[1]

        if project not in projects:
            # Add to dictionary
            projects[project] = address

            # Create directory 
            directory = rootProjectsDirectory + project
            if not os.path.exists(directory):
                os.mkdir(directory)
    
    #listOfAddresses()

def listOfAddresses():
    for project in projects:
        address = projects[project]
        print("Project: ", project)
        print("Address: ", address)
 
def listOfSha(dicFlaky):
    for flakyTest in dicFlaky:
        projectName = dicFlaky[flakyTest][0]
        commitNumber = dicFlaky[flakyTest][1]

        # If Project not in dictionary, we add it
        if projectName not in finalResults:
            finalResults[projectName] = {commitNumber: [flakyTest]}
        else:
            # Project exists, If commitNumber not in project, we add it
            if commitNumber not in finalResults[projectName]:
                finalResults[projectName][commitNumber] = [flakyTest]
            # CommitNumber exists, we add flaky test
            else:
                finalResults[projectName][commitNumber].append(flakyTest)
    saveResults(finalResults)

def saveResults(dicFlaky):
    with open('flakyTests.json', 'w') as json_file:
        json.dump(dicFlaky, json_file)

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 3 or not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
        print("Usage: python3 generateFlakyByProjectDictionary.py [path/to/flakyTests.json] [path/to/historical_projects.csv]")
        sys.exit(1)

if __name__ == "__main__":
    main()