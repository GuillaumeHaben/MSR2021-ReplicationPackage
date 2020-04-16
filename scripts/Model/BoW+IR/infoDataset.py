import sys
import os
import glob
import json
import pandas as pd


def main():
    # Checks
    checkUsage()

    datasetPath = sys.argv[1]

    # Read dataset.json
    with open(datasetPath, 'r') as json_file:
        data = json.load(json_file)
        ft = []
        nft = []
        projects = set()

        for test in data:
            projects.add(test["ProjectName"])
            if test["Label"] == 1:
                ft.append(test)
            if test["Label"] == 0:
                nft.append(test)

        print("Info about dataset: ", datasetPath)
        print("Number of Projets: ", len(projects))
        print("Number of total tests: ", len(data))
        print("Number of Flaky Tests: ", len(ft))
        print("Number of Non Flaky Tests: ", len(nft))

        # Check if FT are in NFT
        # for flakyTest in ft:
        #     for nonFlakyTest in nft:
        #         if flakyTest["MethodName"] == nonFlakyTest["MethodName"] and flakyTest["ClassName"] == nonFlakyTest["ClassName"]:
        #             print(flakyTest)

        # for project in projects:
        #     infoPerProject(data, project)
    


def infoPerProject(data, project):
    ft = []
    nft = []

    print("[PROJECT] ", project)

    for test in data:
        if test["ProjectName"] == project:
            if test["Label"] == 1:
                ft.append(test)
            if test["Label"] == 0:
                nft.append(test)
                
    print("Number of Flaky Tests: ", len(ft))
    print("Number of Non Flaky Tests: ", len(nft))
    print("_______________")

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 infoDataset.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()