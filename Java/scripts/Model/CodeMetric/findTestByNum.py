import sys
import os
import json
import glob
import pandas as pd


def main():
    # Checks
    checkUsage()

    dataSetPath = sys.argv[1]
    dataSet = pd.read_json(dataSetPath)

    num = int(sys.argv[2])

    print("Test Number: ", num)
    print(dataSet.iloc[num])


def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 3 or not os.path.isfile(sys.argv[1]) or not isinstance(int(sys.argv[2]), int):
        print("Usage: python3 findTestByNum.py [path/to/flakyTests.json] num")
        sys.exit(1)

if __name__ == "__main__":
    main()