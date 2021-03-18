from analyzer import Analyzer
from functionObject import FunctionObject
import astor
import ast
import sys
import os
from pprint import pprint
from tqdm import tqdm
import traceback
import pytest

folderPath = sys.argv[1]
filePath = sys.argv[2]

def main():
    checkUsage()

    # Get All Test

    getAllTests(filePath)

    # Get All Test annotated Flaky

    getAllFlakyTests()

    # Get All Test non annotated Flaky

    # Check the sets

def getAllTests(filePath):
    print("[STEP] Get All Tests")
    print("Processing:", filePath)
    with open(filePath, 'r') as f:
        collections = f.readlines()

    allTests = []
    currentModule = ""
    currentClass = ""
    currentFunction = ""
    
    for line in collections:
        line = line.strip()
        if line.startswith("<Module"):
            currentModule = line.replace("<Module ", "")[:-1]
            currentClass = ""
        if line.startswith("<Class"):
            currentClass = line.replace("<Class ", "")[:-1]
        if line.startswith("<Function"):
            currentFunction = line.replace("<Function ", "")[:-1]
            # Avoid parametrized test
            currentFunction = currentFunction.split("[")[0]
            # Create test object
            test = FunctionObject()
            test.setFunctionName(currentFunction)
            test.setClassName(currentClass)
            test.setFileName(currentModule)
            # Add to list
            if not alreadyInList(test, allTests):
                allTests.append(test)
            else:
                test.toMinString()
    
    print("Done.")
    print("Number of items:", len(allTests))

def getAllFlakyTests():
    return

def alreadyInList(currentTest, testsList):
        """ Check if test object is already in list """
        isInList = False
        for element in testsList:
            if currentTest.__eq__(element):
                isInList = True
        return isInList

# Check arguments
def checkUsage():
    """Check the programs' arguments"""
    if len(sys.argv) != 3 or not os.path.isdir(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
        print("Usage: python buildDataset.py [path/to/project] [path/to/collection.txt]")
        sys.exit(1)

if __name__ == "__main__":
    main()