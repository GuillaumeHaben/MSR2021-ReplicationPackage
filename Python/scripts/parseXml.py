from lxml import etree
from lxml import objectify
from functionObject import FunctionObject
from collections import Counter
from pprint import pprint
from tqdm import tqdm
import json
import sys
import os
import traceback

# Parse report of XML file generated with : 
# for i in {0..9}; do pytest -p no:flaky --junitxml=../../results/PROJECT/report_$i.xml; done
# -p no:flaky, to disable @flaky
# -rA to print summary info about all tests
# --junitxml to specify the path of the report file.

def main():
    checkUsage()

    # Variables
    folderPath = sys.argv[1]
    allFailures = []
    numberOfFiles = 0
    manifestFlakyTest = []

    # For every report file
    for report in os.listdir(folderPath):
        numberOfFiles += 1
        filePath = os.path.join(folderPath, report)
        # print("Processing:", filePath)
        if os.path.isfile(filePath):
            # Get failing test lists
            failList = process(filePath)
            # For each failing test
            for failTest in failList:
                # Add it to list of all failures
                allFailures.append(failTest)

    # A flaky test is a test that, accross all reruns, failed at least once, and passed at least once.
    # For test that failed at least once, if they failed less than the number of reruns, they are flaky
    for k, v in Counter(allFailures).items():
        print(k.getFunctionName(), k.getClassName(), v)
        if v < numberOfFiles:
            manifestFlakyTest.append(k)


    # Print number of test that failed at least once (== that are in allFailures) and that passed at least once
    print("\nNumber of reruns:", numberOfFiles)
    print("Number of test that failed at least once:", len(Counter(allFailures)))
    print("Number of manifest Flaky Tests:", len(manifestFlakyTest))

    manifestFlakyTest = [f.toJSON() for f in manifestFlakyTest]
    saveManifestFTToFile(manifestFlakyTest)

def process(file):
    # Parse File
    tree = objectify.parse(file)
    root = tree.getroot()

    # TODO: Handle cases where testsuites.getchildren > 1
    if root.tag == "testsuites":
        root = root.testsuite

    # Set Variables
    failList = []
    
    for testcase in root.getchildren():
        testObject = FunctionObject()
        testObject.setFunctionName(testcase.attrib["name"])
        testObject.setClassName(testcase.attrib["classname"])
        # TODO: Is error considered a flaky failure? -> No because if it passes once and then fails, it is still a flaky test
        if hasattr(testcase, "failure") or hasattr(testcase, "error"):
            failList.append(testObject)
            
    return failList

def alreadyInList(test, testList):
    isInList = False
    for element in testList:
        if test.__eq__(element):
            isInList = True
    return isInList

def saveManifestFTToFile(manifestFlakyTest):
    project = sys.argv[1].split("/")[-1]
    print(project)
    with open("./manifest/manifestFT." + str(project) + ".json", "w") as json_file:
        json.dump(manifestFlakyTest, json_file, indent=4)

def checkUsage():
    """Check the programs' arguments"""
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print("Usage: python main.py [path/to/reportFolder]")
        sys.exit(1)

if __name__ == "__main__":
    main()