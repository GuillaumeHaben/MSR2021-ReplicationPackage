import sys
import os
import glob
import json


def main():
    # Checks
    checkUsage()

    TEST_folderPath = sys.argv[1]
    METHOD_folderPath = sys.argv[2]

    # Variables
    dic = []


    print("Processing TEST folder...")

    for txtFile in os.listdir(TEST_folderPath):
        txtFilePath = os.path.join(TEST_folderPath, txtFile)
        
        with open(txtFilePath, 'r') as txtFileRead:
            className = txtFile.split(".")[0]
            methodName = txtFile.split(".")[1]
            # projectName = projectPath
            body = txtFileRead.read()
            dic.append({"ClassName": className, "MethodName": methodName, "Body": body, "Label": "test"})

    print("Processing METHOD folder...")

    for txtFile in os.listdir(METHOD_folderPath):
        txtFilePath = os.path.join(METHOD_folderPath, txtFile)
        
        with open(txtFilePath, 'r') as txtFileRead:
            className = txtFile.split(".")[0]
            methodName = txtFile.split(".")[1]
            # projectName = projectPath
            body = txtFileRead.read()
            dic.append({"ClassName": className, "MethodName": methodName, "Body": body, "Label": "method"})

    print("Done.")

    # Save new dic 
    saveResults(dic)
    print("File saved to `./dic.json`")

def saveResults(dic):
    with open('dic.json', 'w') as json_file:
        json.dump(dic, json_file) 

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 3 or not os.path.isdir(sys.argv[1]) or not os.path.isdir(sys.argv[2]):
        print("Usage: python3 buildDataset.py [path/to/TEST_folder] [path/to/METHOD_folder]")
        sys.exit(1)

if __name__ == "__main__":
    main()