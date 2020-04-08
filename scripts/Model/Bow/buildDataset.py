import sys
import os
import glob
import json


def main():
    # Checks
    checkUsage()

    folderPath = sys.argv[1]
    dic = []

    print("Processing folders...")
    for project in os.listdir(folderPath):
        projectPath = os.path.join(folderPath, project)
        print(project)

        for txtFile in os.listdir(projectPath):
            txtFilePath = os.path.join(projectPath, txtFile)
            
            with open(txtFilePath, 'r') as txtFileRead:
                className = txtFile.split(".")[0]
                methodName = txtFile.split(".")[1]
                projectName = projectPath
                body = txtFileRead.read()
                dic.append({"ClassName": className, "MethodName": methodName, "ProjectName": projectName, "Body": body, "Label": 1})
    
    print("Done.")
    # Save new dic 
    saveResults(dic)
    print("File saved to `./dic.json`")

def saveResults(dic):
    with open('dic.json', 'w') as json_file:
        json.dump(dic, json_file) 

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print("Usage: python3 buildDataset.py [path/to/folder]")
        sys.exit(1)

if __name__ == "__main__":
    main()