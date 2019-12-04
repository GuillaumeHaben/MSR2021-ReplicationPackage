import sys
import os
import json
import glob

def main():
    # Checks
    checkUsage()

    nonFlakyTestsPath = sys.argv[1]
    dicNonFlaky = []

    with open(nonFlakyTestsPath, 'r') as json_file:
        flakyTest = json.load(json_file)
        dicNonFlaky = flakyTest
    
    for project in dicNonFlaky:
        f = open("./listReady4ME/" + project["ProjectName"].split("/")[-1] + "/master.txt", "w")
        for classes in project["Classes"]:
            for currentClass in classes:
                for currentMethod in classes[currentClass]:
                    print(currentClass + "." + currentMethod)
                    f.write(currentClass + "." + currentMethod + "\n")
        f.close()

        

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 createNFList4ME.py [path/to/nonFlakyTests.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()