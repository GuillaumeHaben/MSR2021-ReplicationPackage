import sys
import os
import glob
import json
import pprint
import subprocess
from subprocess import STDOUT
import shutil
from tqdm import tqdm
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def main():
    # Checks
    checkUsage()

    # VARIABLES
    sourcesPath = sys.argv[1]
    listPath = sys.argv[2]
    nbSimilarMethods = int(sys.argv[3])
    dataset = []

    print("\n[STEP] main")

    projectSources = [ f.path for f in os.scandir(sourcesPath) if f.is_dir() ]
    projectList = [ f.path for f in os.scandir(listPath) if f.is_dir() ]

    counterProject = 0
    for projectSource in projectSources:
        counterProject += 1
        # TO TEST, WITH SMALL PROJECT, TO REMOVE LATER
        # if getProjectName(projectSource) == "httpcore":
        print("    [INFO] Project: ", getProjectName(projectSource), "[", counterProject ,"/", len(projectSources), "]")
        # Should be able to delete two lines below
        for project in projectList:
            if getProjectName(project) == getProjectName(projectSource):
                commitFiles = [ f.path for f in os.scandir(project) if f.is_file() ]
                
                counterCommit = 0
                for commitFile in commitFiles:
                    # if getCommitID(commitFile) == "7473f5b70e94d44fbc253aa1cecb7b5d76b25684":
                    counterCommit += 1
                    print("____________________________________________________________")
                    print("[INFO] Commit ", getCommitID(commitFile), "[", counterCommit ,"/", len(commitFiles), "]")
                    allTests, allMethods = getMetrics(commitFile, projectSource)
                    displayResultsForCommit(allTests, allMethods, projectSource, commitFile)

                    print("\n[STEP] Find CUT for each test")
                    for i in tqdm(range(len(allTests))):
                        test = allTests[i]
                        testWithCUT = getTestAndCUT(test, allMethods, nbSimilarMethods)
                        saveToDataset(commitFile, testWithCUT, dataset)
        displayResultsForProject(projectSource, dataset)
                            

        # Save to disk
        saveResults(dataset)

def getMetrics(commitFile, projectSource):
    commitID = getCommitID(commitFile)
    
    # Prepare project
    prepareProject(commitID, projectSource)

    # Get allMethods body
    allMethods = getBodies(projectSource, commitFile, "method")
    # Get allTests body
    allTests = getBodies(projectSource, commitFile, "test")

    # Clean project
    cleanProject(projectSource)

    return allTests, allMethods

def getTestAndCUT(test, allMethods, nb):
    """
    Return a test with its most similar methods, supposed to be part of the CUT.

    Parameters
    ----------
    test: A test object, {ClassName, MethodName, ProjectName, Body, Label:"test"}
    allMethods: A list of method objects. {ClassName, MethodName, ProjectName, Body, Label:"method"}

    Returns
    -------
    test: A final test object, {ClassName, MethodName, ProjectName, Body, "test", cut_1, cut_2...}
    """

    # Build Arrays of bodies
    testBody = [ test["Body"] ]
    methodsBody = list(map(itemgetter('Body'), allMethods))

    # Vectorize

    # To use if you want to deal with CamelCase:
    # vectorizer = TfidfVectorizer(preprocessor=CustomPreProcessor)
    
    # TF-IDF Approach
    vectorizer = TfidfVectorizer()
    # Fit to all Tests + Methods bodies vocabulary length
    vectorizer.fit(testBody+methodsBody)
    # Vectorize all Tests, and all Methods based on vector size established line before
    X_Test = vectorizer.transform(testBody)
    X_Methods = vectorizer.transform(methodsBody)
    
    # print("Feature names: ", vectorizer.get_feature_names())
    # print("Vocabulary size: ", len(vectorizer.get_feature_names()))

    # Similarity

    # Computing similarities between selected test and all methods
    cosine_similarities = linear_kernel(X_Test, X_Methods).flatten()
    # Retrieving 5 most similar methods to selected test
    similarMethodsIndex = cosine_similarities.argsort()[:-nb-1:-1]
    # print(nb, " most similar methods:")
    # print(similarMethodsIndex)

    similarMethods = []
    for i in similarMethodsIndex:
        # Get whole method based on index of similar method, append to final array
        similarMethods.append(allMethods[i])

    newTest = buildTestWithCUT(test, similarMethods)

    return newTest

def displayResultsForCommit(allTests, allMethods, projectSource, commitFile):
    print("\n[STEP] displayResultsForCommit")
    print("    [INFO] Results for ", getProjectName(projectSource), ":", getCommitID(commitFile), sep='')
    print("    [INFO] len(allTests): ", len(allTests),)
    print("    [INFO] len(allMethods): ", len(allMethods))
    print("    [INFO] Done.")
    return

def displayResultsForProject(projectSource, dataset):
    print("____________________________________________________________")
    print("[INFO] Project", getProjectName(projectSource), "done.")
    print("[INFO] len(dataset): ", len(dataset))
    return

def saveToDataset(commitFile, test, dataset):
    # Check if we have a FT or NFT
    if getCommitID(commitFile) == "master":
        test["Label"] = 0
    else:
        test["Label"] = 1
    dataset.append(test)
    return dataset

def buildTestWithCUT(test, similarMethods):
    c = 1
    for method in similarMethods:
        keyName = "CUT_" + str(c)
        test[keyName] = method["Body"]
        c += 1
    return test

# Check ./scripts/Analysis/findNonFlaky.py
def removeFlakyFromNonFlaky(dataset):
    print("[STEP] removeFlakyFromNonFlaky")
    return

def prepareProject(commitID, projectSource):
    print("\n[STEP] prepareProject")
    # Create log file
    logFile = "/Users/guillaume.haben/Desktop/logBuildDataset/" + getProjectName(projectSource) + "-prepare.txt"
    with open(logFile, "w+") as outfile:
        subprocess.call(["./prepareProject.sh", projectSource, commitID], stdout=outfile, stderr=STDOUT)
    print("    [INFO] On", commitID, ", mvn clean.")
    return

def cleanProject(projectSource):
    print("\n[STEP] cleanProject")
    logFile = "/Users/guillaume.haben/Desktop/logBuildDataset/" + getProjectName(projectSource) + "-clean.txt"
    with open(logFile, "w+") as outfile:
        subprocess.call(["./cleanProject.sh", projectSource], stdout=outfile, stderr=STDOUT)
    print("    [INFO] Sources reset, HEAD is now master.")
    return

def getCommitID(commitFile):
    """
    Return basename of a commitFile.

    Parameters
    ----------
    commitFile: "/sample/path/to/commitID.txt"

    Returns
    -------
    commitID: "commitID"
    """
    commitFileName = commitFile.split("/")[-1]
    commitID = commitFileName.split(".")[0]
    return commitID

def getProjectName(projectSource):
    """
    Return basename of a project.

    Parameters
    ----------
    projectSource: "/sample/path/to/project/Name/"

    Returns
    -------
    projectName: "Name"
    """
    return projectSource.split("/")[-1]

def getBodies(projectSource, commitFile, label):
    print("\n[STEP] Get " , label, " bodies")
    cleanResultsFolder(projectSource)
    print("    [INFO] Extraction...")
    if label == "method":
        fileName = "getMethodsBody"
    elif label == "test":
        fileName = "getTestsBody"
    else:
        print("[ERROR] Wrong label")
        sys.exit(1)
    logFile = "/Users/guillaume.haben/Desktop/logBuildDataset/" + getProjectName(projectSource) + "-" + fileName + ".txt"
    with open(logFile, "w+") as outfile:
        subprocess.call(["./" + fileName + ".sh", projectSource, commitFile], stdout=outfile)
    print("    [INFO] Done.")
    return processResultsFolder(projectSource, label)

def cleanResultsFolder(projectSource):
    # Clean result folder
    projectName = getProjectName(projectSource)
    resultFolder = "/Users/guillaume.haben/Documents/Work/projects/MetricExtractor/results/" + projectName
    print("    [INFO] Cleaning old ME results folder.")
    try:
        shutil.rmtree(resultFolder)
    except:
        print("    [INFO] Nothing to clean.")
    return

def processResultsFolder(projectSource, label):
    # Get data
    projectName = getProjectName(projectSource)
    resultFolder = "/Users/guillaume.haben/Documents/Work/projects/MetricExtractor/results/" + projectName

    dataset = buildBodyDataset(resultFolder, label)
    return dataset

def buildBodyDataset(folderPath, label):
    """
    Build a JSON dataset of body of tests / methods.

    Parameters
    ----------
    folderPath: path to MetricExtractor results folder
    label: flag each body as being a "body" or "test"

    Returns
    -------
    dataset
    """
    dataset = []
    if not os.path.isdir(folderPath):
        return dataset
    
    for txtFile in os.listdir(folderPath):
        txtFilePath = os.path.join(folderPath, txtFile)
        
        with open(txtFilePath, 'r') as txtFileRead:
            className = txtFile.split(".")[0]
            methodName = txtFile.split(".")[1]
            
            body = txtFileRead.read()
            dataset.append({"ClassName": className, "MethodName": methodName, "ProjectName": getProjectName(folderPath), "Body": body, "Label": label})
    return dataset

def saveResults(dataset):
    """
    Save an array to a file.

    Parameters
    ----------
    dataset: the array to save

    Returns
    -------
    Nothing
    """
    with open('dataset.json', 'w') as json_file:
        json.dump(dataset, json_file) 
    print("File saved to `./dataset.json`")

def checkUsage():
    """
    Check parameters before running script.
    Exit if misusage
    """
    if len(sys.argv) != 4 or not os.path.isdir(sys.argv[1]) or not os.path.isdir(sys.argv[2]):
        print("Usage: python3 buildDataset.py [path/to/listProjectSources] [path/to/listTestPerProject] [nbSimilarMethods]")
        sys.exit(1)

if __name__ == "__main__":
    main()

