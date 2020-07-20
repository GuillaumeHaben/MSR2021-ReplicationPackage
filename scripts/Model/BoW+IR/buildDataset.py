import sys
import os
import glob
import json
import pprint
import subprocess
from subprocess import STDOUT, check_output
import shutil
from tqdm import tqdm
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Dependencies
# ./getMethodsBody.sh
# ./getTestsBody.sh
# ./cleanProject.sh
# ./prepareProject.sh

# Configuration
logFileFolder = "/Users/guillaume.haben/Desktop/logBuildDataset/"
metricExtractorResultsFolder = "/Users/guillaume.haben/Documents/Work/projects/MetricExtractor/results/"
percentages = [0.6, 0.7, 0.8, 0.9, 1]

def main():
    # Checks
    checkUsage()

    # Variables
    sourcesPath = sys.argv[1]
    listPath = sys.argv[2]
    nbSimilarMethods = int(sys.argv[3])

    print("\n[STEP] main")

    projectSources = [ f.path for f in os.scandir(sourcesPath) if f.is_dir() ]
    projectList = [ f.path for f in os.scandir(listPath) if f.is_dir() ]

    counterProject = 0
    for projectSource in projectSources:
        counterProject += 1
        projectName = getProjectName(projectSource)
        dataset = []
        # TO TEST, WITH SMALL PROJECT, TO REMOVE LATER
        if projectName == "oozie":
            print("    [INFO] Project: ", projectName, "[", counterProject ,"/", len(projectSources), "]")
            # Should be able to delete two lines below
            for project in projectList:
                if getProjectName(project) == projectName:
                    commitFiles = [ f.path for f in os.scandir(project) if f.is_file() ]
                    
                    counterCommit = 0
                    for commitFile in commitFiles:
                        commitID = getCommitID(commitFile)
                        counterCommit += 1
                        print("____________________________________________________________")
                        print("[INFO] Commit ", commitID, "[", counterCommit ,"/", len(commitFiles), "]")
                        getFinalTestsAndAddToDataset(commitID, projectSource, nbSimilarMethods, dataset, 1)

            # If project contains no FT, we jump to next project
            if len(dataset) == 0:
                displayResultsForProject(projectSource, dataset)  
                saveResults(dataset, projectName)
                dataset = []
                continue

            # Find commit where I want to take NFT
            NFTcommits = []
            print("[INFO]: Adding Non Flaky Tests to Dataset")
            for percentage in percentages:
                dataset, splittingCommit = handleDataset(dataset, percentage)
                print("[INFO]: Found commit", splittingCommit, "at percentage", percentage)
                if splittingCommit not in NFTcommits:
                    NFTcommits.append(splittingCommit)
            
            print("[INFO]: Processing", len(NFTcommits), "commits.")
            for NFTcommit in NFTcommits:
                dataset = getFinalTestsAndAddToDataset(NFTcommit, projectSource, nbSimilarMethods, dataset, 0)

            # Save to disk
            displayResultsForProject(projectSource, dataset)  
            saveResults(dataset, projectName)
            dataset = []

def getFinalTestsAndAddToDataset(commitID, projectSource, nbSimilarMethods, dataset, label):
    # TODO: give commitID info, give Label 0 info, make this code not duplicate
    allTests, allMethods, date, timestamp = getMetrics(commitID, projectSource, label)
    # In the case of NFT, we remove all FT from NFT for this particular commit
    if label == 0:
        allTests = removeFlakyFromNonFlaky(allTests, commitID, getProjectName(projectSource))
    displayResultsForCommit(allTests, allMethods, projectSource, commitID)

    print("\n[STEP] Find CUT for each test")
    for i in tqdm(range(len(allTests))):
        test = allTests[i]
        testWithCUT = getTestAndCUT(test, allMethods, nbSimilarMethods)
        testWithDate = getTestAndDate(testWithCUT, date, timestamp)
        dataset = saveToDataset(commitID, testWithDate, dataset, label)
    return dataset

def getMetrics(commitID, projectSource, label):
    """
    Prepare the current project, get the metrics (bodies) of all Tests and Methods.

    Parameters
    ----------
    commitFile: The commitFile to know on which commitID to do the analysis
    projectSource: The source path of the current project

    Returns
    -------
    allTests: Bodies of all Tests
    allMethods: Bodies of all Methods
    date: Date of the current commit
    timestamp: Timestamp of the current commit
    """
       
    # Prepare project
    date, timestamp = prepareProject(commitID, projectSource)

    # Get bodies
    allMethods = getBodies(projectSource, commitID, "method", label)
    allTests = getBodies(projectSource, commitID, "test", label)

    # Clean project
    cleanProject(projectSource)

    return allTests, allMethods, date, timestamp

def getBodies(projectSource, commitID, methodType, label):
    """
    Get bodies of given method (test or method)

    Parameters
    ----------
    projectSource: "/sample/path/to/project/Name/"
    commitFile: "/sample/path/to/commitID.txt"
    methodType: Either "method" or "test" (Different attribute of MetricExtractor)
    label: 1 (FT) or 0 (NFT)

    Returns
    -------
    dataset: Constructed with processResultsFolder()
    """
    print("\n[STEP] Get " , methodType, " bodies")
    cleanResultsFolder(projectSource)
    print("    [INFO] Extraction...")
   
    # Which SH script to call depending on the method type 
    if methodType == "method":
        fileName = "getMethodsBody"
    elif methodType == "test":
        fileName = "getTestsBody"
    else:
        print("[ERROR] Wrong methodType")
        sys.exit(1)
    
    logFile = logFileFolder + getProjectName(projectSource) + "-" + fileName + ".txt"
    commitFile = getCommitFile(commitID, getProjectName(projectSource))
    
    # Non Flaky Tests 
    if label == 0 and methodType == "test":
        with open(logFile, "w+") as outfile:
            subprocess.call(["./getAllTestsBody.sh", projectSource], stdout=outfile)
        print("    [INFO] Done.")

    # Flaky Tests
    else:
        with open(logFile, "w+") as outfile:
            subprocess.call(["./" + fileName + ".sh", projectSource, commitFile], stdout=outfile)
        print("    [INFO] Done.")

    return processResultsFolder(projectSource, commitID, methodType)

def getTestAndCUT(test, allMethods, nb):
    """
    Return a test with its most similar methods, supposed to be part of the CUT.

    Parameters
    ----------
    test: A test object, {ClassName, MethodName, ProjectName, Body, Label:"test"}
    allMethods: A list of method objects. {ClassName, MethodName, ProjectName, Body, Label:"method"}
    nb: Number of similar methods to find

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

def getTestAndDate(test, date, timestamp):
    test["Date"] = date
    test["Timestamp"] = timestamp
    return test

def buildTestWithCUT(test, similarMethods):
    """
    Add CUT to a test.

    Parameters
    ----------
    test: The test to add the CUT
    similarMethods: The CUT we want to add to the test

    Returns
    -------
    test: The test with its CUT
    """
    c = 1
    for method in similarMethods:
        keyName = "CUT_" + str(c)
        test[keyName] = method["Body"]
        c += 1
    return test

def removeFlakyFromNonFlaky(allTests, commitID, projectName):
    print("\n[STEP] removeFlakyFromNonFlaky")
    commitFile = getCommitFile(commitID, projectName)
    print("    [DEBUG] Length of allTests before removing FT", len(allTests))
    indexesToDelete = []
    for i in range(len(allTests)):
        test = allTests[i]
        commitFileRead = open(commitFile, 'r')
        lines = commitFileRead.readlines()
        for line in lines:
            className = line.split(".")[-2]
            methodName = line.split(".")[-1].rstrip("\n")
            if test["ClassName"] == className and test["MethodName"] == methodName:
                indexesToDelete.append(i)
    cleanedTests = [i for j, i in enumerate(allTests) if j not in set(indexesToDelete)]

    print("    [DEBUG] Nb of tests removed", len(indexesToDelete))
    print("    [DEBUG] Length of allTests after removing FT from NFT", len(cleanedTests))
    return cleanedTests

def prepareProject(commitID, projectSource):
    """
    Call prepareProject.sh, checkout to the correct version of the project, mvn clean (to remove potential target classes).
    Log output to a file

    Parameters
    ----------
    commitID: Commit we wish to switch to
    projectSource: The source path of the current project
    """
    print("\n[STEP] prepareProject")
    # Create log file
    try:
        output = check_output(['./prepareProject.sh', projectSource, commitID]).decode('UTF-8').rstrip("\n")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    date = output.split(".")[0]
    timestamp = output.split(".")[1]
    print("    [INFO] On", commitID, ", mvn clean.")
    return date, timestamp

def cleanProject(projectSource):
    """
    Call cleanProject.sh. If any, remove changes (due to mvn clean for example) and checkout to master.
    Log output to a file

    Parameters
    ----------
    projectSource: The source path of the current project
    """
    print("\n[STEP] cleanProject")
    logFile = logFileFolder + getProjectName(projectSource) + "-clean.txt"
    with open(logFile, "w+") as outfile:
        subprocess.call(["./cleanProject.sh", projectSource], stdout=outfile, stderr=STDOUT)
    print("    [INFO] Sources reset, HEAD is now master.")
    return

def cleanResultsFolder(projectSource):
    """
    Clear results folder (MetricExtractor)

    Parameters
    ----------
    projectSource: To get the name of the current project, to know which folder to clean
    """
    # Clean result folder
    projectName = getProjectName(projectSource)
    resultFolder = metricExtractorResultsFolder + projectName
    print("    [INFO] Cleaning old ME results folder.")
    try:
        shutil.rmtree(resultFolder)
    except:
        print("    [INFO] Nothing to clean.")
    return

def processResultsFolder(projectSource, commitID, label):
    """
    Process the result folder from MetricExtractor for the specified project

    Parameters
    ----------
    projectSource: "/sample/path/to/project/Name/"
    label: Either "method" or "test"

    Returns
    -------
    dataset: The created dataset
    """
    projectName = getProjectName(projectSource)
    resultFolder = metricExtractorResultsFolder + projectName

    dataset = buildBodyDataset(resultFolder, commitID, label)
    return dataset

def buildBodyDataset(folderPath, commitID, label):
    """
    Build a JSON dataset of body of tests / methods.

    Parameters
    ----------
    folderPath: path to MetricExtractor results folder
    label: flag each body as being a "method" or "test"

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
            dataset.append({"Commit": commitID, "ClassName": className, "MethodName": methodName, "ProjectName": getProjectName(folderPath), "Body": body, "Label": label})
    return dataset

def handleDataset(dataset, percentage):
    sortedDataset = orderByDate(dataset)
    commit = findSplittingCommit(sortedDataset, percentage)
    if commit == -1:
        commit = sortedDataset[-1]["Commit"]
    return sortedDataset, commit

def findSplittingCommit(sortedDataset, percentage):
    # TODO Dangerous to let else branch empty
    found = False
    for i in range(0, len(sortedDataset)):
        if i / len(sortedDataset) >= percentage and found == False:
            commit = sortedDataset[i]["Commit"]
            found = True
            return commit
    return -1

def orderByDate(dataset):
    sortedDataset = sorted(dataset, key=lambda x: x["Timestamp"])
    return sortedDataset

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

def getCommitFile(commitID, projectName):
    """
    Return path of commitFile based on the given commitID.

    Parameters
    ----------
    commitID: 

    Returns
    -------
    commitFile: 
    """
    commitFile = sys.argv[2] + "/" + projectName + "/" + commitID +".txt"
    return commitFile

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

def displayResultsForCommit(allTests, allMethods, projectSource, commitID):
    print("\n[STEP] displayResultsForCommit")
    print("    [INFO] Results for ", getProjectName(projectSource), ": ", commitID, sep='')
    print("    [INFO] len(allTests): ", len(allTests),)
    print("    [INFO] len(allMethods): ", len(allMethods))
    print("    [INFO] Done.")
    return

def displayResultsForProject(projectSource, dataset):
    print("____________________________________________________________")
    print("[INFO] Project", getProjectName(projectSource), "done.")
    print("[INFO] len(dataset): ", len(dataset))
    return

def saveToDataset(commitID, test, dataset, label):
    """
    Add a test to the dataset.

    Parameters
    ----------
    commitFile: The commitFile to know on which commitID to do the analysis
    test: The test to add to the dataset
    dataset: Current dataset
    label: 1 (FT) or 0 (NFT)

    Returns
    -------
    dataset: The dataset with the new test
    """
    # Check if we have a FT or NFT
    if label == 0:
        test["Label"] = 0
    else:
        test["Label"] = 1
    dataset.append(test)
    return dataset

def saveResults(dataset, projectName):
    """
    Save an array to a file.

    Parameters
    ----------
    dataset: the array to save

    Returns
    -------
    Nothing
    """
    filename = "./results/dataset." + projectName + ".json"
    with open(filename, 'w') as json_file:
        json.dump(orderByDate(dataset), json_file) 
    print("File saved to `", filename, "`")

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

