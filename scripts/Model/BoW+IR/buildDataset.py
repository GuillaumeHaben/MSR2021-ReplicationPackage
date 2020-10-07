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
import math as m

# Dependencies
# ./getAllMethodsMetrics.sh
# ./getAllTestMethodsMetrics.sh
# ./getTestsMetrics.sh
# ./cleanProject.sh
# ./prepareProject.sh

# Configuration
logFileFolder = "/Users/guillaume.haben/Documents/Work/sandbox/preDesktopTrash/logBuildDataset/"
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
        # List of projects to consider
        if projectName in ["togglz", "achilles", "hbase", "logback", "okhttp", "oozie", "oryx"]:
        # if projectName in ["togglz"]:
            print("    [INFO] Project: ", projectName, "[", counterProject ,"/", len(projectSources), "]")
            # For each project
            for project in projectList:
                if getProjectName(project) == projectName:
                    commitFiles = [ f.path for f in os.scandir(project) if f.is_file() ]
                    
                    counterCommit = 0
                    # For each commit of this project
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

            # Find commits where I want to get NFT
            NFTcommits = []
            print("[INFO]: Adding Non Flaky Tests to Dataset")
            for percentage in percentages:
                dataset, splittingCommit = handleDataset(dataset, percentage)
                print("[INFO]: Found commit", splittingCommit, "at percentage", percentage)
                if splittingCommit not in NFTcommits:
                    NFTcommits.append(splittingCommit)
            
            print("[INFO]: Processing", len(NFTcommits), "commits.")
            # For each commit where I want to get NFT
            for NFTcommit in NFTcommits:
                dataset = getFinalTestsAndAddToDataset(NFTcommit, projectSource, nbSimilarMethods, dataset, 0)

            print("[STEP] Remove Non Flaky similar to Flaky.")
            dataset = removeFlakyFromNonFlaky(dataset, projectName)

            # Add CUT metrics
            dataset = addCutMetric(dataset)

            # Save to disk
            displayResultsForProject(projectSource, dataset)  
            saveResults(dataset, projectName)
            dataset = []

def getFinalTestsAndAddToDataset(commitID, projectSource, nbSimilarMethods, dataset, label):
    allTests, allMethods, date, timestamp = getMetrics(commitID, projectSource, label)
    displayResultsForCommit(allTests, allMethods, projectSource, commitID)

    print("\n[STEP] Find CUT for each test")
    for i in tqdm(range(len(allTests))):
        test = allTests[i]
        # testWithCUT = getTestAndCUT(test, allMethods, nbSimilarMethods)
        # testWithCUT = getStaticCut(test, projectSource, commitID)
        testWithDate = getTestAndDate(test, date, timestamp)
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
    allMethods = []
    #allMethods = getBodies(projectSource, commitID, "method", label)
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
    print("\n[STEP] Get " , methodType, " metrics")
    cleanResultsFolder(projectSource)
    print("    [INFO] Extraction...")
   
    # Which SH script to call depending on the method type 
    if methodType == "method":
        fileName = "getAllMethodsMetrics"
    elif methodType == "test":
        fileName = "getTestsMetrics"
    else:
        print("[ERROR] Wrong methodType")
        sys.exit(1)
    
    logFile = logFileFolder + getProjectName(projectSource) + "-" + fileName + ".txt"
    commitFile = getCommitFile(commitID, getProjectName(projectSource))
    
    # Non Flaky Tests 
    if label == 0 and methodType == "test":
        with open(logFile, "w+") as outfile:
            subprocess.call(["./getAllTestMethodsMetrics.sh", projectSource], stdout=outfile)
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
        test[keyName] = {
            "ClassName": method["ClassName"],
            "MethodName": method["MethodName"],
            "Body": method["Body"],
            "CyclomaticComplexity": method["CyclomaticComplexity"],
            "HasTimeoutInAnnotations": method["HasTimeoutInAnnotations"],
            "NumberOfAsserts": method["NumberOfAsserts"],
            "NumberOfAsynchronousWaits": method["NumberOfAsynchronousWaits"],
            "NumberOfDates": method["NumberOfDates"],
            "NumberOfFiles": method["NumberOfFiles"],
            "NumberOfLines": method["NumberOfLines"],
            "NumberOfRandoms": method["NumberOfRandoms"],
            "NumberOfThreads": method["NumberOfThreads"]
        }
        c += 1
    return test

def removeFlakyFromNonFlaky(dataset, projectName):
    print("\n[STEP] removeFlakyFromNonFlaky")
    FT = []
    indexesToDelete = []

    for el in dataset:
        if el["Label"] == 1:
            FT.append(el)
    for i in range(len(dataset)):
        for flaky in FT:
            if dataset[i]["MethodName"] == flaky["MethodName"] and dataset[i]["ClassName"] == flaky["ClassName"] and dataset[i]["Label"] == 0:
                indexesToDelete.append(i)
    print("[INFO] Number of NFT similar to FT to remove:", len(indexesToDelete))
    cleanedDataset = [i for j, i in enumerate(dataset) if j not in set(indexesToDelete)]
    return cleanedDataset

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
        with open(txtFilePath, 'r') as json_file:
            data = json.load(json_file)
            className = txtFile.split(".")[0]
            methodName = txtFile.split(".")[1]

            dataset.append({"Commit": commitID, "ClassName": className, "MethodName": methodName, "ProjectName": getProjectName(folderPath), 
                "CyclomaticComplexity": data["CyclomaticComplexity"], 
                "HasTimeoutInAnnotations": data["HasTimeoutInAnnotations"], 
                "NumberOfAsserts": data["NumberOfAsserts"], 
                "NumberOfAsynchronousWaits": data["NumberOfAsynchronousWaits"], 
                "NumberOfDates": data["NumberOfDates"], 
                "NumberOfFiles": data["NumberOfFiles"], 
                "NumberOfLines": data["NumberOfLines"], 
                "NumberOfRandoms": data["NumberOfRandoms"], 
                "NumberOfThreads": data["NumberOfThreads"], 
                "StaticCUT": data["StaticCUT"],
                "Body": data["Body"],
                "Label": label})    
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

def addCutMetric(dataset):
    # Metrics to consider for the CUT
    metrics = ["CyclomaticComplexity", "NumberOfAsynchronousWaits", "NumberOfDates", "NumberOfFiles", "NumberOfLines", "NumberOfRandoms", "NumberOfThreads"]
    
    # For each test
    for i in range(len(dataset)):
        # Init scores for test
        # dataset[i]["Mean"] = {}
        # dataset[i]["Total"] = {}
        # dataset[i]["Maximum"] = {}
        # dataset[i]["Minimum"] = {}

        for metric in metrics:
            # Init metric score
            mean = 0
            total = 0
            maximum = 0
            minimum = m.inf
            # For each method in the CUT
            for j in range(len(dataset[i]["StaticCUT"])):
                # Total
                total += dataset[i]["StaticCUT"][j][metric]
                # Max
                if dataset[i]["StaticCUT"][j][metric] > maximum:
                    maximun = dataset[i]["StaticCUT"][j][metric]
                # Min
                if dataset[i]["StaticCUT"][j][metric] < minimum:
                    minimum = dataset[i]["StaticCUT"][j][metric]
            # Mean
            if len(dataset[i]["StaticCUT"]) != 0:
                mean = total / len(dataset[i]["StaticCUT"])
            else:
                minimum = 0
            # Clean for JSON
            # dataset[i]["Mean"][metric] = int(mean)
            # dataset[i]["Total"][metric] = total
            # dataset[i]["Maximum"][metric] = maximum
            # dataset[i]["Minimum"][metric] = minimum
            # Clean for Panda DATAFRAME
            dataset[i]["Mean"+metric] = int(mean)
            dataset[i]["Total"+metric] = total
            dataset[i]["Maximum"+metric] = maximum
            dataset[i]["Minimum"+metric] = minimum

    return dataset

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

