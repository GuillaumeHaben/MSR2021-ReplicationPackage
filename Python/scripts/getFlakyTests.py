import os
import sys
import ast
import json
import astor
import traceback
from tqdm import tqdm
from pprint import pprint
from analyzer import Analyzer
from operator import itemgetter
from functionObject import FunctionObject
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Main variables
folderPath = sys.argv[1]
projectName = folderPath.split("/")[-1]

def main():
    checkUsage()

    # Get all python files in the project folder
    pythonFiles = findPythonFiles(folderPath)
    functionsInProject = []

    print("Project: ", projectName)
    print("Number of Python files to analyze: ", len(pythonFiles), "\n")

    # For each file, build AST, get function and test objects
    for i in tqdm(range(0, len(pythonFiles))):
        pythonFilePath = pythonFiles[i]
        pythonFile = getPath(pythonFilePath)

        # Exceptions
        if "hue" in projectName and "/desktop/core/ext-py/" in pythonFilePath:
            continue
        if "jira" in projectName and ".tox" in pythonFilePath:
            continue

        
        try:
            # Create source code's AST
            tree = astor.code_to_ast.parse_file(pythonFilePath)
        except SyntaxError:
            print("Syntax Error in file, skiping. ", pythonFilePath)
            continue
        except UnicodeDecodeError:
            print("Decode Error in file, skiping. ", pythonFilePath)
            continue
        
        # Help to find functions' class name
        tree = setParents(tree)

        # Browse the AST
        analyzer = Analyzer()
        analyzer.visit(tree)


        # Set projectName and fileName for each function, add to list
        for function in analyzer.objects:
            function.setProjectName(projectName)
            function.setFileName(pythonFile[1:])
            fileName = function.getFileName().split("/")[-1]
            if fileName.startswith("test_") or fileName.endswith("_test.py"):
                function.setIsTest(True)
            functionsInProject.append(function)

    countFunction, countTest, countFlaky, countNonFlaky = getGlobalStatistics(functionsInProject)

    # Display statistics
    print("\nNumber of Functions: ", countFunction)
    print("Number of Tests: ", countTest)
    print("Number of Flaky Tests: ", countFlaky)
    print("Number of Non Flaky Tests: ", countNonFlaky)

    # To JSON
    functionsInProject = [f.toJSON() for f in functionsInProject]

    # Add Most similar methods to each test
    # dataset = addCut2Test(functionsInProject)
    # print("Number of Flaky Tests with CUT added to dataset: ", len(dataset))

    # # Save results to JSON file
    # saveResults(dataset, projectName)
    
def getGlobalStatistics(functionsInProject):
    # Counter for statistics
    countFlaky = 0
    countNonFlaky = 0
    countTest = 0
    countFunction = 0

    # Count numbers for the whole list
    for obj in functionsInProject:
        if obj.getIsTest():
            countTest += 1
            if obj.getIsMarkedFlaky():
                countFlaky += 1
                print(obj.getFunctionName(), obj.getClassName(), obj.getFileName())
            else:
                countNonFlaky += 1
        else:
            countFunction += 1

    return countFunction, countTest, countFlaky, countNonFlaky

def findPythonFiles(filepath):
    """Return a list of all .py files in the given filepath"""
    paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.lower().endswith(".py"):
                paths.append(os.path.join(root, file))
    return paths
 
def setParents(tree):
    """Add info about parent Node"""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    return tree

def getPath(pythonFilePath):
    """Get relative path (intra project)"""
    pythonFile = pythonFilePath.replace(folderPath, '.')
    return pythonFile

def getTestAndCUT(test, functionsInProject, nb):
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

    # Init allFunctions
    allFunctions = []
    for function in functionsInProject:
        if function["isTest"] == False:
            allFunctions.append(function)

    # Build Arrays of bodies
    testBody = [ test["Body"] ]
    methodsBody = list(map(itemgetter('Body'), allFunctions))

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
        similarMethods.append(allFunctions[i])

    newTest = buildTestWithCUT(test, similarMethods)

    return newTest

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

def addCut2Test(functionsInProject):
    print("\nAdding CUT to tests...")
    dataset = []
    # For each function in list
    for function in tqdm(functionsInProject):
        # If it is a test
        if function["isTest"] == True:
            # We find its 5 most similar functions
            test = function
            test = getTestAndCUT(test, functionsInProject, 5)
            dataset.append(test)
    print("Done.\n")
    return dataset

def saveResults(dic, name):
    """Save results to file"""
    fileName = "./results/" + name + ".json"
    with open(fileName, 'w') as json_file:
        json.dump(dic, json_file, indent=4) 

def checkUsage():
    """Check the programs' arguments"""
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print("Usage: python main.py [path/to/project]")
        sys.exit(1)

if __name__ == "__main__":
    main()