import sys
import os
import glob
import re
import json
import pandas as pd
import pprint
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def main():
    # Checks
    checkUsage()

    # Load Data
    datasetPath = sys.argv[1]

    # Variables
    tests = []
    methods = []
    
    # Read dataset.json
    with open(datasetPath, 'r') as json_file:
        dic = json.load(json_file)

        # Build tests and methods arrays
        for el in dic:
            if el["Label"] == "test":
                tests.append(el)
            else:
                methods.append(el)

        # Info
        print("Number of Test methods: ", len(tests))
        print("Number of Methods: ", len(methods))

        # Build Arrays of bodies
        testsBody = list(map(itemgetter('Body'), tests))
        methodsBody = list(map(itemgetter('Body'), methods))

        # Vectorize
        # To use if you want to deal with CamelCase:
        # vectorizer = TfidfVectorizer(preprocessor=CustomPreProcessor)
        
        # TF-IDF Approach
        vectorizer = TfidfVectorizer()
        # Fit to all Tests + Methods bodies vocabulary length
        vectorizer.fit(testsBody+methodsBody)
        # Vectorize all Tests, and all Methods based on vector size established line before
        X_Tests = vectorizer.transform(testsBody)
        X_Methods = vectorizer.transform(methodsBody)
        

        #print("Feature names: ", vectorizer.get_feature_names())
        print("Vocabulary size: ", len(vectorizer.get_feature_names()))

        # Similarity
        # Index of test I want to compare
        indice = 7

        print("Finding methods similar to Test: ", indice)
        pprint.pprint(tests[indice])

        # Computing similarities between selected test and all methods
        cosine_similarities = linear_kernel(X_Tests[indice], X_Methods).flatten()
        # Retrieving 5 most similar methods to selected test
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]
        print("5 most similar methods:")
        print(related_docs_indices)

        for i in related_docs_indices:
            print("ClassName: ", methods[i]["ClassName"])
            print("MethodName: ", methods[i]["MethodName"])
            pprint.pprint(methods[i])
        

def saveResults(dic):
    with open('dic.json', 'w') as json_file:
        json.dump(dic, json_file) 

# Separate CamelCase to Camel Case
def CustomPreProcessor(doc):
    return re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', doc).lower()

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 testSKlearnProcessing.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()