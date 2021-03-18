from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
import sys
import os
import json 
import pandas as pd
import pickle
import numpy as np
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    # Checks
    checkUsage()

    # Load Data
    datasetPath = sys.argv[1]

    # Variables
    FT = []
    NFT = []

    # Read dataset.json
    with open(datasetPath, 'r') as json_file:
        dic = json.load(json_file)

        # Build tests and methods arrays
        for el in dic:
            if el["ProjectName"] == "/Users/guillaume.haben/Desktop/results-NFT-bodyText/oozie" or el["ProjectName"] == "/Users/guillaume.haben/Desktop/results-FT-bodyText/oozie":
                if el["Label"] == 0:
                    NFT.append(el)
                else:
                    FT.append(el)


        # Info
        print("Number of FT: ", len(FT))
        print("Number of NFT: ", len(NFT))

        # Build Arrays of bodies
        FTBody = list(map(itemgetter('Body'), FT))
        NFTBody = list(map(itemgetter('Body'), NFT))
        
        # TF-IDF Approach
        vectorizer = TfidfVectorizer()
        # Fit to all Tests + Methods bodies vocabulary length
        vectorizer.fit(FTBody+NFTBody)
        # Vectorize all Tests, and all Methods based on vector size established line before
        X_FT = vectorizer.transform(FTBody)
        X_NFT = vectorizer.transform(NFTBody)

        # Most common words
        sum_weighted_words = X_FT.sum(axis=0)
        words_freq = [(word, sum_weighted_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=False)
        
        print("Most Common words: \n")
        for word, freq in words_freq:
            print(word, freq)

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 analyze.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()