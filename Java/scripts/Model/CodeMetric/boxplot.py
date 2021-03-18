import sys
import os
import json
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Checks
    checkUsage()

    # Get data
    dataSetPath = sys.argv[1]
    dataSet = pd.read_json(dataSetPath)

    f, axes = plt.subplots(2, 4)

    ax = sns.boxplot(x="CyclomaticComplexity", y="Label", data=dataSet, linewidth=1, hue="ProjectName", whis=10, orient="h", ax=axes[0][0]).legend_.remove()
    ax = sns.boxplot(x="NumberOfAsserts", y="Label", data=dataSet, linewidth=1, hue="ProjectName", whis=10, orient="h", ax=axes[0][1]).legend_.remove()
    ax = sns.boxplot(x="NumberOfAsynchronousWaits", y="Label", data=dataSet, linewidth=1, hue="ProjectName", whis=10, orient="h", ax=axes[0][2]).legend_.remove()
    ax = sns.boxplot(x="NumberOfThreads", y="Label", data=dataSet, linewidth=1, hue="ProjectName", whis=10, orient="h", ax=axes[0][3]).legend_.remove()
    ax = sns.boxplot(x="NumberOfDates", y="Label", data=dataSet, linewidth=1, hue="ProjectName", whis=10, orient="h", ax=axes[1][0]).legend_.remove()
    ax = sns.boxplot(x="NumberOfFiles", y="Label", data=dataSet, linewidth=1, hue="ProjectName", whis=10, orient="h", ax=axes[1][1]).legend_.remove()
    ax = sns.boxplot(x="NumberOfLines", y="Label", data=dataSet, linewidth=1, hue="ProjectName", whis=10, orient="h", ax=axes[1][2]).legend_.remove()
    ax = sns.boxplot(x="NumberOfRandoms", y="Label", data=dataSet, linewidth=1, hue="ProjectName", whis=10, orient="h", ax=axes[1][3]).legend_.remove()

    plt.show()

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 boxplot.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()