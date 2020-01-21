import sys
import os
import json
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Checks
    checkUsage()

    dataSetPath = sys.argv[1]
    data = pd.read_json(dataSetPath)

    sns_plot = sns.pairplot(data, hue="Label", 
    vars=["DepthOfInheritance", "HasTimeoutInAnnotations", "NumberOfAsynchronousWaits", "NumberOfFiles"])
    #plt.show()
    sns_plot.savefig("output.png")



def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 eda.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()