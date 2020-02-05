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

    dataSetPath = sys.argv[1]
    data = pd.read_json(dataSetPath)
    # Only take features columns
    df1 = data[["CyclomaticComplexity", "HasTimeoutInAnnotations", "NumberOfAsserts", "NumberOfAsynchronousWaits",
    "NumberOfDates", "NumberOfFiles", "NumberOfLines", "NumberOfRandoms", "NumberOfThreads", "Label"]]

    # Mask for Triangular Matrix
    matrix = np.triu(df1.corr())

    # Seaborn Correlation Heatmap
    sns_plot = sns.heatmap(df1.corr(), annot=True, vmin=0, vmax=1, center= 0.5, cmap= 'coolwarm', mask=matrix, square=True)
    plt.show()

    # Optiona, to save output as .png file
    # fig = sns_plot.get_figure()
    # fig.savefig("output.png")

def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 eda.py [path/to/dataset.json]")
        sys.exit(1)

if __name__ == "__main__":
    main()