import sys
import os
import pprint
import json

# Global Variables

def main():
    # Checks
    checkUsage()

    # Variables
    path = sys.argv[1]

    # Logic
    with open(path, 'r') as f:
        dicFlaky = json.load(f)


def checkUsage():
    #Check the programs' arguments
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 downloadProjects.py [path/to/file]")
        sys.exit(1)

if __name__ == "__main__":
    main()