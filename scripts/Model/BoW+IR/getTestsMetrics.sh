#!/bin/sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 /path/to/projects/sources /path/to/list/commit.txt" >&2
  exit 1
fi

PROJECTS_SOURCES=$1
LIST_PATH=$2

# Check if directories exist ###
if [ ! -d $PROJECTS_SOURCES ] || [ ! -f $LIST_PATH ] 
then
    echo "Directory DOES NOT exists." 
    exit 9999
fi

echo "Calling script:"
echo "/Users/guillaume.haben/Documents/Work/projects/DeFlaker/scripts/DeFlakerDataset/metricExtractor.sh -projectPath "$PROJECTS_SOURCES" -listPath "$LIST_PATH

sh "/Users/guillaume.haben/Documents/Work/projects/DeFlaker/scripts/DeFlakerDataset/metricExtractor.sh" -projectPath $PROJECTS_SOURCES -listPath $LIST_PATH

