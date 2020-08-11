#!/bin/sh

PROJECTS_SOURCES=$1

# Check if directories exist ###
if [ ! -d $PROJECTS_SOURCES ]
then
    echo "Directory DOES NOT exists." 
    exit 9999
fi

echo "Calling script:"
echo "/Users/guillaume.haben/Documents/Work/projects/DeFlaker/scripts/DeFlakerDataset/metricExtractor.sh -projectPath "$PROJECTS_SOURCES" -getAllMethods"

sh "/Users/guillaume.haben/Documents/Work/projects/DeFlaker/scripts/DeFlakerDataset/metricExtractor.sh" -projectPath $PROJECTS_SOURCES -getAllMethods

