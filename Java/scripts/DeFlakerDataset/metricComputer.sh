#!/bin/sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 /path/to/projects/sources /path/to/listPath/folder" >&2
  exit 1
fi

PROJECTS_SOURCES=$1
LIST_PATH_FOLDER=$2

# Check if directories exist ###
if [ ! -d $PROJECTS_SOURCES ] || [ ! -d $LIST_PATH_FOLDER ] 
then
    echo "Directory DOES NOT exists." 
    exit 9999
fi

# For each project 
for PROJECT in $LIST_PATH_FOLDER/*; do
    # Get Project name
    PROJECT_NAME="$(basename -- $PROJECT)"
    # Get Project path
    INFO_PROJECT=$LIST_PATH_FOLDER"/"$PROJECT_NAME
    PROJECT_PATH=$PROJECTS_SOURCES"/"$PROJECT_NAME
    cd $INFO_PROJECT
    echo "PROJECT: "$PROJECT_NAME
    # DEBUG echo "PROJECT PATH: "$PROJECT_PATH

    # For each commit of this project
    for COMMIT in ./*; do
        # Get Commit name
        COMMIT_NUM="$(basename -- $COMMIT .txt)"
        LIST_PATH=$INFO_PROJECT"/"$COMMIT_NUM".txt"
        # Go to project sources
        cd $PROJECTS_SOURCES"/"$PROJECT_NAME
        # Checkout and run MetricExtractor
        git checkout $COMMIT_NUM
        # DEBUG echo "LIST FILE: "$LIST_PATH
        echo "PROJECT PATH: "$PROJECT_PATH
        echo "LIST FILE: "$LIST_PATH
        /opt/apache-maven-3.6.2/bin/mvn clean > /dev/null 2>&1
        #cd "/Users/guillaume.haben/Documents/Work/projects/MetricExtractor"
        sh "/Users/guillaume.haben/Documents/Work/projects/DeFlaker/scripts/DeFlakerDataset/metricExtractor.sh" -projectPath $PROJECT_PATH -listPath $LIST_PATH
    done
    # Switch back to master before processing next project.
    git checkout master
    
done

