#!/bin/sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 /path/to/projects/sources commitID" >&2
  exit 1
fi

PROJECTS_SOURCES=$1
COMMIT_ID=$2

# Check if directories exist ###
if [ ! -d $PROJECTS_SOURCES ] 
then
    echo "Directory DOES NOT exists." 
    exit 9999
fi

cd $PROJECTS_SOURCES

git checkout $COMMIT_ID
/opt/apache-maven-3.6.2/bin/mvn clean


