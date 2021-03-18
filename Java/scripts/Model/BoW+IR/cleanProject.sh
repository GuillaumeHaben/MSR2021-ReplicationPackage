#!/bin/sh

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/projects/sources" >&2
  exit 1
fi

PROJECTS_SOURCES=$1

# Check if directories exist ###
if [ ! -d $PROJECTS_SOURCES ] 
then
    echo "Directory DOES NOT exists." 
    exit 9999
fi

cd $PROJECTS_SOURCES
git reset --hard HEAD
git clean -fx > /dev/null 2>&1
git clean -f -d > /dev/null 2>&1
git checkout master


