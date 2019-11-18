# Scripts to manipulate DeFlaker dataset

## 1. generateFlakyDictionary.py

`python3 generateFlakyDictionary.py ./path/to/historical_rerun_flaky_tests.csv`

#### Features

* Open `../data/historical_rerun_flaky_tests.csv`
* Create a dictionary of flaky tests with [project name, sha]
* Print and save results to JSON file

## 2. generateFlakyByProjectDictionary.py

`python3 generateFlakyByProjectDictionary.py ./path/to/previouslyCreatedFile.json ./path/to/historical_projects.csv`

#### Features

* Open `../data/historical_projects.csv` and `flakyTests.json`
* Create a dictionary of projects containing a list of commit and their flaky tests.
* Save results to JSON file

## 3. downloadProjects.sh

`cat gitProjects.txt | xargs -I {} ./downloadProjects.sh {}`

### Features

* Take a git URL as parameter and clone it in the current directory
* This command takes all URL from gitProjects.txt and pass them to the script

## Miscellaneous

Beautify a JSON file:
`python -m json.tool sample.json > sample.beautify.json`