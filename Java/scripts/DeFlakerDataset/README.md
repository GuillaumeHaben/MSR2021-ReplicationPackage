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

* Open `../data/historical_projects.csv` and `flakyTests.json` (from step 1)
* Create a dictionary of projects containing a list of commit and their flaky tests.
* Save results to JSON file

## 3. downloadProjects.sh

`cat gitProjects.txt | xargs -I {} ./downloadProjects.sh {}`

### Features

* Take a git URL as parameter and clone it in the current directory
* This command takes all URL from gitProjects.txt and pass them to the script

## 4. generateListPaths.py

`python3 generateListPaths.py ./path/to/flakyTests.json`

### Features

* Take `flakyTests.json` (from step 2)
* Create files containing list of flaky tests per commit for each projects, ready for MetricExtractor program, -listPath option

## 5. metricComputer.sh

`./metricComputer.sh ../projects ../results`

### Features

* Take folder containing project sources and folder containing information about commit/flaky tests generated previously
* Compute metrics for every flaky tests and save the results in MetricExtractor/results
* Requires MetricExtractor project

## 5. bis metricExtractor.sh

`./metricExtractor.sh -projectPath ... -listPath ...`

### Features

* Simple script to run the Java project Metric Extractor

## Miscellaneous

Beautify a JSON file:
`python -m json.tool sample.json > sample.beautify.json`