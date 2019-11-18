# Scripts to manipulate DeFlaker dataset

### 1. generateFlakyDictionary.py

`python3 generateFlakyDictionary.py ./path/to/historical_rerun_flaky_tests.csv`

### Features

* Open `../data/historical_rerun_flaky_tests.csv`
* Create a dictionary of flaky tests with [project name, sha]
* Print and save results to JSON file

### 2. downloadProjects.py

`python3 downloadProjects.py ./path/to/previouslyCreatedFile.json`

### Features

