## Python experimentations

The goal is to create a dataset of flaky tests based on developers opinions when they put @flaky.
We select projects containing lots of @flaky annotations from https://sourcegraph.com/.

### Structure

* `collections` contains files listing all tests for each project. They have been generated using `pytest --collect-only`

* `projects` contains cloned repository of subject projects.

* `results` contains xml reports of all reruns. They have been generated with bash `for` loops and `pytest`

* `scripts` contains scripts used to get information about projects and process the results

### Dependencies

* python 3.7.3
* astor 0.7.1
* tqdm
* lxml

### Setup

* Create virtual env: `virtualenv venv && source venv/bin/activate`

* Install the dependencies: `pip install -r requirements.txt`

### Author

@GuillaumeHaben