# Script about SciKit ML Model

## 1. main.py

`python3 main.py ./path/to/dataset.json`

#### Features

* Open `dataset.json` containing flaky and non flaky tests.

## 2. eda.py

`python3 eda.py ./path/to/dataset.json`

#### Features

* Open `dataset.json` containing flaky and non flaky tests.
* Create correlation matrix for all features.

## 2. testModel.py

`python3 testModel.py ./path/to/dataset.json ./path/to/testMethods.json`

#### Features

* Open `dataset.json` containing flaky and non flaky tests.
* Open `testMethods.json` containing information about tests.
* See model performing on `testMethods.json`

