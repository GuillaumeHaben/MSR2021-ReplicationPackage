# Data created during experimentations

- `v1` contains all files related to the first dataset created with limited metrics

* `flakyTests.json` contains code metrics of 1385 tests that has been flagged as flaky by DeFlaker.
* `nonFlakyTests.json` contains code metrics of 2288 tests taken from the same projects as DeFlaker and that are apparently not flaky.
* `dataset.json` contains both flaky and non flaky tests in one file.
* `dataset.test.json` contains only flaky and non flaky with DepthOfInheritance = 0.

- `v2` contains all files related to the second dataset with extended metrics

* `flakyTestsV2.json` contains code metrics of 1359 tests that has been flagged as flaky by DeFlaker.
* `nonFlakyTests.json` contains code metrics of 14661 tests taken from the same projects as DeFlaker and that are apparently not flaky.
* `dataset.json` contains both flaky and non flaky tests in one file.

- `v3` contains JSON files per project. Features per test are the following:

* MethodName
* ClassName
* ProjectName
* Commit (date timestamp and ID)
* Raw source code
* Code metrics
* 5 most similar methods representing the CUT
* Raw source code and code metrics for each CUT. 