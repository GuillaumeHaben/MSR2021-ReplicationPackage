# Python scripts

- buildDataset.py
 
Build dataset of flaky / non flaky tests with their CUT for all projects.

- crossValidateModel.py

Cross validation with trained RFC model.

- infoDataset.py

Gives information about the generated dataset.

- model.py

Create and save RFC model

# SH scripts

- getMethodsBody.sh

Get all methods of a proejct to be considered as CUT

- getTestsBody.sh

Get junit test of a project based on a list of wanted test (all or specific tests)

- prepareProject.sh

Checkout to the commit of interest and mvn clean the project

- cleanProject.sh

Go back to master

# Folders

- results

Contains dataset for projects when built for 20% training set, 80% test set. (Experience deprecated)