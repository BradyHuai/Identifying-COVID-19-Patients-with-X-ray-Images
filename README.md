# Identifying-COVID-19-Patients-with-X-ray-Images

Clone the Github repository.
To run the train.py and test.py files, you need to have Python installed on your system along with the required dependencies. You can follow the below steps to run this file:

Install Python on your system if you don't have it.
Install the required dependencies by running the following command in your terminal/command prompt:

pip install pretrainedmodels cnn_finetune torch torchvision matplotlib

The dataset can be downloaded from this link with a kaggle account:
https://www.kaggle.com/datasets/jonathanchan/dlai3-hackathon-phase3-covid19-cxr-challenge

Sign up for a kaggle account if you don't have one. It is free to sign up.

Unzip the dataset and place it in the same project directory as the Python files.
Rename the outer folder to 'CSC413Project' and the inner folder to 'COVID-19 Radiography Database'.

Open a terminal/command prompt and navigate to the project directory.

Run the following command in your terminal/command prompt:
python filename.py
Here, replace 'filename.py' with the name of the file you want to run.

For example, if the name of the file is 'train.py', then the command will be:
python train.py

Run the train.py file first before running test.py because train.py will save the best model which test.py will use for evaluation.
