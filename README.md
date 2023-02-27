# cogpilotdatachallenge
## Introduction
This repository contains the submission notebook and helperfunctions file for the [cogpilot data challenge](http://pilotperformance.mit.edu/cogpilot-data-challenge-20-description) hosted by the US Department of the Air Force - MIT Artificial Intelligence (AI) Accelerator.

There are 2 tasks for this challenge, and our submission focuses on Task 1. Task 1 requires us to classify the difficulty level (1-4) of a flight simulation run using only physiological metrics. This task will be evaluated using F1 score and Area Under the ROC Curve (AUC) to assess classification accuracy between predicted and actual difficulty level.

## Setup
Do create a new virtual environment using the supplied requirements.txt to ensure that you have the required packages.

Once the virtual environment is set up, the folder structure is as such:
```
Main Directory
| Code - put code files here: 'Submission.ipynb' and 'helperfunctions.py'
| dataPackage - the original dataPackage with training set data
| dataPackageEval - the eval dataset; put the EvalSet_StartEndTimes.csv file in this folder
```

## Method
### Packages
We primarily use the sktime package for time-series predictions via machine learning. The documentation can be found [here](https://www.sktime.net/en/latest/index.html).

We also utilised pycaret to train the final ensemble model as it provided a fast and convenient way to test multiple regression classifiers rapidly on non time-series data. Documentation can be found [here](https://pycaret.gitbook.io/docs/)

### Steps
#### Generating ensembled probabilities
Utilising the time series classification APIs available in sktime, we methodically trained the APIs on each physiological signal and tested the models against 20% holdout data across 5 folds. (Certain datasets were too large to feasibly train on local machines. For these we only conducted 1 fold testing as each fold require up to 10 hours to run)

Through this process, we optimised the hyperparameters and selected the best classifier for that particular physiological signal.

Following this, the classifier with optimised parameters were used to generate the prediction probabilities for each physiological signal. The predictions of the five holdout sets of data was combined to one table. Here is an example generated for the ECG signal:

![pred_proba](https://github.com/skulu/cogpilotdatachallenge/blob/main/readme_pics/prediction_probabilities.png)

This was repeated for all the signals. The probabilities were joined on the `subject`, `difficulty` and `run` columns to form a large ensembled table.

#### Training models to generate probabilities from evaluation dataset
With the optimised classifiers and hyperparameters found in the previous section, we trained the classifiers on the entire physiological signal training set for each signal. These models were then used on the evaluation datasets to generate the probabilities like the above picture and once again ensembled across all physiological signals.

With these probabilities we now have a training set of ensembled probabilities and an evaluation set of ensembled probabilities with all physiological signals included.

#### Predicting landing difficulty
With the ensembled training table, we utilised pycaret to determine the importance of features. It was found that pupil diameter had the largest predictive power when validated against holdouts of the training data:

![feature importance](https://github.com/skulu/cogpilotdatachallenge/blob/main/readme_pics/feature_importance.png)

We hypothesize that this is because pupil diameter is an involuntary response that cannot be controlled by training, as compared to other responses such as heart rate, respiration rate etc. We also hypothesize that the poorer visibility of higher difficulty landings correspond well to dilated pupil diameters as the pilot's eyes will naturally dilate when visibility is poor to gather more light.

Utilising only the pupil diameter training set, we obtained the below accuracy, F1 and AUC scores (do note that due to random seeds results can change from run to run):

![diameter classifier metrics](https://github.com/skulu/cogpilotdatachallenge/blob/main/readme_pics/diameter_classifier_metrics.png)

## Results

## Team Members
1. Tan Kah Ming - [LinkedIn](https://www.linkedin.com/in/tankahming/) [GitHub](https://github.com/kmt112)
2. Skyler Tan - [LinkedIn](https://www.linkedin.com/in/skyler-tan/) [GitHub](https://github.com/skulu)
