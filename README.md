# cogpilotdatachallenge

This repository contains the submission notebook and helperfunctions file for the [cogpilot data challenge](http://pilotperformance.mit.edu/cogpilot-data-challenge-20-description) hosted by the US Department of the Air Force - MIT Artificial Intelligence (AI) Accelerator.

There are 2 tasks for this challenge, and our submission focuses on Task 1. Task 1 requires us to classify the difficulty level (1-4) of a flight simulation run using only physiological metrics. This task will be evaluated using F1 score and Area Under the ROC Curve (AUC) to assess classification accuracy between predicted and actual difficulty level.

After testing the predictive power of the various physiological metrics supplied, we found that pupil diameter had the largest predictive potential when validated against holdouts of the training data set. We hypothesize that this is because pupil diameter is an involuntary response that cannot be controlled by training, as compared to other responses such as heart rate, respiration rate etc. We also hypothesize that the poorer visibility of higher difficulty landings can correspond well to dilated pupil diameters as the pilot's eyes will naturally dilate when visibility is poor to gather more light.
