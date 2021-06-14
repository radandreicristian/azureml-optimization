# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
💸 The dataset used is related to bank marketing. It contains data about respondents to marketing phone calls, and whether or not they accepted an offer for a deposit.
📊 The presented task is binary classification, where we aim to predict whether a person would accept a deposit offer, being given its profile.
⚙ The best performing model for this task was a voting ensemble classifier, consisting of multiple gradient-boosted tree-based classifiers, with an accuracy of 0.9166.

## Scikit-learn Pipeline
🧠 For the Scikit-Learn part of the pipeline, the chosen estimator was a simple **logistic regressor**, and its hyperparmeters were tuned using HyperDrive. The two hyperparameters were **C** (the inverse of the regularization strenght), and **max_iter** (the number of maximum gradient iterations).  

🏃‍♂️ A random parameter sampling method is faster than a grid search, and yields a similar best selection of hyperparameters in a fracion of time compared to the grid search.

🤚 Bandit early-stopping policy stops a model training if its accuracy is not within the specified slack factor/amount when compared to the best performing model.

## AutoML
⚗️ The pipelines used in the AutoML experiments included two steps, usually a preprocessor and an estimator.
  * Preprocessors: MaxAbsScaler, SparseNormalizer, StandardScalerWrapper, etc.
  * Estimators: Usually gradient-boosted tree-based classifiers (XGBoost, LightGBM, Random Forests), but also Logistic Regression. 
 

## Pipeline comparison
🧪 The test accuracies reported by the two pipelines are:
  * 0.9089 for Logistic Regression after hyperparameter tuning.
  * 0.9160 for the Voting Ensemble classifier selected by AutoML.

🌳 Tree-based models are known to be state-of-the-art for many datasets, and this one was no exception. These models are more advanced and complex, and are able to capture better dependencies and relationships between variables.

## Future work
📈 As the model part of the task has been optimized, it is likely that in order to further improve the performance, the data has to undergo better preprocessing techniques.
