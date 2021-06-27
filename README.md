# Optimizing an ML Pipeline in Azure
Part of the Udacity Azure ML Nanodegree 🧠 Most notebooks / scripts are not written from scratch, but provided as-is and modified to a certain extent.

## Summary
💸 The dataset used is related to bank marketing. It contains data about respondents to marketing phone calls, and whether or not they accepted an offer for a deposit. The training data was explored, analyzed and feature engineered prior to the training.

📊 The presented task is binary classification, where we aim to predict whether a person would accept a deposit offer, being given its profile. The dataset is imbalanced towards the negative class.

⚙ The best performing model for this task was an XGBoost classifier, consisting of regularized gradient-boosted tree-based classifiers, with a test accuracy of 0.787.

## Scikit-learn Pipeline
🧠 For the Scikit-Learn part of the pipeline, the chosen estimator was a simple **logistic regressor**, and its hyperparmeters were tuned using HyperDrive. The two hyperparameters were **C** (the inverse of the regularization strenght), and **max_iter** (the number of maximum gradient iterations).  

🏃‍♂️ A random parameter sampling method is faster than a grid search, and yields a similar best selection of hyperparameters in a fracion of time compared to the grid search.

🤚 Bandit early-stopping policy stops a model training if its accuracy is not within the specified slack factor/amount when compared to the best performing model.

## AutoML
⚗️ The pipelines used in the AutoML experiments included two steps, usually a preprocessor and an estimator.
  * Preprocessors: MaxAbsScaler, SparseNormalizer, StandardScalerWrapper, etc.
  * Estimators: Logistic Regression, LightGBM, GradientBoosting, DecisionTree, KNN, SCV, RandomForest, XGBoost, SGD.
More details about the parameters for the AutoML run can be found [here](https://gist.github.com/radandreicristian/c42bda8e0b60320162ac7bda38edd399), and the according documentation can be foudn [here](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig).

## Pipeline comparison
🧪 Since the dataset is imbalanced, an appropriate metric for validating model performance is the Weighted ROC AUC, not the accuracy. The values for the two best performing models on the test set were the following:
  * **0.770** for Logistic Regression after hyperparameter tuning.
  * **0.787** for the XGBoost classifier selected by AutoML.

🌳 Tree-based models are known to be state-of-the-art for many datasets, and this one was no exception. These models are more advanced and complex, and are able to capture better dependencies and relationships between variables.

## Future work
📈 As the model part of the task has been optimized (selected best model, performed cross-validation), it is likely that, in order to further improve the performance, the data needs improvemets. Some feature engineering has already been done on the data (as can be seen in the eda.ipynb notebook). Futhermore, better data can be accomplished by adding more data to balance the classes or doing better feature engineering. Adding more data could help the model generalize more and avoid overfitting and class imbalance, which generally increases performance, while feature engineering and selection could provide better variables for classification.
