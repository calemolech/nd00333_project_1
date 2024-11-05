# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The aim of this project is to develop a binary classification model to determine whether a client will sign up for a term deposit at the bank using the UCI Bank Marketing dataset, which includes client details like age, job, marital status, and education levels.

To achieve this, several models were developed through two primary methods: AzureML HyperDrive and Microsoft’s AutoML. Out of the 36 LogisticRegression models produced by HyperDrive, the highest-performing model achieved an accuracy of 91. Meanwhile, the top-performing model from AutoML was a VotingEnsemble, which is a little bit better.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
- Data Ingestion: This first phase involves gathering data from various file sources.
- Data Preprocessing:
    - Data Cleaning: Addressed missing values in the initial data. Implemented one-hot encoding for categorical variables such as contact and education levels. Preprocessed temporal features like months and weekdays. Employed integer encoding on the target variable.
- Data Splitting: The dataset is segmented into training, validation, and testing subsets to ensure that the model undergoes training, validation, and testing on different sections of the data.
- Model Training:
    - Hyperparameter Optimization: Used Azure ML's Hyperdrive service for hyperparameter tuning, which is the process of determining the best hyperparameter settings to optimize model performance. This process is generally resource-intensive and complex, thus we opted for Hyperdrive to automate it. Techniques like grid search, random search, or Bayesian optimization were employed to identify ideal model parameters.
    - Cross-validation: Implemented during the training phase to verify the stability of the model's performance across various data subsets.
    - Classification Algorithm: The project's primary objective was classification, for which we utilized the logistic regression algorithm. This algorithm is well-suited for binary classification tasks, where the outcome variable is categorical.

**What are the benefits of the parameter sampler you chose?**
The parameter sampler I used allows for the adjustment of hyperparameters by testing various values within a specified range for each hyperparameter. I selected discrete options through the choice method for the parameters C and max_iter, where C represents the Regularization factor and max_iter signifies the maximum iterations allowed. I opted for RandomParameterSampling due to its speed and its capability to halt underperforming runs early. This method randomly picks values for hyperparameters from the established range.

**What are the advantages of the early stopping policy you selected?**
The early stopping policy I implemented terminates runs prematurely based on their performance against the primary metric, effectively identifying and stopping low-performing runs to save time and resources.

## AutoML
The AutoML process involved many different executions, employing various algorithms and hyperparameters. The most successful execution produced a VotingEnsemble model, which utilized a combination of hyperparameters including algorithms such as ['LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'LogisticRegression', 'SGD']

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The overall performance of the models was similar, with the Voting Ensemble model produced using AutoML showing only a slight improvement of 0.83% in accuracy compared to the Logistic Regression model trained with HyperDrive.

#### AutoML result
24    VotingEnsemble                                0:01:00             0.9176    0.9176

#### Hyperdrive result
Best Run Id:  HD_2e477c33-057e-4fb4-a711-fad47c57378b_4
Accuracy: 0.9092564491654022
['--C', '1.3311456981265843', '--max_iter', '250']

**C is the Regularization while max_iter is the maximum number of iterations.**

### Reasons for Performance Differences
- Model Complexity and Diversity
- Hyperparameter Tuning and Model Optimization
    - AutoML: Automatically experiments with many different models and hyperparameters
    - HyperDrive: Focuses on optimizing the hyperparameters of a single model (logistic regression in this case)

## Future work
- Improve imbalance with other algorithm
- Create more features with feature engieering

## Proof of cluster clean up
I already clean up with my coding in notebook.
```cluster.delete()``