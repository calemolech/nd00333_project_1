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

**What are the benefits of the parameter sampler you chose?**
The parameter sampler I used allows for the adjustment of hyperparameters by testing various values within a specified range for each hyperparameter. I selected discrete options through the choice method for the parameters C and max_iter, where C represents the Regularization factor and max_iter signifies the maximum iterations allowed. I opted for RandomParameterSampling due to its speed and its capability to halt underperforming runs early. This method randomly picks values for hyperparameters from the established range.

**What are the advantages of the early stopping policy you selected?**
The early stopping policy I implemented terminates runs prematurely based on their performance against the primary metric, effectively identifying and stopping low-performing runs to save time and resources.

## AutoML
The AutoML process involved many different executions, employing various algorithms and hyperparameters. The most successful execution produced a VotingEnsemble model, which utilized a combination of hyperparameters including algorithms such as ['LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'LogisticRegression', 'SGD']

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The overall performance of the models was similar, with the Voting Ensemble model produced using AutoML showing only a slight improvement of 0.2% in accuracy compared to the Logistic Regression model trained with HyperDrive.

## Future work
- Improve imbalance with other algorithm
- Create more features with feature engieering

## Proof of cluster clean up
I already clean up with my coding in notebook.
```cluster.delete()``