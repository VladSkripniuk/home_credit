import logging
from typing import Any, Dict

import mlflow

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def split_train_data(data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Node for splitting the dataset into training and test
    sets, each split into features and labels.
    The split ratio parameter is taken from conf/project/parameters.yml.
    The data and the parameters will be loaded and provided to your function
    automatically when the pipeline is executed and it is time to run this node.
    """
    test_size = parameters['test_size']

    data = data[[
        'TARGET',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
        'DAYS_BIRTH'
    ]]
   
    # Shuffle all the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split to training and testing data
    n = data.shape[0]
    n_test = int(n * test_size)
    training_data = data.iloc[n_test:, :].reset_index(drop=True)
    test_data = data.iloc[:n_test, :].reset_index(drop=True)

    # Split the data to features and labels
    train_data_x = training_data.loc[:, "EXT_SOURCE_1":"DAYS_BIRTH"]
    train_data_y = training_data['TARGET']
    test_data_x = test_data.loc[:, "EXT_SOURCE_1":"DAYS_BIRTH"]
    test_data_y = test_data['TARGET']

    # When returning many variables, it is a good practice to give them names:
    return train_data_x, test_data_x, train_data_y, test_data_y


def train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]
) -> LogisticRegression:
    """Node for training a simple logistic regression model.
    """
    X = X_train.to_numpy()
    y = y_train.to_numpy()

    model = LogisticRegression()
    model.fit(X, y)

    mlflow.sklearn.log_model(sk_model=model, artifact_path="logistic_regression_estimator")

    return model


def evaluate_model(
    model:LogisticRegression, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> None:
    """Node for evaluating a simple logistic regression model.
    """

    predictions_proba = predict_proba(model, X_test)
    predictions = predict(model, X_test)

    report_accuracy(predictions, y_test)
    report_rocauc(predictions_proba[:, 1], y_test)


def predict_proba(model: LogisticRegression, X_test: pd.DataFrame) -> np.ndarray:
    """Node for making probability predictions given a pre-trained model and a test set.
    """
    X = X_test.to_numpy()

    # Predict probabilities
    y_pred_proba = model.predict_proba(X)

    return y_pred_proba


def predict(model: LogisticRegression, X_test: pd.DataFrame) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set.
    """
    X = X_test.to_numpy()

    # Predict probabilities
    predictions = model.predict(X)

    return predictions


def report_accuracy(predictions: np.ndarray, y_test: pd.DataFrame) -> None:
    """Node for reporting the accuracy of the predictions performed by the
    previous node. Notice that this function has no outputs, except logging.
    """
    # Calculate accuracy of predictions
    accuracy = np.sum(predictions == y_test) / y_test.shape[0] * 100.0
    mlflow.log_metric("accuracy", accuracy)
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    log.info("Model accuracy on test set: %0.2f%%", accuracy)

def report_rocauc(predictions_proba: np.ndarray, y_test: pd.DataFrame) -> None:
    """Node for reporting the ROC-AUC of the predictions performed by the
    previous node. Notice that this function has no outputs, except logging.
    """
    # Calculate ROC-AUC of predictions
    rocauc = roc_auc_score(y_test, predictions_proba)
    mlflow.log_metric("ROC-AUC", rocauc)
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    log.info("Model ROC-AUC on test set: %0.4f", rocauc)


def prepare_submission(model: LogisticRegression, data: pd.DataFrame) -> None:
    X = data.to_numpy()

    # Predict probabilities
    scores = model.predict_proba(X)[:,1]
