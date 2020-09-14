import logging
from typing import Any, Dict

import mlflow

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def split_train_data(df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Node for splitting the dataset into training and test
    sets, each split into features and labels.
    The split ratio parameter is taken from conf/project/parameters.yml.
    """
    test_size = parameters['test_size']

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
   
    training_data, val_data = train_test_split(train_df, test_size=test_size,
                stratify=train_df['TARGET'], random_state=parameters['random_state'])
    training_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Split the data to features and labels
    train_data_x = training_data.drop(labels=['TARGET'], axis=1)
    train_data_y = training_data['TARGET']
    val_data_x = val_data.drop(labels=['TARGET'], axis=1)
    val_data_y = val_data['TARGET']
    test_data_x = test_df.drop(labels=['TARGET'], axis=1)

    return train_data_x, val_data_x, test_data_x, train_data_y, val_data_y


def train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]
) -> LogisticRegression:
    """Node for training a simple logistic regression model.
    """
    feats = [f for f in X_train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X_train = X_train[feats]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(sk_model=model, artifact_path="logistic_regression_estimator")

    return model


def evaluate_model(
    model:LogisticRegression, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> None:
    """Node for evaluating a simple logistic regression model.
    """
    feats = [f for f in X_test.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X_test = X_test[feats]

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
