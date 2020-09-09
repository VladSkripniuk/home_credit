import logging
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# LightGBM GBDT
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, parameters: Dict[str, Any]) -> None:

    feats = [f for f in X_train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    X_train = X_train[feats]
    X_val = X_val[feats]

    # LightGBM parameters found by Bayesian optimization
    clf = LGBMClassifier(
        nthread=4,
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=34,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.041545473,
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775,
        silent=-1,
        verbose=-1, )

    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)

    y_val_pred = clf.predict_proba(X_val, num_iteration=clf.best_iteration_)[:, 1]

    log = logging.getLogger(__name__)
    log.info('ROC-AUC score %.6f' % roc_auc_score(y_val, y_val_pred))
    mlflow.log_metric("ROC-AUC", roc_auc_score(y_val, y_val_pred))
    mlflow.lightgbm.log_model(lgb_model=clf.booster_, artifact_path="LightGBM_estimator")

    return clf
