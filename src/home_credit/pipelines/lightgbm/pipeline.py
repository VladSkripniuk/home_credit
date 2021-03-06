from kedro.pipeline import Pipeline, node

from home_credit.pipelines.logistic_regression.nodes import (
    split_train_data,
)
from home_credit.pipelines.lightgbm.nodes import (
    train_model,
    make_predictions,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_train_data,
                inputs=["features_table", "parameters"],
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val"],
            ),
            node(func=train_model, inputs=["X_train", "y_train", "X_val", "y_val", "parameters"], outputs="lgbm_estimator"),
            node(func=make_predictions, inputs=["X_test", "lgbm_estimator", "parameters"], outputs="predictions"),
        ]
    )
