from kedro.pipeline import Pipeline, node

from home_credit.pipelines.logistic_regression.nodes import (
    split_train_data,
    train_model,
    evaluate_model,
    prepare_submission,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_train_data,
                inputs=["application_EXT_BIRTH_train_test_df", "parameters"],
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val"],
            ),
            node(func=train_model, inputs=["X_train", "y_train", "parameters"], outputs="logistic_regression_estimator"),
            node(
                func=evaluate_model,
                inputs=["logistic_regression_estimator", "X_val", "y_val"],
                outputs=None,
            ),
        ]
    )
