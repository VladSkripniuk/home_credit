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
                inputs=["application_train_preprocessed", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
            ),
            node(func=train_model, inputs=["X_train", "y_train", "parameters"], outputs="logistic_regression_estimator"),
            node(
                func=evaluate_model,
                inputs=["logistic_regression_estimator", "X_test", "y_test"],
                outputs=None,
            ),
            node(
                func=prepare_submission,
                inputs=["logistic_regression_estimator", "application_test_preprocessed"],
                outputs=None,
                name='logistic_regression_prepare_submission',
            ),

        ]
    )
