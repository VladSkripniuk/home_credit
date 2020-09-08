from kedro.pipeline import Pipeline, node

from .nodes import (
    preprocess_applications_train,
    preprocess_applications_test,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_applications_train,
                inputs="application_train",
                outputs="application_train_preprocessed",
                name="preprocessing_application_train",
            ),
            node(
                func=preprocess_applications_test,
                inputs="application_test",
                outputs="application_test_preprocessed",
                name="preprocessing_application_test",
            ),
        ]
    )
