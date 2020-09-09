from kedro.pipeline import Pipeline, node

from .nodes import (
    preprocess_application_train_test_EXT_BIRTH,
    preprocess_application_train_test,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_application_train_test_EXT_BIRTH,
                inputs=["application_train", "application_test", "parameters"],
                outputs="application_EXT_BIRTH_train_test_df",
                name="preprocessing_application_EXT_BIRTH",
            ),
            node(
                func=preprocess_application_train_test,
                inputs=["application_train", "application_test", "parameters"],
                outputs="application_train_test_df",
                name="preprocessing_application",
            ),
        ]
    )
