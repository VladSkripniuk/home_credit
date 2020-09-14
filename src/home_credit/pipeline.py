from typing import Dict

from kedro.pipeline import Pipeline

from home_credit.pipelines import data_engineering
from home_credit.pipelines import logistic_regression
from home_credit.pipelines import lightgbm


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    data_engineering_pipeline = data_engineering.create_pipeline()
    logistic_regression_pipeline = logistic_regression.create_pipeline()
    lightgbm_pipeline = lightgbm.create_pipeline()

    return {
        "de": data_engineering_pipeline,
        "logistic_regression_pipeline": logistic_regression_pipeline,
        "lightgbm_pipeline": lightgbm_pipeline,
        "__default__": data_engineering_pipeline + lightgbm_pipeline,
    }
