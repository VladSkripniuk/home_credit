from typing import Any, Dict

import pandas as pd


def preprocess_applications_train(applications: pd.DataFrame) -> pd.DataFrame:
    """Preprocess application_train.csv data.

        Args:
            applications: Source data.
        Returns:
            Preprocessed data.

    """
    applications = applications[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    applications = applications.fillna(applications.mean())

    return applications

def preprocess_applications_test(applications: pd.DataFrame) -> pd.DataFrame:
    """Preprocess application_test.csv data.

        Args:
            applications: Source data.
        Returns:
            Preprocessed data.

    """
    applications = applications[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    applications = applications.fillna(applications.mean())

    return applications
