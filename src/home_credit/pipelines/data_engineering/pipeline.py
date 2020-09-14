from kedro.pipeline import Pipeline, node

from .nodes import (
    preprocess_application_train_test_EXT_BIRTH,
    preprocess_application_train_test,
    preprocess_bureau_and_balance,
    preprocess_previous_application,
    preprocess_pos_cash,
    preprocess_installments_payments,
    preprocess_credit_card_balance,
    join_all_tables,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_application_train_test_EXT_BIRTH,
                inputs=["application_train", "application_test", "parameters"],
                outputs="application_EXT_BIRTH_train_test_df",
                name="preprocess_application_train_test_EXT_BIRTH",
            ),
            node(
                func=preprocess_application_train_test,
                inputs=["application_train", "application_test", "parameters"],
                outputs="application_train_test_df",
                name="preprocess_application_train_test",
            ),
            node(
                func=preprocess_bureau_and_balance,
                inputs=["bureau", "bureau_balance", "parameters"],
                outputs="bureau_and_balance_preprocessed",
                name="preprocess_bureau_and_balance",
            ),
            node(
                func=preprocess_previous_application,
                inputs=["previous_application", "parameters"],
                outputs="previous_application_preprocessed",
                name="preprocess_previous_application",
            ),
            node(
                func=preprocess_pos_cash,
                inputs=["POS_CASH_balance", "parameters"],
                outputs="pos_cash_preprocessed",
                name="preprocess_pos_cash",
            ),
            node(
                func=preprocess_installments_payments,
                inputs=["installments_payments", "parameters"],
                outputs="installments_payments_preprocessed",
                name="preprocess_installments_payments",
            ),
            node(
                func=preprocess_credit_card_balance,
                inputs=["credit_card_balance", "parameters"],
                outputs="credit_card_balance_preprocessed",
                name="preprocess_credit_card_balance",
            ),
            node(
                func=join_all_tables,
                inputs=["application_train_test_df", "bureau_and_balance_preprocessed",
                    "previous_application_preprocessed", "pos_cash_preprocessed",
                    "installments_payments_preprocessed", "credit_card_balance_preprocessed"],
                outputs="features_table",
                name="join_all_tables",
            ),


        ]
    )
