import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.data import TabularDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def clean_data(data: TabularDataset) -> pd.DataFrame:
    """
    Preprocessing of tabular data. Includes encoding and missing value removal. For reasoning of the preprocessing, check the bank_management_eda notebook.
    :param data: Data in the format of a Azure TabularDataset
    :return: Preprocessed data in the format of a Pandas DataFrame
    """

    # Convert the data to a pandas dataframe and remove missing values
    df = data.to_pandas_dataframe().dropna()

    df.loc[:, 'age_group'] = '<30'
    df.loc[(df['age'] >= 30) & (df['age'] <= 60), 'age_group'] = '30-60'
    df.loc[(df['age'] > 60), 'age_group'] = '>60'

    df.loc[:, 'previous_group'] = '0-1'
    df.loc[(df['previous'] == 2), 'previous_group'] = '2-3-4'
    df.loc[(df['previous'] == 3), 'previous_group'] = '2-3-4'
    df.loc[(df['previous'] == 4), 'previous_group'] = '2-3-4'
    df.loc[(df['previous'] == 5), 'previous_group'] = '5-6'
    df.loc[(df['previous'] == 6), 'previous_group'] = '5-6'
    df.loc[(df['previous'] == 7), 'previous_group'] = '7'

    used_variables = ['age_group', 'previous_group','job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                      'poutcome', 'day_of_week', 'y']
    df = df[used_variables]

    unknown_variables = ['marital', 'default', 'housing', 'loan']

    # Remove data with 'unknown' in it
    for variable in unknown_variables:
        df.drop(df[df[variable] == 'unknown'].index, inplace=True)

    # Binary encode binary variables
    binary_variables = ['default', 'housing', 'loan', 'y']
    dictionary = {'no': 0, 'yes': 1}

    for variable in binary_variables:
        df[variable] = df[variable].apply(lambda x: dictionary[x])

    categorical_variables = ['age_group', 'previous_group','job', 'marital', 'education', 'contact', 'month', 'poutcome', 'day_of_week']

    df = pd.get_dummies(df, columns=categorical_variables, drop_first=False)

    # Label encode the months and days of week variables
    return df


def split_variables(df):
    """
    Split the target variable from the predictors.
    :param df: A Pandas Dataframe
    :return: Two Pandas Dataframes, (x_df, y_df), corresponding to the predictor and target variables
    """
    x_df = deepcopy(df)
    y_df = x_df.pop("y")
    return x_df, y_df


def main():
    # Bank Marketing Dataset - https://archive.ics.uci.edu/ml/datasets/Bank+Marketing (Moro et. al., 2014)
    # Predict whether a customer would accept an offer from a bank to open a deposit
    csv_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

    ds = TabularDatasetFactory.from_delimited_files(path=csv_path)

    dataframe = clean_data(ds)
    x, y = split_variables(dataframe)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, stratify=y)

    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    y_pred = model.predict(x_val)

    auc_weighted = roc_auc_score(y_pred, y_val, average='weighted')

    run.log("auc_weighted", np.float(auc_weighted))


if __name__ == '__main__':
    main()