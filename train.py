import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.data import TabularDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def clean_data(data: TabularDataset) -> pd.DataFrame:
    """
    Preprocessing of tabular data. Includes encoding and missing value removal
    :param data: Data in the format of a Azure TabularDataset
    :return: Preprocessed data in the format of a Pandas DataFrame
    """

    # Convert the data to a pandas dataframe and remove missing values
    df = data.to_pandas_dataframe().dropna()

    # Label encode the job, contract and eduction variables
    jobs = pd.get_dummies(df.job, prefix="job")
    df.drop("job", inplace=True, axis=1)
    df = df.join(jobs)

    contact = pd.get_dummies(df.contact, prefix="contact")
    df.drop("contact", inplace=True, axis=1)
    df = df.join(contact)

    education = pd.get_dummies(df.education, prefix="education")
    df.drop("education", inplace=True, axis=1)
    df = df.join(education)

    # Binary encode the marital/default/housing/loan/poutcome variables
    df["marital"] = df.marital.apply(lambda s: 1 if s == "married" else 0)
    df["default"] = df.default.apply(lambda s: 1 if s == "yes" else 0)
    df["housing"] = df.housing.apply(lambda s: 1 if s == "yes" else 0)
    df["loan"] = df.loan.apply(lambda s: 1 if s == "yes" else 0)
    df["poutcome"] = df.poutcome.apply(lambda s: 1 if s == "success" else 0)
    df["y"] = df.y.apply(lambda s: 1 if s == "yes" else 0)

    # Label encode the months and days of week variables
    months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
              "nov": 11, "dec": 12}
    weekdays = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}

    df["month"] = df.month.map(months)
    df["day_of_week"] = df.day_of_week.map(weekdays)
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

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

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

    accuracy = model.score(x_val, y_val)
    run.log("accuracy", np.float(accuracy))


if __name__ == '__main__':
    main()