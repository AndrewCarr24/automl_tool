
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import Optional

def ts_train_test_split(X: pd.DataFrame, y: pd.Series, outcome_col: str, date_col: str, fdw: int, holdout_window: int):
    """
    Apply preprocessing and split the data into training and testing sets for time series modeling.
    """

    # Helper function to preprocess ts data
    def _ts_preproc(inp_tbl, inp_y):   
        preproc_tbl = (inp_tbl
        .pipe(lambda x: x.assign(**{f"lagged_{outcome_col}_{i}m": x[outcome_col].shift(i) for i in range(1, fdw + 1)}))
        # Drop the original date and outcome columns
        .drop([date_col, outcome_col], axis=1)
        # Rowwise deletion of missing values
        .dropna(axis=0)
        )
        preproc_y = inp_y.loc[preproc_tbl.index]

        return preproc_tbl, preproc_y

    # Reset index of X and y
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # Calculate the index to split the data
    train_end_index = X.shape[0] - (holdout_window)
    test_start_index = X.shape[0] - (fdw + holdout_window)

    # Split the data
    X_train = X.iloc[:train_end_index]
    X_test = X.iloc[test_start_index:]
    y_train = y.iloc[:train_end_index]
    y_test = y.iloc[test_start_index:]

    # Set the indices of both X and y train/test to the 'date' column 
    X_train.set_index(date_col, drop=False, inplace=True)
    y_train.index = X_train.index
    X_test.set_index(date_col, drop=False, inplace=True)
    y_test.index = X_test.index

    # Preprocess the data
    X_train, y_train = _ts_preproc(X_train, y_train)
    X_test, y_test = _ts_preproc(X_test, y_test)

    return X_train, X_test, y_train, y_test

class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_var: str, outcome_var: str, fdw: Optional[int]=None):
        self.date_var = date_var
        self.outcome_var = outcome_var
        self.fdw = fdw

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # No fitting necessary for this transformer
        return self
    
    def transform(self, X: pd.DataFrame):
        """
        Preprocess a time series DataFrame by creating lagged and logged lagged features.
        """
        return (X
                .pipe(lambda x: x.assign(**{f"lagged_{self.outcome_var}_{i}m": x[self.outcome_var].shift(i) for i in range(1, self.fdw + 1)}))
                # Drop the original date and outcome columns
                .drop([self.date_var, self.outcome_var], axis=1)
               )
    
# Custom preprocessor 
class Prepreprocessor:
    """
    A class to dynamically build preprocessing pipelines for numeric, categorical, and text data.
    """
    def __init__(self):
        pass

    def build_preprocessor(self, X: pd.DataFrame):
        """
        Builds a ColumnTransformer pipeline for preprocessing numeric, categorical, and text columns.

        Parameters:
        - X (pd.DataFrame): The input DataFrame.

        Returns:
        - ColumnTransformer: The preprocessing pipeline.
        """

        ## Preprocessing pipeline
        # Identify numerical, categorical, and text columns 
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        text_cols = X.select_dtypes(include=['string']).columns

        # Preprocessing for numerical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Preprocessing for textual data
        text_transformer = Pipeline(steps=[
            ('vectorizer', CountVectorizer())
        ])

        # ('ts', time_series_transformer, outcome_col)

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols),
                *[('text', text_transformer, text_col) for text_col in text_cols]
            ])
        
        return preprocessor
    


#####


    def build_preprocessor_experimental(self, X: pd.DataFrame, outcome_col: Optional[str] = None, fdw: Optional[str] = None):
        """
        Builds a ColumnTransformer pipeline for preprocessing numeric, categorical, and text columns.

        Parameters:
        - X (pd.DataFrame): The input DataFrame.
        - outcome_col (str) (optional): The name of the outcome column.

        Returns:
        - ColumnTransformer: The preprocessing pipeline.
        """

        ## Preprocessing pipeline
        # Identify numerical, categorical, text, and outcome columns 
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != outcome_col]
        categorical_cols = X.select_dtypes(include=['object']).columns
        text_cols = X.select_dtypes(include=['string']).columns
        outcome_cols = [outcome_col, "date"]

        # Preprocessing for numerical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Preprocessing for textual data
        text_transformer = Pipeline(steps=[
            ('vectorizer', CountVectorizer())
        ])

        transformer_lst = [('num', numeric_transformer, numeric_cols),
         ('cat', categorical_transformer, categorical_cols),
         *[('text', text_transformer, text_col) for text_col in text_cols]
         ]

        if outcome_cols[0]:
            print("Using time series transformer")
            ts_transformer = Pipeline(steps=[
                ('ts_preproc', TimeSeriesTransformer(date_var='date', outcome_var=outcome_cols[0], fdw=fdw)),
                ('imputer', SimpleImputer(strategy='median'))
                ])
            transformer_lst.append(('ts', ts_transformer, outcome_cols))

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(transformers=transformer_lst)

        return preprocessor