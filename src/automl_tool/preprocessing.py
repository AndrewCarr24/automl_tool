
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import Optional, Tuple

def ts_train_test_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    outcome_col: str, 
    date_col: str, 
    fdw: int, 
    holdout_window: int,
    forecast_window: Optional[int] = 1 
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Apply preprocessing and split the data into training and testing sets for time series modeling.
    """

    # Helper function to preprocess ts data
    def _ts_preproc(inp_tbl: pd.DataFrame, inp_y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:   
        preproc_tbl = (inp_tbl
        .pipe(lambda x: x.assign(**{f"lagged_{outcome_col}_{i}m": x[outcome_col].shift(i) for i in range(forecast_window, fdw + 1)}))
        .pipe(lambda x: x.assign(**{f"logged_lagged_{outcome_col}_{i}m": np.log1p(x[outcome_col].shift(i)) for i in range(forecast_window, fdw + 1)}))
        .pipe(lambda x: x.assign(**{f"rolling_avg_{outcome_col}_{i}m": x[outcome_col].shift(1).rolling(window=i).mean() for i in range(forecast_window, fdw + 1)}))
        .pipe(lambda x: x.assign(**{f"min_{outcome_col}_{i}m": x[outcome_col].shift(1).rolling(window=i).min() for i in range(forecast_window, fdw + 1)}))
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

# Custom preprocessor 
class Prepreprocessor:
    """
    A class to dynamically build preprocessing pipelines for numeric, categorical, and text data.
    """
    def __init__(self) -> None:
        pass

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
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

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols),
                *[('text', text_transformer, text_col) for text_col in text_cols]
            ])

        # Store column metadata for efficient feature mapping without refitting
        preprocessor._numeric_cols = list(numeric_cols)
        preprocessor._categorical_cols = list(categorical_cols)
        preprocessor._text_cols = list(text_cols)
        
        return preprocessor
    
