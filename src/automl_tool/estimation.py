
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import warnings
import re

warnings.filterwarnings(
    "ignore",
	message=r"(?s).*Parameters:\s*\{\s*\"time_series\"\s*\}\s*are\s*not\s*used.*",
    category=UserWarning,
    module="xgboost.callback"
)

# Utility function to split data and train with early stopping
def _fit_with_early_stopping(
	xgb_model: Union[XGBClassifier, XGBRegressor], 
	X_train: np.ndarray, 
	y_train: pd.Series
) -> Union[XGBClassifier, XGBRegressor]:
	"""Utility function to train a model with early stopping."""

	# if time_series:    
	# 	tscv = TimeSeriesSplit(n_splits=3)
	# 	# Use the last split for validation
	# 	splits = list(tscv.split(X_train))
	# 	train_index, val_index = splits[-1]
	# 	# Handles class called directly or through automl (which converts pd.dataframe to np.array)
	# 	if type(X_train) == pd.DataFrame:
	# 		X_inner_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
	# 	else:
	# 		X_inner_train, X_val = X_train[train_index], X_train[val_index]  
	# 	y_inner_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
	# else:

	X_inner_train, X_val, y_inner_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

	xgb_model.fit(X_inner_train, y_inner_train, eval_set=[(X_val, y_val)], verbose=False)

	return xgb_model

# Custom XGBoost classifier with early stopping
class XGBWithEarlyStoppingClassifier(XGBClassifier):
	"""XGBoost Classifier with built-in early stopping."""

	def __init__(self, n_estimators: int = 800, early_stopping_rounds: int = 5, **kwargs) -> None:
		super().__init__(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, **kwargs)

	def fit(self, X_train: np.ndarray, y_train: pd.Series) -> 'XGBWithEarlyStoppingClassifier':
		return _fit_with_early_stopping(self, X_train, y_train)

# Custom XGBoost regressor with early stopping
class XGBWithEarlyStoppingRegressor(XGBRegressor):
	"""XGBoost Regressor with built-in early stopping."""

	def __init__(self, n_estimators: int = 800, early_stopping_rounds: int = 5, **kwargs) -> None:
		super().__init__(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, **kwargs)
	
	def fit(self, X_train: np.ndarray, y_train: pd.Series) -> 'XGBWithEarlyStoppingRegressor':
		return _fit_with_early_stopping(super(), X_train, y_train)


