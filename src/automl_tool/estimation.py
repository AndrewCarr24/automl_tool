
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
	model: Union['XGBWithEarlyStoppingClassifier', 'XGBWithEarlyStoppingRegressor'],
	X_train: Union[np.ndarray, pd.DataFrame],
	y_train: pd.Series
) -> Union['XGBWithEarlyStoppingClassifier', 'XGBWithEarlyStoppingRegressor']:
	"""Train wrapper model with early stopping on a simple validation split.

	Splits off 10% of the provided training data (random split with fixed seed) to form an
	evaluation set for XGBoost early stopping. Calls the BASE class fit (not the overridden
	wrapper fit) to avoid recursion. Returns the fitted wrapper instance.
	"""

	# Simple random validation split (future: make fraction configurable / stratify for classification)
	X_inner_train, X_val, y_inner_train, y_val = train_test_split(
		X_train, y_train, test_size=0.1, random_state=42
	)

	# Dispatch to base estimator fit via explicit super to avoid calling overridden fit again.
	if isinstance(model, XGBWithEarlyStoppingClassifier):
		super(XGBWithEarlyStoppingClassifier, model).fit(
			X_inner_train, y_inner_train,
			eval_set=[(X_val, y_val)],
			verbose=False
		)
	elif isinstance(model, XGBWithEarlyStoppingRegressor):
		super(XGBWithEarlyStoppingRegressor, model).fit(
			X_inner_train, y_inner_train,
			eval_set=[(X_val, y_val)],
			verbose=False
		)
	else:
		# Fallback: assume model behaves like base XGB; call its fit directly.
		model.fit(X_inner_train, y_inner_train, eval_set=[(X_val, y_val)], verbose=False)

	return model

# Custom XGBoost classifier with early stopping
class XGBWithEarlyStoppingClassifier(XGBClassifier):
	"""XGBoost Classifier with built-in early stopping using an internal 10% validation split."""

	def __init__(self, n_estimators: int = 800, early_stopping_rounds: int = 5, **kwargs) -> None:
		super().__init__(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, **kwargs)

	def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: pd.Series) -> 'XGBWithEarlyStoppingClassifier':
		return _fit_with_early_stopping(self, X_train, y_train)

# Custom XGBoost regressor with early stopping
class XGBWithEarlyStoppingRegressor(XGBRegressor):
	"""XGBoost Regressor with built-in early stopping using an internal 10% validation split."""

	def __init__(self, n_estimators: int = 800, early_stopping_rounds: int = 5, **kwargs) -> None:
		super().__init__(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, **kwargs)

	def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: pd.Series) -> 'XGBWithEarlyStoppingRegressor':
		return _fit_with_early_stopping(self, X_train, y_train)


