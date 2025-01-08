
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split

# Utility function to split data and train with early stopping
def _fit_with_early_stopping(xgb_model, X_train, y_train):
    """Utility function to train a model with early stopping."""
    X_inner_train, X_val, y_inner_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    xgb_model.fit(X_inner_train, y_inner_train, eval_set=[(X_val, y_val)], verbose=False)
    return xgb_model

# Custom XGBoost classifier with early stopping
class XGBWithEarlyStoppingClassifier(XGBClassifier):
    """XGBoost Classifier with built-in early stopping."""

    def __init__(self, n_estimators=800, early_stopping_rounds=4, **kwargs):
        super().__init__(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, **kwargs)

    def fit(self, X_train, y_train):
        return _fit_with_early_stopping(super(), X_train, y_train)

# Custom XGBoost regressor with early stopping
class XGBWithEarlyStoppingRegressor(XGBRegressor):
    """XGBoost Regressor with built-in early stopping."""

    def __init__(self, n_estimators=800, early_stopping_rounds=4, **kwargs):
        super().__init__(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds, **kwargs)

    def fit(self, X_train, y_train):
        return _fit_with_early_stopping(super(), X_train, y_train)


