import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from automl_tool.automl import AutoML
import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
from typing import Tuple

@pytest.fixture
def data() -> Tuple[pd.DataFrame, pd.Series]:
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

@pytest.fixture
def split_data(data: Tuple[pd.DataFrame, pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = data
    # Split the dataset into training and holdout sets
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_holdout, y_train, y_holdout

@pytest.fixture
def automl_estimator(split_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]) -> AutoML:
    X_train, _, y_train, _ = split_data
    # Initialize and fit the AutoML estimator
    estimator = AutoML(X_train, y_train, "target")
    estimator.fit_pipeline()
    return estimator

def test_fitted_pipeline_is_gridsearchcv(automl_estimator: AutoML) -> None:
    assert isinstance(automl_estimator.fitted_pipeline, GridSearchCV), "fitted_pipeline is not an instance of GridSearchCV"

def test_pipeline_not_none(automl_estimator: AutoML) -> None:
    assert automl_estimator.fitted_pipeline is not None, "fitted_pipeline should not be None"

def test_pipeline_has_best_estimator(automl_estimator: AutoML) -> None:
    assert hasattr(automl_estimator.fitted_pipeline, 'best_estimator_'), "fitted_pipeline should have a best_estimator_ attribute"

def test_estimator_log_loss(automl_estimator: AutoML, split_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]) -> None:
    _, X_holdout, _, y_holdout = split_data
    y_preds = automl_estimator.fitted_pipeline.predict_proba(X_holdout)
    loss = log_loss(y_holdout, y_preds)
    assert loss <= 0.32, f"Log loss should be <= 0.32 but got {loss}"

def test_get_feature_importance_scores(automl_estimator: AutoML) -> None:
    # Test the get_feature_importance_scores method
    automl_estimator.get_feature_importance_scores()
    feature_importances = automl_estimator.feature_importance_scores
    assert isinstance(feature_importances, pd.DataFrame), "Feature importances should be a pandas DataFrame"
    assert feature_importances["importance_norm"].iloc[0] == 100, "Feature importances should be normalized so most important feature is 100."

def test_get_partial_dependence_plots(automl_estimator: AutoML) -> None:
    # Test the get_partial_dependence_plots method
    automl_estimator.get_partial_dependence_plots()
    pdp_dict = automl_estimator.partial_dependence_plots
    assert isinstance(pdp_dict, dict), "Partial dependence plots should be returned as a dictionary"
    assert len(pdp_dict) > 0, "Partial dependence plots dictionary should not be empty"

if __name__ == "__main__":
    pytest.main()