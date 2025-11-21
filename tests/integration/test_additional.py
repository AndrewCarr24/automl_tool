import numpy as np
import pandas as pd
from automl_tool.automl import AutoML, SimpleESRegressor, AutoARIMARegressor
from automl_tool.estimation import XGBWithEarlyStoppingClassifier
from sklearn.linear_model import SGDClassifier


def test_classification_non_time_series_only_sklearn_models_false():
    """Binary classification (non-time-series) should only include sklearn models even when only_sklearn_models=False.
    Verifies that no time-series univariate model classes (SimpleESRegressor/AutoARIMARegressor) appear in the param grid.
    """
    # Synthetic binary classification dataset
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        'feat1': rng.normal(size=n),
        'feat2': rng.normal(loc=2.0, scale=1.5, size=n),
        'feat3': rng.integers(0, 5, size=n)
    })
    # Create a probabilistic target and threshold for class labels
    logits = 0.8 * X['feat1'] - 0.5 * X['feat2'] + 0.2 * X['feat3']
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    y = pd.Series(y, name='target')

    automl = AutoML(X, y, 'target', time_series=False, only_sklearn_models=False)
    automl.fit_pipeline()

    # Access underlying GridSearchCV
    gs = automl.fitted_pipeline

    # Assert param grid contains only sklearn models (boosting + SGD and optionally ElasticNet if regression, but classification path excludes ES/ARIMA)
    for entry in gs.param_grid:
        model_instance = entry['model'][0]
        assert not isinstance(model_instance, SimpleESRegressor), "SimpleESRegressor should not appear for classification."
        assert not isinstance(model_instance, AutoARIMARegressor), "AutoARIMARegressor should not appear for classification."

    best_model = gs.best_estimator_['model']
    assert not isinstance(best_model, (SimpleESRegressor, AutoARIMARegressor)), \
        "Best model should not be a univariate time series model for non-time-series classification."


def test_time_series_regression_includes_univariate_models():
    """Time series regression with only_sklearn_models=False should include univariate ES & ARIMA wrappers in the param grid."""
    n = 120
    t = np.arange(n)
    # Generate a simple seasonal + trend series as target
    y_vals = 10 + 0.1 * t + np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.5, n)
    y = pd.Series(y_vals.astype(float), name='value')

    # Exogenous feature(s) separate from target
    X_ts = pd.DataFrame({'trend_feature': t})

    automl_ts = AutoML(X_ts, y, 'value', time_series=True, only_sklearn_models=False)
    automl_ts.fit_pipeline(holdout_window=12)

    gs_ts = automl_ts.fitted_pipeline

    # Verify ES and ARIMA models are present in parameter grid
    es_present = any(isinstance(entry['model'][0], SimpleESRegressor) for entry in gs_ts.param_grid)
    arima_present = any(isinstance(entry['model'][0], AutoARIMARegressor) for entry in gs_ts.param_grid)

    assert es_present, "Expected SimpleESRegressor variants in param grid for time series regression."
    assert arima_present, "Expected AutoARIMARegressor variants in param grid for time series regression."
