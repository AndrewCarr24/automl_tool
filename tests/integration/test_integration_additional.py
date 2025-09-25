import pytest
import pandas as pd
import numpy as np
import pickle
from typing import Tuple
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error
from automl_tool.automl import AutoML


@pytest.fixture
def regression_data() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y


def test_regression_pipeline_basic(regression_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = regression_data
    automl = AutoML(X, y, 'target')
    automl.fit_pipeline()
    preds = automl.fitted_pipeline.predict(X)
    mae = mean_absolute_error(y, preds)
    # Loose bound; adjust with empirical stability if pipeline changes
    assert mae < 75, f"MAE unexpectedly high ({mae}) for diabetes regression integration test"


def test_feature_importance_permutation_path(regression_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = regression_data
    automl = AutoML(X, y, 'target')
    automl.fit_pipeline()
    # Only supported for XGB / SGD models in current implementation; skip otherwise
    model = automl.fitted_pipeline.best_estimator_['model']
    model_cls_name = model.__class__.__name__
    supported = {"XGBWithEarlyStoppingRegressor", "SGDRegressor", "XGBWithEarlyStoppingClassifier", "SGDClassifier"}
    if model_cls_name not in supported:
        pytest.skip(f"Permutation importance not implemented for model type {model_cls_name} in current plotting logic.")
    automl.get_feature_importance_scores(type='permutation')
    fi = automl.feature_importance_scores
    assert isinstance(fi, pd.DataFrame)
    assert 'importance_norm' in fi.columns
    assert fi['importance_norm'].iloc[0] == 100, 'Top feature should normalize to 100.'


def test_serialization_round_trip(regression_data: Tuple[pd.DataFrame, pd.Series], tmp_path) -> None:
    X, y = regression_data
    automl = AutoML(X, y, 'target')
    automl.fit_pipeline()
    path = tmp_path / 'automl.pkl'
    with open(path, 'wb') as f:
        pickle.dump(automl, f)
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    p1 = automl.fitted_pipeline.predict(X.head(20))
    p2 = loaded.fitted_pipeline.predict(X.head(20))
    # Allow exact equality; if stochastic components added later consider atol tolerance
    np.testing.assert_allclose(p1, p2)


def test_partial_dependence_end_to_end(regression_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = regression_data
    automl = AutoML(X, y, 'target')
    automl.fit_pipeline()
    automl.get_partial_dependence_plots()
    pdp = automl.partial_dependence_plots
    assert isinstance(pdp, dict)
    assert len(pdp) > 0


if __name__ == '__main__':  # pragma: no cover
    pytest.main()
