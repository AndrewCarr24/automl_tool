import os
import numpy as np
import pandas as pd
from automl_tool.automl import AutoML


def test_feature_importance_plot_saved():
    """Fit a simple classification AutoML run, compute feature importance, generate plot, and save PNG artifact.

    This test ensures the plotting code executes without error and produces a non-empty file. It is an
    integration / smoke test for the visual pipeline rather than a pixel-perfect regression test.
    """
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame({
        'x1': rng.normal(size=n),
        'x2': rng.normal(loc=1.5, scale=0.7, size=n),
        'x3': rng.integers(0, 4, size=n)
    })
    logits = 0.9 * X['x1'] - 0.4 * X['x2'] + 0.25 * X['x3']
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)
    y = pd.Series(y, name='target')

    automl = AutoML(X, y, 'target', time_series=False, only_sklearn_models=False)
    automl.fit_pipeline()
    automl.get_feature_importance_scores(type='shap')  # exercise SHAP path for feature-driven model
    automl.plot_feature_importance_scores(logo=False, top_k=10)

    fig = automl.feature_importance_plot
    assert fig is not None, "Feature importance plot figure should be created."

    out_dir = os.path.join(os.path.dirname(__file__), 'test_plots')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'feature_importance.png')

    fig.savefig(out_path, dpi=120)

    assert os.path.exists(out_path), f"Plot file was not saved at {out_path}."
    assert os.path.getsize(out_path) > 0, "Saved plot file is empty."