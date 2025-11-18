
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from .preprocessing import Prepreprocessor
from .estimation import XGBWithEarlyStoppingClassifier, XGBWithEarlyStoppingRegressor
from xgboost import XGBRegressor
from .plotting import PlotTools
from sklearn.metrics import make_scorer, log_loss, mean_absolute_error
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
import warnings
from typing import Optional

from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning as StatsmodelsConvergenceWarning

# Suppress convergence warnings 
warnings.filterwarnings("ignore", category=SklearnConvergenceWarning)
warnings.filterwarnings("ignore", category=StatsmodelsConvergenceWarning)

# Suppress worker stopping warnings
warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing as _HWES
 
import pmdarima as pmd

class AutoARIMARegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper around pmdarima.auto_arima (univariate).
    Ignores X except for length (like SimpleESRegressor).
    """
    def __init__(
        self,
        seasonal=True,
        m=12,
        max_p=5,
        max_q=5,
        max_P=2,
        max_Q=2,
        max_d=2,
        max_D=1,
        stepwise=True,
        suppress_warnings=True,
        information_criterion="aic"
    ):
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_d = max_d
        self.max_D = max_D
        self.stepwise = stepwise
        self.suppress_warnings = suppress_warnings
        self.information_criterion = information_criterion
        self._model = None
        self._y_len = None

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        self._y_len = len(y_arr)
        # If insufficient length for seasonality, turn it off
        seasonal_flag = self.seasonal and (self.m is not None) and (self._y_len >= 2 * self.m)
        try:
            self._model = pmd.auto_arima(
                y_arr,
                seasonal=seasonal_flag,
                m=self.m if seasonal_flag else 1,
                max_p=self.max_p,
                max_q=self.max_q,
                max_P=self.max_P,
                max_Q=self.max_Q,
                max_d=self.max_d,
                max_D=self.max_D,
                stepwise=self.stepwise,
                suppress_warnings=self.suppress_warnings,
                information_criterion=self.information_criterion,
                error_action="ignore",
                trace=False,
            )
        except Exception:
            # Fallback: simple mean model
            class _MeanModel:
                def __init__(self, mu): self.mu = mu
                def predict(self, n_periods): return np.full(n_periods, self.mu)
            self._model = _MeanModel(float(np.mean(y_arr)))
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        horizon = len(X)
        try:
            return np.asarray(self._model.predict(n_periods=horizon), dtype=float)
        except Exception:
            # Fallback: last value repeat
            return np.full(horizon, np.nan if self._y_len is None else np.nan)

class SimpleESRegressor(BaseEstimator, RegressorMixin):
    """
    Exponential Smoothing regressor that ignores any time index.
    Treats y as an ordered sequence only. X is ignored except for length.
    Provides fit, predict, and transform (returns in-sample fitted values).
    """
    def __init__(
        self,
        trend='add',            # 'add', 'mul', or None
        damped_trend=False,
        seasonal=None,          # 'add', 'mul', or None
        seasonal_periods=None,  # int or None
        use_boxcox=False,
        initialization_method='estimated',
        optimized=True
    ):
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.use_boxcox = use_boxcox
        self.initialization_method = initialization_method
        self.optimized = optimized
        self._res = None
        self._y_len = None

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        self._y_len = len(y_arr)
        sp = self.seasonal_periods if self.seasonal is not None else None
        # Guard: need at least 2*sp points for seasonal; else drop seasonality
        if sp is not None and self._y_len < 2 * sp:
            sp = None

        self._res = _HWES(
            y_arr,
            trend=self.trend,
            damped_trend=(self.damped_trend if self.trend else False),
            seasonal=self.seasonal if sp else None,
            seasonal_periods=sp,
            use_boxcox=self.use_boxcox,
            initialization_method=self.initialization_method,
        ).fit(optimized=self.optimized)

        return self

    def predict(self, X):
        if self._res is None:
            raise RuntimeError("Model not fitted.")
        horizon = len(X)
        return self._res.forecast(horizon)

    def transform(self, X):
        # Return in-sample fitted values aligned to X length (truncate or pad)
        if self._res is None:
            raise RuntimeError("Model not fitted.")
        fitted = np.asarray(getattr(self._res, 'fittedvalues', np.full(self._y_len, np.nan)))
        if len(X) <= self._y_len:
            return fitted[:len(X)]
        # pad with last fitted for excess length
        pad = np.full(len(X) - self._y_len, fitted[-1])
        return np.concatenate([fitted, pad])
    
# AutoML class to automate the machine learning pipeline
class AutoML:
    """
    AutoML class to automate the machine learning pipeline.

    This class provides functionality to automate the process of building, training, and evaluating machine learning models. It supports both time series and non-time series data, and can handle binary classification and regression tasks.

    Attributes:
    X (pd.DataFrame): The input feature matrix.
    y (pd.Series): The target variable.
    target (str): The name of the target variable.
    time_series (bool): Whether the data is time series data. Default is False.
    boosting_model: The boosting model used for classification or regression.
    elastic_net_model: The ElasticNet model used for classification or regression.
    scoring_func: The scoring function used for model evaluation.
    response_method: The method used to get the model's response (e.g., 'predict_proba' for classification).

    Methods:
    __init__(self, X: pd.DataFrame, y: pd.Series, outcome: str, time_series: bool = False):
        Initializes the AutoML class with the input data and target variable.

    fit_pipeline(self, holdout_window: int = None):
        Fits the pipeline with cross-validation and grid search.

    get_feature_importance_scores(self, X_pred: pd.DataFrame = None, y_pred: pd.Series = None, type: str = 'shap'):
        Calculates and stores feature importance scores for the fitted pipeline.

    plot_feature_importance_scores(self, logo: bool = False, top_k: Optional[int] = None):
        Generates and stores a plot of the feature importance scores.

    get_partial_dependence_plots(self, logo: bool = False):
        Generates partial dependence plots for the fitted pipeline.

    get_backtest_plots(self):
        Generates backtest plots for the fitted pipeline (only works for time series).
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, outcome: str, time_series: bool = False) -> None:
        self.X = X
        self.y = y
        self.target = outcome
        self.time_series = time_series

        # Throw error if target variable is in the input feature matrix
        if self.target in self.X.columns:
            raise ValueError('Target variable cannot be in the input feature matrix.')

        # Store boosting and linear classification/regression models
        if self.y.dtype in [object, bool]:
            raise ValueError('Target variable must be numeric. For binary classification, convert target variable to integer with values 0 and 1. For regression, convert target variable to float.')
        elif self.y.dtype == int:
            if self.y.value_counts().shape[0] != 2:
                raise ValueError('Target variable must be binary for binary classification. Multiclass modeling is currently not supported.')
            self.boosting_model = XGBWithEarlyStoppingClassifier()
            self.elastic_net_model_sgd = SGDClassifier(loss='log_loss', penalty='elasticnet', random_state=42)
            self.scoring_func = log_loss
            self.response_method = 'predict_proba'
        elif self.y.dtype == float:
            self.boosting_model = XGBWithEarlyStoppingRegressor()
            self.elastic_net_model_sgd = SGDRegressor(loss='squared_error', penalty='elasticnet', random_state=42)
            self.elastic_net_model_coord_desc = ElasticNet(random_state=42)
            self.scoring_func = mean_absolute_error
            self.response_method = 'predict'

    def fit_pipeline(self, holdout_window: Optional[int] = None) -> None:
        """
        Fit the pipeline with cross-validation and grid search.

        Parameters:
        holdout_window (int): The number of observations to use as the holdout window for time series modeling. Required if self.time_series is True.

        Returns:
        None: The method sets the fitted_pipeline attribute with the fitted pipeline.
        """
        # Validate that target variable has no missing values
        if self.y.isnull().any():
            missing_count = self.y.isnull().sum()
            total_count = len(self.y)
            raise ValueError(
                f"""Target variable '{self.target}' contains {missing_count} missing values out of {total_count} total observations. AutoML is built on top of scikit-learn, which requires complete target data for training. Please remove or impute missing target values."""
            )
        
        # Define the cross-validation object based on whether time series or not
        if self.time_series:
            if holdout_window is None:
                raise ValueError('For time series modeling, the holdout_window parameters must be specified.')
            cv_obj = TimeSeriesSplit(n_splits=5, test_size=holdout_window, gap=0)
            self.holdout_window = holdout_window
        else:
            cv_obj = 5

        # Convert boolean features to ints 
        self.X = self.X.apply(lambda col: col.astype(int) if col.dtype == 'bool' else col)

        # Build the preprocessor
        preprocessor = Prepreprocessor().build_preprocessor(self.X)
        
        # Create the pipeline
        tmp_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.boosting_model)
        ])

        # Define the parameter grid for GridSearchCV
        parameters = [
            {
                'model': [self.boosting_model],
                'model__n_estimators': [1000],
                'model__early_stopping_rounds': [5],
                'model__learning_rate': [0.08],
                'model__max_depth': [2, 3],
                'model__colsample_bytree': [0.3],
            },
            {
                'model': [self.elastic_net_model_sgd],
                'model__l1_ratio': [0, .05, .1, .3, .5, .6, .8, .9, 1],
                'model__alpha': [2, 1, .5, .1, .01, .005, .001, .00001, 5e-6],
                'model__max_iter': [3000],
            }
            ]
        
        # Include enet coordinate descent model if regression problem
        if self.y.dtype == float:

            # # ENet model 
            parameters.append({
                'model': [self.elastic_net_model_coord_desc],
                'model__l1_ratio': [0, .05, .1, .3, .5, .6, .8, .9, 1],
                'model__alpha': [2, 1, .5, .1, .01, .005, .001, .00001, 5e-6],
                'model__max_iter': [3000],
            })

            if self.time_series:

                # Add ES models - split into 4 groups to avoid redundant parameter combinations
                # 1. Trend + Seasonal (both damped_trend and seasonal_periods matter)
                parameters.append({
                    'model': [SimpleESRegressor(trend='add', seasonal='add')],
                    'model__damped_trend': [False, True],
                    'model__seasonal_periods': [4, 6, 8, 12],
                    'model__use_boxcox': [False, True],
                    'model__initialization_method': ['estimated', 'heuristic'],
                })
                
                # 2. Trend only, no seasonal (damped_trend matters, seasonal_periods ignored)
                parameters.append({
                    'model': [SimpleESRegressor(trend='add', seasonal=None)],
                    'model__damped_trend': [False, True],
                    'model__seasonal_periods': [12],  # doesn't matter, pick one
                    'model__use_boxcox': [False, True],
                    'model__initialization_method': ['estimated', 'heuristic'],
                })
                
                # 3. Seasonal only, no trend (seasonal_periods matters, damped_trend ignored)
                parameters.append({
                    'model': [SimpleESRegressor(trend=None, seasonal='add')],
                    'model__damped_trend': [False],  # doesn't matter, pick one
                    'model__seasonal_periods': [4, 6, 8, 12],
                    'model__use_boxcox': [False, True],
                    'model__initialization_method': ['estimated', 'heuristic'],
                })
                
                # 4. Simple exponential smoothing (no trend, no seasonal)
                parameters.append({
                    'model': [SimpleESRegressor(trend=None, seasonal=None)],
                    'model__damped_trend': [False],  # doesn't matter
                    'model__seasonal_periods': [12],  # doesn't matter
                    'model__use_boxcox': [False, True],
                    'model__initialization_method': ['estimated', 'heuristic'],
                })

                # Add AutoARIMA models - split into seasonal and non-seasonal to avoid redundancy
                # Seasonal ARIMA: try different seasonal periods
                parameters.append({
                    'model': [AutoARIMARegressor(seasonal=True)],
                    'model__m': [4, 6, 12],
                    'model__max_p': [5],
                    'model__max_q': [5],
                    'model__max_P': [2],
                    'model__max_Q': [2],
                    'model__stepwise': [True],
                    'model__information_criterion': ['aic'],
                })
                
                # Non-seasonal ARIMA: m doesn't matter, only need one
                parameters.append({
                    'model': [AutoARIMARegressor(seasonal=False)],
                    'model__m': [12],  # value doesn't matter when seasonal=False
                    'model__max_p': [5],
                    'model__max_q': [5],
                    'model__max_P': [2],
                    'model__max_Q': [2],
                    'model__stepwise': [True],
                    'model__information_criterion': ['aic'],
                })
                
        # Perform grid search with cross-validation
        scoring = make_scorer(self.scoring_func, greater_is_better=False, response_method=self.response_method)
        grid_tmp = GridSearchCV(tmp_pipeline, parameters, cv=cv_obj, n_jobs=-1, verbose=0, scoring=scoring)
        
        self.fitted_pipeline = grid_tmp.fit(self.X, self.y)

    def get_feature_importance_scores(
        self, 
        X_pred: Optional[pd.DataFrame] = None, 
        y_pred: Optional[pd.Series] = None, 
        type: str = 'shap'
    ) -> None:
        """
        Calculate and store feature importance scores for the fitted pipeline.

        Parameters:
        X_pred (pd.DataFrame, optional): The input features for prediction. If None, uses self.X.
        y_pred (pd.Series, optional): The target values for prediction. If None, uses self.y.
        type (str): The type of feature importance to calculate. Options are 'shap' for SHAP values and 'permutation' for permutation importance. Default is 'shap'.

        Returns:
        None: The method sets the feature_importance_scores and feature_importance_type attributes with the calculated scores and type.
        """
        if X_pred is None:
            X_pred = self.X
        if y_pred is None:
            y_pred = self.y 

        if type == 'shap':
            importance_df = PlotTools().get_shap_values(self.fitted_pipeline, X_pred, y_pred)
        elif type == 'permutation':
            importance_df = PlotTools().get_permutation_importance(self.fitted_pipeline, X_pred, y_pred)    

        self.feature_importance_scores = importance_df
        self.feature_importance_type = type

    def plot_feature_importance_scores(self, logo: bool = False, top_k: Optional[int] = None) -> None:
        """
        Generate and store a plot of the feature importance scores.

        Parameters:
        logo (bool): Whether to include a logo in the plot. Default is False.
        top_k (int, optional): The number of top features to display in the plot. If None, all features are displayed.

        Returns:
        None: The method sets the feature_importance_plot attribute with the generated plot.
        """
        tmp_plt = PlotTools().plot_feature_importance(self.feature_importance_scores, logo, top_k)
        self.feature_importance_plot = tmp_plt

    def get_partial_dependence_plots(self, logo: bool = False) -> None:
        """
        Generate partial dependence plots for the fitted pipeline.

        Parameters:
        logo (bool): Whether to include a logo in the plots. Default is False.

        Returns:
        None: The method sets the partial_dependence_plots attribute with a dict of the generated plots.
        """
        tmp_plts = PlotTools().get_pdp(self.fitted_pipeline, self.X, logo, self.target)
        self.partial_dependence_plots = tmp_plts

    def get_backtest_plots(self) -> None:
        """
        Generate backtest plots for the fitted pipeline.

        Returns:
        None: The method sets the backtest_plots attribute with the generated plots.
        """
        if not self.time_series:
            raise ValueError('Backtest plots are only available for time series data.')
        
        tmp_plts = PlotTools().get_bt_plts(self.fitted_pipeline, self.X, self.y, self.holdout_window)
        self.backtest_plots = tmp_plts

    

