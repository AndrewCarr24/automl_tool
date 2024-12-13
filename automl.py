
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from automl_tool.preprocessing import Prepreprocessor
from automl_tool.estimation import XGBWithEarlyStoppingClassifier, XGBWithEarlyStoppingRegressor
from automl_tool.plotting import PlotTools
from sklearn.metrics import make_scorer, log_loss, mean_absolute_error
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.model_selection import TimeSeriesSplit
import warnings
# Suppress worker stopping warnings
warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")
# Suppress warnings about max iterations reached before convergence
# warnings.filterwarnings("ignore", message="Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.")


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
    def __init__(self, X: pd.DataFrame, y: pd.Series, outcome: str, time_series: bool = False):
        self.X = X
        self.y = y
        self.target = outcome
        self.time_series = time_series

        # Throw error if target variable is in the input feature matrix
        if self.target in self.X.columns:
            raise ValueError('Target variable cannot be in the input feature matrix.')

        # Store boosting and linear classification/regression models
        if self.y.dtype == object:
            raise ValueError('Target variable must be numeric. For binary classification, convert target variable to integer with values 0 and 1. For regression, convert target variable to float.')
        elif self.y.dtype == int:
            if self.y.value_counts().shape[0] != 2:
                raise ValueError('Target variable must be binary for binary classification. Multiclass modeling is currently not supported.')
            self.boosting_model = XGBWithEarlyStoppingClassifier()
            self.elastic_net_model = SGDClassifier(loss='log_loss', penalty='elasticnet')
            self.scoring_func = log_loss
            self.response_method = 'predict_proba'
        elif self.y.dtype == float:
            self.boosting_model = XGBWithEarlyStoppingRegressor()
            self.elastic_net_model = SGDRegressor(loss='squared_error', penalty='elasticnet')
            self.scoring_func = mean_absolute_error
            self.response_method = 'predict'

    def fit_pipeline(self, holdout_window: int = None):
        """
        Fit the pipeline with cross-validation and grid search.

        Parameters:
        holdout_window (int): The number of observations to use as the holdout window for time series modeling. Required if self.time_series is True.

        Returns:
        None: The method sets the fitted_pipeline attribute with the fitted pipeline.
        """
        # Define the cross-validation object based on whether time series or not
        if self.time_series:
            if holdout_window is None:
                raise ValueError('For time series modeling, the holdout_window parameters must be specified.')
            cv_obj = TimeSeriesSplit(n_splits=5, test_size=holdout_window, gap=0)
            self.holdout_window = holdout_window
        else:
            cv_obj = 5

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
                'model': [self.elastic_net_model],
                'model__l1_ratio': [0, .05, .1, .5, .8, 1],
                'model__alpha': [.1, .01, .005, .001, .00001, 5e-6],
                'model__max_iter': [3000],
            }
            ]

        # Perform grid search with cross-validation
        scoring = make_scorer(self.scoring_func, greater_is_better=False, response_method=self.response_method)
        grid_tmp = GridSearchCV(tmp_pipeline, parameters, cv=cv_obj, n_jobs=-1, verbose=0, scoring=scoring)
        self.fitted_pipeline = grid_tmp.fit(self.X, self.y)

    def get_feature_importance_scores(self, X_pred: pd.DataFrame = None, y_pred: pd.Series = None, type: str = 'shap'):
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

    def plot_feature_importance_scores(self, logo: bool = False, top_k: int = None):
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

    def get_partial_dependence_plots(self, logo: bool = False):
        """
        Generate partial dependence plots for the fitted pipeline.

        Parameters:
        logo (bool): Whether to include a logo in the plots. Default is False.

        Returns:
        None: The method sets the partial_dependence_plots attribute with a dict of the generated plots.
        """
        tmp_plts = PlotTools().get_pdp(self.fitted_pipeline, self.X, logo, self.target)
        self.partial_dependence_plots = tmp_plts

    def get_backtest_plots(self):
        """
        Generate backtest plots for the fitted pipeline.

        Returns:
        None: The method sets the backtest_plots attribute with the generated plots.
        """
        if not self.time_series:
            raise ValueError('Backtest plots are only available for time series data.')
        
        tmp_plts = PlotTools().get_bt_plts(self.fitted_pipeline, self.X, self.y, self.holdout_window)
        self.backtest_plots = tmp_plts

    

