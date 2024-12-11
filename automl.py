
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from automl_tool.preprocessing import Prepreprocessor
from automl_tool.estimation import XGBWithEarlyStoppingClassifier, XGBWithEarlyStoppingRegressor
from automl_tool.plotting import PlotTools
from sklearn.metrics import make_scorer, log_loss, mean_absolute_error
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.model_selection import TimeSeriesSplit
import warnings
# Suppress the specific warning about a worker stopping
warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")


    
# AutoML class to automate the machine learning pipeline
class AutoML:
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

    def build_pipeline(self):
    
        preprocessor = Prepreprocessor().build_preprocessor(self.X)
        
        # Create the pipeline
        tmp_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.boosting_model)
        ])

        self.pipeline = tmp_pipeline

    def fit_pipeline(self, ts_outcome=None, fdw=None, holdout_window=None):

        # Define the cross-validation object based on whether time series or not
        if self.time_series:
            ts_outcome = self.target
            cv_obj = TimeSeriesSplit(n_splits=5, test_size=holdout_window, gap=0)
            if fdw is None or holdout_window is None or ts_outcome is None:
                raise ValueError('For time series modeling, the fdw (feature derivation window) and holdout_window parameters must be specified.')
        else:
            cv_obj = 5

        # Build the preprocessor
        preprocessor = Prepreprocessor().build_preprocessor_experimental(self.X, ts_outcome, fdw)
        
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
            }
            ]

        # Perform grid search with cross-validation
        scoring = make_scorer(self.scoring_func, greater_is_better=False, response_method=self.response_method)
        grid_tmp = GridSearchCV(tmp_pipeline, parameters, cv=cv_obj, n_jobs=-1, verbose=0, scoring=scoring)
        self.fitted_pipeline = grid_tmp.fit(self.X, self.y)

    def get_feature_importance_scores(self, X_pred: pd.DataFrame = None, y_pred: pd.Series = None, type: str = 'shap'):

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

    def plot_feature_importance_scores(self, logo = False, top_k = None):
        
        tmp_plt = PlotTools().plot_feature_importance(self.feature_importance_scores, logo, top_k)
        self.feature_importance_plot = tmp_plt

    def get_partial_dependence_plots(self, logo = False):
        
        tmp_plts = PlotTools().get_pdp(self.fitted_pipeline, self.X, logo, self.target)
        self.partial_dependence_plots = tmp_plts
    
    
    ##########




    def fit_pipeline2(self, holdout_window=None):

        # Define the cross-validation object based on whether time series or not
        if self.time_series:
            cv_obj = TimeSeriesSplit(n_splits=5, test_size=holdout_window, gap=0)
            if holdout_window is None:
                raise ValueError('For time series modeling, the holdout_window parameters must be specified.')
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
            }
            ]

        # Perform grid search with cross-validation
        scoring = make_scorer(self.scoring_func, greater_is_better=False, response_method=self.response_method)
        grid_tmp = GridSearchCV(tmp_pipeline, parameters, cv=cv_obj, n_jobs=-1, verbose=0, scoring=scoring)
        self.fitted_pipeline = grid_tmp.fit(self.X, self.y)
