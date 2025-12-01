
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
from .preprocessing import Prepreprocessor
import shap
import numpy as np
from sklearn.inspection import permutation_importance, partial_dependence
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from .estimation import XGBWithEarlyStoppingClassifier, XGBWithEarlyStoppingRegressor
from .automl import SimpleESRegressor, AutoARIMARegressor   
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.utils import Bunch
import os 
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Optional, Union, Any, Type, Tuple, List
import matplotlib.figure

# Cache frequently used paths at module level for efficiency
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_STYLE_PATH = os.path.join(_CURRENT_DIR, 'styles', 'opinionated_rc.mplstyle')
_LOGO_PATH = os.path.join(_CURRENT_DIR, 'assets', 'logo_placeholder.png')

class PlotTools:
    def __init__(self) -> None:
        pass

    def get_shap_values(self, fitted_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate SHAP values for the fitted pipeline and return a DataFrame of feature importance scores.

        Parameters:
        fitted_pipeline (Pipeline): The fitted pipeline containing the preprocessor and model.
        X (pd.DataFrame): The input feature matrix.
        y (pd.Series): The target variable.

        Returns:
        pd.DataFrame: A DataFrame containing the feature importance scores based on SHAP values.
        """
        # Transform X once
        preprocessor = fitted_pipeline.best_estimator_['preprocessor']
        X_transformed = preprocessor.transform(X)
        model = fitted_pipeline.best_estimator_['model']

        # Guard: univariate models (ignore engineered predictors)
        if isinstance(model, (SimpleESRegressor, AutoARIMARegressor)):
            raise ValueError(
                "Model uses only the dependent variable for prediction and cannot compute feature importance (univariate smoothing/ARIMA)."
            )

        # Get feature names once (avoid multiple calls)
        feature_names = preprocessor.get_feature_names_out()

        # Choose masker logic for supported models
        if isinstance(model, (XGBWithEarlyStoppingClassifier, XGBWithEarlyStoppingRegressor)):
            inp_masker = None  # let tree explainer infer background
        elif isinstance(model, (SGDClassifier, SGDRegressor)):
            inp_masker = X_transformed  # linear models: pass matrix directly
        else:
            # Fallback: attempt generic masking; may not work for unsupported custom models
            inp_masker = X_transformed

        # Build SHAP explainer (will dispatch to specific explainer if recognized)
        explainer = shap.Explainer(
            model=model,
            masker=inp_masker,
            feature_names=feature_names
        )
        shap_values = explainer(X_transformed)

        feature_imp_tbl = (pd.DataFrame({'transformed_feature': feature_names,
                                         'sv_contrib': np.abs(shap_values.values).mean(0)})
                                         .merge(self._build_feature_mapping(preprocessor), on='transformed_feature', how='left')
                                         .groupby('original_column')
                                         .agg({'sv_contrib': 'sum'})
                                         .sort_values('sv_contrib', ascending = False)
                                         .reset_index()
                                         .rename(columns={'original_column': 'feature'})
                                         .assign(importance_norm = lambda df_: np.round((100 * df_['sv_contrib']/df_['sv_contrib'].iloc[0]).astype(np.float64), 1))
                                         [['feature', 'importance_norm', 'sv_contrib']]
              )
        
        return feature_imp_tbl        

    def _build_feature_mapping(self, fitted_preprocessor: ColumnTransformer) -> pd.DataFrame:
        """Build mapping between transformed feature names and original columns using fitted ColumnTransformer.
        
        Avoids refitting per column (performance) and stays consistent with the actually fitted transformer.
        
        Parameters:
        fitted_preprocessor (ColumnTransformer): The fitted preprocessor from the pipeline.
        
        Returns:
        pd.DataFrame: Mapping table with 'transformed_feature' and 'original_column' columns.
        """
        if not hasattr(fitted_preprocessor, 'transformers_'):
            raise ValueError("Preprocessor must be fitted before building feature mapping.")
        
        transformed_names = fitted_preprocessor.get_feature_names_out()
        # Use a set for O(1) membership checks instead of O(n) list lookups
        transformed_names_set = set(transformed_names)
        original_map: Dict[str, str] = {}
        
        for name, transformer, cols in fitted_preprocessor.transformers_:
            if transformer == 'drop' or transformer is None:
                continue
                
            # Handle pipeline objects
            if hasattr(transformer, 'named_steps'):
                final_step = list(transformer.named_steps.values())[-1]
            else:
                final_step = transformer
            
            # OneHotEncoder expands columns
            if isinstance(final_step, OneHotEncoder):
                ohe_cols = cols if isinstance(cols, list) else list(cols)
                # Skip if no columns to encode
                if len(ohe_cols) == 0:
                    continue
                # Use get_feature_names_out to get the expanded feature names
                expanded_features = final_step.get_feature_names_out(ohe_cols)
                for exp_feature in expanded_features:
                    prefixed_feature = f"{name}__{exp_feature}"
                    if prefixed_feature in transformed_names_set:
                        # Extract original column name from expanded feature
                        # Format is typically "original_col_category"
                        for orig_col in ohe_cols:
                            if exp_feature.startswith(f"{orig_col}_"):
                                original_map[prefixed_feature] = orig_col
                                break
            
            # CountVectorizer for text columns  
            elif isinstance(final_step, CountVectorizer):
                vocab_features = final_step.get_feature_names_out()
                base_col = cols if isinstance(cols, str) else cols[0]
                for vf in vocab_features:
                    prefixed_feature = f"{name}__{vf}"
                    if prefixed_feature in transformed_names_set:
                        original_map[prefixed_feature] = base_col
            
            # Numeric features (scaler/imputer) - one-to-one mapping
            else:
                base_cols = cols if isinstance(cols, list) else list(cols)
                for c in base_cols:
                    prefixed_feature = f"{name}__{c}"
                    if prefixed_feature in transformed_names_set:
                        original_map[prefixed_feature] = c
        
        feature_mapping_tbl = (pd.DataFrame({
            'transformed_feature': list(original_map.keys()),
            'original_column': list(original_map.values())
        })
        .sort_values('transformed_feature')
        .reset_index(drop=True))
        
        return feature_mapping_tbl    
    
    def get_permutation_importance(self, fitted_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        model = fitted_pipeline.best_estimator_['model']

        # Guard: univariate models do not use engineered features
        if isinstance(model, (SimpleESRegressor, AutoARIMARegressor)):
            raise ValueError(
                "Model uses only the dependent variable for prediction and cannot compute permutation feature importance (univariate smoothing/ARIMA)."
            )

        if isinstance(model, (XGBWithEarlyStoppingClassifier, SGDClassifier)):
            perm_scorer = 'neg_log_loss'
        elif isinstance(model, (XGBWithEarlyStoppingRegressor, SGDRegressor)):
            perm_scorer = 'neg_mean_squared_error'
        else:
            # Fallback for other regressors/classifiers: default to neg MSE for regressors, log_loss for classifiers
            if hasattr(model, 'predict_proba'):
                perm_scorer = 'neg_log_loss'
            else:
                perm_scorer = 'neg_mean_squared_error'

        # Compute permutation importance scores (takes a couple minutes or more depending on size of prediction dataset and n_repeats)
        result = permutation_importance(fitted_pipeline, X, y, n_repeats=5, random_state=42,
                                         n_jobs=-1, scoring=perm_scorer)
        
        # Create pandas df of importance scores 
        feature_imp_tbl = (pd.DataFrame({"feature": X.columns,
                                         "importance_mean": result.importances_mean,
                                         "importance_std": result.importances_std
                                         })
                                         .sort_values(by="importance_mean", ascending=False)
                                         .assign(importance_norm = lambda df_: np.round(100 * df_['importance_mean']/df_['importance_mean'].iloc[0], 1))
                                         [['feature', 'importance_norm', 'importance_mean', 'importance_std']]
                                         .reset_index(drop = True)
        )

        return feature_imp_tbl
    
    def plot_feature_importance(
        self, 
        feature_importance_scores: pd.DataFrame, 
        logo: bool = False, 
        top_k: Optional[int] = None
    ) -> matplotlib.figure.Figure:

        # Filter to top k features if top_k specified
        if top_k:
            feature_importance_scores = feature_importance_scores.iloc[:top_k]

        # Plot feature importance scores

        # Use cached style path
        plt.style.use(_STYLE_PATH)

        # Set the rcParams
        plt.rcParams.update({
            'grid.linestyle': '-',
        })

        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(8.3, 5))

        sns.barplot(x='importance_norm', y='feature', data=feature_importance_scores, orient='h', color='#038747')

        # Calculate deciles
        deciles = np.arange(0, 110, 10)

        # Set x-axis labels at every decile with '%' sign
        plt.xticks(ticks=deciles, labels=[f'{decile}%' for decile in deciles])

        # Customize labels and title
        plt.xlabel('', fontsize=9, color='#3b5f60')
        plt.ylabel('', fontsize=9, color='#3b5f60')
        plt.title('Feature Impact', fontsize=17, pad=14, loc='center')

        # Customize tick parameters
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10) # , font='Roboto Condensed')

        # Remove axis spines
        sns.despine(left=False, bottom=False)

        if logo:
            # Use cached logo path
            logo_img = mpimg.imread(_LOGO_PATH)
            # Create a new set of axes for the logo
            logo_ax = fig.add_axes([.8, .02, 0.1, 0.1], anchor='SE', zorder=10)
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')  # Hide the axes

        # Close the figure to prevent it from displaying
        plt.close(fig)

        return fig
    
    def get_pdp(
        self, 
        fitted_pipeline: Pipeline, 
        X: pd.DataFrame, 
        logo: bool = False, 
        target: Optional[str] = None
    ) -> Dict[str, matplotlib.figure.Figure]:
        """
        Generate Partial Dependence Plots (PDPs) for each feature in `X` using the fitted model pipeline and
        return a dictionary of matplotlib figures for each feature.

        Parameters
        ----------
        fitted_pipeline : Pipeline
            A fitted pipeline from AutoML.

        X : pd.DataFrame
            The input data on which partial dependence is computed. Should match the training data format.

        logo : bool, optional
            If True, appends a logo image to the plot (default is False).

        target : str, optional
            The name of the target variable, used in axis labels (default is None).

        Returns
        -------
        pdp_plots : dict
            A dictionary where keys are feature names and values are matplotlib figures.

        Notes
        -----
        - For categorical variables, only the top N categories (default 14) are used for plotting.
        """

        # Guard: PDP not meaningful for univariate models that ignore engineered predictors
        model = fitted_pipeline.best_estimator_['model']
        if isinstance(model, (SimpleESRegressor, AutoARIMARegressor)):
            raise ValueError(
                "Model uses only the dependent variable for prediction and cannot compute partial dependence plots (univariate smoothing/ARIMA)."
            )

        # Transform many-category variable
        def _transform_cat_var(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
            # Get the most common values
            top_n = df[column_name].value_counts().nlargest(14).index
            # Replace values not in the top nlargest with 'Other' using vectorized operation
            top_n_set = set(top_n)
            df[column_name] = df[column_name].where(df[column_name].isin(top_n_set), 'Other')
            # Filter out rows where the value is 'Other'
            ## TODO -- add parm to control whether to include the Other row
            df = df[df[column_name] != 'Other']
    
            return df
        
        # Shorten string for x axis tick labels
        def _shorten_string(s: str) -> str:
            if len(s) > 17:
                return f"{s[:8]}...{s[-8:]}"
            return s
        
        # Get plot parms for pdp
        def _get_pdp_vals(pdp_object: Bunch) -> Tuple[Union[List[str], np.ndarray], float, int, str]:
            # Shorten string and put ellipsis in middle for x axis tick labels, scale plot len based on number of ticks
            if pdp_object['grid_values'][0].dtype == 'O':
                grid_vals = [_shorten_string(i) for i in pdp_object['grid_values'][0]]
                plt_len = min(4.5 + .5*len(grid_vals), 13)
            else:
                grid_vals = pdp_object['grid_values'][0]
                plt_len = 8
            # Rotate x axis labels and right position if there are more than 7 ticks
            if len(pdp_object['grid_values'][0]) > 7:
                rotation_val = 30
                xtick_label_loc = 'right'
            else:
                rotation_val = 0
                xtick_label_loc = 'center'

            return grid_vals, plt_len, rotation_val, xtick_label_loc

        # Plot pdp  
        def _plot_pdp(
            pdp_output: Bunch, 
            grid_vals: Union[list, np.ndarray], 
            plt_len: float, 
            rotation_val: int, 
            xtick_label_loc: str, 
            logo: bool
        ) -> matplotlib.figure.Figure:
                
            fig, ax = plt.subplots(figsize=(plt_len, 4.5))

            # Use cached style path
            plt.style.use(_STYLE_PATH)

            plt.rcParams.update({'grid.linestyle': '-',})
            plt.xlabel(f"Feature ({varname})", fontsize=11, loc='center')
            plt.ylabel(f"Target ({target})", fontsize=11, loc='center')        
            plt.xticks(fontsize=11, rotation=rotation_val, ha=xtick_label_loc)
            plt.yticks(fontsize=11)
            plt.title(f"Partial Dependence of {varname}", fontsize=15, loc='center')
            if pdp_output['grid_values'][0].dtype in [int, float]:
                plt.plot(grid_vals, pdp_output['average'][0], color='#038747')
            plt.scatter(grid_vals, pdp_output['average'][0], color='#038747')

            if logo:
                # Use cached logo path
                logo_img = mpimg.imread(_LOGO_PATH)
                # Create a new set of axes for the logo
                logo_ax = fig.add_axes([0.82, -0.03, 0.1, 0.1], anchor='SE', zorder=10)
                logo_ax.imshow(logo_img)
                logo_ax.axis('off')  # Hide the axes

            plt.close()

            return fig

        pdp_plots = {}
        # Create StringDtype instance once outside loop for efficient comparison
        string_dtype = pd.StringDtype()
        for varname in X.columns:

            if X[varname].dtype == object:
                X = X.assign(**{varname: lambda df_: df_[varname].astype(str)}).pipe(_transform_cat_var, varname)

            if X[varname].dtype != string_dtype:

                pdp_output = partial_dependence(fitted_pipeline, X, features = [varname])

                grid_vals, plt_len, rotation_val, xtick_label_loc = _get_pdp_vals(pdp_output)

                pdp_plots[varname] = _plot_pdp(pdp_output, grid_vals, plt_len, rotation_val, xtick_label_loc, logo)

        return pdp_plots
    
    def get_bt_plts(
        self, 
        fitted_pipeline: Pipeline, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        inp_holdout_window: int
    ) -> Dict[str, matplotlib.figure.Figure]:
        """
        Generate backtest plots for time series model.
        """    
        ts_splt = TimeSeriesSplit(n_splits=5, test_size=inp_holdout_window, gap=0)
        outcome = y_train.name.replace('_', ' ').title()

        backtest_plts = {}
        for idx, idx_arr in enumerate(ts_splt.split(X_train)):

            fig, ax = plt.subplots(figsize=(20, 5))
            plt.style.use("opinionated_rc")

            X_splt = X_train.iloc[idx_arr[1]]
            splt1_preds = fitted_pipeline.best_estimator_.predict(X_splt)

            # Extract the actual values for the last 18 entries
            y_splt = y_train.iloc[idx_arr[1]]
            actual_values = y_splt.to_numpy()

            # Plot the actual values
            ax.plot(X_splt.index, actual_values, label='Actual', color='black')

            # Plot the predicted values
            ax.plot(X_splt.index, splt1_preds, label='Predicted', color='#038747', linestyle='dashed')

            # Add labels and title
            ax.set_xlabel('')
            ax.set_ylabel(f'{outcome}', size = 11, loc = 'center')
            ax.set_title(f'{outcome} Backtest {idx+1}', size=18)

            # Add a legend and nudge it down to the lower right
            ax.legend(fontsize=11, loc='lower right', bbox_to_anchor=(1.04, .2))

            ax.grid(True, which='both', linestyle='-', linewidth=0.8)
            
            # Add a caption
            fig.text(.7, -.03, "Note: Predictions are based on a forecast window of 1. Each prediction is made from a forecast point 1 period prior to the prediction date.", wrap=True, horizontalalignment='center', fontsize=10)

            plt.close(fig)
            backtest_plts[f"bt{idx+1}"] = fig

        return backtest_plts

 






