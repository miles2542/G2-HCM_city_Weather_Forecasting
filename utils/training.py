# utils/training.py - APPEND THIS CODE

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm.notebook import tqdm
import joblib
import optuna
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner
import os
from functools import partial
from typing import Dict, List
from rich.console import Console
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV, _CalibratedClassifier
from sklearn.base import MetaEstimatorMixin
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


def _calculate_mape(y_true, y_pred):
    """Helper to calculate MAPE safely."""
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100


def run_bakeoff(
    models_to_run: Dict,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv_splitter,
    horizons: List[int],
) -> Dict:
    """
    Performs a systematic, multi-horizon bake-off for a dictionary of models.

    Captures comprehensive results including scores per fold, predictions,
    and true values for deep diagnostics.

    Args:
        models_to_run (Dict): A dictionary of {model_name: model_object}.
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.DataFrame): The training target set with all horizon columns.
        cv_splitter: An instantiated scikit-learn CV splitter (e.g., TimeSeriesSplit).
        horizons (List[int]): A list of integer horizons to evaluate.

    Returns:
        Dict: A nested dictionary containing all evaluation results.
              Structure: {model: {horizon: {scores: {metric: [scores]}, ...}}}
    """
    bakeoff_results = {}

    # Outer loop with progress bar for models
    for model_name, model_obj in tqdm(models_to_run.items(), desc="Overall Model Bake-Off"):
        model_results = {}

        # Inner loop for horizons
        for h in tqdm(horizons, desc=f"Training {model_name}", leave=False):
            target_col = f"target_temp_t+{h}"
            y_train_horizon = y_train[target_col]

            fold_scores = {"r2": [], "rmse": [], "mae": [], "mape": []}
            fold_predictions = []
            fold_true_values = []

            for train_idx, val_idx in cv_splitter.split(X_train):
                # 1. Split data for the current fold
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train_horizon.iloc[train_idx], y_train_horizon.iloc[val_idx]

                # 2. Fit model
                model_obj.fit(X_train_fold, y_train_fold)

                # 3. Predict and evaluate
                y_pred_fold = model_obj.predict(X_val_fold)

                # 4. Store scores and predictions for deep analysis
                fold_scores["r2"].append(r2_score(y_val_fold, y_pred_fold))
                fold_scores["rmse"].append(np.sqrt(mean_squared_error(y_val_fold, y_pred_fold)))
                fold_scores["mae"].append(mean_absolute_error(y_val_fold, y_pred_fold))
                fold_scores["mape"].append(_calculate_mape(y_val_fold.values, y_pred_fold))

                fold_predictions.append(y_pred_fold)
                fold_true_values.append(y_val_fold.values)

            # Store all results for this horizon
            model_results[f"t+{h}"] = {
                "scores": fold_scores,
                "predictions": fold_predictions,
                "true_values": fold_true_values,
            }

        bakeoff_results[model_name] = model_results

    return bakeoff_results


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    A custom Stacking Ensemble model designed for our two feature sets.
    - Base Models: One for a linear feature set, one for a tree feature set.
    - Meta-Learner: Learns to combine the predictions of the base models.
    """

    def __init__(self, base_model_linear, base_model_tree, meta_learner=None):
        self.base_model_linear = base_model_linear
        self.base_model_tree = base_model_tree
        self.meta_learner = meta_learner if meta_learner is not None else RidgeCV(cv=TimeSeriesSplit(n_splits=3))
        self.trained_models = {}

    def fit(self, X, y):
        """
        Fits the stacking ensemble. Expects X to be a dictionary:
        {'linear': X_linear, 'tree': X_tree}
        """
        X_linear = X["linear"]
        X_tree = X["tree"]

        tscv_base = TimeSeriesSplit(n_splits=5)
        meta_features = np.zeros((len(y), 2))

        # --- DEFINITIVELY CORRECTED LOOP ---
        for train_idx, val_idx in tscv_base.split(X_linear):
            # Fit linear model on its feature set
            lin_model_fold = clone(self.base_model_linear).fit(X_linear.iloc[train_idx], y.iloc[train_idx])
            meta_features[val_idx, 0] = lin_model_fold.predict(X_linear.iloc[val_idx])

            # Fit tree model on its feature set
            tree_model_fold = clone(self.base_model_tree).fit(X_tree.iloc[train_idx], y.iloc[train_idx])
            meta_features[val_idx, 1] = tree_model_fold.predict(X_tree.iloc[val_idx])

        # Train the meta-learner on the generated meta-features
        first_val_idx_start = tscv_base.split(X_linear).__next__()[1][0]
        self.meta_learner.fit(meta_features[first_val_idx_start:], y.iloc[first_val_idx_start:])

        # Retrain base models on the full training data for final prediction
        self.base_model_linear = clone(self.base_model_linear).fit(X_linear, y)
        self.base_model_tree = clone(self.base_model_tree).fit(X_tree, y)

        self.trained_models = {
            "base_linear": self.base_model_linear,
            "base_tree": self.base_model_tree,
            "meta_learner": self.meta_learner,
        }
        return self

    def predict(self, X):
        """
        Generates final predictions. Expects X to be a dictionary:
        {'linear': X_linear, 'tree': X_tree}
        """
        X_linear = X["linear"]
        X_tree = X["tree"]

        pred_linear = self.trained_models["base_linear"].predict(X_linear)
        pred_tree = self.trained_models["base_tree"].predict(X_tree)

        meta_features_pred = np.column_stack([pred_linear, pred_tree])
        return self.trained_models["meta_learner"].predict(meta_features_pred)


def run_ensemble_bakeoff(
    ensembles_to_run: Dict,
    X_train_linear: pd.DataFrame,
    X_train_tree: pd.DataFrame,
    y_train: pd.DataFrame,
    cv_splitter,
    horizons: List[int],
) -> Dict:
    """
    A specialized bake-off runner for ensembles that require multiple feature sets.
    """
    bakeoff_results = {}

    for model_name, model_obj in tqdm(ensembles_to_run.items(), desc="Ensemble Bake-Off"):
        model_results = {}

        for h in tqdm(horizons, desc=f"Training {model_name}", leave=False):
            target_col = f"target_temp_t+{h}"
            y_train_horizon = y_train[target_col]

            fold_scores = {"r2": [], "rmse": [], "mae": [], "mape": []}

            for train_idx, val_idx in cv_splitter.split(X_train_linear):
                # Prepare fold-specific data dictionary
                X_train_fold_dict = {"linear": X_train_linear.iloc[train_idx], "tree": X_train_tree.iloc[train_idx]}
                X_val_fold_dict = {"linear": X_train_linear.iloc[val_idx], "tree": X_train_tree.iloc[val_idx]}
                y_train_fold = y_train_horizon.iloc[train_idx]
                y_val_fold = y_train_horizon.iloc[val_idx]

                # For simple averaging, we need to train base models manually
                if model_name == "SimpleAveragingEnsemble":
                    # Train base models
                    linear_base = clone(model_obj["linear"]).fit(X_train_fold_dict["linear"], y_train_fold)
                    tree_base = clone(model_obj["tree"]).fit(X_train_fold_dict["tree"], y_train_fold)
                    # Predict and average
                    pred_linear = linear_base.predict(X_val_fold_dict["linear"])
                    pred_tree = tree_base.predict(X_val_fold_dict["tree"])
                    y_pred_fold = (pred_linear + pred_tree) / 2.0
                else:  # For StackingEnsemble or other sklearn-compatible ensembles
                    model_obj.fit(X_train_fold_dict, y_train_fold)
                    y_pred_fold = model_obj.predict(X_val_fold_dict)

                # Evaluate and store scores
                fold_scores["r2"].append(r2_score(y_val_fold, y_pred_fold))
                fold_scores["rmse"].append(np.sqrt(mean_squared_error(y_val_fold, y_pred_fold)))
                fold_scores["mae"].append(mean_absolute_error(y_val_fold, y_pred_fold))
                fold_scores["mape"].append(_calculate_mape(y_val_fold.values, y_pred_fold))

            model_results[f"t+{h}"] = {"scores": fold_scores}

        bakeoff_results[model_name] = model_results

    return bakeoff_results


def _define_hpo_search_space(trial: optuna.Trial, model_name: str, custom_space: Dict = None) -> Dict:
    """
    Defines the hyperparameter search space for a given model_name or a custom space.
    This centralized function makes it easy to add new models.
    """

    if custom_space:
        # If a custom space is provided, build it dynamically
        params = {}
        for param, settings in custom_space.items():
            if settings["type"] == "int":
                params[param] = trial.suggest_int(param, settings["low"], settings["high"])
            elif settings["type"] == "float":
                params[param] = trial.suggest_float(
                    param, settings["low"], settings["high"], log=settings.get("log", False)
                )
            elif settings["type"] == "categorical":
                params[param] = trial.suggest_categorical(param, settings["choices"])
        return params

    # --- Pre-defined Search Spaces ---
    if "CatBoost" in model_name:
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
    elif "LGBM" in model_name:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
    else:
        raise ValueError(
            f"No pre-defined search space for model containing '{model_name}'. Please provide a custom_space."
        )


def create_optuna_objective(
    model_name: str,
    X_train_dict: Dict,
    y_train: pd.DataFrame,
    cv_splitter,
    horizon_to_tune: int,
    cv_weights: np.ndarray,
    custom_search_space: Dict = None,
):
    """
    (Version 4.0 - With Sanity Check)
    Includes a "sanity check" for ensembles to ensure they outperform their
    own internal tree-based component.
    """

    def objective(trial: optuna.Trial):
        is_ensemble = "Stacking" in model_name

        # 1. Get model-specific parameters
        if "CatBoost" in model_name:
            base_params = _define_hpo_search_space(trial, "CatBoost", custom_search_space)
            base_params.update({"verbose": 0, "random_state": 105, "thread_count": -1})
            tree_model_class = CatBoostRegressor
        elif "LGBM" in model_name:
            base_params = _define_hpo_search_space(trial, "LGBM", custom_search_space)
            base_params.update({
                # New addition, since tree models can't extrapolate
                "linear_tree": True,
                # General settings for all models
                "verbose": -1,
                "random_state": 105,
                "n_jobs": -1,  # n_jobs is often ignored when device=gpu, but good to have
                # --- GPU ACCELERATION PARAMETERS ---
                # "device": "gpu",
                # "gpu_platform_id": 0,  # Usually 0 for AMD
                # "gpu_device_id": 0,  # Usually 0 for your primary GPU
                # # --- PERFORMANCE TIPS FROM DOCS ---
                # "max_bin": 63,
                # "gpu_use_dp": False,  # Use single precision
            })
            tree_model_class = LGBMRegressor
        else:
            raise ValueError(f"Unsupported model class for objective: {model_name}")

        # 2. Target the correct horizon
        target_col = f"target_temp_t+{horizon_to_tune}"
        y_train_horizon = y_train[target_col]

        # --- Cross-Validation Loop with Sanity Check & Pruning ---
        fold_scores_ensemble = []
        fold_scores_tree_only = []  # For the sanity check

        for fold_num, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_dict["linear"])):
            y_train_fold, y_val_fold = y_train_horizon.iloc[train_idx], y_train_horizon.iloc[val_idx]

            # Instantiate the tuned tree model for this trial
            tuned_tree_model_for_fold = tree_model_class(**base_params)

            if is_ensemble:
                # --- Ensemble Logic ---
                X_train_fold_dict = {
                    "linear": X_train_dict["linear"].iloc[train_idx],
                    "tree": X_train_dict["tree"].iloc[train_idx],
                }
                X_val_fold_dict = {
                    "linear": X_train_dict["linear"].iloc[val_idx],
                    "tree": X_train_dict["tree"].iloc[val_idx],
                }

                ensemble_instance = StackingEnsemble(
                    base_model_linear=ElasticNetCV(cv=TimeSeriesSplit(n_splits=3), n_jobs=-1, random_state=105),
                    base_model_tree=tuned_tree_model_for_fold,
                    meta_learner=RidgeCV(cv=TimeSeriesSplit(n_splits=3)),
                )
                ensemble_instance.fit(X_train_fold_dict, y_train_fold)
                y_pred_ensemble = ensemble_instance.predict(X_val_fold_dict)
                score_ensemble = r2_score(y_val_fold, y_pred_ensemble)
                fold_scores_ensemble.append(score_ensemble)

                # --- SANITY CHECK ---
                # Also evaluate the performance of just the tree model component from the ensemble
                # This uses the model already fitted within the ensemble stack
                y_pred_tree_only = ensemble_instance.trained_models["base_tree"].predict(X_val_fold_dict["tree"])
                score_tree_only = r2_score(y_val_fold, y_pred_tree_only)
                fold_scores_tree_only.append(score_tree_only)

            else:  # Single model logic
                X_train_fold, X_val_fold = X_train_dict["tree"].iloc[train_idx], X_train_dict["tree"].iloc[val_idx]
                tuned_tree_model_for_fold.fit(X_train_fold, y_train_fold)
                y_pred_fold = tuned_tree_model_for_fold.predict(X_val_fold)
                score = r2_score(y_val_fold, y_pred_fold)
                fold_scores_ensemble.append(score)

            # --- Cumulative Weighted Average Reporting ---
            # The value reported to the pruner is ALWAYS the main ensemble score
            current_scores = np.array(fold_scores_ensemble)
            current_weights = cv_weights[: len(fold_scores_ensemble)]
            cumulative_weighted_avg = np.sum(current_scores * current_weights) / np.sum(current_weights)
            trial.report(cumulative_weighted_avg, fold_num)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # --- Final Objective Score Calculation ---
        final_weighted_score_ensemble = np.sum(np.array(fold_scores_ensemble) * cv_weights)

        if is_ensemble:
            final_weighted_score_tree_only = np.sum(np.array(fold_scores_tree_only) * cv_weights)
            # PENALTY: If the ensemble is not better than its tree component, penalize its score.
            # We use a soft penalty: return the average of the two scores. This punishes underperformance
            # without creating a harsh discontinuity that can confuse the optimizer.
            if final_weighted_score_ensemble < final_weighted_score_tree_only:
                return (final_weighted_score_ensemble + final_weighted_score_tree_only) / 2.0

        return final_weighted_score_ensemble

    return objective


# Version without penalizing for ensembles models (those that don't perform better than their base model alone)
# def create_optuna_objective(
#     model_name: str,
#     X_train_dict: Dict,
#     y_train: pd.DataFrame,
#     cv_splitter,
#     horizon_to_tune: int,
#     cv_weights: np.ndarray,  # NEW: Weights for the CV folds
#     custom_search_space: Dict = None,
# ):
#     """
#     (Version 3.0 - Weighted CV)
#     A flexible factory for Optuna objectives, now with weighted CV scoring.
#     """

#     def objective(trial: optuna.Trial):
#         # 1. Get model-specific parameters
#         if "CatBoost" in model_name:
#             base_params = _define_hpo_search_space(trial, "CatBoost", custom_search_space)
#             base_params.update({"verbose": 0, "random_state": 105, "thread_count": -1})
#             model_class = CatBoostRegressor
#         elif "LGBM" in model_name:
#             base_params = _define_hpo_search_space(trial, "LGBM", custom_search_space)
#             base_params.update({"verbosity": -1, "random_state": 105, "n_jobs": -1})
#             model_class = LGBMRegressor
#         else:
#             raise ValueError(f"Unsupported model class for objective: {model_name}")

#         # 2. Instantiate the correct model architecture
#         if "Stacking" in model_name:
#             # Note: The specific base/meta learners are now hardcoded for robustness
#             # This logic assumes the name contains the component info.
#             linear_base_model = ElasticNetCV(cv=TimeSeriesSplit(n_splits=3), n_jobs=-1, random_state=105)
#             meta_learner_model = RidgeCV(cv=TimeSeriesSplit(n_splits=3))

#             model_instance = StackingEnsemble(
#                 base_model_linear=linear_base_model,
#                 base_model_tree=model_class(**base_params),
#                 meta_learner=meta_learner_model,
#             )
#             X_data = X_train_dict
#         else:  # It's a single model
#             model_instance = model_class(**base_params)
#             X_data = X_train_dict["tree"]

#         # 3. Target the correct horizon
#         target_col = f"target_temp_t+{horizon_to_tune}"
#         y_train_horizon = y_train[target_col]

#         # --- Cross-Validation Loop with ENHANCED Pruning ---
#         fold_scores = []
#         for fold_num, (train_idx, val_idx) in enumerate(
#             cv_splitter.split(X_data if isinstance(X_data, pd.DataFrame) else X_data["linear"])
#         ):
#             if isinstance(X_data, dict):
#                 X_train_fold = {"linear": X_data["linear"].iloc[train_idx], "tree": X_data["tree"].iloc[train_idx]}
#                 X_val_fold = {"linear": X_data["linear"].iloc[val_idx], "tree": X_data["tree"].iloc[val_idx]}
#             else:
#                 X_train_fold, X_val_fold = X_data.iloc[train_idx], X_data.iloc[val_idx]

#             y_train_fold, y_val_fold = y_train_horizon.iloc[train_idx], y_train_horizon.iloc[val_idx]

#             model_instance.fit(X_train_fold, y_train_fold)
#             y_pred_fold = model_instance.predict(X_val_fold)
#             score = r2_score(y_val_fold, y_pred_fold)
#             fold_scores.append(score)

#             # --- Cumulative Weighted Average Reporting ---
#             current_scores = np.array(fold_scores)
#             current_weights = cv_weights[: len(fold_scores)]
#             cumulative_weighted_avg = np.sum(current_scores * current_weights) / np.sum(current_weights)

#             trial.report(cumulative_weighted_avg, fold_num)

#             if trial.should_prune():
#                 raise optuna.exceptions.TrialPruned()

#         # The final return value is the weighted average of all 5 folds.
#         return np.sum(np.array(fold_scores) * cv_weights)

#     return objective


def run_optuna_study(
    study_name: str,
    objective_func,
    n_trials: int,
    n_jobs: int = -1,
) -> optuna.Study:
    """
    A wrapper function to create, run, and manage an Optuna study with
    persistent storage, pruning, and multiprocessing.
    """
    storage_path = f"sqlite:///experiments/{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=105),
        pruner=SuccessiveHalvingPruner(min_resource=3),
        # pruner=HyperbandPruner(min_resource=3, max_resource="auto"),
    )

    study.optimize(
        objective_func,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    return study


def save_tuned_params(study: optuna.Study, file_path: str, per_horizon_studies: Dict = None):
    """
    Saves the best parameters from a study or a collection of per-horizon studies.
    """
    console = Console()

    if per_horizon_studies:
        # We have a dictionary of {horizon: study}
        final_params = {}
        console.print(f"[bold]Best Per-Horizon Parameters saved to [cyan]{file_path}[/cyan][/bold]")
        for horizon, horizon_study in per_horizon_studies.items():
            final_params[f"t+{horizon}"] = horizon_study.best_params
            console.print(f"  - [bold]Horizon t+{horizon}:[/bold] R² = {horizon_study.best_value:.4f}")
            console.print(f"    {horizon_study.best_params}")
    else:
        # A single study was run for the average case
        best_params = study.best_params
        horizons = [1, 2, 3, 4, 5]  # Assuming default horizons
        final_params = {f"t+{h}": best_params for h in horizons}
        console.print(f"Best parameters saved to [cyan]{file_path}[/cyan]")
        console.print("[bold]Best Trial's Parameters (for average performance):[/bold]")
        console.print(study.best_params)
        console.print(f"[bold]Best Trial's Average R² Score:[/bold] [green]{study.best_value:.4f}[/green]")

    joblib.dump(final_params, file_path)


def consolidate_best_params(base_study_name: str, horizons: List[int], num_runs: int, output_file_path: str):
    """
    Iterates through all study database files for a given model, finds the best
    trial for each horizon across all runs, and saves the consolidated parameters.
    """
    console = Console()
    final_best_params = {}
    console.print(f"[bold]Consolidating Best Parameters for [cyan]{base_study_name}[/cyan]...[/bold]")

    for h in horizons:
        best_value_for_horizon = -np.inf
        best_params_for_horizon = None
        best_study_name = None

        for i in range(num_runs):
            study_name = f"{base_study_name}_h{h}_run{i + 1}"
            storage_path = f"sqlite:///experiments/{study_name}.db"
            try:
                loaded_study = optuna.load_study(study_name=study_name, storage=storage_path)
                if loaded_study.best_value > best_value_for_horizon:
                    best_value_for_horizon = loaded_study.best_value
                    best_params_for_horizon = loaded_study.best_params
                    best_study_name = study_name
            except KeyError:
                console.print(
                    f"  - [yellow]Warning:[/yellow] Study '{study_name}' not found or has no trials. Skipping."
                )
                continue

        if best_params_for_horizon:
            final_best_params[f"t+{h}"] = best_params_for_horizon
            console.print(
                f"  - [bold]Horizon t+{h}:[/bold] Best R² = {best_value_for_horizon:.4f} (from [italic]{best_study_name}[/italic])"
            )

    joblib.dump(final_best_params, output_file_path)
    console.print(f"\n[bold green]Successfully saved consolidated parameters to '{output_file_path}'[/bold green]")


class WeightedAverageEnsemble(BaseEstimator, RegressorMixin):
    """(Version 3.0 - Robust Input Handling)"""

    def __init__(self, base_model_linear, base_model_tree):
        self.base_model_linear = base_model_linear
        self.base_model_tree = base_model_tree
        self.optimal_weight_ = 0.5

    def fit(self, X, y):
        # Input can be a dict or a single DataFrame. If it's a dict, use it.
        # If not, this class is being used as a component and cannot fit itself
        # with a single DataFrame. This is handled by the parent estimator.
        if not isinstance(X, dict):
            raise ValueError(
                "WeightedAverageEnsemble expects a dictionary {'linear': X_lin, 'tree': X_tree} as input for .fit()"
            )

        X_linear, X_tree = X["linear"], X["tree"]
        self.base_model_linear_ = clone(self.base_model_linear).fit(X_linear, y)
        self.base_model_tree_ = clone(self.base_model_tree).fit(X_tree, y)
        return self

    def predict(self, X):
        if not isinstance(X, dict):
            raise ValueError(
                "WeightedAverageEnsemble expects a dictionary {'linear': X_lin, 'tree': X_tree} as input for .predict()"
            )

        X_linear, X_tree = X["linear"], X["tree"]
        pred_linear = self.base_model_linear_.predict(X_linear)
        pred_tree = self.base_model_tree_.predict(X_tree)
        return (self.optimal_weight_ * pred_linear) + ((1 - self.optimal_weight_) * pred_tree)


def _get_model_instance(model_class, params=None):
    """
    A factory function to instantiate models with sensible, centralized defaults.
    """
    if params is None:
        params = {}

    DEFAULT_KWARGS = {
        # --- Tree Models (no scaling needed) ---
        CatBoostRegressor: {
            "verbose": 0,
            "random_state": 105,
            "thread_count": -1,
        },
        LGBMRegressor: {
            "verbose": -1,
            "random_state": 105,
            "n_jobs": -1,  # Ignored if device is set to GPU
            # # --- GPU ACCELERATION PARAMETERS ---
            # "device": "gpu",
            # "gpu_platform_id": 0, # AMD
            # "gpu_device_id": 0,
            # # --- PERFORMANCE TIPS FROM DOCS ---
            # "max_bin": 63,
            # "gpu_use_dp": False,  # Use single precision
        },
        XGBRegressor: {
            "random_state": 105,
            "n_jobs": -1,
        },
        LinearRegression: {"n_jobs": -1},  # Also typically used in a pipeline, but can be standalone
    }

    # The provided params (e.g., from Optuna) override the defaults
    kwargs = DEFAULT_KWARGS.get(model_class, {})
    kwargs.update(params)

    # --- Linear Models (wrap in a Pipeline with a scaler) ---
    if model_class in [RidgeCV, ElasticNetCV]:
        return Pipeline([
            ("scaler", RobustScaler()),
            (
                "model",
                model_class(
                    cv=TimeSeriesSplit(n_splits=5),
                    **params,  # Pass any specific linear model params here
                ),
            ),
        ])

    return model_class(**kwargs)


def train_and_save_final_models(
    models_to_train: Dict,
    X_train_linear: pd.DataFrame,
    X_train_tree: pd.DataFrame,
    y_train: pd.DataFrame,
    horizons: List[int],
    save_dir: str = "assets/models",
):
    """
    (Version 2.0 - Corrected Instantiation)
    Trains final models on 100% of the training data and saves them to disk.
    """
    console = Console()
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"[bold]Training final models on 100% of training data...[/bold]")

    for model_name, model_config in tqdm(models_to_train.items(), desc="Training Final Models"):
        if "tuned" in model_config.get("type", ""):
            tuned_params_all_horizons = joblib.load(model_config["params_path"])

        for h in horizons:
            horizon_name = f"t+{h}"
            y_train_horizon = y_train[f"target_temp_{horizon_name}"]

            model_type = model_config.get("type")
            if model_type == "weighted_averaging_tuned":
                # Instantiate the unfitted base models
                linear_base_model = _get_model_instance(model_config["linear_model_class"])
                tree_params = tuned_params_all_horizons[horizon_name]
                tree_base_model = _get_model_instance(model_config["tree_model_class"], params=tree_params)

                # Create the ensemble instance with the UNFITTED base models
                final_model_instance = WeightedAverageEnsemble(
                    base_model_linear=linear_base_model, base_model_tree=tree_base_model
                )
                final_model_instance.optimal_weight_ = model_config["horizon_weights"][h]

                # The .fit() method of the ensemble will handle fitting its components
                X_train_dict = {"linear": X_train_linear, "tree": X_train_tree}
                final_model_instance.fit(X_train_dict, y_train_horizon)

            elif model_type == "oob" and model_config.get("feature_set") == "linear":
                final_model_instance = _get_model_instance(model_config["model_class"])
                final_model_instance.fit(X_train_linear, y_train_horizon)
            else:
                raise NotImplementedError(f"Final training not implemented for model type: {model_type}")

            file_path = os.path.join(save_dir, f"{model_name}_h{h}.pkl")
            joblib.dump(final_model_instance, file_path)

    console.print(f"\n[bold green]All final models trained and saved to '{save_dir}'.[/bold green]")


def load_and_predict_on_test(
    models_to_predict: Dict,
    X_test_linear: pd.DataFrame,
    X_test_tree: pd.DataFrame,
    horizons: List[int],
    load_dir: str = "assets/models",
) -> Dict:
    """
    Loads final, pre-trained models and generates predictions on the test set,
    now correctly using the model_config dictionary.
    """
    all_predictions = {}
    for model_name, model_config in models_to_predict.items():
        model_preds_all_horizons = {}
        for h in horizons:
            horizon_name = f"t+{h}"
            file_path = os.path.join(load_dir, f"{model_name}_h{h}.pkl")

            try:
                loaded_model = joblib.load(file_path)

                # Use the model's type to determine the correct input format
                model_type = model_config.get("type", "oob")

                if "averaging" in model_type:
                    X_test_input = {"linear": X_test_linear, "tree": X_test_tree}
                elif model_config.get("feature_set") == "linear":
                    X_test_input = X_test_linear
                else:  # Default to tree feature set
                    X_test_input = X_test_tree

                preds = loaded_model.predict(X_test_input)
                model_preds_all_horizons[horizon_name] = pd.Series(preds, index=X_test_linear.index)

            except FileNotFoundError:
                print(f"Warning: Model file not found at '{file_path}'. Skipping prediction.")
                continue

        all_predictions[model_name] = pd.DataFrame(model_preds_all_horizons)

    return all_predictions
