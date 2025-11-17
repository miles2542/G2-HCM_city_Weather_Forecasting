# utils/evaluation.py

import os
from typing import Dict, List
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone, MetaEstimatorMixin
from sklearn.calibration import CalibratedClassifierCV, _CalibratedClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import joblib
import optuna
from optuna.importance import FanovaImportanceEvaluator

from rich.console import Console
from rich.table import Table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from tqdm.notebook import tqdm

from utils.training import StackingEnsemble, WeightedAverageEnsemble


def _calculate_mape(y_true, y_pred):
    """Helper to calculate MAPE safely."""
    # Ensure y_true is a numpy array for consistent behavior
    y_true = np.asarray(y_true)
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def evaluate_persistence_model(
    df_model_ready: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train_all_horizons: pd.DataFrame,
    horizons: List[int],
    cv_splitter,
) -> Dict:
    """
    (Version 3.1 - Corrected Structure)
    Evaluates the persistence model and returns results in the standard,
    bake-off-compatible dictionary format.
    """
    # The dictionary for this specific model's results
    model_horizon_results = {}

    original_temp_series = df_model_ready.loc[X_train.index, "temp"]

    for h in horizons:
        horizon_name = f"t+{h}"
        y_train_horizon = y_train_all_horizons[f"target_temp_{horizon_name}"]

        fold_scores = {"r2": [], "rmse": [], "mae": [], "mape": []}

        for train_idx, val_idx in cv_splitter.split(X_train):
            y_pred_fold = original_temp_series.iloc[val_idx]
            y_true_fold = y_train_horizon.iloc[val_idx]

            fold_scores["r2"].append(r2_score(y_true_fold, y_pred_fold))
            fold_scores["rmse"].append(np.sqrt(mean_squared_error(y_true_fold, y_pred_fold)))
            fold_scores["mae"].append(mean_absolute_error(y_true_fold, y_pred_fold))
            fold_scores["mape"].append(_calculate_mape(y_true_fold.values, y_pred_fold.values))

        model_horizon_results[horizon_name] = {"scores": fold_scores}

    # Return the results in a dictionary with the model name as the single top-level key.
    return {"Baseline (Persistence)": model_horizon_results}


def display_baseline_results(baseline_cv_scores: Dict[str, np.ndarray]):
    """
    Takes the output of evaluate_persistence_model and displays it in a rich table.
    """
    console = Console()
    baseline_table = Table(
        title="[bold]Persistence (Naive) Baseline Performance[/bold]\n(Cross-Validated on Training Set)",
        show_header=True,
        header_style="bold #063163",
    )
    baseline_table.add_column("Horizon", justify="center", style="cyan")
    baseline_table.add_column("Mean CV R²", justify="center", style="#0771A4")
    baseline_table.add_column("Std Dev of CV Scores", justify="center", style="#690120")

    all_mean_scores = []
    all_std_scores = []
    for horizon, scores in baseline_cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        all_mean_scores.append(mean_score)
        all_std_scores.append(std_score)
        baseline_table.add_row(horizon, f"{mean_score:.4f}", f"{std_score:.4f}")

    # Add an average row
    baseline_table.add_row(
        "[bold]Average[/bold]",
        f"[bold dark_orange]{np.mean(all_mean_scores):.4f}[/bold dark_orange]",
        f"[bold dark_orange]{np.mean(all_std_scores):.4f}[/bold dark_orange]",
    )

    console.print(baseline_table)


def _get_model_family(model_name: str) -> str:
    """A helper function to dynamically determine the model family from its name."""
    name = model_name.lower()

    # --- Tuned Models ---
    if "tuned" in name:
        if "stack" in name:
            return "Tuned stacking"
        if "avg" in name or "simple" in name:
            return "Tuned simple avg"
        return "Tuned tree"
    if "weighted_avg" in name:
        return "Tuned weighted avg"

    # --- OOB Ensembles ---
    if ("ensemble" in name) or ("stack" in name) or ("avg" in name) or ("with" in name):
        return "Ensemble (OOB)"

    # --- OOB Single Models & Baselines ---
    if "ridge" in name:
        return "Linear"
    if "elasticnet" in name:
        return "Linear"
    if "linearregression" in name:
        return "Linear"
    if "lgbm" in name:
        return "Tree-based"
    if "catboost" in name:
        return "Tree-based"
    if "xgb" in name:
        return "Tree-based"
    if "prophet" in name:
        return "Advanced time series"
    if "sarimax" in name:
        return "Advanced time series"
    if "lstm" in name:
        return "Deep learning"
    if "baseline" in name:
        return "Baseline"

    return "Unknown"


def generate_leaderboard_data(results_files: List[str], baseline_scores: Dict = None) -> pd.DataFrame:
    """
    (Version 4.1 - Corrected & Simplified)
    Loads all bake-off artifacts and baseline results to create a comprehensive leaderboard DataFrame.
    """
    all_results = {}
    if baseline_scores:
        all_results.update(baseline_scores)

    for file_path in results_files:
        try:
            results = joblib.load(file_path)
            all_results.update(results)
        except FileNotFoundError:
            print(f"Warning: Results file not found at '{file_path}'. Skipping.")
            continue

    leaderboard_data = []
    metrics_to_process = ["r2", "rmse", "mae", "mape"]

    for model_name, model_results in all_results.items():
        horizon_summary = {metric: [] for metric in metrics_to_process}

        # This single, clean loop now works for ALL model result types.
        # It correctly processes the dictionary for each model.
        for horizon, horizon_data in model_results.items():
            if "scores" in horizon_data:  # Defensive check
                for metric in metrics_to_process:
                    fold_scores = horizon_data["scores"].get(metric, [])
                    if fold_scores:
                        horizon_summary[metric].append(np.mean(fold_scores))
                    else:
                        horizon_summary[metric].append(np.nan)

        # Use np.nanmean to safely calculate averages, ignoring NaNs if a metric was missing.
        leaderboard_data.append({
            "Model": model_name,
            "Avg CV R²": np.nanmean(horizon_summary["r2"]),
            "Avg CV RMSE": np.nanmean(horizon_summary["rmse"]),
            "Avg CV MAE": np.nanmean(horizon_summary["mae"]),
            "Avg CV MAPE (%)": np.nanmean(horizon_summary["mape"]),
            "Model Family": _get_model_family(model_name),
        })

    df = pd.DataFrame(leaderboard_data)
    df.fillna(value=np.nan, inplace=True)  # Final cleanup for display
    df = df.sort_values(by="Avg CV R²", ascending=False).reset_index(drop=True)
    return df


def display_leaderboard(
    df_leaderboard: pd.DataFrame,
    title: str,
    highlight_models: List[str] = None,
    exclude_models: List[str] = None,
):
    """
    Displays a flexible leaderboard with optional highlighting for specific models. Shows all metrics.
    """
    console = Console()
    table = Table(
        title=f"[bold]{title}[/bold]\n(Metrics averaged across all 5 horizons & 5 CV folds)",
        show_header=True,
        header_style="bold #063163",
        show_lines=True,
    )
    table.add_column("Rank", justify="center", style="grey70")
    table.add_column("Model", style="cyan", width=23, overflow="fold")
    table.add_column("Model Family", justify="center")
    table.add_column("Avg CV R² ↑", justify="center")
    table.add_column("Avg CV RMSE ↓", justify="center")
    table.add_column("Avg CV MAE ↓", justify="center")
    table.add_column("Avg CV MAPE (%) ↓", justify="center")

    family_colors = {
        "Baseline": "grey50",
        "Linear": "#24C0D2",
        "Tree-based": "#AAB94B",
        "Advanced time series": "#963D4D",
        "Deep learning": "#AA8B96",
        "Ensemble (OOB)": "#369D8E",
        "Tuned tree": "#E7B142",
        "Tuned simple avg": "#dc6f6f",
        "Tuned Weighted Avg": "bold #92617f",
        "Tuned stacking": "#0771A4",
    }

    for i, row in df_leaderboard.iterrows():
        if exclude_models and row["Model"] in exclude_models:
            continue
        is_highlighted = highlight_models and row["Model"] in highlight_models
        row_style = "bold on #fffbe8" if is_highlighted else ""
        if (i == 0) and is_highlighted:  # Top model and in highlight models
            row_style = "bold dark_orange on #fffbe8"
        elif (i == 0) and not is_highlighted:  # Top model and not in highlight models
            row_style = "bold dark_orange"

        # Format MAPE specifically
        mape_str = f"{row['Avg CV MAPE (%)']:.2f}%" if pd.notna(row["Avg CV MAPE (%)"]) else "N/A"

        # Use a dictionary to handle potential NaNs from baseline
        metrics = {
            "r2": f"{row['Avg CV R²']:.4f}",
            "rmse": f"{row['Avg CV RMSE']:.4f}" if pd.notna(row["Avg CV RMSE"]) else "N/A",
            "mae": f"{row['Avg CV MAE']:.4f}" if pd.notna(row["Avg CV MAE"]) else "N/A",
            "mape": mape_str,
        }

        table.add_row(
            f"#{i + 1}",
            row["Model"],
            f"[{family_colors.get(row['Model Family'], 'white')}]{row['Model Family']}[/]",
            metrics["r2"],
            metrics["rmse"],
            metrics["mae"],
            metrics["mape"],
            style=row_style,
        )

    console.print(table)


def generate_finalist_horizon_data(
    results_files: List[str],
    finalists: List[str],
    baseline_results: Dict = None,
    metrics: List[str] = ["r2", "rmse", "mae"],
) -> pd.DataFrame:
    """(Version 2.0: Now correctly integrates baseline results)"""
    all_results = {}
    if baseline_results:
        all_results.update(baseline_results)
        finalists.append("Baseline (Persistence)")  # Ensure baseline is included

    for file_path in results_files:
        try:
            results = joblib.load(file_path)
            all_results.update(results)
        except FileNotFoundError:
            continue

    horizon_data = []
    for model_name, model_results in all_results.items():
        if model_name in finalists:
            for horizon, horizon_data_raw in model_results.items():
                row = {"Model": model_name, "Horizon": horizon}
                for metric in metrics:
                    mean_score = np.mean(horizon_data_raw["scores"][metric])
                    row[metric.upper()] = mean_score
                horizon_data.append(row)

    df = pd.DataFrame(horizon_data)
    df["Horizon"] = pd.Categorical(df["Horizon"], categories=[f"t+{i}" for i in range(1, 6)], ordered=True)
    df = df.sort_values(by=["Model", "Horizon"])
    return df


def display_finalist_deep_dive_plots(df_horizon_data: pd.DataFrame):
    """
    Creates a set of high-impact, faceted dot plots to compare finalist model
    performance across all five forecast horizons for multiple key metrics.
    (Version 3.0: Corrected colors, axes, sorting, and added baseline context).
    """
    # --- 1. Data Preparation ---
    model_order = ["RidgeCV", "CatBoostRegressor", "Baseline (Persistence)"]
    df_plot_data = df_horizon_data[df_horizon_data["Model"].isin(model_order)].copy()
    df_plot_data["Model"] = pd.Categorical(df_plot_data["Model"], categories=model_order, ordered=True)

    family_colors_map = {"RidgeCV": "#0771A4", "CatBoostRegressor": "#E7B142", "Baseline (Persistence)": "#963D4D"}

    # --- 2. Create Subplots ---
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.15,
        subplot_titles=(
            "<b>Mean CV R² (Higher is Better)</b>",
            "<b>Mean CV RMSE (Lower is Better)</b>",
            "<b>Mean CV MAE (Lower is Better)</b>",
        ),
    )

    # --- 3. Populate Plots ---
    metrics = ["R2", "RMSE", "MAE"]
    for i, metric in enumerate(metrics):
        row_num = i + 1
        # Draw dumbbell lines first
        for horizon in df_plot_data["Horizon"].unique():
            horizon_df = df_plot_data[df_plot_data["Horizon"] == horizon]
            fig.add_trace(
                go.Scatter(
                    x=[horizon_df[metric].min(), horizon_df[metric].max()],
                    y=[horizon, horizon],
                    mode="lines",
                    line=dict(color="lightgrey", width=4),
                    showlegend=False,
                ),
                row=row_num,
                col=1,
            )
        # Draw model performance dots on top
        for model in model_order:
            model_df = df_plot_data[df_plot_data["Model"] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_df[metric],
                    y=model_df["Horizon"],
                    mode="markers",
                    name=model,
                    marker=dict(color=family_colors_map.get(model), size=14, line=dict(width=1.5, color="black")),
                    legendgroup=model,
                    showlegend=(i == 0),
                ),
                row=row_num,
                col=1,
            )

    # --- 4. Final Layout ---
    fig.update_layout(
        title_text="<b>Stage 2 Deep-Dive: Linear vs. Tree-based Performance by Horizon</b><br><sup>Comparing the best model from each family against each other, for each forecast horizon.</sup>",
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Metric Value", row=3, col=1)
    fig.update_traces(hovertemplate="<b>%{x:.4f}</b>")

    fig.show()


def generate_finalist_fold_data(
    results_files: List[str],
    finalists: List[str],
    horizons_to_plot: List[str] = ["t+1", "t+5"],
) -> pd.DataFrame:
    """
    Extracts the per-fold R² scores for specified models and horizons to
    allow for a granular performance comparison.
    """
    all_results = {}
    for file_path in results_files:
        try:
            results = joblib.load(file_path)
            all_results.update(results)
        except FileNotFoundError:
            continue

    fold_data = []
    for model_name, model_results in all_results.items():
        if model_name in finalists:
            for horizon, horizon_data_raw in model_results.items():
                if horizon in horizons_to_plot:
                    for i, r2_score in enumerate(horizon_data_raw["scores"]["r2"]):
                        fold_data.append({
                            "Model": model_name,
                            "Horizon": horizon,
                            "Fold": f"Fold {i + 1}",
                            "R2": r2_score,
                        })

    df = pd.DataFrame(fold_data)
    df["Horizon"] = pd.Categorical(df["Horizon"], categories=horizons_to_plot, ordered=True)
    df["Fold"] = pd.Categorical(df["Fold"], categories=[f"Fold {i + 1}" for i in range(5)], ordered=True)
    return df.sort_values(by=["Model", "Horizon", "Fold"])


def display_finalist_fold_plot(df_fold_data: pd.DataFrame):
    """
    Generates an enhanced, publication-quality faceted line plot to compare per-fold R² scores.
    (Version 3.0: Dynamic model coloring, titles, and improved layout).
    """
    # --- 1. Dynamic Setup ---
    horizons = df_fold_data["Horizon"].unique().tolist()
    models_to_plot = df_fold_data["Model"].unique().tolist()

    # Create a dynamic color map to ensure consistency
    # Uses the 'te' template's colorway and cycles through it
    template_colors = pio.templates["te"].layout.colorway
    model_colors = {model: template_colors[i % len(template_colors)] for i, model in enumerate(models_to_plot)}

    # --- 2. Create Subplots ---
    vertical_spacing = -0.05 * len(horizons) + 0.3
    fig = make_subplots(
        rows=len(horizons),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=[f"<b>Performance on Horizon: {h}</b>" for h in horizons],
    )

    # --- 3. Populate Plots ---
    for i, horizon in enumerate(horizons):
        row_num = i + 1
        horizon_df = df_fold_data[df_fold_data["Horizon"] == horizon]

        for model_name in models_to_plot:
            model_df = horizon_df[horizon_df["Model"] == model_name]
            if model_df.empty:
                continue

            # Add the main performance line
            fig.add_trace(
                go.Scatter(
                    x=model_df["Fold"],
                    y=model_df["R2"],
                    mode="lines+markers",
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=(i == 0),
                    line=dict(color=model_colors.get(model_name), width=2.5),
                    marker=dict(size=8),
                ),
                row=row_num,
                col=1,
            )

            # Add the mean performance line for context
            mean_r2 = model_df["R2"].mean()
            fig.add_hline(
                y=mean_r2,
                line=dict(color=model_colors.get(model_name), width=1.5, dash="dash"),
                opacity=0.7,
                row=row_num,
                col=1,
            )

    # --- 4. Final Dynamic Layout & Annotations ---
    # Create dynamic subtitle text based on the horizons being plotted
    horizon_text = " and ".join(horizons)
    subtitle = (
        f"<sup>Comparing per-fold R² for horizon(s): {horizon_text}. Dashed lines indicate mean performance.</sup>"
    )

    fig.update_layout(
        title_text=f"<b>Deep-Dive: Model Stability & Performance Across CV Folds</b><br>{subtitle}",
        height=400 * len(horizons),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, r=40),
        xaxis_title="Cross-Validation Fold",
        # Dynamically set the title for the bottom-most x-axis
        **{f"xaxis{len(horizons)}_title": "Cross-Validation Fold"},
    )

    # Unify y-axis titles
    fig.update_yaxes(title_text="CV R² Score")

    return fig


def evaluate_tuned_models(
    models_to_evaluate: Dict,
    tuned_params_paths: Dict,
    X_train_linear: pd.DataFrame,
    X_train_tree: pd.DataFrame,
    y_train: pd.DataFrame,
    cv_splitter,
    horizons: List[int],
) -> Dict:
    """
    (Version 3.0 - Final)
    Evaluates a set of models that use pre-tuned, per-horizon parameters.
    Now supports SimpleAveragingEnsemble.
    """
    evaluation_results = {}

    for model_name, model_config in tqdm(models_to_evaluate.items(), desc="Evaluating Tuned Models"):
        model_results = {}

        # Simple Averaging has a different structure and doesn't need pre-loaded params for itself
        if model_config["type"] != "simple_averaging":
            params_path = tuned_params_paths.get(model_config["tree_model_name"])
            if not params_path or not os.path.exists(params_path):
                print(f"Warning: Tuned params not found for {model_config['tree_model_name']}. Skipping {model_name}.")
                continue
            tuned_params_all_horizons = joblib.load(params_path)

        for h in tqdm(horizons, desc=f"Evaluating {model_name}", leave=False):
            horizon_name = f"t+{h}"
            y_train_horizon = y_train[f"target_temp_{horizon_name}"]

            # --- Manual CV Loop to Calculate All Metrics ---
            fold_scores = {"r2": [], "rmse": [], "mae": [], "mape": []}
            for train_idx, val_idx in cv_splitter.split(X_train_linear):
                y_train_fold, y_val_fold = y_train_horizon.iloc[train_idx], y_train_horizon.iloc[val_idx]

                # --- Model-specific instantiation and fitting logic ---
                if model_config["type"] == "simple_averaging":
                    linear_base = model_config["linear_model_class"]().fit(X_train_linear.iloc[train_idx], y_train_fold)

                    tree_params_path = tuned_params_paths[model_config["tree_model_name"]]
                    tree_params_all = joblib.load(tree_params_path)
                    params_for_horizon = tree_params_all[horizon_name]

                    if model_config["tree_model_name"] == "CatBoostRegressor":
                        tree_base_class = CatBoostRegressor
                        tree_base_params = {"verbose": 0, "random_state": 105, "thread_count": -1}
                    else:  # LGBM
                        tree_base_class = LGBMRegressor
                        tree_base_params = {"verbose": -1, "random_state": 105, "n_jobs": -1}

                    tree_base = tree_base_class(**params_for_horizon, **tree_base_params).fit(
                        X_train_tree.iloc[train_idx], y_train_fold
                    )

                    pred_linear = linear_base.predict(X_train_linear.iloc[val_idx])
                    pred_tree = tree_base.predict(X_train_tree.iloc[val_idx])
                    y_pred_fold = (pred_linear + pred_tree) / 2.0

                else:  # Stacking or Single
                    params_for_horizon = tuned_params_all_horizons[horizon_name]
                    if model_config["tree_model_name"] == "CatBoostRegressor":
                        tuned_tree_model = CatBoostRegressor(
                            **params_for_horizon, verbose=0, random_state=105, thread_count=-1
                        )
                    else:  # LGBM
                        tuned_tree_model = LGBMRegressor(**params_for_horizon, verbose=-1, random_state=105, n_jobs=-1)

                    if model_config["type"] == "single":
                        model_instance = tuned_tree_model
                        X_train_fold, X_val_fold = X_train_tree.iloc[train_idx], X_train_tree.iloc[val_idx]
                    else:  # Stacking
                        model_instance = StackingEnsemble(
                            base_model_linear=model_config["linear_model_class"](),
                            base_model_tree=tuned_tree_model,
                            meta_learner=model_config["meta_learner_class"](),
                        )
                        X_train_fold = {"linear": X_train_linear.iloc[train_idx], "tree": X_train_tree.iloc[train_idx]}
                        X_val_fold = {"linear": X_train_linear.iloc[val_idx], "tree": X_train_tree.iloc[val_idx]}

                    model_instance.fit(X_train_fold, y_train_fold)
                    y_pred_fold = model_instance.predict(X_val_fold)

                # --- Shared evaluation logic ---
                fold_scores["r2"].append(r2_score(y_val_fold, y_pred_fold))
                fold_scores["rmse"].append(np.sqrt(mean_squared_error(y_val_fold, y_pred_fold)))
                fold_scores["mae"].append(mean_absolute_error(y_val_fold, y_pred_fold))
                fold_scores["mape"].append(_calculate_mape(y_val_fold.values, y_pred_fold))

            model_results[horizon_name] = {"scores": fold_scores}

        evaluation_results[model_name] = model_results

    return evaluation_results


def _get_model_instance(model_class, params=None):
    """
    A factory function to instantiate models with sensible, centralized defaults.
    """
    if params is None:
        params = {}

    # Centralized default keyword arguments for our models
    DEFAULT_KWARGS = {
        RidgeCV: {"cv": TimeSeriesSplit(n_splits=5)},
        ElasticNetCV: {"cv": TimeSeriesSplit(n_splits=5), "n_jobs": -1, "random_state": 105},
        CatBoostRegressor: {"verbose": 0, "random_state": 105, "thread_count": -1},
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
        LinearRegression: {"n_jobs": -1},
        XGBRegressor: {"random_state": 105, "n_jobs": -1},
    }

    # Get the default kwargs for the given class, if any
    kwargs = DEFAULT_KWARGS.get(model_class, {})

    # The provided params (e.g., from Optuna) override the defaults
    kwargs.update(params)

    return model_class(**kwargs)


def evaluate_models(
    models_to_evaluate: Dict,
    X_train_linear: pd.DataFrame,
    X_train_tree: pd.DataFrame,
    y_train: pd.DataFrame,
    cv_splitter,
    horizons: List[int],
) -> Dict:
    """
    Universal Evaluator - v3.0 of the evaluate tuned models function
    Evaluates a dictionary of models with diverse configurations (OOB, tuned, ensembles),
    calculating and returning all key metrics for each fold.
    """
    evaluation_results = {}

    for model_name, model_config in tqdm(models_to_evaluate.items(), desc="Evaluating Models"):
        model_results = {}
        model_type = model_config.get("type", "oob")

        # Load parameters if the model type requires them
        if "tuned" in model_type:
            params_path = model_config.get("params_path")
            if not params_path or not os.path.exists(params_path):
                print(f"Warning: Tuned params file not found at '{params_path}'. Skipping {model_name}.")
                continue
            tuned_params_all_horizons = joblib.load(params_path)

        for h in tqdm(horizons, desc=f"Evaluating {model_name}", leave=False):
            horizon_name = f"t+{h}"
            y_train_horizon = y_train[f"target_temp_{horizon_name}"]

            fold_scores = {"r2": [], "rmse": [], "mae": [], "mape": []}
            for train_idx, val_idx in cv_splitter.split(X_train_linear):
                y_train_fold, y_val_fold = y_train_horizon.iloc[train_idx], y_train_horizon.iloc[val_idx]

                # --- Model-specific instantiation and fitting logic ---

                # A. SIMPLE AVERAGING LOGIC
                if model_type == "simple_averaging_tuned":
                    # Use the factory to get instances with defaults
                    linear_base = _get_model_instance(model_config["linear_model_class"]).fit(
                        X_train_linear.iloc[train_idx], y_train_fold
                    )

                    tree_params_for_horizon = tuned_params_all_horizons[horizon_name]
                    # Use the factory and override with tuned params
                    tree_base = _get_model_instance(
                        model_config["tree_model_class"], params=tree_params_for_horizon
                    ).fit(X_train_tree.iloc[train_idx], y_train_fold)

                    pred_linear = linear_base.predict(X_train_linear.iloc[val_idx])
                    pred_tree = tree_base.predict(X_train_tree.iloc[val_idx])
                    y_pred_fold = (pred_linear + pred_tree) / 2.0

                # B. WEIGHTED AVERAGING LOGIC
                elif model_type == "weighted_averaging_tuned":
                    linear_base = _get_model_instance(model_config["linear_model_class"]).fit(
                        X_train_linear.iloc[train_idx], y_train_fold
                    )
                    tree_params_for_horizon = tuned_params_all_horizons[horizon_name]
                    tree_base = _get_model_instance(
                        model_config["tree_model_class"], params=tree_params_for_horizon
                    ).fit(X_train_tree.iloc[train_idx], y_train_fold)

                    pred_linear_val = linear_base.predict(X_train_linear.iloc[val_idx])
                    pred_tree_val = tree_base.predict(X_train_tree.iloc[val_idx])

                    # Grid search for the best weight for this fold
                    best_fold_score = -np.inf
                    best_fold_weight = 0.5
                    for w in np.linspace(0, 1, 21):  # Search weights from 0.0 to 1.0 in 0.05 increments
                        y_pred_fold_candidate = (w * pred_linear_val) + ((1 - w) * pred_tree_val)
                        score = r2_score(y_val_fold, y_pred_fold_candidate)
                        if score > best_fold_score:
                            best_fold_score = score
                            best_fold_weight = w

                    # The prediction for this fold uses the best weight found for this fold
                    y_pred_fold = (best_fold_weight * pred_linear_val) + ((1 - best_fold_weight) * pred_tree_val)

                    # Store the optimal weight to analyze later
                    if "optimal_weights" not in fold_scores:
                        fold_scores["optimal_weights"] = []
                    fold_scores["optimal_weights"].append(best_fold_weight)

                # C. STACKING / SINGLE / OOB LOGIC
                else:
                    if "tuned" in model_type:
                        tree_params_for_horizon = tuned_params_all_horizons[horizon_name]
                        tree_base_model = _get_model_instance(
                            model_config["tree_model_class"], params=tree_params_for_horizon
                        )
                    else:  # OOB model
                        tree_base_model = _get_model_instance(model_config["model_class"])

                    if model_type == "stacking_tuned":
                        model_instance = StackingEnsemble(
                            base_model_linear=_get_model_instance(model_config["linear_model_class"]),
                            base_model_tree=tree_base_model,
                            meta_learner=_get_model_instance(model_config["meta_learner_class"]),
                        )
                        X_train_fold, X_val_fold = (
                            {"linear": X_train_linear.iloc[train_idx], "tree": X_train_tree.iloc[train_idx]},
                            {"linear": X_train_linear.iloc[val_idx], "tree": X_train_tree.iloc[val_idx]},
                        )

                    else:  # Single Tuned or OOB
                        model_instance = (
                            tree_base_model
                            if "tuned" in model_type
                            else _get_model_instance(model_config["model_class"])
                        )
                        feature_set = model_config.get("feature_set", "tree")
                        X_to_use = X_train_tree if feature_set == "tree" else X_train_linear
                        X_train_fold, X_val_fold = X_to_use.iloc[train_idx], X_to_use.iloc[val_idx]

                    model_instance.fit(X_train_fold, y_train_fold)
                    y_pred_fold = model_instance.predict(X_val_fold)

                # --- Shared evaluation logic ---
                fold_scores["r2"].append(r2_score(y_val_fold, y_pred_fold))
                fold_scores["rmse"].append(np.sqrt(mean_squared_error(y_val_fold, y_pred_fold)))
                fold_scores["mae"].append(mean_absolute_error(y_val_fold, y_pred_fold))
                fold_scores["mape"].append(_calculate_mape(y_val_fold.values, y_pred_fold))

            model_results[horizon_name] = {"scores": fold_scores}
        evaluation_results[model_name] = model_results

    return evaluation_results


def analyze_hyperparameter_importance(
    base_study_name: str,
    horizons: List[int],
    num_runs: int,
) -> pd.DataFrame:
    """
    Loads all study databases for a given model, calculates the average fANOVA
    importance for each hyperparameter across all runs for each horizon,
    and returns a consolidated DataFrame.
    """
    console = Console()
    all_importances = defaultdict(lambda: defaultdict(list))

    for h in horizons:
        for i in range(num_runs):
            study_name = f"{base_study_name}_h{h}_run{i + 1}"
            storage_path = f"sqlite:///experiments/{study_name}.db"
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_path)
                if len(study.trials) > 1:  # Need at least 2 trials for importance
                    evaluator = FanovaImportanceEvaluator()
                    importances = evaluator.evaluate(study)
                    for param, value in importances.items():
                        all_importances[h][param].append(value)
            except (KeyError, ValueError) as e:
                console.print(
                    f"  - [yellow]Warning:[/yellow] Could not process study '{study_name}'. Skipping. Reason: {e}"
                )
                continue

    # Average the importances
    avg_importances_list = []
    for h, params in all_importances.items():
        for param, values in params.items():
            avg_importances_list.append({"horizon": f"t+{h}", "param": param, "importance": np.mean(values)})

    df = pd.DataFrame(avg_importances_list)
    # Normalize importance to be a percentage within each horizon
    df["importance_pct"] = df.groupby("horizon")["importance"].transform(lambda x: (x / x.sum()) * 100)
    return df


def _get_text_color_for_bg(bg_color: str) -> str:
    """Determines if text should be light or dark based on background color luminance."""
    hex_color = bg_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white"
    # return "white" if luminance < 0.5 else "#38545f"


def plot_importance_bump_chart(df_importance: pd.DataFrame) -> go.Figure:
    """
    Plots all available hyperparameters with enhanced text placement and a high-contrast custom legend.
    """
    # 1. Data Preparation (No filtering, use all params)
    all_params = df_importance["param"].unique().tolist()
    df_plot = df_importance.copy()
    horizons = sorted(df_plot["horizon"].unique(), key=lambda h: int(h.split("+")[1]))
    num_params = len(all_params)

    # 2. Styling & Color Configuration
    param_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_map = {param: param_colors[i % len(param_colors)] for i, param in enumerate(all_params)}

    fig = go.Figure()
    shapes = []
    annotations = []

    # 3. Draw Bezier curve connectors
    for param in all_params:
        param_df = df_plot[df_plot["param"] == param].set_index("horizon").reindex(horizons)
        for i in range(len(horizons) - 1):
            h_start, h_end = horizons[i], horizons[i + 1]
            rank_start, rank_end = param_df.loc[h_start, "rank"], param_df.loc[h_end, "rank"]
            imp_start, imp_end = param_df.loc[h_start, "importance_pct"], param_df.loc[h_end, "importance_pct"]

            if pd.notna(rank_start) and pd.notna(rank_end):
                avg_importance = (imp_start + imp_end) / 2
                shapes.append(
                    go.layout.Shape(
                        type="path",
                        path=f"M {i},{rank_start} C {i + 0.5},{rank_start} {i + 0.5},{rank_end} {i + 1},{rank_end}",
                        line=dict(color=color_map[param], width=avg_importance / 2.0),
                        opacity=0.8,
                        layer="below",
                    )
                )

    # 4. Draw the dots
    for param in all_params:
        param_df = df_plot[df_plot["param"] == param]
        fig.add_trace(
            go.Scatter(
                x=param_df["horizon"],
                y=param_df["rank"],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color=color_map[param],
                    size=param_df["importance_pct"] / 1.2,
                    line=dict(width=2, color="white"),
                ),
                hoverinfo="text",
                hovertemplate=f"<b>{param}</b><br>Rank: {int(param_df['rank'].iloc[0]) if not param_df.empty else 'N/A'}<br>Importance: {param_df['importance_pct'].round(1).iloc[0] if not param_df.empty else 'N/A'}%<extra></extra>",
                showlegend=False,
            )
        )

    # 5. Add Text Labels (on top of dots)
    for _, row in df_plot.iterrows():
        annotations.append(
            dict(
                x=row["horizon"],
                y=row["rank"],
                text=f"<b>{row['importance_pct']:.1f}%</b>",
                showarrow=False,
                yanchor="bottom",
                yshift=4 + (row["importance_pct"] / 3.0),
                font=dict(color=color_map[row["param"]], size=12),
            )
        )

    # 6. Create the Custom "Pill" Legend
    t1_ranks = df_plot[df_plot["horizon"] == "t+1"].set_index("param")["rank"]
    sorted_params_by_t1_rank = t1_ranks.sort_values().index.tolist()

    for param in sorted_params_by_t1_rank:
        rank_at_t1 = t1_ranks.get(param)
        if pd.notna(rank_at_t1):
            bg_color = color_map[param]
            text_color = _get_text_color_for_bg(bg_color)
            annotations.append(
                dict(
                    xref="paper",
                    x=-0.017,
                    xanchor="right",
                    y=rank_at_t1,
                    yanchor="middle",
                    text=f"<b> {param} </b>",
                    showarrow=False,
                    font=dict(color=text_color, size=14),
                    bgcolor=bg_color,
                    borderpad=5,
                )
            )

    # 7. Final Layout
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        title_text="<b>Shifting Priorities: Hyperparameter Importance Across Horizons</b><br><sup>fANOVA importance (%) for all tuned parameters. Dot size and line width reflect importance.</sup>",
        height=900,
        plot_bgcolor="#f7fafa",
        paper_bgcolor="#f7fafa",
        showlegend=False,
        hovermode=False,
        xaxis=dict(title="Forecast Horizon", showgrid=False, zeroline=False, tickfont=dict(size=12)),
        yaxis=dict(
            title="",
            showgrid=True,
            gridwidth=1,
            gridcolor="#D9D9D9",
            zeroline=False,
            autorange="reversed",
            range=[num_params + 0.5, 0.5],  # Dynamically set range
            tickvals=list(range(1, num_params + 1)),
            tickfont=dict(size=12),
        ),
        margin=dict(l=160, r=50, t=80, b=80),
    )
    return fig


class CalibratedRegressor(BaseEstimator, RegressorMixin):
    """
    (Version 3.0 - Correct Implementation)
    A meta-estimator that calibrates a base regressor using Isotonic Regression.
    It learns the calibration mapping on a chronological hold-out set.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        # We need to handle both dict and DataFrame inputs
        if isinstance(X, dict):
            # Use chronological split for calibration set
            calib_split_idx = int(len(X["linear"]) * 0.7)  # 70% for training, 30% for calibration
            train_indices = X["linear"].index[:calib_split_idx]
            calib_indices = X["linear"].index[calib_split_idx:]

            X_train = {"linear": X["linear"].loc[train_indices], "tree": X["tree"].loc[train_indices]}
            y_train = y.loc[train_indices]
            X_calib = {"linear": X["linear"].loc[calib_indices], "tree": X["tree"].loc[calib_indices]}
            y_calib = y.loc[calib_indices]
        else:
            calib_split_idx = int(len(X) * 0.7)
            X_train, X_calib = X.iloc[:calib_split_idx], X.iloc[calib_split_idx:]
            y_train, y_calib = y.iloc[:calib_split_idx], y.iloc[calib_split_idx:]

        # 1. Fit the base estimator on the first part of the training data
        self.base_estimator_ = clone(self.base_estimator).fit(X_train, y_train)

        # 2. Generate uncalibrated predictions on the hold-out calibration set
        uncalibrated_preds = self.base_estimator_.predict(X_calib)

        # 3. Fit the Isotonic Regressor to learn the calibration map.
        # It learns to map the uncalibrated predictions to the true values.
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_.fit(uncalibrated_preds, y_calib)

        return self

    def predict(self, X):
        # First, get the base model's prediction
        uncalibrated_pred = self.base_estimator_.predict(X)
        # Then, apply the learned calibration
        return self.calibrator_.predict(uncalibrated_pred)


class ResidualStackingEnsemble(BaseEstimator, RegressorMixin):
    """(Version 2.0 - Correct Input Handling for Nested Estimators)"""

    def __init__(self, primary_model, error_model, error_model_feature_set="tree"):
        self.primary_model = primary_model
        self.error_model = error_model
        self.error_model_feature_set = error_model_feature_set

    def fit(self, X, y):
        # The 'X' input to this fit method is now expected to be the full dictionary
        # {'linear': X_lin, 'tree': X_tree}

        # 1. Fit the primary model using the full dictionary
        self.primary_model_ = clone(self.primary_model).fit(X, y)

        # 2. Calculate residuals on the training data
        primary_preds = self.primary_model_.predict(X)
        residuals = y - primary_preds

        # 3. Fit the error model on its designated feature set to predict residuals
        X_for_error_model = X[self.error_model_feature_set]
        self.error_model_ = clone(self.error_model).fit(X_for_error_model, residuals)

        return self

    def predict(self, X):
        # Get the primary prediction using the full dictionary
        primary_pred = self.primary_model_.predict(X)

        # Get the error prediction using its designated feature set
        X_for_error_model = X[self.error_model_feature_set]
        error_pred = self.error_model_.predict(X_for_error_model)

        return primary_pred + error_pred


def evaluate_enhancements(
    base_model_config: Dict,
    enhancement_configs: Dict,
    X_train_linear: pd.DataFrame,
    X_train_tree: pd.DataFrame,
    y_train: pd.DataFrame,
    cv_splitter,
    horizons: List[int],
) -> Dict:
    """
    (Version 2.0 - Scikit-learn Compliant)
    Evaluates a series of enhancements applied to a base champion model.
    This version correctly uses scikit-learn compatible estimators throughout.
    """
    all_results = {}

    # 1. First, evaluate the baseline champion model
    base_model_name = "Champion (Unenhanced)"
    base_model_results = evaluate_models(
        {base_model_name: base_model_config}, X_train_linear, X_train_tree, y_train, cv_splitter, horizons
    )
    all_results.update(base_model_results)

    # 2. Evaluate each enhancement
    for enhancement_name, enhancement_config in enhancement_configs.items():
        results_for_this_enhancement = {}

        base_params_path = base_model_config.get("params_path")
        tuned_params_all_horizons_base = joblib.load(base_params_path) if base_params_path else {}

        error_params_path = enhancement_config.get("error_model_params_path")
        tuned_params_all_horizons_error = joblib.load(error_params_path) if error_params_path else {}

        for h in tqdm(horizons, desc=f"Evaluating Enhancement: {enhancement_name}"):
            horizon_name = f"t+{h}"
            y_train_horizon = y_train[f"target_temp_{horizon_name}"]

            # --- Instantiate the base champion model for this HORIZON ---
            linear_base_model = _get_model_instance(base_model_config["linear_model_class"])
            tree_params_for_horizon = tuned_params_all_horizons_base[horizon_name]
            tree_base_model = _get_model_instance(base_model_config["tree_model_class"], params=tree_params_for_horizon)

            champion_for_horizon = WeightedAverageEnsemble(
                base_model_linear=linear_base_model,
                base_model_tree=tree_base_model,
            )
            # Manually set the optimal weight for this horizon
            champion_for_horizon.optimal_weight_ = base_model_config["horizon_weights"][h]

            # --- Build the final pipeline of enhancements for this HORIZON ---
            enhancement_type = enhancement_config["type"]
            final_model_for_horizon = champion_for_horizon  # Start with the champion

            if "residual_stacking" in enhancement_type:
                error_model_params = tuned_params_all_horizons_error.get(horizon_name, {})
                error_model = _get_model_instance(enhancement_config["error_model_class"], params=error_model_params)

                # NOTE: For residual stacking, we must decide which feature set the error model uses.
                # Here we assume it uses the 'tree' feature set if not specified.
                error_model_X_key = (
                    "tree" if enhancement_config.get("error_model_feature_set") != "linear" else "linear"
                )

                final_model_for_horizon = ResidualStackingEnsemble(
                    primary_model=final_model_for_horizon, error_model=error_model
                )

            if "calibration" in enhancement_type:
                # Wrap the current model with our new, correct CalibratedRegressor
                final_model_for_fold = CalibratedRegressor(base_estimator=final_model_for_horizon)

            # --- Now, evaluate this final, potentially complex, model using a standard CV loop ---
            fold_scores = {"r2": [], "rmse": [], "mae": [], "mape": []}
            for train_idx, val_idx in cv_splitter.split(X_train_linear):
                y_train_fold, y_val_fold = y_train_horizon.iloc[train_idx], y_train_horizon.iloc[val_idx]

                X_train_fold_dict = {"linear": X_train_linear.iloc[train_idx], "tree": X_train_tree.iloc[train_idx]}
                X_val_fold_dict = {"linear": X_train_linear.iloc[val_idx], "tree": X_train_tree.iloc[val_idx]}

                # Fit the final model pipeline
                final_model_for_fold = clone(final_model_for_horizon)

                final_model_for_fold.fit(X_train_fold_dict, y_train_fold)
                y_pred_fold = final_model_for_fold.predict(X_val_fold_dict)

                # --- Shared evaluation logic ---
                fold_scores["r2"].append(r2_score(y_val_fold, y_pred_fold))
                fold_scores["rmse"].append(np.sqrt(mean_squared_error(y_val_fold, y_pred_fold)))
                fold_scores["mae"].append(mean_absolute_error(y_val_fold, y_pred_fold))
                fold_scores["mape"].append(_calculate_mape(y_val_fold.values, y_pred_fold))

            results_for_this_enhancement[horizon_name] = {"scores": fold_scores}
        all_results[enhancement_name] = results_for_this_enhancement

    return all_results


def display_final_performance_table(predictions: Dict, y_test: pd.DataFrame, display_names: Dict = None):
    """
    Calculates final test set metrics and displays them in a grouped,
    publication-quality rich table with averages.
    """
    if display_names is None:
        display_names = {key: key for key in predictions.keys()}

    console = Console()
    table_data = []
    model_keys = list(predictions.keys())
    horizons = y_test.columns

    # --- 1. Calculate Metrics for each horizon ---
    for horizon in horizons:
        row_data = {"Horizon": horizon.replace("target_temp_", "")}
        for model_key in model_keys:
            y_true_h = y_test[horizon]
            y_pred_h = predictions[model_key][horizon.replace("target_temp_", "")]

            row_data[f"{model_key}_R2"] = r2_score(y_true_h, y_pred_h)
            row_data[f"{model_key}_RMSE"] = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
            row_data[f"{model_key}_MAE"] = mean_absolute_error(y_true_h, y_pred_h)
            row_data[f"{model_key}_MAPE"] = _calculate_mape(y_true_h.values, y_pred_h.values)
        table_data.append(row_data)

    # --- 2. Calculate Averages ---
    avg_row = {"Horizon": "[bold]Average[/bold]"}
    for model_key in model_keys:
        for metric in ["R2", "RMSE", "MAE", "MAPE"]:
            avg_row[f"{model_key}_{metric}"] = np.mean([row[f"{model_key}_{metric}"] for row in table_data])

    table_data.append(avg_row)

    # --- 3. Build the Rich Table ---
    table = Table(
        title="[bold]Final Model Performance on Unseen Test Set[/bold]",
        show_header=True,
        header_style="bold #063163",
        show_lines=True,
    )
    table.add_column("Horizon", justify="center", style="#063163", no_wrap=True)

    model_colors = ["#0771A4", "#B58A24"]  # Blue, Darker Yellow
    metrics = {"R² ↑": "R2", "RMSE ↓": "RMSE", "MAE ↓": "MAE", "MAPE ↓": "MAPE"}

    for metric_display, metric_key in metrics.items():
        for i, model_key in enumerate(model_keys):
            model_display_name = display_names.get(model_key, model_key)
            color = model_colors[i % len(model_colors)]
            header_style = f"bold {color}" if metric_key == "R2" else color
            table.add_column(f"{model_display_name}\n({metric_display})", justify="center", style=header_style)

    # Populate rows
    for row in table_data:
        row_cells = [row["Horizon"]]
        for _, metric_key in metrics.items():
            for model_key in model_keys:
                value = row[f"{model_key}_{metric_key}"]
                cell_str = f"{value:.4f}" if metric_key != "MAPE" else f"{value:.2f}%"
                row_cells.append(cell_str)
        table.add_row(*row_cells)

    console.print(table)
