# utils/diagnostics.py

import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import acf, pacf
import statsmodels as sm
from sklearn.inspection import permutation_importance
import shap
from typing import Dict, List, Literal, Union
from rich.console import Console
import os
from tqdm.notebook import tqdm
from utils.evaluations import _get_model_instance
from scipy.stats import norm


def generate_diagnostic_data(
    models_to_run: Dict,
    X_train_linear: pd.DataFrame,
    X_train_tree: pd.DataFrame,
    y_train: pd.DataFrame,
    cv_splitter,
    horizons_to_analyze: List[int],
    artifact_path: str,
) -> Dict:
    """
    Performs a CV run and captures all data (residuals, predictions, full feature sets)
    for deep diagnostics. Includes a robust caching mechanism.
    """
    console = Console()
    if os.path.exists(artifact_path):
        console.print(f"Loading cached diagnostic data from [cyan]{artifact_path}[/cyan]")
        return joblib.load(artifact_path)

    console.print(f"No cache found. Generating new diagnostic data for {list(models_to_run.keys())}...")
    diagnostic_results = {}

    for model_name, model_config in tqdm(models_to_run.items(), desc="Generating Diagnostic Data"):
        model_results = {}

        if model_config.get("params_path"):
            tuned_params_all_horizons = joblib.load(model_config["params_path"])

        for h in tqdm(horizons_to_analyze, desc=f"Analyzing {model_name}", leave=False):
            horizon_name = f"t+{h}"
            y_train_horizon = y_train[f"target_temp_{horizon_name}"]

            # Storage for all data from all folds
            horizon_data_folds = {
                "y_true": [],
                "y_pred": [],
                "residuals": [],
                "X_linear_fold": [],
                "X_tree_fold": [],
            }

            for train_idx, val_idx in cv_splitter.split(X_train_linear):
                y_train_fold, y_val_fold = y_train_horizon.iloc[train_idx], y_train_horizon.iloc[val_idx]

                # --- Model-specific instantiation and fitting ---
                model_type = model_config.get("type")

                if model_type == "weighted_averaging_tuned":
                    linear_base = _get_model_instance(model_config["linear_model_class"]).fit(
                        X_train_linear.iloc[train_idx], y_train_fold
                    )
                    tree_params = tuned_params_all_horizons[horizon_name]
                    tree_base = _get_model_instance(model_config["tree_model_class"], params=tree_params).fit(
                        X_train_tree.iloc[train_idx], y_train_fold
                    )

                    pred_linear_val = linear_base.predict(X_train_linear.iloc[val_idx])
                    pred_tree_val = tree_base.predict(X_train_tree.iloc[val_idx])

                    w = model_config["horizon_weights"][h]
                    y_pred_fold = (w * pred_linear_val) + ((1 - w) * pred_tree_val)

                elif model_type == "oob" and model_config.get("feature_set") == "linear":
                    model_instance = _get_model_instance(model_config["model_class"])
                    model_instance.fit(X_train_linear.iloc[train_idx], y_train_fold)
                    y_pred_fold = model_instance.predict(X_train_linear.iloc[val_idx])

                else:
                    raise NotImplementedError(
                        f"Diagnostic data generation not implemented for model type: {model_type}"
                    )

                # --- Consolidate and store all necessary data for this fold ---
                horizon_data_folds["y_true"].append(y_val_fold)
                horizon_data_folds["y_pred"].append(pd.Series(y_pred_fold, index=y_val_fold.index))
                horizon_data_folds["residuals"].append(y_val_fold - y_pred_fold)
                horizon_data_folds["X_linear_fold"].append(X_train_linear.iloc[val_idx])
                horizon_data_folds["X_tree_fold"].append(X_train_tree.iloc[val_idx])

            # Concatenate results from all folds into single DataFrames/Series for easy access
            model_results[horizon_name] = {
                "y_true": pd.concat(horizon_data_folds["y_true"]),
                "y_pred": pd.concat(horizon_data_folds["y_pred"]),
                "residuals": pd.concat(horizon_data_folds["residuals"]),
                "X_linear": pd.concat(horizon_data_folds["X_linear_fold"]),
                "X_tree": pd.concat(horizon_data_folds["X_tree_fold"]),
            }

        diagnostic_results[model_name] = model_results

    joblib.dump(diagnostic_results, artifact_path)
    console.print(f"Diagnostic data saved to [cyan]{artifact_path}[/cyan]")
    return diagnostic_results


def plot_residuals_vs_time(diagnostic_data: Dict, horizons_to_plot: List[str]):
    """
    Creates a 2x2 faceted plot to compare model residuals over time,
    now with LOESS trendlines AND bootstrapped 95% confidence intervals.
    """
    models = list(diagnostic_data.keys())
    fig = make_subplots(
        rows=len(models),
        cols=len(horizons_to_plot),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[f"Horizon: {h}" for h in horizons_to_plot] * len(models),
        vertical_spacing=0.15,
        horizontal_spacing=0.05,
    )

    model_colors = {models[0]: "#0771A4", models[1]: "#d59b3b"}
    trend_color = "#963d4d"
    N_BOOTSTRAP = 50  # Number of bootstrap samples for CI calculation

    for i, model_name in enumerate(models):
        row_num = i + 1
        for j, horizon in enumerate(horizons_to_plot):
            col_num = j + 1

            data = diagnostic_data[model_name][horizon]
            residuals = data["residuals"].sort_index()

            # 1. Scatter plot of residuals
            fig.add_trace(
                go.Scattergl(
                    x=residuals.index,
                    y=residuals,
                    mode="markers",
                    name=model_name,
                    marker=dict(color=model_colors[model_name], size=3.5, opacity=0.3),
                    showlegend=False,
                ),
                row=row_num,
                col=col_num,
            )

            # 2. Calculate main LOESS trendline
            lowess = sm.api.nonparametric.lowess
            loess_fit = lowess(residuals.values, residuals.index, frac=0.2)
            x_loess = pd.to_datetime(loess_fit[:, 0])
            y_loess = loess_fit[:, 1]

            # 3. Manually bootstrap the confidence interval
            bootstrap_fits = []
            for _ in range(N_BOOTSTRAP):
                sample_indices = np.random.choice(len(residuals), size=len(residuals), replace=True)
                res_sample = residuals.iloc[sample_indices]

                # Fit lowess on the bootstrapped sample
                boot_fit = lowess(res_sample.values, res_sample.index, frac=0.2)

                # Interpolate to the same x-grid as the main fit for comparison
                boot_df = pd.DataFrame(boot_fit, columns=["x", "y"]).sort_values("x")
                y_interp = np.interp(loess_fit[:, 0], boot_df["x"], boot_df["y"])
                bootstrap_fits.append(y_interp)

            bootstrap_fits = np.array(bootstrap_fits)
            ci_lower = np.percentile(bootstrap_fits, 2.5, axis=0)
            ci_upper = np.percentile(bootstrap_fits, 97.5, axis=0)

            # 4. Plot the CI as a filled area
            fig.add_trace(
                go.Scattergl(
                    x=x_loess.tolist() + x_loess.tolist()[::-1],
                    y=ci_upper.tolist() + ci_lower.tolist()[::-1],
                    fill="toself",
                    fillcolor=trend_color,
                    opacity=0.4,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="none",
                    name="95% CI",
                    showlegend=False,
                ),
                row=row_num,
                col=col_num,
            )

            # 5. Plot the main LOESS trendline on top
            fig.add_trace(
                go.Scattergl(
                    x=x_loess,
                    y=y_loess,
                    mode="lines",
                    name="LOESS Trend",
                    line=dict(color=trend_color, width=2),
                    showlegend=False,
                ),
                row=row_num,
                col=col_num,
            )

    # --- Final Layout & Annotations ---
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="black", opacity=0.7)

    fig.add_annotation(
        text="<b>Ensemble</b>",
        xref="paper",
        yref="paper",
        x=-0.085,
        y=0.8,
        showarrow=False,
        font=dict(color=model_colors[models[0]], size=14),
    )
    fig.add_annotation(
        text="<b>RidgeCV</b>",
        xref="paper",
        yref="paper",
        x=-0.085,
        y=0.19,
        showarrow=False,
        font=dict(color=model_colors[models[1]], size=14),
    )

    fig.update_layout(
        title_text="<b>Diagnostic: Residuals vs. Time for Ensemble and Single Models</b><br><sup>Scatter plot shows error distribution. Red LOESS line and 95% CI reveal systematic bias - the CI interval should include the y=0 line.</sup>",
        height=350 * len(models),
        margin=dict(l=140),
    )

    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)

    fig.update_yaxes(title_text="Residual Error (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Residual Error (°C)", row=2, col=1)

    for annotation in fig["layout"]["annotations"]:
        if "Horizon:" in annotation["text"]:
            annotation["y"] = 1.0

    return fig


def plot_acf_pacf(diagnostic_data: Dict, horizons_to_plot: List[str]):
    """
    Creates a 2x2 faceted plot to compare the ACF and PACF of model residuals.
    """
    models = list(diagnostic_data.keys())
    fig = make_subplots(
        rows=len(models),
        cols=2,  # One for ACF, one for PACF
        subplot_titles=(
            "<b>ACF (Autocorrelation)</b>",
            "<b>PACF (Partial Autocorrelation)</b>",
            "",  # No title for the second row's subplots
            "",
        ),
        vertical_spacing=0.2,
        horizontal_spacing=0.1,
    )

    model_colors = {models[0]: "#0771A4", models[1]: "#d59b3b"}
    CI_COLOR = "rgba(184, 213, 230, 0.35)"
    # CI_COLOR = "rgba(150, 150, 150, 0.3)"

    for i, model_name in enumerate(models):
        row_num = i + 1
        for j, horizon in enumerate(horizons_to_plot):
            # We will overlay the horizons on the same plot for direct comparison
            data = diagnostic_data[model_name][horizon]
            residuals = data["residuals"].dropna()

            # --- ACF ---
            acf_vals, confint_acf = acf(residuals, nlags=30, alpha=0.05)
            lags = np.arange(len(acf_vals))

            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([lags, lags[::-1]]),
                    y=np.concatenate([confint_acf[:, 1] - acf_vals, (confint_acf[:, 0] - acf_vals)[::-1]]),
                    fill="toself",
                    fillcolor=CI_COLOR,
                    line=dict(color="rgba(255,255,255,0)"),
                    name="ACF/PACF 95% CI interval",
                    legendgroup="ci",
                    showlegend=(i == 0 and j == 0),
                ),
                row=row_num,
                col=1,
            )
            # Stems and markers
            for k in range(1, len(acf_vals)):
                fig.add_trace(
                    go.Scatter(
                        x=[k, k],
                        y=[0, acf_vals[k]],
                        mode="lines",
                        line_color=model_colors[model_name],
                        showlegend=False,
                    ),
                    row=row_num,
                    col=1,
                )
            fig.add_trace(
                go.Scatter(
                    x=lags[1:],
                    y=acf_vals[1:],
                    mode="markers",
                    name=model_name,
                    legendgroup=model_name,
                    marker_color=model_colors[model_name],
                    showlegend=True,
                ),
                row=row_num,
                col=1,
            )

            # --- PACF ---
            pacf_vals, confint_pacf = pacf(residuals, nlags=30, alpha=0.05)
            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([lags, lags[::-1]]),
                    y=np.concatenate([confint_pacf[:, 1] - pacf_vals, (confint_pacf[:, 0] - pacf_vals)[::-1]]),
                    fill="toself",
                    fillcolor=CI_COLOR,
                    line=dict(color="rgba(255,255,255,0)"),
                    name="PACF 95% CI",
                    legendgroup="ci",
                    showlegend=False,
                ),
                row=row_num,
                col=2,
            )
            # Stems and markers
            for k in range(1, len(pacf_vals)):
                fig.add_trace(
                    go.Scatter(
                        x=[k, k],
                        y=[0, pacf_vals[k]],
                        mode="lines",
                        line_color=model_colors[model_name],
                        showlegend=False,
                    ),
                    row=row_num,
                    col=2,
                )
            fig.add_trace(
                go.Scatter(
                    x=lags[1:],
                    y=pacf_vals[1:],
                    mode="markers",
                    name=model_name,
                    legendgroup=model_name,
                    marker_color=model_colors[model_name],
                    showlegend=False,
                ),
                row=row_num,
                col=2,
            )

    # --- Final Layout & Annotations ---
    fig.add_annotation(
        text=f"<b>{models[0]}</b>",
        xref="paper",
        yref="paper",
        x=-0.1,
        y=0.8,
        showarrow=False,
        font=dict(color=model_colors[models[0]], size=14),
        textangle=-90,
    )
    fig.add_annotation(
        text=f"<b>{models[1]}</b>",
        xref="paper",
        yref="paper",
        x=-0.1,
        y=0.19,
        showarrow=False,
        font=dict(color=model_colors[models[1]], size=14),
        textangle=-90,
    )

    fig.update_layout(
        title_text="<b>Diagnostic: Autocorrelation of Model Residuals (Horizon t+1)</b><br><sup>Checks if errors are random. Significant spikes outside the grey CI indicate remaining patterns.</sup>",
        height=300 * len(models),
        margin=dict(l=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="Correlation", range=[-0.25, 0.25])
    fig.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor="black")

    return fig


def plot_residuals_vs_variable(
    diagnostic_data: Dict,
    horizon_to_plot: str,
    x_variable: str | Literal["predicted"],
):
    """
    (Version 2.0 - Faceted)
    Creates a 2x1 faceted plot to compare model residuals against a specified variable,
    with shared axes for direct visual comparison.
    """
    models = list(diagnostic_data.keys())
    display_names = {"WeightedAvg_Ridge_TunedLGBM": "Ensemble", "RidgeCV": "RidgeCV"}

    fig = make_subplots(
        rows=len(models),
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[f"<b>{display_names.get(m, m)}<b>" for m in models],
        vertical_spacing=0.15,
    )

    model_colors = {models[0]: "#0771A4", models[1]: "#d59b3b"}

    for i, model_name in enumerate(models):
        row_num = i + 1
        data = diagnostic_data[model_name][horizon_to_plot]
        residuals = data["residuals"]

        if x_variable == "predicted":
            x_values = data["y_pred"]
            x_title = "Predicted Temperature (°C)"
            main_title = "Residuals vs. Predicted Values"
        else:
            # Check both feature sets
            if x_variable in data["X_tree"].columns:
                X_to_use = data["X_tree"]
            elif x_variable in data["X_linear"].columns:
                X_to_use = data["X_linear"]
            else:
                raise ValueError(f"Feature '{x_variable}' not found in the diagnostic data.")
            x_values = X_to_use[x_variable]
            x_title = f"Value of '{x_variable}'"
            main_title = f"Residuals vs. '{x_variable}'"

        # 1. ScatterGL plot of residuals for performance
        fig.add_trace(
            go.Scattergl(
                x=x_values,
                y=residuals,
                mode="markers",
                name=display_names.get(model_name, model_name),  # This sets the text in the legend
                marker=dict(color=model_colors[model_name], size=4, opacity=0.3),
                legendgroup=display_names.get(model_name, model_name),  # Group traces by model name
                showlegend=True,  # Show all model names in the legend
            ),
            row=row_num,
            col=1,
        )

        # 2. LOESS trendline
        lowess = sm.api.nonparametric.lowess
        clean_idx = pd.Series(residuals.values).notna() & pd.Series(x_values.values).notna()
        loess_fit = lowess(residuals.values[clean_idx], x_values.values[clean_idx], frac=0.3)

        fig.add_trace(
            go.Scatter(
                x=loess_fit[:, 0],
                y=loess_fit[:, 1],
                mode="lines",
                name="LOESS Trend",  # Generic name for all trendlines
                line=dict(color="#963d4d", width=2.7),
                legendgroup="LOESS Trend",  # Group all trendlines together
                showlegend=(i == 0),  # Only show this legend item once
            ),
            row=row_num,
            col=1,
        )

    # --- Final Layout & Annotations ---
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="black", opacity=0.7)

    fig.update_layout(
        title_text=f"<b>Diagnostic: {main_title} (Horizon: {horizon_to_plot})</b><br><sup>Checks for systematic bias. A perfect model shows a flat trendline at y=0.</sup>",
        height=350 * len(models),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(title_text=x_title, row=len(models), col=1)
    fig.update_yaxes(title_text="Residual Error (°C)", range=[-4, 4])  # Set a fixed range for direct comparison

    # Clean up subplot titles to just be the model name
    for annotation in fig["layout"]["annotations"]:
        annotation["y"] = annotation["y"] + 0.04

    return fig


def generate_shap_vs_perm_data(
    champion_model_config: Dict,
    X_train_linear: pd.DataFrame,
    X_train_tree: pd.DataFrame,
    y_train: pd.DataFrame,
    cv_splitter,
    horizons_to_analyze: List[int],
    artifact_path: str,
    n_repeats: int = 15,
    random_state: int = 105,
    verbose: bool = False,
) -> Dict:
    """
    Generates and caches data for comparing Weighted SHAP importance against
    Permutation Importance for a weighted average ensemble model.
    """
    console = Console()
    if os.path.exists(artifact_path):
        if verbose:
            console.print(f"Loading cached SHAP vs Permutation data from [cyan]{artifact_path}[/cyan]")
        return joblib.load(artifact_path)

    if verbose:
        console.print("No cache found. Generating new SHAP vs Permutation data...")
    all_results = {}

    # Use the last fold of the CV splitter as our hold-out validation set
    all_folds = list(cv_splitter.split(X_train_linear))
    train_idx, val_idx = all_folds[-1]

    # --- Load Tuned Parameters for the Tree Model ---
    tuned_params_all_horizons = joblib.load(champion_model_config["params_path"])

    for h in tqdm(horizons_to_analyze, desc="Analyzing Horizons"):
        horizon_name = f"t+{h}"
        y_train_horizon = y_train[f"target_temp_{horizon_name}"]
        y_train_fold, y_val_fold = y_train_horizon.iloc[train_idx], y_train_horizon.iloc[val_idx]

        # --- 1. Fit Base Models on the Training Fold ---
        linear_base = _get_model_instance(champion_model_config["linear_model_class"]).fit(
            X_train_linear.iloc[train_idx], y_train_fold
        )
        tree_params = tuned_params_all_horizons[horizon_name]
        tree_base = _get_model_instance(champion_model_config["tree_model_class"], params=tree_params).fit(
            X_train_tree.iloc[train_idx], y_train_fold
        )
        w_linear = champion_model_config["horizon_weights"][h]

        # --- 2. Calculate Permutation Importance (as before) ---
        class WeightedAvgPredictor:  # Wrapper class for permutation importance
            def __init__(self, linear_model, tree_model, weight):
                self.linear_model, self.tree_model, self.weight = linear_model, tree_model, weight

            def fit(self, X, y):
                return self

            def predict(self, X_dict):
                pred_linear = self.linear_model.predict(X_dict["linear"])
                pred_tree = self.tree_model.predict(X_dict["tree"])
                return (self.weight * pred_linear) + ((1 - self.weight) * pred_tree)

        model_instance_for_perm = WeightedAvgPredictor(linear_base, tree_base, w_linear)
        X_val_combined = pd.concat([X_train_linear.iloc[val_idx], X_train_tree.iloc[val_idx]], axis=1).loc[
            :, ~pd.concat([X_train_linear.iloc[val_idx], X_train_tree.iloc[val_idx]], axis=1).columns.duplicated()
        ]

        def scoring_func(estimator, X_permuted, y):
            X_perm_dict = {"linear": X_permuted[X_train_linear.columns], "tree": X_permuted[X_train_tree.columns]}
            y_perm_pred = estimator.predict(X_perm_dict)
            return r2_score(y_val_fold, y_perm_pred)

        perm_result = permutation_importance(
            model_instance_for_perm,
            X_val_combined,
            y_val_fold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
            scoring=scoring_func,
        )

        # --- 3. Calculate Weighted SHAP Importance ---
        # SHAP for Linear Component
        explainer_linear = shap.LinearExplainer(linear_base, X_train_linear.iloc[train_idx])
        shap_values_linear = explainer_linear(X_train_linear.iloc[val_idx])
        global_shap_linear = pd.Series(np.abs(shap_values_linear.values).mean(axis=0), index=X_train_linear.columns)

        # SHAP for Tree Component
        explainer_tree = shap.TreeExplainer(tree_base)
        shap_values_tree = explainer_tree(X_train_tree.iloc[val_idx])
        global_shap_tree = pd.Series(np.abs(shap_values_tree.values).mean(axis=0), index=X_train_tree.columns)

        # Combine into a single DataFrame for weighted calculation
        df_shap = pd.DataFrame({
            "shap_linear": global_shap_linear,
            "shap_tree": global_shap_tree,
        }).fillna(0)

        # Calculate the final weighted SHAP importance
        w_tree = 1 - w_linear
        weighted_shap_importance = (w_linear * df_shap["shap_linear"]) + (w_tree * df_shap["shap_tree"])

        # --- 4. Store Results ---
        all_results[horizon_name] = {
            "permutation_importance": perm_result,
            "permutation_feature_names": X_val_combined.columns.tolist(),
            "weighted_shap_importance": weighted_shap_importance.sort_values(ascending=False),
        }

    joblib.dump(all_results, artifact_path)
    if verbose:
        console.print(f"SHAP vs Permutation data saved to [cyan]{artifact_path}[/cyan]")
    return all_results


def plot_shap_vs_perm_comparison(
    shap_vs_perm_data: Dict,
    horizons_to_plot: List[str],
    n_top_features: int = 20,
    sort_by: Union[Literal["shap"], Literal["permutation"]] = "permutation",
):
    """
    Creates a comprehensive faceted bar chart comparing SHAP vs. Permutation
    importance, with correct sorting for all subplots and improved layout.
    """
    if sort_by.lower() not in ["shap", "permutation"]:
        raise ValueError("sort_by must be either 'shap' or 'permutation'")

    num_horizons = len(horizons_to_plot)
    v_spacing = 0.12
    fig = make_subplots(
        rows=num_horizons,
        cols=2,
        horizontal_spacing=0.04,
        vertical_spacing=v_spacing,
    )

    SHAP_COLORSCALE = [[0, "#D0DDE3"], [0.5, "#6c8794"], [1.0, "#38545f"]]
    PERM_COLORSCALE = [[0, "#CFE4E1"], [0.5, "#369D8E"], [1.0, "#194942"]]

    # Pre-calculate all data and sorting orders
    plot_data = {}
    for horizon in horizons_to_plot:
        data = shap_vs_perm_data[horizon]
        df_shap = data["weighted_shap_importance"].to_frame("importance")
        perm_result = data["permutation_importance"]
        df_perm = pd.DataFrame({
            "feature": data["permutation_feature_names"],
            "importance": perm_result["importances_mean"],
        }).set_index("feature")

        sort_df = df_shap if sort_by.lower() == "shap" else df_perm
        y_labels = sort_df.sort_values("importance", ascending=False).head(n_top_features).index.tolist()

        plot_data[horizon] = {"y_labels": y_labels, "df_shap": df_shap, "df_perm": df_perm}

    # Main plotting loop
    for i, horizon in enumerate(horizons_to_plot):
        row_num = i + 1
        horizon_plot_data = plot_data[horizon]
        y_labels = horizon_plot_data["y_labels"]

        df_shap_plot = horizon_plot_data["df_shap"].reindex(y_labels).fillna(0)
        df_shap_plot["normalized"] = df_shap_plot["importance"] / df_shap_plot["importance"].max()

        df_perm_plot = horizon_plot_data["df_perm"].reindex(y_labels).fillna(0)
        df_perm_plot["normalized"] = df_perm_plot["importance"] / df_perm_plot["importance"].max()

        fig.add_trace(
            go.Bar(
                x=df_shap_plot["normalized"],
                y=y_labels,
                orientation="h",
                text=df_shap_plot["normalized"].apply(lambda x: f"{x:.2f}"),
                textposition="outside",
                marker=dict(color=df_shap_plot["normalized"], colorscale=SHAP_COLORSCALE, showscale=False),
                customdata=df_shap_plot["importance"],
                hovertemplate="<b>%{y}</b><br>SHAP (Raw): %{customdata:.4f}<extra></extra>",
            ),
            row=row_num,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df_perm_plot["normalized"],
                y=y_labels,
                orientation="h",
                text=df_perm_plot["normalized"].apply(lambda x: f"{x:.2f}"),
                textposition="outside",
                marker=dict(color=df_perm_plot["normalized"], colorscale=PERM_COLORSCALE, showscale=False),
                customdata=df_perm_plot["importance"],
                hovertemplate="<b>%{y}</b><br>Permutation (R² Drop): %{customdata:.4f}<extra></extra>",
            ),
            row=row_num,
            col=2,
        )

        subplot_height_fraction = (1 - v_spacing * (num_horizons - 1)) / num_horizons
        y_top_edge = 1 - (i * (subplot_height_fraction + v_spacing))
        title_y_position = y_top_edge + 0.007

        xref_col1 = f"x{2 * i + 1} domain" if i > 0 else "x domain"
        xref_col2 = f"x{2 * i + 2} domain"

        # Annotation for Column 1 Title (SHAP)
        fig.add_annotation(
            text="<b>Model's Belief (Weighted SHAP)</b>",
            xref=xref_col1,
            yref="paper",
            x=0.5,
            y=title_y_position,
            showarrow=False,
            font=dict(size=17),
            xanchor="center",
            yanchor="bottom",
        )

        # Annotation for Column 2 Title (Permutation)
        fig.add_annotation(
            text="<b>Actual Impact (Permutation Importance)</b>",
            xref=xref_col2,
            yref="paper",
            x=0.5,
            y=title_y_position,
            showarrow=False,
            font=dict(size=17),
            xanchor="center",
            yanchor="bottom",
        )

    # --- Final Layout & Annotation ---
    total_height_per_row = 40 * n_top_features  # Increased height per bar
    fig.update_layout(
        title_text=f"<b>SHAP vs. Permutation Importance (Sorted by {sort_by.title()})</b><br><sup>Comparing model's internal belief (SHAP) vs. its actual performance impact (Permutation).</sup>",
        height=(total_height_per_row + (v_spacing * total_height_per_row)) * num_horizons + 150,
        showlegend=False,
        margin=dict(l=240, t=140, b=60, r=50),
        hovermode="y unified",
    )
    fig.update_xaxes(
        title_text="Normalized Importance",
        range=[0, 1.15],
        zeroline=True,
        zerolinewidth=1.8,
        zerolinecolor="lightgrey",
    )

    # --- Corrected Sorting and Annotation Logic ---
    for i, horizon in enumerate(horizons_to_plot):
        row_num = i + 1
        y_labels_for_row = plot_data[horizon]["y_labels"]

        # Explicitly set the y-axis for each subplot in the row
        fig.update_yaxes(
            row=row_num,
            col=1,
            autorange="reversed",
            tickfont=dict(size=14),
            categoryorder="array",
            categoryarray=y_labels_for_row,
            ticklabelstandoff=7,
        )
        fig.update_yaxes(
            row=row_num,
            col=2,
            autorange="reversed",
            categoryorder="array",
            categoryarray=y_labels_for_row,
            showticklabels=False,
            ticklabelstandoff=7,
        )

        # Correct annotation positioning
        subplot_height_fraction = (1 - v_spacing * (num_horizons - 1)) / num_horizons
        y_center = 1 - (i * (subplot_height_fraction + v_spacing) + subplot_height_fraction / 2)

        fig.add_annotation(
            text=f"<b>Horizon {horizon}</b>",
            xref="paper",
            yref="paper",
            x=-0.17,
            y=y_center,
            showarrow=False,
            font=dict(size=18),
            textangle=-90,
        )

    return fig


def plot_performance_timeseries_dashboard(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    horizon: str,
    add_rolling_mean: int = None,  # Optional rolling mean window
):
    """
    Creates a high-quality, 2x1 dashboard showing the actual vs. predicted
    time series and the corresponding residuals over time. Can optionally overlay
    a rolling mean of the actual values for context.
    """
    df_plot = pd.DataFrame({"Actual": y_true, "Predicted": y_pred}).sort_index()
    df_plot["Error"] = df_plot["Actual"] - df_plot["Predicted"]

    # Calculate key metrics for the title
    r2 = r2_score(df_plot["Actual"], df_plot["Predicted"])
    rmse = np.sqrt(mean_squared_error(df_plot["Actual"], df_plot["Predicted"]))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.095,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            "<b>Actual vs. Predicted Temperature</b>",
            "<b>Prediction Error (Residuals)</b>",
        ),
    )

    # --- Panel 1: Actual vs. Predicted Time Series ---
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["Actual"],
            name="Actual",
            mode="lines",
            line=dict(color="#0771A4", width=1.8),
            opacity=0.8,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["Predicted"],
            name="Predicted",
            mode="lines",
            line=dict(color="#A22222", width=1.9, dash="dot"),
            opacity=0.9,
        ),
        row=1,
        col=1,
    )

    # --- Add Optional Rolling Mean ---
    if add_rolling_mean and isinstance(add_rolling_mean, int):
        rolling_mean = df_plot["Actual"].rolling(window=add_rolling_mean).mean()
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=rolling_mean,
                name=f"{add_rolling_mean}-Day Avg. (Actual)",
                mode="lines",
                line=dict(color="#E7B142", width=1.7, dash="dash"),
                opacity=0.9,
            ),
            row=1,
            col=1,
        )

    # --- Panel 2: Residuals Over Time ---
    fig.add_trace(
        go.Scatter(
            x=df_plot.index, y=df_plot["Error"], name="Residual", mode="lines", line=dict(color="#38545f", width=1.7)
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="black", opacity=0.8, row=2, col=1)

    # Find and annotate max/min errors
    max_error_idx = df_plot["Error"].idxmax()
    min_error_idx = df_plot["Error"].idxmin()
    max_error_val = df_plot["Error"].max()
    min_error_val = df_plot["Error"].min()

    fig.add_annotation(
        x=max_error_idx,
        y=max_error_val,
        text=f"Max Error:<br>{max_error_val:+.2f}°C",
        showarrow=True,
        arrowhead=2,
        ax=-40,
        ay=-40,
        row=2,
        col=1,
        font=dict(color="#A22222"),
        bgcolor="rgba(255,255,255,0.7)",
    )
    fig.add_annotation(
        x=min_error_idx,
        y=min_error_val,
        text=f"Min Error:<br>{min_error_val:+.2f}°C",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=50,
        row=2,
        col=1,
        font=dict(color="#0771A4"),
        bgcolor="rgba(255,255,255,0.7)",
    )

    # --- Final Layout ---
    fig.update_layout(
        title_text=f"<b>Final Performance Dashboard: {model_name} (Horizon: {horizon})</b><br><sup>Test Set Results: R² = {r2:.4f}, RMSE = {rmse:.4f}°C</sup>",
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(t=100),
    )
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Error (°C)", row=2, col=1)

    return fig


def plot_performance_statistical_dashboard(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    horizon: str,
):
    """
    Creates a high-quality, 1x2 dashboard showing the prediction calibration
    (Actual vs. Predicted scatter) and the distribution of errors.
    """
    df_plot = pd.DataFrame({"Actual": y_true, "Predicted": y_pred}).sort_index()
    df_plot["Error"] = df_plot["Actual"] - df_plot["Predicted"]

    # Calculate key metrics for the title
    r2 = r2_score(df_plot["Actual"], df_plot["Predicted"])
    rmse = np.sqrt(mean_squared_error(df_plot["Actual"], df_plot["Predicted"]))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>Actual vs. Predicted</b>",
            "<b>Distribution of Prediction Errors</b>",
        ),
        horizontal_spacing=0.12,
        column_widths=[0.35, 0.55],
    )

    # --- Panel 1: Actual vs. Predicted Scatter ---
    fig.add_trace(
        go.Scattergl(
            x=df_plot["Predicted"],
            y=df_plot["Actual"],
            mode="markers",
            name="Predictions",
            marker=dict(color="#0771A4", size=5, opacity=0.5),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    # Add the y=x line for a perfect calibration reference
    min_val = min(df_plot["Actual"].min(), df_plot["Predicted"].min())
    max_val = max(df_plot["Actual"].max(), df_plot["Predicted"].max())
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="#A22222", width=2, dash="dash"),
        row=1,
        col=1,
    )

    # --- Panel 2: Error Distribution ---
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=df_plot["Error"],
            name="Error Distribution",
            marker_color="#0771A4",
            opacity=0.6,
            histnorm="probability density",  # Normalize for KDE overlay
        ),
        row=1,
        col=2,
    )
    # KDE Curve
    x_kde = np.linspace(df_plot["Error"].min(), df_plot["Error"].max(), 500)
    kde = sm.nonparametric.kde.KDEUnivariate(df_plot["Error"].values)
    kde.fit()
    fig.add_trace(
        go.Scatter(
            x=x_kde,
            y=kde.evaluate(x_kde),
            mode="lines",
            name="KDE",
            line=dict(color="#A22222", width=2.5),
        ),
        row=1,
        col=2,
    )
    # Normal distribution reference
    mean_err, std_err = df_plot["Error"].mean(), df_plot["Error"].std()
    y_norm = norm.pdf(x_kde, mean_err, std_err)
    fig.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_norm,
            mode="lines",
            name="Normal Dist.",
            line=dict(color="#38545f", width=1.9, dash="dot"),
            opacity=0.8,
        ),
        row=1,
        col=2,
    )

    # --- Final Layout ---
    fig.update_layout(
        title_text=f"<b>Statistical Diagnostics: {model_name} (Horizon: {horizon})</b><br><sup>Test Set Results: R² = {r2:.4f}, RMSE = {rmse:.4f}°C</sup>",
        height=550,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=100),
    )
    fig.update_xaxes(title_text="Predicted Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Actual Temperature (°C)", scaleanchor="x1", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="Error (°C)", row=1, col=2, range=[-4, 4])
    fig.update_yaxes(title_text="Density", row=1, col=2)

    return fig
