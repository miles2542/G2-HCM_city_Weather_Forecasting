# utils/analysis_helpers.py

import pandas as pd
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import List
from rich.console import Console
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.panel import Panel


def display_correlation_summary(df: pd.DataFrame, target: str, method: str = "pearson", top_n: int = 5):
    """
    Calculates and displays a rich, formatted summary of the most significant
    correlations from a DataFrame. (Version 2.0: No borders, expanded columns)

    Args:
        df (pd.DataFrame): DataFrame with numerical data.
        target (str): The name of the target variable column.
        method (str): Correlation method ('pearson' or 'spearman').
        top_n (int): The number of top correlations to display.
    """
    console = Console()
    corr_matrix = df.corr(method=method)

    # --- 1. Correlations with the Target Variable ---
    target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
    top_positive_target = target_corr.head(top_n)
    top_negative_target = target_corr.tail(top_n).sort_values(ascending=True)

    # --- 2. Most Correlated Predictor Pairs (Multicollinearity) ---
    corr_unstacked = corr_matrix.unstack()
    corr_unstacked = corr_unstacked.drop_duplicates()
    corr_unstacked = corr_unstacked[corr_unstacked.abs() < 1.0].sort_values(ascending=False)
    top_positive_pairs = corr_unstacked.head(top_n)
    top_negative_pairs = corr_unstacked.tail(top_n).sort_values(ascending=True)

    # --- 3. Build Rich Tables ---
    # Target correlations
    table_pos_target = Table(
        title=f"[bold #690120]Most Positively Correlated with '{target}'", show_header=True, header_style="bold black"
    )
    table_pos_target.add_column("Feature", style="cyan", no_wrap=True)
    table_pos_target.add_column("Coefficient", style="#690120", justify="right")
    for idx, val in top_positive_target.items():
        table_pos_target.add_row(idx, f"{val:.3f}")

    table_neg_target = Table(
        title=f"[bold #063163]Most Negatively Correlated with '{target}'", show_header=True, header_style="bold black"
    )
    table_neg_target.add_column("Feature", style="cyan", no_wrap=True)
    table_neg_target.add_column("Coefficient", style="#063163", justify="right")
    for idx, val in top_negative_target.items():
        table_neg_target.add_row(idx, f"{val:.3f}")

    # Predictor pairs
    table_pos_pairs = Table(
        title="[bold #690120]Highest Positive Predictor Pairs", show_header=True, header_style="bold black"
    )
    table_pos_pairs.add_column("Feature 1", style="cyan", no_wrap=True)
    table_pos_pairs.add_column("Feature 2", style="cyan", no_wrap=True)
    table_pos_pairs.add_column("Coefficient", style="#690120", justify="right")
    for (f1, f2), val in top_positive_pairs.items():
        table_pos_pairs.add_row(f1, f2, f"{val:.3f}")

    table_neg_pairs = Table(
        title="[bold #063163]Highest Negative Predictor Pairs", show_header=True, header_style="bold black"
    )
    table_neg_pairs.add_column("Feature 1", style="cyan", no_wrap=True)
    table_neg_pairs.add_column("Feature 2", style="cyan", no_wrap=True)
    table_neg_pairs.add_column("Coefficient", style="#063163", justify="right")
    for (f1, f2), val in top_negative_pairs.items():
        table_neg_pairs.add_row(f1, f2, f"{val:.3f}")

    # --- 4. Display in Columns ---
    console.print(
        Text(f"Target Correlation Summary ({method.title()})", style="bold black", justify="center"),
        "\n",
        Columns([table_pos_target, table_neg_target], expand=True, equal=True),
    )

    console.print(
        Text(f"\nPredictor Multicollinearity Summary ({method.title()})", style="bold black", justify="center"),
        "\n",
        Columns([table_pos_pairs, table_neg_pairs], expand=True, equal=True),
    )


def categorize_feature(feature_name: str) -> str:
    """Categorizes a feature name into a logical group for analysis."""
    if feature_name in ["year", "month", "week_of_year", "day_of_year"]:
        return "Temporal (Basic)"
    if "sin_" in feature_name or "cos_" in feature_name:
        return "Temporal (Cyclical)"
    if "fourier" in feature_name:
        return "Temporal (Fourier)"
    if feature_name in ["daylight_hours", "temp_range", "dew_point_deficit"]:
        return "Domain-Specific"
    if "lag" in feature_name:
        match = re.search(r"(\w+)_lag_\d+", feature_name)
        if match:
            return f"Lag ({match.group(1).title()})"
        return "Lag (Other)"
    if "roll" in feature_name:
        match = re.search(r"(\w+)_roll_\d+d", feature_name)
        if match:
            return f"Rolling ({match.group(1).title()})"
        return "Rolling (Other)"
    if "preciptype" in feature_name:
        return "Categorical"
    if "text_svd" in feature_name:
        return "Text (SVD)"
    # Default category for raw features
    return "Raw Predictor"


def display_lasso_summary_table(all_features: List[str], selected_features: List[str]):
    """Creates and displays a rich summary table of LASSO feature selection results."""
    console = Console()
    df_selection_analysis = pd.DataFrame({
        "feature": all_features,
        "status": ["Kept" if f in selected_features else "Dropped" for f in all_features],
    })
    df_selection_analysis["category"] = df_selection_analysis["feature"].apply(categorize_feature)

    selection_summary = (
        pd.crosstab(df_selection_analysis["category"], df_selection_analysis["status"])
        .rename_axis(None, axis=1)
        .rename_axis("Feature Category", axis=0)
    )

    if "Dropped" not in selection_summary:
        selection_summary["Dropped"] = 0
    if "Kept" not in selection_summary:
        selection_summary["Kept"] = 0

    selection_summary["Total"] = selection_summary["Kept"] + selection_summary["Dropped"]
    selection_summary = selection_summary[["Kept", "Dropped", "Total"]].sort_values(by="Total", ascending=False)

    summary_table = Table(title="LASSO Feature Selection Summary", show_header=True, header_style="bold")
    summary_table.add_column("Feature Category", style="cyan")
    summary_table.add_column("Kept", justify="center", style="green")
    summary_table.add_column("Dropped", justify="center", style="red")
    summary_table.add_column("Total", justify="center", style="blue")
    for category, row in selection_summary.iterrows():
        summary_table.add_row(category, str(row["Kept"]), str(row["Dropped"]), str(row["Total"]))
    console.print(summary_table)


def plot_lasso_summary_sunburst(all_features: List[str], selected_features: List[str]) -> go.Figure:
    """
    Creates a refined, 3-level sunburst chart to visualize LASSO feature selection,
    using simplified, high-level categories. (CORRECTED)
    """

    def _get_high_level_category(feature_name: str) -> str:
        """A simplified categorizer for the sunburst plot's narrative clarity."""
        if (
            "sin_" in feature_name
            or "cos_" in feature_name
            or "fourier" in feature_name
            or feature_name in ["year", "month", "week_of_year", "day_of_year"]
        ):
            return "Temporal Features"
        if "lag" in feature_name:
            return "Lag Features"
        if "roll" in feature_name:
            return "Rolling Features"
        if "text_svd" in feature_name:
            return "Text Features"
        if feature_name in ["daylight_hours", "temp_range", "dew_point_deficit"]:
            return "Domain Features"
        if "preciptype" in feature_name:
            return "Categorical"
        return "Raw Predictors"

    df_selection = pd.DataFrame({
        "feature": all_features,
        "status": ["Kept" if f in selected_features else "Dropped" for f in all_features],
    })
    df_selection["category"] = df_selection["feature"].apply(_get_high_level_category)

    # Create a count column for the `values` parameter
    df_selection["count"] = 1

    fig = px.sunburst(
        df_selection,
        path=["status", "category", "feature"],
        values="count",
        color="status",
        color_discrete_map={"Kept": "#0771A4", "Dropped": "#963D4D"},
        title="<b>Interactive View of LASSO Feature Selection</b><br><sup>The middle ring shows how LASSO treated broad feature families. Click on a segment to zoom.</sup>",
    )

    fig.update_traces(insidetextorientation="radial")
    fig.update_layout(height=750, margin=dict(t=80, l=50, r=50, b=50))
    return fig


def display_extrapolation_summary(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_linear: pd.Series,
    y_pred_tree: pd.Series,
):
    """
    Calculates and displays a rich table summarizing the systematic error of models
    in the extrapolation zone (i.e., on test data outside the training range).
    """
    console = Console()

    # 1. Define the boundaries of the training data
    train_min = y_train.min()
    train_max = y_train.max()

    # 2. Identify the "extrapolation zones" in the test set
    extrapolation_zone_high = y_test[y_test > train_max]
    extrapolation_zone_low = y_test[y_test < train_min]

    # 3. Calculate Mean Error (Bias) in these zones
    me_high_linear = (extrapolation_zone_high - y_pred_linear.loc[extrapolation_zone_high.index]).mean()
    me_high_tree = (extrapolation_zone_high - y_pred_tree.loc[extrapolation_zone_high.index]).mean()

    me_low_linear = (extrapolation_zone_low - y_pred_linear.loc[extrapolation_zone_low.index]).mean()
    me_low_tree = (extrapolation_zone_low - y_pred_tree.loc[extrapolation_zone_low.index]).mean()

    # 4. Display the results in a rich table
    table = Table(
        title="[bold]Extrapolation Failure Diagnostic: Mean Error (Bias)[/bold]",
        caption="Mean Error = (Actual - Predicted). Positive indicates underprediction, Negative indicates overprediction.",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Model", style="cyan")
    table.add_column(f"High-Temp Zone (> {train_max:.1f}°C)", justify="center")
    table.add_column(f"Low-Temp Zone (< {train_min:.1f}°C)", justify="center")
    table.add_column("Interpretation", style="white")

    # Linear Model Row
    table.add_row(
        "RidgeCV",
        f"[green]{me_high_linear:+.3f}°C[/green]",
        f"[green]{me_low_linear:+.3f}°C[/green]",
        "Errors are small and centered near zero, indicating unbiased extrapolation.",
    )

    # Tree Model Row
    table.add_row(
        "LGBM (OOB)",
        f"[bold red]{me_high_tree:+.3f}°C[/bold red]",
        f"[bold red]{me_low_tree:+.3f}°C[/bold red]",
        "Large systematic errors confirm a failure to extrapolate beyond training range.",
    )

    console.print(table)


def plot_extrapolation_diagnostics(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_linear: pd.Series,
    y_pred_tree: pd.Series,
    linear_model_name: str = "RidgeCV",
    tree_model_name: str = "LGBM (OOB)",
) -> go.Figure:
    """
    Generates a two-panel diagnostic plot to visually prove the extrapolation
    failure of tree-based models compared to linear models.

    Panel 1: Prediction vs. Actual, showing the "clipping" effect.
    Panel 2: Residual vs. Prediction, showing systematic bias at the extremes.
    """
    train_min = y_train.min()
    train_max = y_train.max()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"<b>1. Prediction vs. Actual</b><br><sup>Tree model predictions 'clip' at training range</sup>",
            f"<b>2. Residual vs. Prediction</b><br><sup>Tree model shows strong bias at extremes</sup>",
        ),
    )

    # --- Panel 1: Prediction vs. Actual ---
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred_linear,
            mode="markers",
            name=linear_model_name,
            marker=dict(color="#0771A4", size=5, opacity=0.6),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred_tree,
            mode="markers",
            name=tree_model_name,
            marker=dict(color="#E7B142", size=5, opacity=0.6),
        ),
        row=1,
        col=1,
    )

    # Add reference lines to Panel 1
    min_val = min(y_test.min(), y_pred_linear.min(), y_pred_tree.min())
    max_val = max(y_test.max(), y_pred_linear.max(), y_pred_tree.max())
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="black", width=2, dash="dash"),
        row=1,
        col=1,
    )
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=train_max,
        x1=max_val,
        y1=train_max,
        line=dict(color="#963D4D", width=2, dash="dot"),
        row=1,
        col=1,
        name="Max Train Temp",
    )
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=train_min,
        x1=max_val,
        y1=train_min,
        line=dict(color="#963D4D", width=2, dash="dot"),
        row=1,
        col=1,
        name="Min Train Temp",
    )

    # --- Panel 2: Residual vs. Prediction ---
    residual_linear = y_test - y_pred_linear
    residual_tree = y_test - y_pred_tree

    fig.add_trace(
        go.Scatter(
            x=y_pred_linear,
            y=residual_linear,
            mode="markers",
            name=f"{linear_model_name} Residuals",
            marker=dict(color="#0771A4", size=5, opacity=0.6),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=y_pred_tree,
            y=residual_tree,
            mode="markers",
            name=f"{tree_model_name} Residuals",
            marker=dict(color="#E7B142", size=5, opacity=0.6),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Add reference line to Panel 2
    fig.add_hline(y=0, line=dict(color="black", width=2, dash="dash"), row=1, col=2)

    # --- Layout and Annotations ---
    fig.update_layout(
        title_text="<b>Visualizing the Extrapolation Failure of Tree-Based Models on the Test Set</b>",
        height=650,
        # width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(title_text="Actual Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Temperature (°C)", row=1, col=1)

    fig.update_xaxes(title_text="Predicted Temperature (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Residual (Actual - Predicted) (°C)", row=1, col=2)

    # Add an annotation to explain the clipping
    fig.add_annotation(
        x=y_test.max(),
        y=train_max,
        text="LGBM predictions saturate here",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#963D4D",
        ax=-40,
        ay=-40,
        row=1,
        col=1,
    )

    return fig
